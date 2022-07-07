#include "bayesship/bayesshipSampler.h"
#include "bayesship/utilities.h"
#include "bayesship/dataUtilities.h"
#include "bayesship/autocorrelationUtilities.h"
#include "bayesship/ThreadPool.h"
#include "bayesship/proposalFunctions.h"
#include "bayesship/standardPriors.h"
#include <cmath>
#include <string>
#include <fstream>
#include <gsl/gsl_randist.h>
#include <nlohmann/json.hpp>



#ifdef _OPENMP
#include <omp.h>
#endif

namespace bayesship{

/*! \file 
 *
 * # bayesshipSampler Source Code
 * 
 * Source code for a majority of the bayesShip sampler routines. This file contains all the definitions for the members and routines for bayesshipSampler objects. This file will also house the definitions for the proposalData class.
 * 
 */


/*! \brief Structure to package swap ``jobs'' for sampling
 *
 * Packages up a job to queue up a chain for swapping
 *
 * See ThreadPool.h to see how this relates to parallel processing, and see the bayesShip.cpp for the implementation of this structure
 */
struct swapJob
{
	/*! Chain ID to be swapped*/
	int chainID;
	/*! samplerData for the sampler storing current data for chainID*/
	samplerData *data;
	/*! The sampler being currently run*/
	bayesshipSampler *sampler;

};

/*! \brief Structure to package sample jobs for sampling
 *
 * Packages up a job to perform Metropolis-Hastings sampling
 * 
 * See ThreadPool.h to see how this relates to parallel processing, and see the bayesShip.cpp for the implementation of this structure
 *
 * */
struct sampleJob
{
	/*! The sampler being currently run*/
	bayesshipSampler *sampler;
	/*! samplerData for the sampler storing current data for chainID*/
	samplerData *data;
	/*! Chain ID to be stepped*/
	int chainID;
	/*! Pool for swapping -- The sampler periodically passes the chain on to be swapped after steping with Metropolis-Hastings*/
	ThreadPoolPair<swapJob> *swapPool;

};


/*! \brief Wrapper function to perform stepMH for sampleJob job and with thread threadID
 *
 * Simply calls stepMH for the chain associated with sampleJob job using thread threadID.
 * 
 * stepMH performs a step in parameter space using the Metropolis-Hastings algorithm.
 */
void sampleThreadedFunctionNoSwap(int threadID, sampleJob job);
void sampleThreadedFunction(int threadID, sampleJob job);
void swapThreadedFunction(int threadID, swapJob job1, swapJob job2);
bool swapPairFunction( swapJob job1, swapJob job2);








/*! \brief Constructor for bayesshipSampler class
 *
 * Needs the likelihood function and the prior function, the bare minimum to run the sampler
 */
//bayesshipSampler::bayesshipSampler( likelihoodFn likelihood, /**< Likelihood function pointer -- See Ptrjmcmc.h for definition of likelihoodFn*/
//	priorFn prior/**< Prior function pointer -- See Ptrjmcmc.h for definition of priorFn*/
//	)
bayesshipSampler::bayesshipSampler( probabilityFn *likelihood, /**< Likelihood function pointer -- See Ptrjmcmc.h for definition of likelihoodFn*/
	probabilityFn *prior/**< Prior function pointer -- See Ptrjmcmc.h for definition of priorFn*/
	)
{
	this->likelihood = likelihood;
	this->prior = prior;
}

/*! \brief Assigns the initial position of the sampler depending on the input arguments
 *
 * Either uses initialPositionEnsemble or initialPosition to populate first positions
 *
 * Populates initial Likelihoods and priors as well.
 */
void bayesshipSampler::assignInitialPosition(samplerData *data)
{
	if(initialPositionEnsemble){
		std::cout<<"Using Initial Ensemble Position"<<std::endl;
		for(int i = 0 ; i<chainN; i++){
			data->positions[i][0]->updatePosition(initialPositionEnsemble[i]);
			//for(int j= 0 ; j<maxDim ; j++){
			//	std::cout<<i<<": "<<data->positions[i][0]->parameters[j];
			//}
			//std::cout<<std::endl;
		}

	}
	else if(initialPosition){
		for(int i = 0 ; i<chainN; i++){
			data->positions[i][0]->updatePosition(initialPosition);
		}

	}
	else{
		errorMessage("Initial position required for sampler -- provide ensemble-wide position, single position, or priorIterations and priorBounds", 1);
	}
	
	/*With initial positions specified, assign prior and likelihood values*/
	for(int i =0; i<chainN; i++){
		//data->priorVals[i][0] = prior(data->positions[i][0],i, this,userParameters[i]);
		data->priorVals[i][0] = prior->eval(data->positions[i][0],i);
		if(data->priorVals[i][0] != limitInf){
			//data->likelihoodVals[i][0] = likelihood(data->positions[i][0],i, this,userParameters[i]);
			data->likelihoodVals[i][0] = likelihood->eval(data->positions[i][0],i);
		}
		else{
			std::cout<<"ERROR -- You have bad initial points!"<<std::endl;
		}
	}
}

/*! \brief Routine to initiate sampling
 *
 * All sampling arguments should be set first through the object member variables, before sampling. See the Ptrjmcmc.h file for all options
 */
void bayesshipSampler::sample()
{
	gsl_error_handler_t *oldHandler = gsl_set_error_handler_off();
	double start = omp_get_wtime();
  		
	/* Overwrite data with checkpoint file if it exists.
 * 		Need memory allocated first
 */
	if( checkDirExist(outputDir+outputFileMoniker+"_checkpoint.json") && !ignoreExistingCheckpoint){
		loadCheckpoint();
	}
	else{
		allocateMemory();
	}

	/*Checks: 
 * 		Threads must be larger than 3 for Thread Pool
 */
	if(independentSamples != 0){
		if(batchSize != 0){
			data = new samplerData(maxDim, ensembleN,ensembleSize, batchSize, proposalFns->proposalN, RJ,betas);
		}
		else{
			data = new samplerData(maxDim, ensembleN,ensembleSize, independentSamples, proposalFns->proposalN, RJ,betas);
		}
	}
	else{
		if( ( iterations < batchSize && batchSize > 0 ) || batchSize == 0  ){
			data = new samplerData(maxDim, ensembleN,ensembleSize, iterations, proposalFns->proposalN, RJ,betas);
		}
		else{
			data = new samplerData(maxDim, ensembleN,ensembleSize, batchSize, proposalFns->proposalN, RJ,betas);
		}
	}
	
	if(priorIterations >0 && priorRanges){
		std::cout<<"Sampling Prior "<<std::endl;

		//likelihoodFn tempL = likelihood;
		probabilityFn *tempL = likelihood;
		likelihood = prior;

		//prior = uniformPrior;
		uniformPrior *tempprior = new uniformPrior();
		tempprior->sampler = this;
		prior = tempprior;

		priorData = new samplerData(maxDim, ensembleN,ensembleSize, priorIterations, proposalFns->proposalN, RJ,betas);
		if(burnPriorIterations >0){
			std::cout<<"Burning in for prior"<<std::endl;
			
			t0 = burnPriorIterations /2.;

			burnPeriod=true;
			adjustTemps=false;

			burnData = new samplerData(maxDim, ensembleN,ensembleSize, burnPriorIterations, proposalFns->proposalN, RJ,betas);
			assignInitialPosition(burnData);
			
			isolateEnsemblesInternal = isolateEnsemblesBurn;

			//swapProb=saveSwapProb;
			sampleLoop(burnPriorIterations,burnData);
			
			burnPeriod=false;

			double betaAverages[ensembleSize];
			for(int j = 0 ; j<ensembleSize; j++){
				betaAverages[j] = 0;	
			}
			for(int i = 1 ; i<ensembleSize-1; i++){
				for(int j = 0 ; j<ensembleN; j++){
					betaAverages[i] += std::log(betas[chainIndex(j,i)]);
				}
				betaAverages[i] /=ensembleN;
			}
			for(int i = 1 ; i<ensembleSize-1; i++){
				for(int j = 0 ; j<ensembleN; j++){
					betas[chainIndex(j,i)] = std::exp(betaAverages[ i]);
				}
			}
			for(int j = 0 ; j<ensembleN; j++){
				betas[chainIndex(j,0)] = 1;
				betas[chainIndex(j,ensembleSize-1)] = 0;
			}
			std::cout<<"Final temperatures after burn in"<<std::endl;
			for(int i = 0 ; i<ensembleSize; i++){
					std::cout<<"Temp: "<<i<<": "<<betas[chainIndex(0,i)]<<std::endl;
			}


			for(int i = 0 ; i<chainN; i++){
				priorData->positions[i][0]->updatePosition(burnData->positions[i][burnData->currentStepID[i]]);
				priorData->likelihoodVals[i][0] = burnData->likelihoodVals[i][burnData->currentStepID[i]];
				priorData->priorVals[i][0] = burnData->priorVals[i][burnData->currentStepID[i]];
			}
			
			

			delete burnData;
			burnData = nullptr;
		}
		else{
			assignInitialPosition(priorData);
		}

		isolateEnsemblesInternal = isolateEnsembles;
		sampleLoop(priorIterations,priorData);

		priorData->updateACs(threads);
		priorData->writeStatFile(outputDir+outputFileMoniker+"Prior_stat.txt");

		delete tempprior;
		prior = likelihood;
		likelihood = tempL;
		
		if(writePriorData){
			priorData->create_data_dump(coldOnlyStorage, true, outputDir+outputFileMoniker+"Prior_output.hdf5");
			
		}
		
	}
	if(burnIterations >0){
		std::cout<<"Burning in "<<std::endl;
		bool saveRandomizeSwappingFlag = randomizeSwapping;
		bool savePool = threadPool;
		double saveSwapProb = swapProb;
		randomizeSwapping = false;
		t0 = burnIterations /8.;

		burnData = new samplerData(maxDim, ensembleN,ensembleSize, burnIterations, proposalFns->proposalN, RJ,betas);
		if(priorData){
		//if(false){
			for(int i = 0 ; i<chainN; i++){
				burnData->positions[i][0]->updatePosition(priorData->positions[i][priorData->currentStepID[i]]);
				//burnData->likelihoodVals[i][0] = likelihood(burnData->positions[i][0],i, this,userParameters[i]);
				burnData->likelihoodVals[i][0] = likelihood->eval(burnData->positions[i][0],i);
				//burnData->priorVals[i][0] = prior(burnData->positions[i][0],i, this,userParameters[i]);
				burnData->priorVals[i][0] = prior->eval(burnData->positions[i][0],i);
			}

		}
		else{
			assignInitialPosition(burnData);
		}
	
		isolateEnsemblesInternal = isolateEnsemblesBurn;

		for( int i = 0 ;i<2;i++){
			burnPeriod=true;
			adjustTemps=true;
			threadPool=false;

			std::cout<<"Adjusting Temps"<<std::endl;

			//swapProb=1;
			//sampleLoop(burnIterations/4,burnData);


			//####################################################
			swapProb=0;
			int steps = 0;
			//int deltaStep = (int)(1./saveSwapProb);
			int deltaStep = 2;
			while(steps < burnIterations/4 - deltaStep){
				sampleLoop(deltaStep,burnData);
				steps+=deltaStep;
				for(int j = 0 ; j<ensembleN; j++){
					for(int k = 0 ; k<ensembleSize-1; k++){

						int ensembleIndex2 = j;
						chainSwap(chainIndex(j,k),chainIndex(ensembleIndex2,k+1),burnData);
				
					}
				}
				if(adjustTemps){
					adjustTemperatures(steps);
					burnData->updateBetas(betas);
				}
			}

			//####################################################
			
			adjustTemps=false;
			swapProb=saveSwapProb;
			threadPool=savePool;
			
			double betaAverages[ensembleSize];
			double acceptAverages[ensembleSize];
			for(int j = 0 ; j<ensembleSize; j++){
				betaAverages[j] = 0;	
				acceptAverages[j] = 0;	
			}
			for(int i = 1 ; i<ensembleSize-1; i++){
				//std::cout<<i<<std::endl;
				for(int j = 0 ; j<ensembleN; j++){
					betaAverages[i] += std::log(betas[chainIndex(j,i)]);
					//std::cout<<betas[chainIndex(j,i)]<<std::endl;
				}
				betaAverages[i]/=ensembleN;
			}
			for(int i = 0 ; i<ensembleSize; i++){
				for(int j = 0 ; j<ensembleN; j++){
					double aveAccRatio =0;
					int ct = 0;
					for(int k = 0 ; k<chainN; k++){
						
						int acc = burnData->swapAccepts[chainIndex(j,i)][k];
						int rej = burnData->swapRejects[chainIndex(j,i)][k];
						if( (acc+rej == 0 ) ){continue;}
						ct++;
						aveAccRatio+=(double)acc/(acc+rej);
					}
				
					acceptAverages[i] += aveAccRatio/ct;
				}
				acceptAverages[i]/=ensembleN;
			}
			for(int i = 1 ; i<ensembleSize-1; i++){
				for(int j = 0 ; j<ensembleN; j++){
					betas[chainIndex(j,i)] = std::exp(betaAverages[i]);
				}
			}
			for(int j = 0 ; j<ensembleN; j++){
				betas[chainIndex(j,0)] = 1;
				betas[chainIndex(j,ensembleSize-1)] = 0;
			}
			for(int i = 0 ; i<ensembleSize; i++){
					
					std::cout<<"Temp/Average Accept: "
						<<i<<": "<<betas[chainIndex(0,i)]<<" "
						<< acceptAverages[i]
						<<std::endl;
			}



			std::cout<<"Exploring"<<std::endl;
			sampleLoop(burnIterations/4,burnData);
		}
		burnPeriod=false;
		swapProb=saveSwapProb;
		std::cout<<"Final temperatures after burn in"<<std::endl;
		for(int i = 0 ; i<ensembleSize; i++){
				std::cout<<"Temp: "<<i<<": "<<betas[chainIndex(0,i)]<<std::endl;
		}



		for(int i = 0 ; i<chainN; i++){
			int currentStep = burnData->currentStepID[i];
			data->positions[i][0]->updatePosition(burnData->positions[i][currentStep]);
			data->likelihoodVals[i][0] = burnData->likelihoodVals[i][currentStep];
			data->priorVals[i][0] = burnData->priorVals[i][currentStep];
			data->currentStepID[i] = 0;
		}
		
		
		randomizeSwapping = saveRandomizeSwappingFlag;
		threadPool=savePool;
	
		//burnData->create_data_dump(coldOnlyStorage, true, outputDir+outputFileMoniker+"Burn_output.hdf5");
	}
	else{
		std::cout<<"Using initial position and skipping burn in"<<std::endl;
		assignInitialPosition(data);
	}

	isolateEnsemblesInternal = isolateEnsembles;
	data->updateBetas(betas);	
	if(independentSamples == 0){
		if( ( iterations < batchSize && batchSize > 0 ) || batchSize == 0  ){
			sampleLoop(iterations,data);
			writeCheckpoint(data);
			if(!RJ){
				data->updateACs(threads);
				int independentSamples = data->countIndependentSamples();
				std::cout<<"Independent samples per chain: "<<independentSamples<<std::endl;
			}
			#ifdef _HDF5
			data->create_data_dump(coldOnlyStorage, true, outputDir+outputFileMoniker+"_output.hdf5");
			#endif
			data->writeStatFile(outputDir+outputFileMoniker+"_stat.txt");
		}
		else  {
			int currentSamples=0;
			int currentIndependentSamples=0;
			bool initializedData=false;
			double AC=0;
			while(currentSamples < iterations){
				sampleLoop(batchSize,data);
				writeCheckpoint(data);
				currentSamples+=batchSize;
	
				if(!RJ){
					data->updateACs(threads);
					currentIndependentSamples = data->countIndependentSamples();
					AC = 0;
					for(int i = 0 ; i<data->ensembleN; i++){
						AC += data->maxACs[i];
					}
					AC /= data->ensembleN;
				}
	

				#ifdef _HDF5
				if(!initializedData){
					data->create_data_dump(coldOnlyStorage, true, outputDir+outputFileMoniker+"_output.hdf5");
					initializedData = true;
				}
				else{
					data->append_to_data_dump(outputDir+outputFileMoniker+"_output.hdf5");

				}
				#endif
				data->writeStatFile(outputDir+outputFileMoniker+"_stat.txt");

				double Lmean = 0;
				double Lmax = data->likelihoodVals[0][0];
				int samples=0;

				for(int i = 0 ; i<ensembleN; i++){
					int maxSteps = data->currentStepID[i];
	
					for(int j = 0 ; j<maxSteps;j++){
						Lmean += data->likelihoodVals[i][j];
						if(data->likelihoodVals[i][j]>Lmax){
							Lmax = data->likelihoodVals[i][j];
						}
						samples++;
					}
				}
				Lmean /= samples;

				if(!RJ){
					std::cout<<"Current independent samples per chain / Average AC: "<<currentIndependentSamples<<" / "<<AC <<std::endl;
				}
				std::cout<<"Mean Log Likelihood / Max Log Likelihood: "<<Lmean<<" / "<<Lmax <<std::endl;
				printProgress( (double) currentSamples/iterations);
				std::cout<<std::endl;
				if(currentSamples<iterations){
					data->extendSize(batchSize-1);
				}
			}
		}
	}
	else{
		int currentIndependentSamples=0;
		bool localBatchSize=false;//Whether to tweak batch size for efficiency
		if(batchSize==0){
			std::cout<<"Optimizing batchSize"<<std::endl;
			batchSize = independentSamples;
			localBatchSize = true;
		}
		bool initializedData=false;
		while(currentIndependentSamples < independentSamples){

			sampleLoop(batchSize,data);
			writeCheckpoint(data);
	
			data->updateACs(threads);
			currentIndependentSamples = data->countIndependentSamples();
			double AC = 0;
			for(int i = 0 ; i<data->ensembleN; i++){
				AC += data->maxACs[i];
			}
			AC /= data->ensembleN;
	

			#ifdef _HDF5
			if(!initializedData){
				data->create_data_dump(coldOnlyStorage, true, outputDir+outputFileMoniker+"_output.hdf5");
				initializedData = true;
			}
			else{
				data->append_to_data_dump(outputDir+outputFileMoniker+"_output.hdf5");

			}
			#endif
			data->writeStatFile(outputDir+outputFileMoniker+"_stat.txt");

			double Lmean = 0;
			double Lmax = data->likelihoodVals[0][0];
			int samples=0;

			for(int i = 0 ; i<ensembleN; i++){
				int maxSteps = data->currentStepID[i];
	
				for(int j = 0 ; j<maxSteps;j++){
					Lmean += data->likelihoodVals[i][j];
					if(data->likelihoodVals[i][j]>Lmax){
						Lmax = data->likelihoodVals[i][j];
					}
					samples++;
				}
			}
			Lmean /= samples;

			std::cout<<"Current independent samples per chain / Average AC: "<<currentIndependentSamples<<" / "<<AC <<std::endl;
			std::cout<<"Mean Log Likelihood / Max Log Likelihood: "<<Lmean<<" / "<<Lmax <<std::endl;
			printProgress( (double) currentIndependentSamples/independentSamples);
			std::cout<<std::endl;
			if(currentIndependentSamples<independentSamples){
				double meanAC;
				mean_list(data->maxACs,ensembleN,&meanAC);

				if(localBatchSize){
					if(batchSize < meanAC*10){
						batchSize*=2;
					}
					else if(batchSize > meanAC*1000){
						batchSize/=2;
					}
				}
				data->extendSize(batchSize-1);
			}
			
		}

	}
		
	gsl_set_error_handler(oldHandler);

	std::cout<<"Total sampling time (seconds): "<<(double)(-start + omp_get_wtime())<<std::endl;

	return;
}

/*! \brief Allocates all the memory needed for the sampler, with all the currently set internal values for the parameters
 *
 * Should be called internally before sampling is initiated
 *
 */
void bayesshipSampler::allocateMemory( )
{

	if(threads <3 && threadPool)
	{
		std::cout<<"Error -- Need more than 3 threads for thread pool option -- using OpenMP instead"<<std::endl;
		threadPool = false;
	}
	omp_set_num_threads(threads);
	chainN = ensembleSize*ensembleN;
	const gsl_rng_type *T = gsl_rng_default;	
	if(!betaSchedule){
		/*If betaSchedule is allocated internally, it should be released internally*/
		internalBetaScheduleFlag = true;

		betaSchedule = new double[ensembleSize];

		betaSchedule[0] = 1;
		betaSchedule[ensembleSize-1]=0;
		/*Geometric spacing -- In principal, we should space these out geometrically between 1 and 0,
 * 		but that proves to be inefficient. Start by spacing out geometrically between 0 and 1/100, 
 * 		which is the likelihood raised to 1/100. That's plenty.*/
		double deltaBeta = pow( (1e2),1./ensembleSize);
		//std::cout<<"Initial Beta Schedule"<<std::endl;
		for(int i = 1 ; i<ensembleSize-1;i++){
			betaSchedule[i] = betaSchedule[i-1]/deltaBeta;
			//std::cout<<betaSchedule[i]<<std::endl;
		}

			
	}
	/*Assign temperature parameters*/
	if(!betas){
		betas = new double[chainN];	
		for(int i = 0 ; i<ensembleSize; i++){
			for(int j = 0 ; j<ensembleN; j++){
				betas[i*ensembleN+j] = betaSchedule[i];
			}
		}
	}
	
	if(!proposalFns){
		/* Set internal flag reminding us to release the object in the end*/
		internalProposalFnsFlag=true;
		this->proposalFns = new proposalData(chainN,maxDim,this,RJ);
	}


	if(!rvec){
		gsl_rng_env_setup();
		rvec = new gsl_rng *[chainN];
		for(int i = 0 ; i<chainN; i++){
 			rvec[i] = gsl_rng_alloc(T);
			gsl_rng_set(rvec[i],seed+i*101);
		}	
	}
	if(!userParameters){
		internalUserParameters=true;
		userParameters = new void*[chainN];
	}
	if(!A){
		A = new double[chainN];
		for(int i = 0 ; i<chainN; i++){
			A[i] = 0;
		}
	}
	if(!waitingSample){
		waitingSample = new bool[chainN];
		for(int i = 0 ; i<chainN; i++){
			waitingSample[i] = true;
		}
	}
	if(!referenceStatus){
		referenceStatus = new bool[ensembleN];
		for(int i = 0 ; i<ensembleN; i++){
			referenceStatus[i] = true;
		}
	}
	if(!waitingMutexes){
		waitingMutexes = new std::mutex[chainN];
	}
	if(!statusMutex){
		statusMutex = new std::mutex;
	}
	
}
/*! \brief helper to return the beta parameter for the chain at index chainID
 */
double bayesshipSampler::getBeta(int chainID){
	return betas[chainID];
}

/*! \brief helper to return the total number of chains in the sampler
 */
int bayesshipSampler::getChainN(){
	return chainN;
}

void bayesshipSampler::deallocateMemory()
{
	if(betas){
		delete [] betas;
		betas = nullptr;
	}
	if(betaSchedule && internalBetaScheduleFlag){
		delete [] betaSchedule;
		betaSchedule = nullptr;
		internalBetaScheduleFlag = false;
	}
	if(proposalFns && internalProposalFnsFlag){
		delete proposalFns;
		proposalFns = nullptr;
		internalProposalFnsFlag=false;
	}
	if(rvec){

		for(int i =0 ;i<chainN; i++){
			gsl_rng_free(rvec[i]);
		}
		delete [] rvec;	
		rvec = nullptr;
	}
	if(data){
		delete data;
		data = nullptr;
	}
	if(burnData){
		delete burnData;
		burnData = nullptr;
	}
	if(priorData){
		delete priorData;
		priorData = nullptr;
	}
	if(userParameters && internalUserParameters){
		delete [] userParameters;
		userParameters=nullptr;
	}
	if(A){
		delete [] A;
		A = nullptr;		
	}
	if(waitingSample){
		delete [] waitingSample;
		waitingSample = nullptr;		
	}
	if(referenceStatus){
		delete [] referenceStatus;
		referenceStatus = nullptr;		
	}
	if(waitingMutexes){
		delete [] waitingMutexes;
		waitingMutexes = nullptr;		
	}
	if(statusMutex){
		delete statusMutex;
		statusMutex = nullptr;		
	}
	if(priorRanges && internalPriorRanges){
		for(int i = 0 ; i<maxDim; i++){
			delete [] priorRanges[i];
		}	
		delete [] priorRanges;
		priorRanges = nullptr;
	}
	if(initialPositionEnsemble && internalInitialPositionEnsemble){
		for(int i = 0 ; i<chainN; i++){
			delete  initialPositionEnsemble[i];
		}	
		delete [] initialPositionEnsemble;
		initialPositionEnsemble = nullptr;
	}

}

/*! \brief Constructor for bayesshipSampler class
 */
bayesshipSampler::~bayesshipSampler()
{
	deallocateMemory();
}

/*! \brief Actually initiates the sampler loop to run ``iterations'' steps
 *
 * Will either use OpenMP loop or the custom thread pool implementation
 */
void bayesshipSampler::sampleLoop(int samples,samplerData *data)
{
	setActiveData(data);	
	if(!threadPool){
		ThreadPool<sampleJob> *samplePool = new ThreadPool<sampleJob>(threads, sampleThreadedFunctionNoSwap,false);
		//#ifdef _OPENMP
		//#pragma omp parallel
		//#endif
		for(int i=0; i<samples-1; i++){
			//#ifdef _OPENMP
			//#pragma omp parallel for schedule(dynamic)
			////#pragma omp for schedule(dynamic)
			//#endif
			//for(int chain = 0 ; chain<chainN; chain++){
			//	stepMH(chain,data);
			//}	

			samplePool->startPool();

			for(int chain = 0 ; chain<chainN; chain++){
				sampleJob job;
				job.sampler = this;
				job.chainID = chain;
				job.data = data;
				samplePool->enqueue(job);	
			}	
			samplePool->stopPool();
			//for(int chain = 0 ; chain<chainN; chain++){
			//	std::cout<<data->currentStepID[chain]<<std::endl;
			//}

			
			//#ifdef _OPENMP
			//#pragma omp single
			//#endif
			if(!adjustTemps)
			{
				for(int j = 0 ; j<ensembleN; j++){
					for(int k = 0 ; k<ensembleSize-1; k++){
						double prob = gsl_rng_uniform(rvec[chainIndex(j,k)]);
						if(prob<swapProb){
							int ensembleIndex2 = j;
							if(randomizeSwapping)
							{
								ensembleIndex2 = (int)(gsl_rng_uniform(rvec[chainIndex(j,k)])*ensembleN);			
							}
							chainSwap(chainIndex(j,k),chainIndex(ensembleIndex2,k+1),data);
						}
					}
				}
				//if(adjustTemps){
				//	adjustTemperatures(i);
				//}
			}
	
		}
		if(samplePool){
			delete samplePool;
		}
	}
	else{
		//the chain at 0 should always be cold, and therefore always start with the largest possible size
		int totalFinalSamples = samples +data->currentStepID[0]-1;
		int *initialStepIDs = new int[chainN];
		for(int i = 0 ; i<chainN; i++){
			initialStepIDs[i] = data->currentStepID[i];
		}
		
		ThreadPool<sampleJob> *samplePool = new ThreadPool<sampleJob>(threads-2, sampleThreadedFunction,false);
		ThreadPoolPair<swapJob> *swapPool = new ThreadPoolPair<swapJob>(1, swapThreadedFunction,swapPairFunction,false);
		//samplePool->randomizeJobs() ;
		//swapPool->randomizeJobs() ;

		//samplePool->startPool();
		//swapPool->startPool();


		samplePool->startPool();
		swapPool->startPool();

		//Reset reference counters and waiting flags -- just precautionary
		for(int i = 0 ; i<ensembleN; i++){
			referenceStatus[i] = true;
		}
		for(int i = 0 ; i<chainN; i++){
			waitingSample[i] = true;
		}

		while(checkStatus()){
			for(int i = 0 ; i<chainN; i++){
				bool chainWaitingSample ; 
				{
					std::unique_lock<std::mutex> lock{waitingMutexes[i]};
					chainWaitingSample = waitingSample[i];
				}
				if(chainWaitingSample){

					if(data->currentStepID[i] < totalFinalSamples){
						sampleJob job;
						job.sampler = this;
						job.chainID = i;
						job.data = data;
						job.swapPool = swapPool;
						{
							std::unique_lock<std::mutex> lock{waitingMutexes[i]};
							waitingSample[i]=false;
						}
						samplePool->enqueue(job);	
					}	
					else{
				
						if(i < ensembleN){
							
							{
								std::unique_lock<std::mutex> lock{*statusMutex};
								referenceStatus[i] = false;
							}
							{
								std::unique_lock<std::mutex> lock{waitingMutexes[i]};
								waitingSample[i]=false;
							}
						}
						else{
							
							//std::cout<<"Resetting Chain "<<i<<std::endl;
							int currentID = data->currentStepID[i];
							int initialID = initialStepIDs[i];
							data->positions[i][initialID]->updatePosition(data->positions[i][currentID]);
							data->likelihoodVals[i][initialID] = data->likelihoodVals[i][currentID];
							data->priorVals[i][initialID] = data->priorVals[i][currentID];
							data->currentStepID[i] = initialID;	

							//Keep stepping
							sampleJob job;
							job.sampler = this;
							job.chainID = i;
							job.data = data;
							job.swapPool = swapPool;

							{
								std::unique_lock<std::mutex> lock{waitingMutexes[i]};
								waitingSample[i]=false;
							}
							samplePool->enqueue(job);	

						}
					}
				}
			}
		}
		samplePool->stopPool();
		swapPool->stopPool();
		//Reset reference counters and waiting flags
		for(int i = 0 ; i<ensembleN; i++){
			referenceStatus[i] = true;
		}
		for(int i = 0 ; i<chainN; i++){
			waitingSample[i] = true;
		}

		//for(int i = 0 ; i<chainN; i++){
		//	std::cout<<i<<" "<<data->currentStepID[i]<<std::endl;
		//}

		delete [] initialStepIDs;
		delete samplePool;
		delete swapPool;
	}
	return;
}

/*! \brief Checks if sampler still needs to continue
 *
 * If sampler is still active, it returns true
 */
bool bayesshipSampler::checkStatus()
{
	bool result = false;
	{
		std::unique_lock<std::mutex> lock{*statusMutex};
		for(int i = 0 ; i<ensembleN; i++){
			if(referenceStatus[i]){
				result =  true;	
				break;
			}
		}
	}
	return result;
}
void sampleThreadedFunctionNoSwap(int threadID, sampleJob job)
{
	job.sampler->stepMH(job.chainID,job.data);
	//job.data->currentStepID[job.chainID] ++;
	//job.data->positions[job.chainID][job.data->currentStepID[job.chainID]]->updatePosition(job.data->positions[job.chainID][job.data->currentStepID[job.chainID-1]]);


	
	return;
}

void sampleThreadedFunction(int threadID, sampleJob job)
{
	job.sampler->stepMH(job.chainID,job.data);
	//job.data->currentStepID[job.chainID] ++;
	//job.data->positions[job.chainID][job.data->currentStepID[job.chainID]]->updatePosition(job.data->positions[job.chainID][job.data->currentStepID[job.chainID-1]]);


	double prob = gsl_rng_uniform(job.sampler->rvec[job.chainID]);
	//If attempting swap
	if(prob< job.sampler->swapProb){
		swapJob j;
		j.sampler = job.sampler;
		j.data = job.data;
		j.chainID = job.chainID;
		job.swapPool->enqueue(j);
	}
	//If just returning to sample
	else{
		std::unique_lock<std::mutex> lock{job.sampler->waitingMutexes[job.chainID]};
		job.sampler->waitingSample[job.chainID]=true;

	}
	
	return;
}
void swapThreadedFunction(int threadID, swapJob job1, swapJob job2)
{
	int j = job1.chainID;
	int k = job2.chainID;
	job1.sampler->chainSwap(j,k,job1.data);
	{
		std::unique_lock<std::mutex> lock{job1.sampler->waitingMutexes[job1.chainID]};
		job1.sampler->waitingSample[job1.chainID]=true;
	}
	{
		std::unique_lock<std::mutex> lock{job2.sampler->waitingMutexes[job2.chainID]};
		job2.sampler->waitingSample[job2.chainID]=true;
	}
	return;
}
bool swapPairFunction( swapJob job1, swapJob job2)
{
	int i =  job1.sampler->betaN(job1.chainID);
	int j =  job2.sampler->betaN(job2.chainID);
	int k =  job1.sampler->ensembleID(job1.chainID);
	int l =  job2.sampler->ensembleID(job2.chainID);
	int diffRung = fabs(i-j);
	int diffLadder = fabs(k-l);
	if( diffRung < 3 && diffRung >0){
		if(!job1.sampler->getCurrentIsolateEnsemblesInternal()){
			return true;
		}
		else if(diffLadder ==0){
			return true;
		}
	}
	return false;
}

bool bayesshipSampler::getCurrentIsolateEnsemblesInternal()
{
	return this->isolateEnsemblesInternal;

}

/*! \brief Swaps two chains in the ensemble identified by chainID1 and chainID2
 */

void bayesshipSampler::chainSwap(int chainID1, int chainID2,samplerData *data)
{

	int currentStep1 = data->currentStepID[chainID1];
	int currentStep2 = data->currentStepID[chainID2];
	//double likelihood1 = positions[chainID1][currentStepID[chainID1]
	double likelihood1 = data->likelihoodVals[chainID1][currentStep1];
	double likelihood2 = data->likelihoodVals[chainID2][currentStep2];
	double beta1 = betas[chainID1];
	double beta2 = betas[chainID2];

	if(fabs(beta1 - beta2)/(fabs(beta1)+fabs(beta2)) < 1e-15){
		data->swapRejects[chainID1][chainID2]++;
		data->swapRejects[chainID2][chainID1]++;
		return;
	}
	
	//double ratio = (likelihood1 - likelihood2)*beta2 - (likelihood1 - likelihood2)*beta1;
	double ratio = (likelihood1 - likelihood2)*(beta2-beta1);
		
	double alpha = gsl_rng_uniform(rvec[chainID1]) ;

	if(exp(ratio) < alpha){
		data->swapRejects[chainID1][chainID2]++;
		data->swapRejects[chainID2][chainID1]++;
		if(adjustTemps){
			//A[chainID2]=0;
			//Assign to hotter chain
			if(beta2 < beta1){
				A[chainID2]=0;
			}
			else{
				A[chainID1] = 0;
			}
		}
		return;
	}
	else{
		if(adjustTemps){
			//A[chainID2]=1;
			//Assign to hotter chain
			if(beta2 < beta1){
				A[chainID2]=1;
			}
			else{
				A[chainID1] = 1;
			}
		}

		data->swapAccepts[chainID1][chainID2]++;
		data->swapAccepts[chainID2][chainID1]++;

		positionInfo *temp = new positionInfo(maxDim, RJ);
		temp->updatePosition(data->positions[chainID1][currentStep1]);
		data->positions[chainID1][currentStep1]->updatePosition(data->positions[chainID2][currentStep2]);
		data->positions[chainID2][currentStep2]->updatePosition(temp);
		delete temp;

		data->likelihoodVals[chainID1][currentStep1] = likelihood2;
		data->likelihoodVals[chainID2][currentStep2] = likelihood1;


		double tempPrior = data->priorVals[chainID1][currentStep1];
		data->priorVals[chainID1][currentStep1] = data->priorVals[chainID2][currentStep2];
		data->priorVals[chainID2][currentStep2] = tempPrior;
	}

	return;
}

double PTDynamicalTimescale(double t0, double nu, int t){

	double kappa = (1./nu) * (double)(t0) / (t + t0);
	return kappa;
}

void bayesshipSampler::adjustTemperatures(int t)
{


	double kappa = PTDynamicalTimescale(t0, nu, t);
	double power = 0;
	double oldBetas[chainN];
	for(int i = 0 ; i<chainN; i++){
		oldBetas[i] = betas[i];
	}
	if(averageDynamics){
		double AverageA[ensembleSize];
		for(int i = 0 ; i<ensembleSize; i++){
			AverageA[i] = 0;
			for(int j = 0 ; j<ensembleN; j++){
				AverageA[i] +=  A[chainIndex(j,i)];
			}
			AverageA[i] /= ensembleN;
		}
		for(int i = 1 ; i<ensembleSize-1; i++){
			for(int j = 0 ; j<ensembleN; j++){
				int currentID = chainIndex(j, i);
				int colderID = chainIndex(j, i-1);

				double TColder = 1./betas[colderID];

				double TOld = 1./oldBetas[currentID];
				double TOldColder = 1./oldBetas[colderID];
					
				power = kappa * (AverageA[i] - AverageA[i+1]);
				double TNew = TColder + (TOld - TOldColder)*std::exp(power);
				betas[currentID] = 1./TNew;
			}

		}
	}
	else{
		for(int i = 1 ; i<ensembleSize-1; i++){
			for(int j = 0 ; j<ensembleN; j++){
				int currentID = chainIndex(j, i);
				int hotterID = chainIndex(j, i+1);
				int colderID = chainIndex(j, i-1);

				double TColder = 1./betas[colderID];
	
				double TOld = 1./oldBetas[currentID];
				double TOldColder = 1./oldBetas[colderID];
					
				//power = kappa * (A[currentID] - A[hotterID]);
				power = kappa * (A[currentID] - A[hotterID]);
				double TNew = TColder + (TOld - TOldColder)*std::exp(power);
				betas[currentID] = 1./TNew;
			}

		}

	}
	for(int i = 0 ; i<chainN; i++){
		A[i] = 0;
	}
	return;
}

/*! \brief Run a single Metropolis-Hastings iteration for a single chain identified by chainID
 *
 * Proposes and evaluates the acceptance of a MH step
 */
void bayesshipSampler::stepMH(
	int chainID,/**< ID of the chain to iterate*/
	samplerData *data
	)
{
	int currentStep = data->currentStepID[chainID];
	int proposalStep = data->currentStepID[chainID]+1;
	
	/*Choose a random proposal*/
	double beta = (gsl_rng_uniform(rvec[chainID]));
	int randStep = 0;
	double runningSum = 0;
	for(int i = 1 ; i<proposalFns->proposalN; i++){
		runningSum+=proposalFns->proposalProb[chainID][i-1];
		double upper = runningSum + proposalFns->proposalProb[chainID][i];
		double lower = runningSum ;
			
		if(beta > lower && beta < upper){
			randStep = i;
		}
	}
	double MHRatioCorrection = 0;
	/*Perform the proposal*/
	double start = omp_get_wtime();	
	proposalFns->proposals[randStep]->propose(data->positions[chainID][currentStep], data->positions[chainID][proposalStep],chainID,  randStep,&MHRatioCorrection);
	double time = omp_get_wtime() - start;
	data->proposalTimes[chainID][randStep] *= (data->rejectN[chainID][randStep] +data->successN[chainID][randStep] );
	data->proposalTimes[chainID][randStep] += time;
	data->proposalTimes[chainID][randStep] /= (data->rejectN[chainID][randStep] +data->successN[chainID][randStep]+1 );
	

	/*Calculate log of the prior values*/
	start = omp_get_wtime();	
	//double logPrior = prior(data->positions[chainID][proposalStep], chainID, this,userParameters[chainID]);
	double logPrior = prior->eval(data->positions[chainID][proposalStep], chainID);
	time = omp_get_wtime() - start;
	data->priorTimes[chainID] *= (data->currentStepID[chainID] );
	data->priorTimes[chainID] += time;
	data->priorTimes[chainID] /= (data->currentStepID[chainID] + 1);
	/*If rejected outright, exitA*/
	if(logPrior == limitInf){
		data->positions[chainID][proposalStep]->updatePosition(data->positions[chainID][currentStep]);
		data->likelihoodVals[chainID][proposalStep] = data->likelihoodVals[chainID][currentStep];
		data->priorVals[chainID][proposalStep] = data->priorVals[chainID][currentStep];
		data->rejectN[chainID][randStep]++;
		data->currentStepID[chainID] +=1;
		return;
	}
	/*Calculate likelihood valeu*/
	start = omp_get_wtime();	
	//double logLikelihood = likelihood(data->positions[chainID][proposalStep], chainID, this,userParameters[chainID]);
	double logLikelihood = likelihood->eval(data->positions[chainID][proposalStep], chainID);
	time = omp_get_wtime() - start;
	data->likelihoodTimes[chainID] *= (data->likelihoodEvals[chainID] );
	data->likelihoodTimes[chainID] += time;
	data->likelihoodEvals[chainID]++;
	data->likelihoodTimes[chainID] /= (data->likelihoodEvals[chainID]) ;
	
	
	/*Calculate the MH ratio*/
	double MHRatio = 
		(logLikelihood - data->likelihoodVals[chainID][currentStep]) * betas[chainID]
		+logPrior - data->priorVals[chainID][currentStep] 
		+ MHRatioCorrection;

	/*Random number representing probability*/
	double prob = log(gsl_rng_uniform(rvec[chainID]));
	/*Reject or accept the step*/
	if(MHRatio < prob){
		//reject
		data->positions[chainID][proposalStep]->updatePosition(data->positions[chainID][currentStep]);
		data->likelihoodVals[chainID][proposalStep] = data->likelihoodVals[chainID][currentStep];
		data->priorVals[chainID][proposalStep] = data->priorVals[chainID][currentStep];
		data->rejectN[chainID][randStep]++;
		
	}
	else{
		//accept
		data->likelihoodVals[chainID][proposalStep] = logLikelihood;
		data->priorVals[chainID][proposalStep] = logPrior ;
		data->successN[chainID][randStep]++;
	}
	
	data->currentStepID[chainID] +=1;
	return;
}



/*! \brief Helper routine to reverse engineer which rung on the beta ladder the index for the chain belongs to
 *
 * For example, say the beta ladder for a sampler is {1,.5,.2,0} with 3 ensembles:
 * 	chainID = 2 -> betaN == 0 and beta == 1
 *
 * 	chainID =4 -> betaN == 1 and beta == 0.5
 *
 * 	chainID =7 -> betaN == 2 and beta == 0.2
 *
 */
int bayesshipSampler::betaN(int chainID){
	return (int)(chainID/ensembleN);
}


/*! \brief Helper routine to reverse engineer which ladder the index for the chain belongs to
 *
 * For example, say the beta ladder for a sampler is {1,.5,.2,0} with 3 ensembles:
 * 	chainID = 2 -> betaN == 0 and beta == 1 and ensembleN = 2
 *
 * 	chainID =4 -> betaN == 1 and beta == 0.5 and ensembleN = 1
 *
 * 	chainID =8 -> betaN == 2 and beta == 0.2 and ensembleN = 2
 *
 */
int bayesshipSampler::ensembleID(int chainID){
	return (int)(chainID%ensembleN);
}

/*! \brief Helper routine to calculate the index for the chain in ensemble ``ensemble'' and with beta ID ''betaN''
 *
 * ensemble goes from 0 to ensembleN
 *
 * betaN goes from 0 to ensembleSize
 */
int bayesshipSampler::chainIndex(int ensemble, int betaN){
	return ensemble + betaN*ensembleN;
}



/*! \brief Converts the structured data in samplerData to a simple 3 dimensional array of primitives 
 *
 * Used for autocorrelation calculations with nonRJ sampling
 *
 * Output is a pointer size [chainN][iterations][dimension]
 *
 * !!NOTE!! first 2 dimensions are allocated and must be allocated but NOT THE LAST!!
 *
 * These are just pointers to the full data set to save on memory and time.
 *
 * Use the deallocation method to properly deallocate the memory of the returned pointer.
 */
double *** samplerData::convertToPrimitivePointer()
{
	double*** newPointer  = new double**[chainN];
	for(int i = 0 ; i<chainN; i++){
		newPointer[i] = new double*[currentStepID[i]+1];
		for(int j = 0 ; j<currentStepID[i]+1; j++){	
	
			newPointer[i][j] = positions[i][j]->parameters;
		}
	}
	return newPointer;
}
void samplerData::deallocatePrimitivePointer(double ***newPointer)
{
	for(int i = 0 ; i<chainN; i++){
		delete [] newPointer[i];
	}
	delete [] newPointer;
	return;	
}





proposalData::proposalData(
	int chainN,
	int proposalN,
	proposal **proposals,
	double *proposalProbFixed ,
	double **proposalProb 
	)
{

	/*User supplied*/

	this->proposalN = proposalN;
	this->chainN = chainN;
	
	this->proposals = new proposal*[proposalN];
	this->proposalProb = new double*[chainN];	

	
	for(int i = 0 ; i<proposalN ; i++){
		this->proposals[i] = proposals[i];
	}
	if(proposalProb){
		for(int i =0  ; i<chainN; i++){
			this->proposalProb[i] = new double[proposalN];	
			for(int j = 0 ; j<proposalN ; j++){
				this->proposalProb[i][j] = proposalProb[i][j];
			}
		}
	}
	else if (proposalProbFixed){
		for(int i =0  ; i<chainN; i++){
			this->proposalProb[i] = new double[proposalN];	
			for(int j = 0 ; j<proposalN ; j++){
				this->proposalProb[i][j] = proposalProbFixed[j];
			}
		}

	}
	else{
		for(int i =0  ; i<chainN; i++){
			this->proposalProb[i] = new double[proposalN];	
			for(int j = 0 ; j<proposalN ; j++){
				this->proposalProb[i][j] = 1./proposalN;
			}
		}
	
	}

}

proposalData::proposalData(
	int chainN,
	int maxDim,
	bayesshipSampler *sampler,
	bool RJ
	)
{

	internalMemorySet=true;
	this->chainN = chainN;
	this->maxDim = maxDim;
	this->proposalProb = new double*[chainN];	

	this->proposalN = 3;

	this->proposals = new proposal*[3];

	this->proposals[0] = new gaussianProposal(chainN, maxDim,sampler);
	this->proposals[1] = new differentialEvolutionProposal(sampler);
	this->proposals[2] = new KDEProposal(chainN, maxDim, sampler,RJ);

	for(int i =0  ; i<chainN; i++){
		this->proposalProb[i] = new double[3];	
		this->proposalProb[i][0] = .4;
		this->proposalProb[i][1] = .5;
		this->proposalProb[i][2] = .1;
	}

	
	

	//#############################################
	//this->proposalFnN = 2;

	//this->proposalFnArray = new proposalFn[2];
	//this->proposalFnProb = new float[2];	
	//this->proposalFnVariables = new void *[2];

	//this->proposalFnArray[0] = gaussianProposal;
	//this->proposalFnArray[1] = differentialEvolutionProposal;

	//this->proposalFnProb[0] = .5;
	//this->proposalFnProb[1] = .5;

	//gaussianProposalVariables *gpv = new gaussianProposalVariables(chainN,maxDim);
	//proposalFnVariables[0] = (void *)gpv;
	//proposalFnVariables[1] = (void *)nullptr;
	//#############################################



	//#############################################
	//this->proposalFnN = 1;

	//this->proposalFnArray = new proposalFn[1];
	//this->proposalFnProb = new float[1];	
	//this->proposalFnVariables = new void *[1];

	//this->proposalFnArray[0] = gaussianProposal;

	//this->proposalFnProb[0] = 1;

	//gaussianProposalVariables *gpv = new gaussianProposalVariables(chainN,maxDim);
	//proposalFnVariables[0] = (void *)gpv;
	//#############################################


}

proposalData::~proposalData()
{
	if(proposals){
		if(internalMemorySet){
			internalMemorySet = false;
			for(int i = 0 ; i<proposalN; i++){
				delete proposals[i];
			}
		}
		delete [] proposals;
		proposals = nullptr;
	}
	if(proposalProb){
		for(int i =0 ; i<chainN; i++){
			delete [] proposalProb[i];
		}
		delete [] proposalProb;
		proposalProb = nullptr;
	}
}

void bayesshipSampler::writeCheckpoint(samplerData *data)
{
	std::string outputFile(outputDir+outputFileMoniker+"_checkpoint.json");
	std::cout<<"Writing Checkpoint File: "<<outputFile<<std::endl;
	//nlohmann::json j;	
	//j["chainN"] = chainN;
	//
	//std::ifstream fileOut(outputDir+outputFileMoniker+"_checkpoint.json");
	//i >> j;
	
	//nlohmann::json j = *this;
	nlohmann::json j;

	j["maxDim"]=this->maxDim;
	j["RJ"]=this->RJ;
	j["threadPool"] = this->threadPool;
	j["minDim"] = this->minDim;
	j["ensembleSize"] = this->ensembleSize;
	j["ensembleN"] = this->ensembleN;
	j["seed"] = this->seed;
	j["swapProb"] = this->swapProb;
	j["averageDynamics"] = this->averageDynamics;
	j["randomizeSwapping"] = this->randomizeSwapping;
	j["betaSchedule"] = std::vector<double>(this->betaSchedule, this->betaSchedule + this->ensembleSize);
	j["betas"] = std::vector<double>(this->betas, this->betas + this->chainN);
	if(this->priorRanges){
		for(int i = 0 ; i<this->maxDim; i++){
			j["priorRanges"]["Dim "+std::to_string(i)] 
				= std::vector<double>(this->priorRanges[i],this->priorRanges[i] + 2);
		}
	}
	for(int i = 0 ; i<this->chainN; i++){
		int finalPos = data->currentStepID[i];
		j["finalPosition"]["Parameters"]["Chain "+std::to_string(i)] 
			= std::vector<double>(data->positions[i][finalPos]->parameters,data->positions[i][finalPos]->parameters + this->maxDim);
		if(this->RJ){
			j["finalPosition"]["Status"]["Chain "+std::to_string(i)] 
				= std::vector<double>(data->positions[i][finalPos]->status,data->positions[i][finalPos]->status + this->maxDim);
			j["finalPosition"]["Model ID"]["Chain "+std::to_string(i)] = data->positions[i][finalPos]->modelID;
		}
	}

	std::ofstream fileOut(outputFile);
	fileOut << j;
	
	for(int i = 0 ; i<proposalFns->proposalN; i++){
			proposalFns->proposals[i]->writeCheckpoint(outputDir, outputFileMoniker);
	}
	

	return;
}
void bayesshipSampler::loadCheckpoint()
{
	std::string inputFile(outputDir+outputFileMoniker+"_checkpoint.json");
	std::cout<<"Loading Checkpoint File: "<<inputFile<<std::endl;
	nlohmann::json j ;
	std::ifstream fileIn(inputFile);
	fileIn>>j;

	j.at("maxDim").get_to(this->maxDim);	
	j.at("RJ").get_to(this->RJ);	
	j.at("threadPool").get_to(this->threadPool);	
	j.at("minDim").get_to(this->minDim);	
	j.at("ensembleSize").get_to(this->ensembleSize);	
	j.at("ensembleN").get_to(this->ensembleN);	
	j.at("seed").get_to(this->seed);	
	j.at("swapProb").get_to(this->swapProb);	
	j.at("averageDynamics").get_to(this->averageDynamics);	
	j.at("randomizeSwapping").get_to(this->randomizeSwapping);	
	std::vector<double> betaScheduleTemp;
	j.at("betaSchedule").get_to(betaScheduleTemp);	
	std::vector<double> betasTemp;
	j.at("betas").get_to(betasTemp);	
		
	allocateMemory();

	for(int i = 0 ; i<ensembleSize; i++){
		this->betaSchedule[i] = betaScheduleTemp[i];
	}
	for(int i = 0 ; i<ensembleSize*ensembleN; i++){
		this->betas[i] = betasTemp[i];
	}
	
	if(j.contains(std::string("priorRanges")) ){
		if(!priorRanges){
			internalPriorRanges = true;
			priorRanges = new double*[maxDim];
			for(int i = 0  ;i<maxDim; i++){
				priorRanges[i] = new double[2];
			}
		}
		for(int i = 0  ;i<maxDim; i++){
			std::vector<double> PRTemp;
			j["priorRanges"]["Dim "+std::to_string(i)].get_to(PRTemp);	
			priorRanges[i][0] = PRTemp[0];
			priorRanges[i][1] = PRTemp[1];
			
		}
	}
	if(j.contains(std::string("finalPosition")) ){
		if(!initialPositionEnsemble){
			internalInitialPositionEnsemble = true;
			initialPositionEnsemble = new positionInfo*[chainN];
			for(int i = 0  ;i<chainN; i++){
				initialPositionEnsemble[i] = new positionInfo(maxDim, RJ);
			}
		}
		for(int i = 0  ;i<chainN; i++){
			std::vector<double> posTemp;
			j["finalPosition"]["Parameters"]["Chain "+std::to_string(i)].get_to(posTemp);	
			for(int j = 0 ;j<maxDim ; j++){
				initialPositionEnsemble[i]->parameters[j] = posTemp[j];
			}
			
		}
		if(j["finalPosition"].contains(std::string("Status"))){
			for(int i = 0  ;i<chainN; i++){
				std::vector<int> statusTemp;
				j["finalPosition"]["Status"]["Chain "+std::to_string(i)].get_to(statusTemp);	
				for(int j = 0 ;j<maxDim ; j++){
					initialPositionEnsemble[i]->status[j] = statusTemp[j];
				}
			}
		}
		if(j["finalPosition"].contains(std::string("Model ID"))){
			for(int i = 0  ;i<chainN; i++){
				j["finalPosition"]["Model ID"]["Chain "+std::to_string(i)].get_to(initialPositionEnsemble[i]->modelID);	
			}
		}
	}
	for(int i = 0 ; i<proposalFns->proposalN; i++){
			proposalFns->proposals[i]->loadCheckpoint(outputDir, outputFileMoniker);
	}
	
		
	return;
}

samplerData * bayesshipSampler::getActiveData()
{
	return this->activeData;
}

void bayesshipSampler::setActiveData(samplerData *newData)
{
	this->activeData = newData;
	return;
}


//void to_json(nlohmann::json& j, const bayesshipSampler& s)
//{
//	//std::vector<double> betaScheduleTemp(s.betaSchedule, s.betaSchedule + s.ensembleSize);
//	//nlohmann::json j_vec(betaScheduleTemp);
//	//j = nlohmann::json{
//	//	{"maxDim",s.maxDim},
//	//	{"RJ",s.RJ},
//	//	{"threadPool",s.threadPool},
//	//	{"minDim",s.minDim},
//	//	{"ensembleSize",s.ensembleSize},
//	//	{"ensembleN",s.ensembleN},
//	//	{"threadPool",s.threadPool},
//	//	{"threadPool",s.threadPool},
//	//	{"betaSchedule",betaScheduleTemp}
//	//};
//	j["maxDim"]=s.maxDim;
//	j["RJ"]=s.RJ;
//	j["threadPool"] = s.threadPool;
//	j["minDim"] = s.minDim;
//	j["ensembleSize"] = s.ensembleSize;
//	j["ensembleN"] = s.ensembleN;
//	j["threadPool"] = s.threadPool;
//	j["seed"] = s.seed;
//	j["swapProb"] = s.swapProb;
//	j["averageDynamics"] = s.averageDynamics;
//	j["randomizeSwapping"] = s.randomizeSwapping;
//	j["betaSchedule"] = std::vector<double>(s.betaSchedule, s.betaSchedule + s.ensembleSize);
//	if(s.priorRanges){
//		for(int i = 0 ; i<s.maxDim; i++){
//			j["priorRanges"]["Dim "+std::to_string(i)] 
//				= std::vector<double>(s.priorRanges[i],s.priorRanges[i] + 2);
//		}
//	}
//}
//void from_json(nlohmann::json& j, bayesshipSampler& s)
//{
//	j.at("maxDim").get_to(s.maxDim);	
//
//}

};

