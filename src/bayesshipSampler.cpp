#include "bayesship/bayesshipSampler.h"
#include "bayesship/utilities.h"
#include "bayesship/dataUtilities.h"
#include "bayesship/autocorrelationUtilities.h"
#include "bayesship/ThreadPool.h"
#include "bayesship/proposalFunctions.h"
#include "bayesship/standardPriors.h"
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <cstring>
#include <csignal>
#include <ctime>
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
	stoppedEarly = false;
	sampleStartTime = omp_get_wtime();
	double worstBatchSeconds = 0;
	int resumedRows = 0;
	bool resuming = false;
	if( checkpointExists() && !ignoreExistingCheckpoint){
		loadCheckpoint();
		resuming = true;
		if(samplingStarted){
			/*Burn-in (and the prior exploration feeding it) already happened in the run that wrote this
			 * checkpoint. The loaded betas are the post-burn-in temperatures and the loaded positions are
			 * the end of a sampling batch, so go straight back to sampling.*/
			std::cout<<"Resuming sampling from checkpoint generation "<<ckptGeneration<<" (batch "<<batchFiles.size()<<", "<<rowsStored<<" stored steps per chain): skipping prior exploration and burn-in"<<std::endl;
			priorIterations = 0;
			burnIterations = 0;
		}
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
			//std::cout<<"Final temperatures after burn in"<<std::endl;
			//for(int i = 0 ; i<ensembleSize; i++){
			//		std::cout<<"Temp: "<<i<<": "<<betas[chainIndex(0,i)]<<std::endl;
			//}


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
		std::cout<<"Sampling prior"<<std::endl;
		sampleLoop(priorIterations,priorData);

		priorData->updateACs(threads);
		priorData->writeStatFile(outputDir+outputFileMoniker+"Prior_stat.txt");
		for(int i = 0 ; i<proposalFns->proposalN; i++){
				proposalFns->proposals[i]->writeStatFile(outputDir, outputFileMoniker);
		}

		delete tempprior;
		prior = likelihood;
		likelihood = tempL;
		
		if(writePriorData){
			priorData->create_data_dump(coldOnlyStorage, true, outputDir+outputFileMoniker+"Prior_output.hdf5");
			
		}
		
	}
	else if( (priorIterations >0 && !priorRanges) || (priorIterations <0 && priorRanges)){
		std::cout<<"Error: If sampling the prior, the priorIterations must be > 0 and the priorRanges (double[dim][2]) must be allocated (! null)"<<std::endl;		
		std::cout<<"Skipping prior exploration"<<std::endl;		
	}
	if(burnIterations >0){
		bool saveRandomizeSwappingFlag = randomizeSwapping;
		bool savePool = threadPool;
		double saveSwapProb = swapProb;
		randomizeSwapping = false;
		if (t0 == 0)
		{
			t0 = burnIterations /8.;
		}

		burnData = new samplerData(maxDim, ensembleN,ensembleSize, burnIterations, proposalFns->proposalN, RJ,betas);
		if(priorData){
		//if(false){
			std::cout<<"Using Prior values for initial points for burn-in"<<std::endl;
			std::cout<<"Calculating likelihood at initial Position"<<std::endl;
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for(int i = 0 ; i<chainN; i++){
				burnData->positions[i][0]->updatePosition(priorData->positions[i][priorData->currentStepID[i]]);
				//burnData->likelihoodVals[i][0] = likelihood(burnData->positions[i][0],i, this,userParameters[i]);
				burnData->likelihoodVals[i][0] = likelihood->eval(burnData->positions[i][0],i);
				//burnData->priorVals[i][0] = prior(burnData->positions[i][0],i, this,userParameters[i]);
				burnData->priorVals[i][0] = prior->eval(burnData->positions[i][0],i);
				burnData->currentStepID[i] = 0;
			}
			std::cout<<"Finished calculating likelihood at initial Position"<<std::endl;

		}
		else{
			assignInitialPosition(burnData);
		}
	
		isolateEnsemblesInternal = isolateEnsemblesBurn;


		std::cout<<"Burning in "<<std::endl;
		std::cout << "(t0, ν) = (" << t0 << ", " << nu << ")\n";

		// Initial time for tracking burn-in duration
		double burn_start_time = 0.;
		// Track time elapsed. See below for implementation.
		double burn_time = 0.;

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
			#ifdef _OPENMP
			burn_start_time = omp_get_wtime();
			#endif // _OPENMP
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
			#ifdef _OPENMP
			burn_time += omp_get_wtime() - burn_start_time;
			#endif // _OPENMP

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
			#ifdef _OPENMP
			burn_start_time = omp_get_wtime();
			#endif // _OPENMP
			sampleLoop(burnIterations/4,burnData);
			#ifdef _OPENMP
			burn_time += omp_get_wtime() - burn_start_time;
			#endif // _OPENMP
		}
		burnPeriod=false;
		swapProb=saveSwapProb;
		printTime(burn_time, "BURN-IN TIME:");
		std::cout<<"Final temperatures after burn in"<<std::endl;
		for(int i = 0 ; i<ensembleSize; i++){
				std::cout<<"Temp: "<<i<<": "<<betas[chainIndex(0,i)]<<std::endl;
		}
		std::cout<<"Writing Burn Data"<<std::endl;
		burnData->create_data_dump(coldOnlyStorage, true, outputDir+outputFileMoniker+"Burn_output.hdf5");



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
	else if(priorData){
		std::cout<<"Using Prior values for initial points for sampling"<<std::endl;
		std::cout<<"Calculating likelihood at initial Position"<<std::endl;
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(int i = 0 ; i<chainN; i++){
			data->positions[i][0]->updatePosition(priorData->positions[i][priorData->currentStepID[i]]);
			//burnData->likelihoodVals[i][0] = likelihood(burnData->positions[i][0],i, this,userParameters[i]);
			data->likelihoodVals[i][0] = likelihood->eval(data->positions[i][0],i);
			//burnData->priorVals[i][0] = prior(burnData->positions[i][0],i, this,userParameters[i]);
			data->priorVals[i][0] = prior->eval(data->positions[i][0],i);
			data->currentStepID[i] = 0;
		}
		std::cout<<"Finished calculating likelihood at initial Position"<<std::endl;

	}
	else{
		std::cout<<"Using initial position and skipping burn in"<<std::endl;
		assignInitialPosition(data);
	}

	isolateEnsemblesInternal = isolateEnsembles;
	data->updateBetas(betas);
	if(resuming && samplingStarted){
		/*Before restoreHistory moves the re-evaluated start behind the stored history*/
		checkResumedValues(data);
	}
	if(resuming && samplingStarted && rowsStored > 0){
		if(RJ){
			/*Status / model ID are not stored in a way read_batch_dump restores yet: keep the batch files
			 * (they are complete) but count only the steps taken since this restart in memory.*/
			std::cout<<"WARNING: resuming an RJ run -- earlier samples stay in "<<batchDirectory()<<" but are not reloaded; the final _output.hdf5 will only cover steps since the restart"<<std::endl;
			dumpedUpTo = std::vector<int>(chainN, 0);
		}
		else{
			restoreHistory(data);
			resumedRows = rowsStored;
		}
	}
	if(resuming && samplingStarted){
		applyEvidenceTotals(data, !RJ && rowsStored > 0);
	}
	installStopHandlers();
	if(independentSamples == 0){
		if( ( iterations < batchSize && batchSize > 0 ) || batchSize == 0  ){
			double batchStart = omp_get_wtime();
			sampleLoop(iterations,data);
			commitBatch(data);
			worstBatchSeconds = std::max(worstBatchSeconds, omp_get_wtime() - batchStart);
			if(!RJ){
				data->updateACs(threads);
				int independentSamples = data->countIndependentSamples();
				std::cout<<"Independent samples per chain: "<<independentSamples<<std::endl;
				data->calculateEvidence();
				std::cout<<"Current ln Evidence: "<<data->evidence<<std::endl;
			}
			writeFinalOutput(data);
			writeStatAtomic(data, outputDir+outputFileMoniker+"_stat.txt");
			for(int i = 0 ; i<proposalFns->proposalN; i++){
					proposalFns->proposals[i]->writeStatFile(outputDir, outputFileMoniker);
			}
		}
		else  {
			int currentSamples=resumedRows;
			int currentIndependentSamples=0;
			double AC=0;
			while(currentSamples < iterations){
				double batchStart = omp_get_wtime();
				sampleLoop(batchSize,data);
				commitBatch(data);
				worstBatchSeconds = std::max(worstBatchSeconds, omp_get_wtime() - batchStart);
				currentSamples+=batchSize;
	
				if(!RJ){
					data->updateACs(threads);
					currentIndependentSamples = data->countIndependentSamples();
					AC = 0;
					for(int i = 0 ; i<data->ensembleN; i++){
						AC += data->maxACs[i];
					}
					AC /= data->ensembleN;
					data->calculateEvidence();
					std::cout<<"Current ln Evidence: "<<data->evidence<<std::endl;
				}
	

				writeStatAtomic(data, outputDir+outputFileMoniker+"_stat.txt");
				for(int i = 0 ; i<proposalFns->proposalN; i++){
						proposalFns->proposals[i]->writeStatFile(outputDir, outputFileMoniker);
				}

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
					if(stopRequested(worstBatchSeconds)){
						stoppedEarly = true;
						break;
					}
					data->extendSize(batchSize-1);
				}
			}
			if(!stoppedEarly){
				writeFinalOutput(data);
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
		while(currentIndependentSamples < independentSamples){

			/*Samples first, then the checkpoint that refers to them -- see commitBatch*/
			double batchStart = omp_get_wtime();
			sampleLoop(batchSize,data);
			commitBatch(data);
			worstBatchSeconds = std::max(worstBatchSeconds, omp_get_wtime() - batchStart);
			data->calculateEvidence();
			std::cout<<"Current ln Evidence: "<<data->evidence<<std::endl;
	
			data->updateACs(threads);
			currentIndependentSamples = data->countIndependentSamples();
			double AC = 0;
			for(int i = 0 ; i<data->ensembleN; i++){
				AC += data->maxACs[i];
			}
			AC /= data->ensembleN;
	

			writeStatAtomic(data, outputDir+outputFileMoniker+"_stat.txt");
			for(int i = 0 ; i<proposalFns->proposalN; i++){
					proposalFns->proposals[i]->writeStatFile(outputDir, outputFileMoniker);
			}

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
				if(stopRequested(worstBatchSeconds)){
					stoppedEarly = true;
					break;
				}
				data->extendSize(batchSize-1);
			}
			
		}
		if(!stoppedEarly){
			writeFinalOutput(data);
		}

	}
	restoreStopHandlers();
		
	gsl_set_error_handler(oldHandler);
	if(stoppedEarly){
		std::cout<<"STOPPED EARLY (signal or wall-time budget).\nCheckpoint generation "<<ckptGeneration<<" is complete; rerun with the same outputDir and ignoreExistingCheckpoint=false to continue."<<std::endl;
	}

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
		geometric_beta_schedule(betaSchedule, ensembleSize, Tmax);
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
	if( (diffRung < job1.sampler->swapRadius && diffRung >0) || !job1.sampler->restrictSwapTemperatures){
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

	/*Zero likelihood: the tempered target L^beta is zero for every beta>0, and so is its beta->0+
	 * limit, which is what the beta==0 rung samples (the prior restricted to where L>0). Reject
	 * outright, like a -inf prior. This keeps -inf out of likelihoodVals (calculateEvidence averages
	 * them per rung) and out of the MHRatio arithmetic below.*/
	if(logLikelihood == limitInf){
		data->positions[chainID][proposalStep]->updatePosition(data->positions[chainID][currentStep]);
		data->likelihoodVals[chainID][proposalStep] = data->likelihoodVals[chainID][currentStep];
		data->priorVals[chainID][proposalStep] = data->priorVals[chainID][currentStep];
		data->rejectN[chainID][randStep]++;
		data->currentStepID[chainID] +=1;
		return;
	}

	/*Calculate the MH ratio*/
	double MHRatio = (betas[chainID] == 0
							? 0.0
							: (logLikelihood -
								data->likelihoodVals[chainID][currentStep]) *
								betas[chainID]) +
						logPrior - data->priorVals[chainID][currentStep] +
						MHRatioCorrection;

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
	
	//Regardless of option employed, make sure its normalized
	for(int i = 0 ; i<chainN;i++){
		double sum = 0 ;
		for(int j = 0 ; j<proposalN; j++){
			sum +=this->proposalProb[i][j];
		}
		for(int j = 0 ; j<proposalN; j++){
			this->proposalProb[i][j]/=sum ;
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
	//this->proposals[1] = new differentialEvolutionProposal(sampler);
	std::vector<std::vector<int>> blocks = std::vector<std::vector<int>>(1);
	for(int i = 0 ; i< maxDim;i++){
		blocks[0].push_back(i);
	}
	std::vector<double> blockProb={1};
	//this->proposals[1] = new differentialEvolutionProposal(sampler);
	this->proposals[1] = new blockDifferentialEvolutionProposal(sampler,blocks,blockProb);
	this->proposals[2] = new GMMProposal(chainN, maxDim, sampler,blocks,blockProb,10,10,10,1e-10, RJ,100*maxDim);

	for(int i =0  ; i<chainN; i++){
		this->proposalProb[i] = new double[4];	
		this->proposalProb[i][0] = .2;
		this->proposalProb[i][1] = .4;
		this->proposalProb[i][2] = .4;
	}



	//this->proposalN = 4;

	//this->proposals = new proposal*[4];

	//this->proposals[0] = new gaussianProposal(chainN, maxDim,sampler);
	////this->proposals[1] = new differentialEvolutionProposal(sampler);
	//std::vector<std::vector<int>> blocks = std::vector<std::vector<int>>(1);
	//for(int i = 0 ; i< maxDim;i++){
	//	blocks[0].push_back(i);
	//}
	//std::vector<double> blockProb={1};
	////this->proposals[1] = new differentialEvolutionProposal(sampler);
	//this->proposals[1] = new blockDifferentialEvolutionProposal(sampler,blocks,blockProb);
	//this->proposals[2] = new KDEProposal(chainN, maxDim, sampler,RJ);
	//this->proposals[3] = new GMMProposal(chainN, maxDim, sampler,10,10,10,1e-10, RJ,100*maxDim);

	//for(int i =0  ; i<chainN; i++){
	//	this->proposalProb[i] = new double[4];	
	//	this->proposalProb[i][0] = .1;
	//	this->proposalProb[i][1] = .3;
	//	this->proposalProb[i][2] = .2;
	//	this->proposalProb[i][3] = .3;
	//}

	//this->proposalN = 1;

	//this->proposals = new proposal*[1];

	//this->proposals[0] = new gaussianProposal(chainN, maxDim,sampler);
	//for(int i =0  ; i<chainN; i++){
	//	this->proposalProb[i] = new double[4];	
	//	this->proposalProb[i][0] = 1;
	//}

	
	

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

/*Signal handling: SIGTERM/SIGUSR1/SIGINT during sampling only set a flag. The sampling loops look at it between batches (after the batch has been committed) and stop cleanly.
 * A second signal restores the default action, so an impatient Ctrl-C still kills the process.*/
static volatile sig_atomic_t bayesshipStopSignal = 0;
static struct sigaction bayesshipOldActions[3];
static bool bayesshipHandlersInstalled = false;
static const int bayesshipStopSignals[3] = {SIGTERM, SIGUSR1, SIGINT};

static void bayesshipSignalHandler(int sig)
{
	if(bayesshipStopSignal){
		signal(sig, SIG_DFL);
		raise(sig);
		return;
	}
	bayesshipStopSignal = sig;
}

void bayesshipSampler::installStopHandlers()
{
	bayesshipStopSignal = 0;
	if(!handleStopSignals || bayesshipHandlersInstalled){return;}
	struct sigaction act;
	memset(&act, 0, sizeof(act));
	act.sa_handler = bayesshipSignalHandler;
	sigemptyset(&act.sa_mask);
	for(int i = 0 ; i<3; i++){
		sigaction(bayesshipStopSignals[i], &act, &bayesshipOldActions[i]);
	}
	bayesshipHandlersInstalled = true;
}

void bayesshipSampler::restoreStopHandlers()
{
	if(!bayesshipHandlersInstalled){return;}
	for(int i = 0 ; i<3; i++){
		sigaction(bayesshipStopSignals[i], &bayesshipOldActions[i], nullptr);
	}
	bayesshipHandlersInstalled = false;
}

/*! \brief Whether sampling should stop after the batch that was just committed
 *
 * True if a stop signal arrived, or if the remaining wall time (the sooner of wallTimeLimitSeconds /
 * BAYESSHIP_WALL_TIME_LIMIT measured from the start of sample(), and the SLURM job end time) cannot
 * fit another batch plus margin. Budget for the next batch is twice the slowest batch so far.
 */
bool bayesshipSampler::stopRequested(double worstBatchSeconds)
{
	if(bayesshipStopSignal){
		std::cout<<"Stop signal "<<(int)bayesshipStopSignal<<" received -- stopping after the committed batch"<<std::endl;
		return true;
	}
	double limit = wallTimeLimitSeconds;
	if(limit <= 0){
		const char *env = getenv("BAYESSHIP_WALL_TIME_LIMIT");
		if(env){limit = atof(env);}
	}
	double margin = wallTimeMarginSeconds;
	const char *envMargin = getenv("BAYESSHIP_WALL_TIME_MARGIN");
	if(envMargin){margin = atof(envMargin);}

	double remaining = std::numeric_limits<double>::infinity();
	if(limit > 0){
		remaining = std::min(remaining, limit - (omp_get_wtime() - sampleStartTime));
	}
	const char *slurmEnd = getenv("SLURM_JOB_END_TIME");
	if(slurmEnd){
		double end = atof(slurmEnd);
		if(end > 0){
			remaining = std::min(remaining, end - (double)time(nullptr));
		}
	}
	if(remaining < 2*worstBatchSeconds + margin){
		std::cout<<"Wall time remaining ("<<remaining<<" s) is below the budget for another batch (2 x "<<worstBatchSeconds<<" s + "<<margin<<" s margin) -- stopping after the committed batch"<<std::endl;
		return true;
	}
	return false;
}

/*! Stat files are derived output, but a truncated one is confusing: write them atomically too*/
void bayesshipSampler::writeStatAtomic(samplerData *data, std::string filename)
{
	std::string tmp = filename + ".tmp";
	data->writeStatFile(tmp);
	if(!atomicRename(tmp, filename)){
		std::cout<<"WARNING: could not update "<<filename<<std::endl;
	}
}

/*! Once a run has reached its target, write the full chain history as a single file (what _output.hdf5 always was)*/
void bayesshipSampler::writeFinalOutput(samplerData *data)
{
	#ifdef _HDF5
	std::string finalName = outputDir+outputFileMoniker+"_output.hdf5";
	std::string tmp = finalName + ".tmp";
	if(data->create_data_dump(coldOnlyStorage, true, tmp) != 0){
		std::cout<<"WARNING: failed to write "<<finalName<<"; the per-batch files in "<<batchDirectory()<<" are intact"<<std::endl;
		remove(tmp.c_str());
		return;
	}
	fsyncPath(tmp);
	if(!atomicRename(tmp, finalName)){
		std::cout<<"WARNING: failed to move "<<tmp<<" to "<<finalName<<std::endl;
	}
	#endif
}

std::string bayesshipSampler::batchDirectory()
{
	return outputDir+outputFileMoniker+"_batches";
}
std::string bayesshipSampler::checkpointDirectory()
{
	return outputDir+outputFileMoniker+"_checkpoints";
}

static std::string hashToString(uint64_t h)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
	return std::string(buf);
}

static std::string generationName(int generation)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "gen_%06d", generation);
	return std::string(buf);
}

/*! generation numbers present in the checkpoint directory (complete ones only: in-progress ones carry a .tmp suffix), newest first*/
static std::vector<int> listGenerations(const std::string &dir)
{
	std::vector<int> gens;
	for(const std::string &name : listDirectory(dir)){
		int g;
		char extra;
		if(name.size() == 10 && sscanf(name.c_str(), "gen_%d%c", &g, &extra) == 1){
			gens.push_back(g);
		}
	}
	std::sort(gens.begin(), gens.end(), std::greater<int>());
	return gens;
}

bool bayesshipSampler::checkpointExists()
{
	if(!listGenerations(checkpointDirectory()).empty()){return true;}
	return checkDirExist(outputDir+outputFileMoniker+"_checkpoint.json");
}

/*! \brief Write one batch of samples, then the checkpoint that refers to it
 *
 * Order matters: the checkpoint's chain positions are the end of the batch, so the samples must be safely
 * on disk first. If we are killed in between, the orphaned batch file is simply rewritten on resume (the
 * checkpoint never lists it); if the batch write fails, the rows are retried together with the next batch
 * (dumpedUpTo is not advanced). Returns false if either step failed.
 */
bool bayesshipSampler::commitBatch(samplerData *data)
{
	bool ok = true;
	#ifdef _HDF5
	if(dumpedUpTo.size() != (size_t)chainN){
		dumpedUpTo = std::vector<int>(chainN, 0);
	}
	std::string batchDir = batchDirectory();
	char name[64];
	snprintf(name, sizeof(name), "batch_%06d.hdf5", (int)batchFiles.size());
	std::string tmp = batchDir+"/batch_tmp.hdf5";
	bool written = makeDirectory(batchDir) && data->write_batch_dump(coldOnlyStorage, dumpedUpTo, tmp) == 0;
	uint64_t hash=0, size=0;
	written = written && fsyncPath(tmp) && fileHashFNV1a(tmp, &hash, &size)
		&& atomicRename(tmp, batchDir+"/"+name);
	if(written){
		batchFiles.push_back(BatchFileInfo{std::string(name), size, hash});
		for(int i = 0 ; i<chainN; i++){
			dumpedUpTo[i] = data->currentStepID[i];
		}
		rowsStored = dumpedUpTo[0];
	}
	else{
		std::cout<<"WARNING: failed to write sample batch "<<name<<" -- these steps will be included with the next batch"<<std::endl;
		remove(tmp.c_str());
		ok = false;
	}
	#endif
	samplingStarted = true;
	if(!writeCheckpoint(data)){
		ok = false;
	}
	if(ok){
		consecutiveCommitFailures = 0;
	}
	else{
		consecutiveCommitFailures++;
		if(consecutiveCommitFailures >= 3){
			errorMessage("FATAL: checkpoint/batch writing failed 3 batches in a row (disk full or quota exceeded?). Stopping so the last good checkpoint generation is not silently left stale.", 1);
		}
	}
	return ok;
}

/*! \brief Write a new checkpoint generation
 *
 * Layout: <outputDir><moniker>_checkpoints/gen_NNNNNN/ holding sampler.json, whatever the proposal functions
 * write, and MANIFEST.json (size + FNV-1a hash of every other file). A generation is assembled in
 * gen_NNNNNN.tmp, fsync'd, and only then renamed into place, so a generation either exists complete or not at
 * all. The newest checkpointGenerationsToKeep generations are retained.
 */
bool bayesshipSampler::writeCheckpoint(samplerData *data)
{
	std::string root = checkpointDirectory();
	if(!makeDirectory(root)){
		std::cout<<"WARNING: cannot create checkpoint directory "<<root<<std::endl;
		return false;
	}
	if(ckptGeneration < 0){ckptGeneration = 0;}
	{
		/*Never reuse a number that exists on disk (e.g. generations from a run we are ignoring)*/
		std::vector<int> existing = listGenerations(root);
		if(!existing.empty() && existing[0] >= ckptGeneration){ckptGeneration = existing[0];}
	}
	int generation = ckptGeneration + 1;
	std::string finalDir = root+"/"+generationName(generation);
	std::string tmpDir = finalDir+".tmp";
	removeDirectoryFlat(tmpDir);
	std::cout<<"Writing Checkpoint Generation: "<<finalDir<<std::endl;
	if(!makeDirectory(tmpDir)){
		std::cout<<"WARNING: cannot create "<<tmpDir<<std::endl;
		return false;
	}

	nlohmann::json j;
	j["checkpointFormat"]=2;
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
	j["samplingStarted"] = this->samplingStarted;
	j["coldOnlyStorage"] = this->coldOnlyStorage;
	j["rowsStored"] = this->rowsStored;
	if(dumpedUpTo.size() == (size_t)chainN){
		/*Steps already in batch files, per stored chain (not every chain necessarily has taken the same number of steps)*/
		int stored = coldOnlyStorage ? ensembleN : chainN;
		j["rowsStoredPerChain"] = std::vector<int>(dumpedUpTo.begin(), dumpedUpTo.begin()+stored);
	}
	for(size_t i = 0 ; i<batchFiles.size(); i++){
		j["batchFiles"].push_back({{"name",batchFiles[i].name},{"size",batchFiles[i].size},{"hash",hashToString(batchFiles[i].hash)}});
	}
	if(batchFiles.empty()){
		j["batchFiles"] = nlohmann::json::array();
	}
	if(this->priorRanges){
		for(int i = 0 ; i<this->maxDim; i++){
			j["priorRanges"]["Dim "+std::to_string(i)] 
				= std::vector<double>(this->priorRanges[i],this->priorRanges[i] + 2);
		}
	}
	/*What calculateEvidence counts for each chain (in-memory history plus any offset carried from an earlier
	 * restart), so a resumed run can carry it on for chains whose history it does not reload*/
	for(int i = 0 ; i<this->chainN; i++){
		double sum = 0;
		for(int k = 0 ; k<data->currentStepID[i]; k++){
			sum += data->likelihoodVals[i][k];
		}
		double steps = data->currentStepID[i];
		if(data->evidenceLogLSumOffset.size() == (size_t)chainN){
			sum += data->evidenceLogLSumOffset[i];
			steps += data->evidenceStepOffset[i];
		}
		j["evidenceTotals"]["LogLikelihoodSum"]["Chain "+std::to_string(i)] = sum;
		j["evidenceTotals"]["Steps"]["Chain "+std::to_string(i)] = steps;
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
		/*Not restored on resume (the chains are re-evaluated) but compared against, to catch a likelihood or prior
		 * that was rebuilt differently. JSON has no -inf: it is written as null and read back as -inf.*/
		j["finalPosition"]["LogLikelihood"]["Chain "+std::to_string(i)] = data->likelihoodVals[i][finalPos];
		j["finalPosition"]["LogPrior"]["Chain "+std::to_string(i)] = data->priorVals[i][finalPos];
	}

	bool ok = atomicWriteFile(tmpDir+"/sampler.json", j.dump());
	
	/*The proposal functions write their own files with whatever directory prefix they are given*/
	for(int i = 0 ; i<proposalFns->proposalN; i++){
			proposalFns->proposals[i]->writeCheckpoint(tmpDir+"/", outputFileMoniker);
	}

	nlohmann::json manifest;
	manifest["generation"] = generation;
	manifest["files"] = nlohmann::json::array();
	for(const std::string &name : listDirectory(tmpDir)){
		if(name == "MANIFEST.json" || name == "MANIFEST.json.tmp"){continue;}
		uint64_t hash, size;
		if(!fsyncPath(tmpDir+"/"+name) || !fileHashFNV1a(tmpDir+"/"+name, &hash, &size)){
			ok = false;
			continue;
		}
		manifest["files"].push_back({{"name",name},{"size",size},{"hash",hashToString(hash)}});
	}
	ok = ok && atomicWriteFile(tmpDir+"/MANIFEST.json", manifest.dump());
	ok = ok && fsyncPath(tmpDir);
	ok = ok && atomicRename(tmpDir, finalDir);
	if(!ok){
		std::cout<<"WARNING: checkpoint generation "<<generation<<" could NOT be written; the previous generation is untouched"<<std::endl;
		removeDirectoryFlat(tmpDir);
		return false;
	}
	ckptGeneration = generation;

	/*Prune*/
	std::vector<int> gens = listGenerations(root);
	for(size_t i = checkpointGenerationsToKeep; i<gens.size(); i++){
		removeDirectoryFlat(root+"/"+generationName(gens[i]));
	}
	for(const std::string &name : listDirectory(root)){
		if(name.size() > 4 && name.compare(name.size()-4, 4, ".tmp") == 0 && root+"/"+name != tmpDir){
			removeDirectoryFlat(root+"/"+name);
		}
	}
	return true;
}

/*! Check a checkpoint generation: every file in its manifest present with the recorded size and hash, and every batch file it refers to as well.*/
bool bayesshipSampler::verifyGeneration(const std::string &genDir, std::string &reason)
{
	try{
		nlohmann::json manifest, j;
		std::ifstream mf(genDir+"/MANIFEST.json");
		if(!mf){reason = "no MANIFEST.json"; return false;}
		mf >> manifest;
		for(const auto &f : manifest.at("files")){
			uint64_t hash, size;
			std::string name = f.at("name").get<std::string>();
			if(!fileHashFNV1a(genDir+"/"+name, &hash, &size)){reason = name+" unreadable"; return false;}
			if(size != f.at("size").get<uint64_t>() || hashToString(hash) != f.at("hash").get<std::string>()){
				reason = name+" does not match the manifest (truncated or corrupt)";
				return false;
			}
		}
		std::ifstream sf(genDir+"/sampler.json");
		if(!sf){reason = "no sampler.json"; return false;}
		sf >> j;
		for(const auto &f : j.at("batchFiles")){
			uint64_t hash, size;
			std::string name = f.at("name").get<std::string>();
			if(!fileHashFNV1a(batchDirectory()+"/"+name, &hash, &size)){reason = "batch file "+name+" is missing"; return false;}
			if(size != f.at("size").get<uint64_t>() || hashToString(hash) != f.at("hash").get<std::string>()){
				reason = "batch file "+name+" does not match the checkpoint (truncated or corrupt)";
				return false;
			}
		}
	}
	catch(std::exception &e){
		reason = std::string("unparseable: ")+e.what();
		return false;
	}
	return true;
}

/*! \brief Load the newest valid checkpoint generation
 *
 * Falls back to older generations when the newest fails verification, and to the pre-generation
 * single-file format (_checkpoint.json) when no generation directory exists. Stops with an error if
 * checkpoints exist but none is usable: silently starting over would discard the run.
 */
void bayesshipSampler::loadCheckpoint()
{
	std::string root = checkpointDirectory();
	std::vector<int> gens = listGenerations(root);
	for(int g : gens){
		std::string genDir = root+"/"+generationName(g);
		std::string reason;
		if(verifyGeneration(genDir, reason)){
			ckptGeneration = g;
			loadCheckpointFrom(genDir+"/sampler.json", genDir+"/");
			return;
		}
		std::cout<<"WARNING: checkpoint generation "<<g<<" is not usable ("<<reason<<")"<<std::endl;
	}
	if(!gens.empty()){
		errorMessage("ERROR: checkpoint generations exist in "+root+" but none passes verification. Not starting over automatically; inspect them, or set ignoreExistingCheckpoint to discard.", 1);
	}
	loadCheckpointFrom(outputDir+outputFileMoniker+"_checkpoint.json", outputDir);
}

void bayesshipSampler::loadCheckpointFrom(const std::string &inputFile, const std::string &proposalPrefix)
{
	std::cout<<"Loading Checkpoint File: "<<inputFile<<std::endl;
	nlohmann::json j ;
	std::ifstream fileIn(inputFile);
	fileIn>>j;

	/*The loaded layout replaces the sampler's. A caller that has already sized per-chain objects from its own
	 * ensemble/dimension settings (userParameters, an initial position) would then index past them, so a
	 * mismatch is fatal in that case rather than silent memory corruption.*/
	if(userParameters || initialPosition){
		int ckptDim = j.at("maxDim").get<int>();
		int ckptSize = j.at("ensembleSize").get<int>();
		int ckptN = j.at("ensembleN").get<int>();
		if(ckptDim != maxDim || ckptSize != ensembleSize || ckptN != ensembleN){
			errorMessage("ERROR: the checkpoint was written for dimension "+std::to_string(ckptDim)+", "+std::to_string(ckptSize)+" temperatures x "+std::to_string(ckptN)+" chains, but this run is set up for dimension "+std::to_string(maxDim)+", "+std::to_string(ensembleSize)+" x "+std::to_string(ensembleN)+". These cannot change when resuming.", 1);
		}
	}

	j.at("maxDim").get_to(this->maxDim);	
	j.at("RJ").get_to(this->RJ);	
	j.at("threadPool").get_to(this->threadPool);	
	j.at("minDim").get_to(this->minDim);	
	j.at("ensembleSize").get_to(this->ensembleSize);	
	j.at("ensembleN").get_to(this->ensembleN);	
	j.at("seed").get_to(this->seed);
	/*A resumed run must not replay the random streams of the run it continues. allocateMemory seeds chain i
	 * with seed + i*101, so move the seed past that whole range. The advanced seed is what the next checkpoint
	 * stores, so every resume gets a fresh range.*/
	this->seed += 101.*ensembleSize*ensembleN;
	std::cout<<"Random number seed advanced to "<<this->seed<<" for the resumed run"<<std::endl;
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

	/*Format 2 additions: where in the run the checkpoint was taken, and the sample batches it refers to*/
	samplingStarted = j.value("samplingStarted", false);
	rowsStored = j.value("rowsStored", 0);
	batchFiles.clear();
	if(j.contains("batchFiles")){
		for(const auto &f : j["batchFiles"]){
			BatchFileInfo info;
			info.name = f.at("name").get<std::string>();
			info.size = f.at("size").get<uint64_t>();
			info.hash = std::stoull(f.at("hash").get<std::string>(), nullptr, 16);
			batchFiles.push_back(info);
		}
	}
	if(j.contains("coldOnlyStorage") && j["coldOnlyStorage"].get<bool>() != coldOnlyStorage){
		errorMessage("ERROR: the checkpoint was written with a different coldOnlyStorage setting than the one now requested; the stored sample batches would not match.", 1);
	}
	dumpedUpTo = std::vector<int>(ensembleSize*ensembleN, rowsStored);
	if(j.contains("rowsStoredPerChain")){
		std::vector<int> perChain = j["rowsStoredPerChain"].get<std::vector<int>>();
		for(size_t i = 0 ; i<perChain.size() && i<dumpedUpTo.size(); i++){
			dumpedUpTo[i] = perChain[i];
		}
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
		resumeLogL.clear();
		resumeLogP.clear();
		if(j["finalPosition"].contains(std::string("LogLikelihood")) && j["finalPosition"].contains(std::string("LogPrior"))){
			for(int i = 0  ;i<chainN; i++){
				const nlohmann::json &l = j["finalPosition"]["LogLikelihood"]["Chain "+std::to_string(i)];
				const nlohmann::json &p = j["finalPosition"]["LogPrior"]["Chain "+std::to_string(i)];
				resumeLogL.push_back(l.is_null() ? limitInf : l.get<double>());
				resumeLogP.push_back(p.is_null() ? limitInf : p.get<double>());
			}
		}
	}
	resumeEvidenceSum.clear();
	resumeEvidenceSteps.clear();
	if(j.contains(std::string("evidenceTotals"))){
		for(int i = 0  ;i<chainN; i++){
			const nlohmann::json &s = j["evidenceTotals"]["LogLikelihoodSum"]["Chain "+std::to_string(i)];
			resumeEvidenceSum.push_back(s.is_null() ? limitInf : s.get<double>());
			resumeEvidenceSteps.push_back(j["evidenceTotals"]["Steps"]["Chain "+std::to_string(i)].get<double>());
		}
	}
	for(int i = 0 ; i<proposalFns->proposalN; i++){
			proposalFns->proposals[i]->loadCheckpoint(proposalPrefix, outputFileMoniker);
	}
	return;
}

/*! \brief Carry the evidence sums of the stopped run into the resumed one
 *
 * calculateEvidence averages each rung's recorded log likelihoods over the history held in memory. Chains whose
 * history is reloaded from the batch files (stored chains, non-RJ) already have it, and here it is only checked
 * against the totals the checkpoint recorded. Every other chain (e.g. hot chains with cold-only storage) gets the
 * recorded totals as an offset, so the evidence continues from where the stopped run left it instead of
 * restarting from the resumed steps.
 */
void bayesshipSampler::applyEvidenceTotals(samplerData *data, bool historyReloaded)
{
	if(resumeEvidenceSum.size() != (size_t)chainN){
		std::cout<<"NOTE: this checkpoint does not record the evidence totals; for chains whose history is not reloaded the evidence only counts steps since the restart"<<std::endl;
		return;
	}
	int stored = coldOnlyStorage ? ensembleN : chainN;
	data->evidenceLogLSumOffset = std::vector<double>(chainN, 0);
	data->evidenceStepOffset = std::vector<double>(chainN, 0);
	int carried = 0;
	for(int i = 0 ; i<chainN; i++){
		if(historyReloaded && i<stored){
			double sum = 0;
			for(int k = 0 ; k<data->currentStepID[i]; k++){
				sum += data->likelihoodVals[i][k];
			}
			if(data->currentStepID[i] != resumeEvidenceSteps[i]
				|| std::abs(sum - resumeEvidenceSum[i]) > 1e-12*std::max(1.0, std::abs(sum))){
				std::cout<<"WARNING: chain "<<i<<": the reloaded history ("<<data->currentStepID[i]<<" steps, log likelihood sum "<<sum<<") does not match what the stopped run counted ("<<resumeEvidenceSteps[i]<<" steps, sum "<<resumeEvidenceSum[i]<<")"<<std::endl;
			}
			continue;
		}
		data->evidenceLogLSumOffset[i] = resumeEvidenceSum[i];
		data->evidenceStepOffset[i] = resumeEvidenceSteps[i];
		carried++;
	}
	std::cout<<"Evidence totals carried over for "<<carried<<" chains whose history is not reloaded"<<std::endl;
}

/*! \brief Compare the re-evaluated start of a resumed run with what the checkpoint recorded
 *
 * The chains restart at their checkpointed positions and assignInitialPosition re-evaluates likelihood and prior
 * there. With one deterministic likelihood at every temperature, that must reproduce the recorded values. If it
 * does not, the likelihood or prior was rebuilt differently (changed source or config, different build,
 * different floating point) and the resumed run would mix two targets, so stop.
 */
void bayesshipSampler::checkResumedValues(samplerData *data)
{
	if(resumeValueRelTol < 0){
		return;
	}
	if(resumeLogL.size() != (size_t)chainN || resumeLogP.size() != (size_t)chainN){
		std::cout<<"NOTE: this checkpoint does not record the chains' log likelihoods; the resumed values cannot be checked"<<std::endl;
		return;
	}
	auto differs = [this](double a, double b){
		if(a == b){return false;}	/*also covers -inf == -inf*/
		if(!std::isfinite(a) || !std::isfinite(b)){return true;}
		return std::abs(a-b) > resumeValueRelTol*std::max(1.0, std::max(std::abs(a), std::abs(b)));
	};
	int mismatches = 0;
	int worst = -1;
	double worstDiff = -1;
	for(int i = 0 ; i<chainN; i++){
		double l = data->likelihoodVals[i][0];
		double p = data->priorVals[i][0];
		if(differs(l, resumeLogL[i]) || differs(p, resumeLogP[i])){
			mismatches++;
			double d = std::isfinite(l - resumeLogL[i]) ? std::abs(l - resumeLogL[i]) : std::numeric_limits<double>::infinity();
			if(d > worstDiff){worstDiff = d; worst = i;}
		}
	}
	if(mismatches){
		errorMessage("ERROR: re-evaluating the checkpointed positions does not reproduce the recorded values for "
			+std::to_string(mismatches)+" of "+std::to_string(chainN)+" chains (e.g. chain "+std::to_string(worst)
			+": recorded log likelihood "+std::to_string(resumeLogL[worst])+", now "+std::to_string(data->likelihoodVals[worst][0])
			+"). The likelihood or prior is not the one this run was sampling with; not resuming.", 1);
	}
	std::cout<<"Resumed log likelihoods and priors match the checkpoint for all "<<chainN<<" chains"<<std::endl;
}

/*! \brief Put the previously stored samples back into a freshly built samplerData
 *
 * Needed so autocorrelations, the independent-sample count and the final merged output cover the whole run and
 * not just what happened since the restart. The start position of the new run (the checkpointed final position,
 * already evaluated by assignInitialPosition at index 0) is moved to index rowsStored, behind the history.
 */
void bayesshipSampler::restoreHistory(samplerData *data)
{
	#ifdef _HDF5
	int stored = coldOnlyStorage ? ensembleN : chainN;
	int maxRows = 0;
	for(int i = 0 ; i<stored; i++){
		maxRows = std::max(maxRows, dumpedUpTo[i]);
	}
	data->extendSize(maxRows);
	for(int i = 0 ; i<stored; i++){
		int rows = dumpedUpTo[i];
		data->positions[i][rows]->updatePosition(data->positions[i][0]);
		data->likelihoodVals[i][rows] = data->likelihoodVals[i][0];
		data->priorVals[i][rows] = data->priorVals[i][0];
	}
	std::vector<int> nextRow(chainN, 0);
	for(const BatchFileInfo &b : batchFiles){
		if(data->read_batch_dump(coldOnlyStorage, batchDirectory()+"/"+b.name, nextRow) != 0){
			errorMessage("ERROR: could not read stored sample batch "+b.name+" while resuming", 1);
		}
	}
	for(int i = 0 ; i<stored; i++){
		if(nextRow[i] != dumpedUpTo[i]){
			errorMessage("ERROR: stored sample batches hold "+std::to_string(nextRow[i])+" steps for chain "+std::to_string(i)+" but the checkpoint expects "+std::to_string(dumpedUpTo[i]), 1);
		}
		data->currentStepID[i] = dumpedUpTo[i];
	}
	std::cout<<"Restored "<<rowsStored<<" stored steps per cold chain from "<<batchFiles.size()<<" batch files"<<std::endl;
	#endif
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

