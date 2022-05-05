#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>

#if _MLPACK
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#endif

namespace bayesship{

/*! \file 
 *
 * # Source file for the KDE proposal template
 */


//###########################################################
//###########################################################
/*! Constructor function*/
KDEProposalVariables::KDEProposalVariables(
	int chainN, /**< Number of chains in the ensemble*/
	int maxDim, /**< Maximum dimension of the space*/
	bool RJ,
	int batchSize ,/**< Maximum number of data points to store*/
	int KDETrainingBatchSize,
	int updateInterval ,/**< number of steps to take before storing a sample*/
	int seed /**< Seed to use for initiating random numbers*/
	)
{
	this->KDETrainingBatchSize = KDETrainingBatchSize;
	this->batchSize = batchSize;
	this->updateInterval = updateInterval;
	this->chainN = chainN;
	this->maxDim = maxDim;
	this->RJ = RJ;
	gsl_rng_env_setup();
	r = new gsl_rng *[chainN];
	runningSTD = new double*[chainN];
	runningCov = new double**[chainN];
	runningCovCholeskyDecomp = new double**[chainN];
	runningMean = new double*[chainN];
	stepNumber = new int[chainN];
	lastUpdatePositionID = new int[chainN];
	currentStorageSize = new int[chainN];
	bandwidth = new double[chainN];
	drawCt = new int[chainN];
	storedSamples = new positionInfo**[chainN];
	//this->kde = new mlpack::kde::KDE<mlpack::kernel::GaussianKernel,mlpack::metric::EuclideanDistance,arma::mat, mlpack::tree::KDTree>*[chainN];
	//kde = new mlpack::kde::KDE<mlpack::kernel::GaussianKernel,mlpack::metric::EuclideanDistance,arma::mat, mlpack::tree::CoverTree>*[chainN];
	this->currentData = new samplerData*[chainN];
	
	this->trainingIDs = new std::vector<int>[chainN];
	const gsl_rng_type *T=gsl_rng_default;
	for(int i =0 ;i<chainN; i++){
		this->currentData[i]= nullptr;
		//this->kde[i]= nullptr;
		drawCt[i] =0;
		stepNumber[i] = 0;
		lastUpdatePositionID[i] = 0;
		currentStorageSize[i] = batchSize;
		r[i] = gsl_rng_alloc(T);
		gsl_rng_set(r[i],seed+i);
		runningSTD[i] =  new double[maxDim];
		runningCov[i] =  new double*[maxDim];
		runningCovCholeskyDecomp[i] =  new double*[maxDim];
		runningMean[i] =  new double[maxDim];
		storedSamples[i] =  new positionInfo*[batchSize];
		for(int j = 0; j < batchSize; j++){
			storedSamples[i][j] = new positionInfo(maxDim,RJ);
		}
		for(int j = 0 ; j<maxDim; j++){
			runningMean[i][j] = 0;
			runningSTD[i][j] = 0;
			runningCov[i][j] = new double[maxDim];
			runningCovCholeskyDecomp[i][j] = new double[maxDim];
			for(int k = 0 ; k<maxDim; k++){
				runningCov[i][j][k] = 0;
				runningCovCholeskyDecomp[i][j][k] = 0;
			}
		}
	}
}

KDEProposalVariables::~KDEProposalVariables()
{
	if(trainingIDs){
		delete [] trainingIDs;
		trainingIDs = nullptr;
	}

	#if _MLPACK
	if(kde){
		for(int i = 0 ; i<chainN; i++){
			if(kde[i]){
				delete kde[i];
			}
		}
		delete [] kde;
	}
	#endif
	if(storedSamples){
		for(int j = 0 ; j<chainN; j++){
			for(int i = 0 ; i < currentStorageSize[j]; i++){
				delete storedSamples[j][i];
			}
			delete [] storedSamples[j];
		}
		delete [] storedSamples;
		storedSamples = nullptr;
	}
	if(drawCt){
		delete [] drawCt;
		drawCt=nullptr;
	}
	if(bandwidth){
		delete [] bandwidth;
		bandwidth=nullptr;
	}
	if(r){
		for(int i =0 ;i<chainN; i++){
			gsl_rng_free(r[i]);
		}
		delete [] r;	
		r=nullptr;
	}
	if(stepNumber){
		delete [] stepNumber;
		stepNumber = nullptr;
	}
	if(lastUpdatePositionID){
		delete [] lastUpdatePositionID;
		lastUpdatePositionID = nullptr;
	}
	if(currentStorageSize){
		delete [] currentStorageSize;
		currentStorageSize = nullptr;
	}
	if(runningSTD){
		for(int i = 0 ; i<chainN;i++){
			delete [] runningSTD[i];
		}
		delete [] runningSTD;
		runningSTD = nullptr;
	}
	if(runningCov){
		for(int i = 0 ; i<chainN;i++){
			for(int j = 0 ; j<maxDim;j++){
				delete [] runningCov[i][j];
			}
			delete [] runningCov[i];
		}
		delete [] runningCov;
		runningCov = nullptr;
	}
	if(runningCovCholeskyDecomp){
		for(int i = 0 ; i<chainN;i++){
			for(int j = 0 ; j<maxDim;j++){
				delete [] runningCovCholeskyDecomp[i][j];
			}
			delete [] runningCovCholeskyDecomp[i];
		}
		delete [] runningCovCholeskyDecomp;
		runningCovCholeskyDecomp = nullptr;
	}
	if(runningMean){
		for(int i = 0 ; i<chainN;i++){
			delete [] runningMean[i];
		}
		delete [] runningMean;
		runningMean = nullptr;
	}
	if(currentData){
		delete [] currentData;	
		currentData = nullptr;
	}
}

//###########################################################
//###########################################################




int KDEDraw(positionInfo *sampleLocation, double **cov, positionInfo *output, gsl_rng *r,double *var)
{
	double mean[sampleLocation->dimension];
	double **temp_out  = new double*[1];
	temp_out[0] = new double[sampleLocation->dimension];
	
	for(int i = 0 ; i<sampleLocation->dimension; i++){
		//mean[i] = sampleLocation->parameters[i]/var[i];
		mean[i] = sampleLocation->parameters[i];
	}
	//gsl_error_handler_t *oldHandler = gsl_set_error_handler_off();
	int status = mvn_sample(1, mean, cov, sampleLocation->dimension, r,temp_out); 
	//gsl_set_error_handler(oldHandler);
	if(status == 0){
		output->updatePosition(sampleLocation);
		for(int i = 0 ; i<sampleLocation->dimension; i++){
			//output->parameters[i] = temp_out[0][i]*var[i];
			output->parameters[i] = temp_out[0][i];
		}
	}
	delete [] temp_out[0];
	delete [] temp_out;
	return status;
}


void KDEProposalVariables::updateStorageSize(int chainID)
{
	positionInfo **temp = storedSamples[chainID];
	currentStorageSize[chainID] +=batchSize;	
	storedSamples[chainID] = new positionInfo*[currentStorageSize[chainID]];
	for(int j = 0 ;j<currentStorageSize[chainID]-batchSize;j++){
		storedSamples[chainID][j]= new positionInfo(maxDim,RJ);
		storedSamples[chainID][j]->updatePosition(temp[j]);
	}
	for(int j = currentStorageSize[chainID] - batchSize ;j<currentStorageSize[chainID];j++){
		storedSamples[chainID][j]= new positionInfo(maxDim,RJ);
	}
	
	for(int i = 0 ; i < currentStorageSize[chainID]-batchSize; i++){
		delete temp[i];
	}
	delete [] temp;
	temp = nullptr;

}




void KDEProposalVariables::reset(int chainID)
{
	stepNumber[chainID] = 0;
	drawCt[chainID] = 0;
	lastUpdatePositionID[chainID] = 0;
	//if(kde[chainID]){
	//	delete kde[chainID];
	//	kde[chainID] = nullptr;
	//}

	for(int i = 0 ; i<currentStorageSize[chainID]; i++){
		delete storedSamples[chainID][i];
	}
	delete [] storedSamples[chainID];
	storedSamples[chainID] = new positionInfo*[batchSize];
	for(int i = 0 ;i<batchSize; i++){
		storedSamples[chainID][i] = new positionInfo(maxDim, RJ);
	}


	currentStorageSize[chainID] = batchSize;
	for(int j = 0 ; j<maxDim; j++){
		runningMean[chainID][j] = 0;
		runningSTD[chainID][j] = 0;	
		for(int k = 0 ; k<maxDim; k++){
			runningCov[chainID][j][k] = 0;	
		}
	}

	return;
}

void KDEProposalVariables::updateCov( int chainID)
{
	for(int i = 0 ; i<maxDim ; i++){
		runningMean[chainID][i] = 0;
		for(int j = 0;j<stepNumber[chainID]; j++){
			runningMean[chainID][i] += storedSamples[chainID][j]->parameters[i];
		}
		runningMean[chainID][i]/=stepNumber[chainID];
	}
	for(int i =0 ; i<maxDim ; i++){
		for(int j =0 ; j<maxDim ; j++){
			runningCov[chainID][i][j] = 0;
			
			for(int k = 0 ; k<stepNumber[chainID]; k++){
				runningCov[chainID][i][j] += (storedSamples[chainID][k]->parameters[i]-runningMean[chainID][i])*(storedSamples[chainID][k]->parameters[j]-runningMean[chainID][j]);
			}
			runningCov[chainID][i][j]/=stepNumber[chainID];
		}
		//If there's only one sample, the variance is 0..
		if(runningCov[chainID][i][i]/fabs(runningMean[chainID][i]) < 1e-15){runningCov[chainID][i][i]=1;}
		//Also just check for weird numerical errors that make variance < 0
		//runningSTD[chainID][i]=sqrt(fabs(runningSTD[chainID][i]));
		runningSTD[chainID][i]=sqrt(runningCov[chainID][i][i]);
		
	}
	return;
}
//void KDEProposalVariables::updateVar( int chainID)
//{
//	int samples = trainingIDs[chainID].size();
//	for(int i = 0 ; i<maxDim ; i++){
//		runningMean[chainID][i] = 0;
//		for(int j = 0;j<samples; j++){
//			runningMean[chainID][i] += storedSamples[chainID][trainingIDs[chainID][j]]->parameters[i];
//		}
//		runningMean[chainID][i]/=samples;
//	}
//	for(int i =0 ; i<maxDim ; i++){
//		runningSTD[chainID][i] = 0;
//		for(int k = 0 ; k<samples; k++){
//			runningSTD[chainID][i] += pow_int((storedSamples[chainID][trainingIDs[chainID][k]]->parameters[i]-runningMean[chainID][i]),2);
//		}
//		//if(samples <=2 || runningSTD[chainID][i] <= 0){runningSTD[chainID][i]=1;}
//		//if(runningSTD[chainID][i]/fabs(runningMean[chainID][i]) < 1e-15){runningSTD[chainID][i]=1;}
//		if(runningSTD[chainID][i] < 1e-15){runningSTD[chainID][i]=1;}
//		else{
//			runningSTD[chainID][i]/=samples;
//		}
//		//If there's only one sample, the variance is 0..
//		//Also just check for weird numerical errors that make variance < 0
//		//if(fabs(runningSTD[chainID][i]/runningMean[chainID][i]) <= 1e-15){runningSTD[chainID][i]+=1e-15;}
//		//std::cout<<runningSTD[chainID][i]<<std::endl;
//		//runningSTD[chainID][i]=sqrt(fabs(runningSTD[chainID][i]));
//		runningSTD[chainID][i]=sqrt(runningSTD[chainID][i]);
//		
//	}
//	return;
//}

#if _MLPACK
int KDEProposalVariables::trainKDEMLPACK(int chainID )
{
	int samples = trainingIDs[chainID].size();
	//Train KDE
	//arma::mat reference(maxDim,stepNumber[chainID] );
	//for(int i = 0 ; i<stepNumber[chainID]; i++){
	//	for(int j = 0 ; j<maxDim; j++){
	//		reference(j,i) = storedSamples[chainID][i]->parameters[j]/runningSTD[chainID][j];
	//	}
	//}
	//kde[chainID] = new mlpack::kde::KDE<mlpack::kernel::GaussianKernel,mlpack::metric::EuclideanDistance,arma::mat, mlpack::tree::KDTree>(0.0,0.01,mlpack::kernel::GaussianKernel(bandwidth[chainID]));
	//kde[chainID]->Train(reference);
	arma::mat reference(maxDim,samples);

	//###########################################
	//TESTING  	
	//###########################################
	//double **testOutput = new double*[samples];
	//for(int i = 0 ; i<samples; i++){
	//	testOutput[i] = new double[maxDim];
	//}
	//for(int j = 0 ; j<maxDim; j++){
	//	std::cout<<j << " "<<runningSTD[chainID][j]<<std::endl;;
	//}
	//std::cout<<std::endl;
	
	//###########################################
	//###########################################

	for(int i = 0 ; i<samples; i++){
		//std::cout<<trainingIDs[chainID][i]<<std::endl;
		for(int j = 0 ; j<maxDim; j++){
			reference(j,i) = storedSamples[chainID][trainingIDs[chainID][i]]->parameters[j]/runningSTD[chainID][j];
			//reference(j,i) = storedSamples[chainID][trainingIDs[chainID][i]]->parameters[j];
			//testOutput[i][j] = storedSamples[chainID][trainingIDs[chainID][i]]->parameters[j]/runningSTD[chainID][j];
		}
	}
	kde[chainID] = new mlpack::kde::KDE<mlpack::kernel::GaussianKernel,mlpack::metric::EuclideanDistance,arma::mat, mlpack::tree::KDTree>(0.0,0.01,mlpack::kernel::GaussianKernel(bandwidth[chainID]));
	//kde[chainID] = new mlpack::kde::KDE<mlpack::kernel::GaussianKernel,mlpack::metric::EuclideanDistance,arma::mat, mlpack::tree::CoverTree>(0.0,0.01,mlpack::kernel::GaussianKernel(bandwidth[chainID]));
	

	//###########################################
	//TESTING  	
	//###########################################
	
	//writeCSVFile("data/KDE_TEST.csv",testOutput, samples,maxDim);	
	//for(int i = 0 ; i<samples; i++){
	//	delete [] testOutput[i];
	//}
	//delete [] testOutput;

	//###########################################
	

	kde[chainID]->Train(reference);

	return 0 ;
}
double KDEProposalVariables::evalKDEMLPACK(positionInfo *position,int chainID)
{
	double val = 0;
	arma::mat query(maxDim, 1);
	arma::vec estimation;
	for(int i = 0 ; i<maxDim; i++){
		query(i,0) = position->parameters[i]/runningSTD[chainID][i];
		//query(i,0) = position->parameters[i];
	}
	kde[chainID]->Evaluate(query,estimation);
	val = estimation(0);
	
	return val;

}

#endif

int KDEProposalVariables::trainKDECustom(int chainID)
{
	gsl_matrix *matrix = gsl_matrix_alloc(maxDim, maxDim);
	gsl_matrix *matrix_inv = gsl_matrix_alloc(maxDim, maxDim);
	for(int i = 0 ; i<maxDim; i++){
		for(int j = 0 ; j<maxDim; j++){
			gsl_matrix_set(matrix, i, j , runningCov[chainID][i][j]);
		}
	}
	//gsl_error_handler_t *oldHandler = gsl_set_error_handler_off();
	int status = 0 ;
	//std::cout<<status<<std::endl;
	gsl_permutation *p = gsl_permutation_alloc(maxDim);
	status = gsl_linalg_pcholesky_decomp(matrix,p);
	//std::cout<<status<<std::endl;
	if(status == 0){
		status = gsl_linalg_pcholesky_invert(matrix, p, matrix_inv);
	}
	if(status == 0){
		status = gsl_linalg_cholesky_decomp1(matrix_inv);
	}
	//gsl_set_error_handler(oldHandler);
	
	if(status == 0){
		for(int i = 0 ; i<maxDim; i++){
			for(int j = 0 ; j<maxDim; j++){
				if(j<=i){
					runningCovCholeskyDecomp[chainID][i][j] = gsl_matrix_get(matrix_inv,i,j);
				}
				else{
					runningCovCholeskyDecomp[chainID][i][j] = 0;
				}
			}
		}
	}
	gsl_permutation_free(p);
	gsl_matrix_free(matrix);
	gsl_matrix_free(matrix_inv);
	return status ;
}

int KDEProposalVariables::trainKDE(int chainID)
{
	//if(kde[chainID]){
	//	delete kde[chainID];
	//	kde[chainID] = nullptr;
	//}
	int samples;
	if(KDETrainingBatchSize > 0 ){
		samples = (stepNumber[chainID] > KDETrainingBatchSize ) ? KDETrainingBatchSize : stepNumber[chainID] ;
	}
	else{
		samples = stepNumber[chainID];
	}

	int localDim = (RJ) ? 1 : maxDim ;	
	bandwidth[chainID] = pow(samples,-1./(localDim + 4));


	//Pick ids

	//if ( trainingIDs[chainID].size() != samples){trainingIDs[chainID].resize(samples);}
	trainingIDs[chainID].resize(stepNumber[chainID]);
	for(int i = 0 ; i<stepNumber[chainID]; i++){
		trainingIDs[chainID].at(i) = i;		
	}
	//std::random_device rd;
	//std::mt19937 g(rd());
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 g(seed);
	std::shuffle(trainingIDs[chainID].begin(),trainingIDs[chainID].end(), g);	
	trainingIDs[chainID].resize(samples);

	//Allocate random number
	//const gsl_rng_type *T = gsl_rng_default;
	//gsl_rng *r = gsl_rng_alloc(T);
	//gsl_rng_set(r, stepNumber[chainID]*chainID);
	//Remove random IDS until you have the right number
	//if(samples != stepNumber[chainID]) {	
	//	for (int i = 0 ; i<stepNumber[chainID] - samples; i++){
	//		int alpha = (int) ( gsl_rng_uniform(r)*trainingIDs[chainID].size());
	//		trainingIDs[chainID].erase(trainingIDs[chainID].begin(),trainingIDs[chainID].begin() +alpha);
	//	}
	//}


	//calculate Covariance 
	//updateVar( chainID);
	updateCov( chainID);

	int status = 0;
	#if _MLPACK
	if(useMLPack){
		status = trainKDEMLPACK(chainID);	
	}
	else{
		status = trainKDECustom(chainID);	
	}
	#else
	status = trainKDECustom(chainID);	
	#endif

	return status;
	
}

double KDEProposalVariables::evalKDE(positionInfo *position,int chainID)
{
	#if _MLPACK
	if(useMLPack){
		return evalKDEMLPACK(position,chainID);
	}
	else{
		return evalKDECustom(position,chainID);
	}
	#else
	return evalKDECustom(position,chainID);
	#endif
}


double KDEProposalVariables::evalKDECustom(positionInfo *position,int chainID)
{
	double val = 0;
	double norm = pow(2.*M_PI, -1*maxDim/2.);
	for(int i = 0 ; i<maxDim; i++){
		norm *= runningCovCholeskyDecomp[chainID][i][i];
	}

	double *data_points_whitened = new double[maxDim];
	double *eval_points_whitened = new double[maxDim];
	matrixDot(runningCovCholeskyDecomp[chainID], position->parameters,eval_points_whitened, maxDim, maxDim );
	for(int i = 0 ; i<trainingIDs[chainID].size(); i++){
		int id = trainingIDs[chainID][i];
		matrixDot(runningCovCholeskyDecomp[chainID], storedSamples[chainID][id]->parameters,data_points_whitened,maxDim, maxDim );
		double arg=0;
		for(int j = 0; j<maxDim; j++){
			double residual = (eval_points_whitened[j] - data_points_whitened[j]);
			arg += residual *residual;
		}
		val += std::exp(-.5 * arg)*norm;
			
	}
	
	delete [] data_points_whitened;
	delete [] eval_points_whitened;
	return val;

}

/*Just use a premade KDE package.. Why do this from scratch?*/
void KDEProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler, double *MHRatioCorrection)
{
	//std::cout<<"KDE"<<" "<<chainID<<std::endl;
	int currentStep = data->currentStepID[chainID];

	positionInfo *currentPosition = data->positions[chainID][currentStep];
	positionInfo *proposedPosition = data->positions[chainID][currentStep+1];

	proposedPosition->updatePosition(currentPosition);
	
	//Not working on RJ yet
	if(sampler->RJ){
		return;
	}
	
	//std::cout<<chainID<<std::endl;
	KDEProposalVariables *kdepv = (KDEProposalVariables *)(sampler->proposalFns->proposalFnVariables[stepID]);

	//Reset data harvesting parameters if data structure has changed
	if(data!=kdepv->currentData[chainID]){
		//std::cout<<chainID<<std::endl;
		//kdepv->lastUpdatePositionID[chainID]=0;
		kdepv->reset(chainID);
		kdepv->currentData[chainID] = data;
	}
	
	if( ( ( (currentStep-1) - kdepv->lastUpdatePositionID[chainID] )/kdepv->updateInterval > 1 )){
		int positionUpdates = ((currentStep-1) - kdepv->lastUpdatePositionID[chainID] )/kdepv->updateInterval;

		//std::cout<<chainID<<" "<<positionUpdates<<std::endl;
		while( ( kdepv->stepNumber[chainID] +positionUpdates - (kdepv->currentStorageSize[chainID]-1)  ) >= 0 ){
			//update storage size to storagesize + batchsize 
			kdepv->updateStorageSize(chainID);
		}
		//Update storage	
		for(int i = 0 ; i<positionUpdates;i++){
			kdepv->lastUpdatePositionID[chainID]+=kdepv->updateInterval;
			kdepv->storedSamples[chainID][kdepv->stepNumber[chainID]]->updatePosition(data->positions[chainID][kdepv->lastUpdatePositionID[chainID] ]);
			kdepv->stepNumber[chainID]+=1;
			//std::cout<<kdepv->lastUpdatePositionID[chainID]<<" "<<currentStep<<" "<<chainID<<std::endl;
		}
		
		//Update var and mean
		//kdepv->updateCov( chainID);
		
		//train kde on new reference matrix
		if(kdepv->stepNumber[chainID] <=100 ){return;}
		int status = kdepv->trainKDE(chainID);
		if(status != 0 ){
			return;
		}
	}

	if(kdepv->stepNumber[chainID] <=100 ){return;}

	//int burnIterations = sampler->burnIterations;
	int internalDim = sampler->maxDim;
	if(sampler->RJ && sampler->minDim !=0 ){
		internalDim = sampler->minDim;
	}
	else if(sampler->RJ){
		std::cout<<"KDE DOESN'T WORK WITH RJ YET"<<std::endl;
	}

	/* Draw random sample and compute probability from KDE for each step*/
	//int sampleID = gsl_rng_uniform(sampler->rvec[chainID])*kdepv->stepNumber[chainID];
	//positionInfo *samplePosition = kdepv->storedSamples[chainID][sampleID];

	int sampleID = kdepv->trainingIDs[chainID][(int)(gsl_rng_uniform(sampler->rvec[chainID])*kdepv->trainingIDs[chainID].size())];
	positionInfo *samplePosition = kdepv->storedSamples[chainID][sampleID];

	double **cov = new double*[sampler->maxDim];
	if(kdepv->useMLPack){
		for(int i = 0 ; i<sampler->maxDim; i++){
			cov[i] = new double[sampler->maxDim];
			for(int j = 0 ; j<sampler->maxDim; j++){
				cov[i][j] = 0;
			}
			cov[i][i] = kdepv->bandwidth[chainID]*kdepv->bandwidth[chainID];
		}
	}
	else{
		for(int i = 0 ; i<sampler->maxDim; i++){
			cov[i] = new double[sampler->maxDim];
			for(int j = 0 ; j<sampler->maxDim; j++){
				cov[i][j] = kdepv->runningCov[chainID][i][j]*kdepv->bandwidth[chainID]*kdepv->bandwidth[chainID];
			}
		}

	}
	int status = KDEDraw(samplePosition,cov,proposedPosition,kdepv->r[chainID],kdepv->runningSTD[chainID]);

	if(status ==0){

	
		double evalFormer, evalProposed;
			
		evalFormer = std::log(kdepv->evalKDE(currentPosition,chainID));
		evalProposed = std::log(kdepv->evalKDE(proposedPosition,chainID));

		//update MH ratio
		*MHRatioCorrection +=evalFormer;
		*MHRatioCorrection -=evalProposed;

		//######################################3
		//TESTING
		//if(chainID < sampler->ensembleN){
		//for(int i = 0 ; i<sampler->maxDim;i++){	
		//	std::cout<<i<<" "<<currentPosition->parameters[i]<<" "<<proposedPosition->parameters[i]<<std::endl;
		//}
		//std::cout<<std::exp(evalFormer)<<" "<<std::exp(evalProposed)<<std::endl;
		//std::cout<<std::endl;
		//}
		
		//######################################3
	}

	for(int i = 0 ; i<sampler->maxDim; i++){
		delete [] cov[i];
	}
	delete [] cov;

	return;
}

}
