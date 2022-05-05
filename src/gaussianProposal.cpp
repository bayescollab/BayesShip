#include "bayesship/proposalFunctions.h"
#include "bayesship/dataUtilities.h"
#include "bayesship/bayesshipSampler.h"
#include "bayesship/utilities.h"
#include <gsl/gsl_randist.h>
#include <nlohmann/json.hpp>
#include <fstream>

/*! \file 
 *
 * # Source file for the Gaussian  proposal template
 */

namespace bayesship{

void gaussianProposalWriteCheckpoint( void *var , bayesshipSampler *sampler)
{
	std::string outfile(sampler->outputDir+sampler->outputFileMoniker + "_checkpoint_gaussianProposalVariables.json");
	std::cout<<"Writing Gaussian checkpoint to "<<outfile<<std::endl;
	gaussianProposalVariables *gpv = (gaussianProposalVariables *)var;	

	nlohmann::json j;
	
	for(int i = 0 ; i<sampler->ensembleN*sampler->ensembleSize; i++){
		j["Gaussian Widths"]["Chain "+std::to_string(i)] 
			= std::vector<double>( gpv->gaussWidths[i],gpv->gaussWidths[i] + sampler->maxDim);
	}
	std::ofstream fileOut(outfile);
	fileOut << j;

	return;
}
void gaussianProposalLoadCheckpoint( void *var,bayesshipSampler *sampler)
{
	std::string inputFile(sampler->outputDir+sampler->outputFileMoniker + "_checkpoint_gaussianProposalVariables.json");
	if(!checkDirExist(inputFile)){
		return;
	}
	std::cout<<"Loading Gaussian checkpoint to "<<inputFile<<std::endl;
	gaussianProposalVariables *gpv = (gaussianProposalVariables *)var;

	nlohmann::json j ;
	std::ifstream fileIn(inputFile);
	fileIn>>j;

	for(int i = 0 ; i<sampler->ensembleN * sampler->ensembleSize; i++){
		std::vector<double> widthTemp;
		j["Gaussian Widths"]["Chain "+std::to_string(i)].get_to(widthTemp);
		for(int j = 0 ; j<sampler->maxDim;j++){
			gpv->gaussWidths[i][j] = widthTemp[j];
		}
	}
	return;

}


/*! Constructor function*/
gaussianProposalVariables::gaussianProposalVariables(
	int chainN, /**< Number of chains in the ensemble*/
	int maxDim, /**< Maximum dimension of the space*/
	int seed /**< Seed to use for initiating random numbers*/
	)
{
	this->chainN = chainN;
	this->maxDim = maxDim;
	gsl_rng_env_setup();
	r = new gsl_rng *[chainN];
	gaussWidths = new double*[chainN];

	const gsl_rng_type *T=gsl_rng_default;

	for(int i =0 ;i<chainN; i++){
		r[i] = gsl_rng_alloc(T);
		gsl_rng_set(r[i],seed+i);
		
		gaussWidths[i] = new double[maxDim];
		for(int j = 0 ; j<maxDim; j++){
			gaussWidths[i][j] = 1;
		}
	}
	previousDimID = new int[chainN];
	
	previousAccepts = new int[chainN];
	for(int i = 0 ; i<chainN; i++){
		previousDimID[i] = 0;
		previousAccepts[i] = 0;
	}
}
gaussianProposalVariables::~gaussianProposalVariables()
{
	if(r){
		for(int i =0 ;i<chainN; i++){
			gsl_rng_free(r[i]);
		}
		delete [] r;	
		r=nullptr;
	}
	if(gaussWidths){
		for(int i =0 ;i<chainN; i++){
			delete [] gaussWidths[i];
		}
		delete [] gaussWidths;	
	
		gaussWidths=nullptr;
	}
	if(previousDimID){
		delete [] previousDimID;
		previousDimID = nullptr;
	}
	if(previousAccepts){
		delete [] previousAccepts;
		previousAccepts = nullptr;
	}
}


/*! \brief Proposal based on random normal distribution around the current point 
 *
 * Symmetric, so MH corrections are 0
 *
 */ 

void gaussianProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler, double *MHRatioCorrection)
{
	positionInfo *currentPosition = data->positions[chainID][data->currentStepID[chainID]];
	positionInfo *proposedPosition = data->positions[chainID][data->currentStepID[chainID]+1];
	gaussianProposalVariables *gpv = (gaussianProposalVariables *)sampler->proposalFns->proposalFnVariables[stepID];	
	proposedPosition->updatePosition(currentPosition);
	
	/*! Update step widths based on accepted fraction*/
	/*! TODO needs revamp -- shouldn't update all the time -- shooting for 20%*/
	if(sampler->burnPeriod){
	//if(false){
		//Accepts haven't changed -- proposal was rejected
		if(gpv->previousAccepts[chainID] == data->successN[chainID][stepID]){
			gpv->gaussWidths[chainID][gpv->previousDimID[chainID]]*=.9;
		}
		//Accepts have changed -- proposal was accepted
		else{
			gpv->gaussWidths[chainID][gpv->previousDimID[chainID]]*=1.1;

		}
	}
	int currentDim = currentPosition->countActiveDimensions();
		
	int beta = (int) (gsl_rng_uniform(gpv->r[chainID])*currentDim);
	int dim = beta;
	if(sampler->RJ){
		dim = 0 ;
		for(int i =  0 ;i<sampler->maxDim; i++){
			if(dim==beta){
				break;
			}
			if(currentPosition->status[i]==1){
				dim +=1;
			}	
		}
	}

	if(sampler->burnPeriod){
		gpv->previousDimID[chainID] = dim;
		gpv->previousAccepts[chainID] = data->successN[chainID][stepID];
	}
	proposedPosition->parameters[dim] +=   gsl_ran_gaussian(gpv->r[chainID],gpv->gaussWidths[chainID][dim]);
	return;
}

}
