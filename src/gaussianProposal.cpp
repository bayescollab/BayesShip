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

void gaussianProposal::writeCheckpoint(  std::string outputDirectory, std::string runMoniker)
{
	
	std::string outfile(outputDirectory+runMoniker + "_checkpoint_gaussianProposalVariables.json");
	std::cout<<"Writing Gaussian checkpoint to "<<outfile<<std::endl;

	nlohmann::json j;
	
	for(int i = 0 ; i<sampler->ensembleN*sampler->ensembleSize; i++){
		j["Gaussian Widths"]["Chain "+std::to_string(i)] 
			= std::vector<double>( gaussWidths[i],gaussWidths[i] + sampler->maxDim);
	}
	std::ofstream fileOut(outfile);
	fileOut << j;

	return;
}
void gaussianProposal::loadCheckpoint(  std::string inputDirectory, std::string runMoniker)
{
	std::string inputFile(inputDirectory+runMoniker + + "_checkpoint_gaussianProposalVariables.json");
	if(!checkDirExist(inputFile)){
		return;
	}
	std::cout<<"Loading Gaussian checkpoint to "<<inputFile<<std::endl;

	nlohmann::json j ;
	std::ifstream fileIn(inputFile);
	fileIn>>j;

	for(int i = 0 ; i<sampler->ensembleN * sampler->ensembleSize; i++){
		std::vector<double> widthTemp;
		j["Gaussian Widths"]["Chain "+std::to_string(i)].get_to(widthTemp);
		for(int j = 0 ; j<sampler->maxDim;j++){
			gaussWidths[i][j] = widthTemp[j];
		}
	}
	return;

}


/*! Constructor function*/
gaussianProposal::gaussianProposal(
	int chainN, /**< Number of chains in the ensemble*/
	int maxDim, /**< Maximum dimension of the space*/
	bayesshipSampler *sampler,
	int seed /**< Seed to use for initiating random numbers*/
	)
{
	this->chainN = chainN;
	this->maxDim = maxDim;
	gsl_rng_env_setup();
	r = new gsl_rng *[chainN];
	gaussWidths = new double*[chainN];
	this->sampler=sampler;

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
gaussianProposal::~gaussianProposal()
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

void gaussianProposal::propose(positionInfo *currentPosition, positionInfo *proposedPosition, int chainID, int stepID, double *MHRatioModifications)
{
	proposedPosition->updatePosition(currentPosition);

	samplerData *data = sampler->getActiveData();
	
	/*! Update step widths based on accepted fraction*/
	/*! TODO needs revamp -- shouldn't update all the time -- shooting for 20%*/
	if(sampler->burnPeriod){
	//if(false){
		//Accepts haven't changed -- proposal was rejected
		if(previousAccepts[chainID] == data->successN[chainID][stepID]){
			gaussWidths[chainID][previousDimID[chainID]]*=.9;
		}
		//Accepts have changed -- proposal was accepted
		else{
			gaussWidths[chainID][previousDimID[chainID]]*=1.1;

		}
	}
	int currentDim = currentPosition->countActiveDimensions();
		
	int beta = (int) (gsl_rng_uniform(r[chainID])*currentDim);
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
	//if(false){
		previousDimID[chainID] = dim;
		previousAccepts[chainID] = data->successN[chainID][stepID];
	}
	proposedPosition->parameters[dim] +=   gsl_ran_gaussian(r[chainID],gaussWidths[chainID][dim]);
	return;
}

}
