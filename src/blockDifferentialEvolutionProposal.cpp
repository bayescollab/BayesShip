#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <gsl/gsl_randist.h>

/*! \file 
 *
 * # Source file for the block differential evolution  proposal template
 *
 */

namespace bayesship{
blockDifferentialEvolutionProposal::blockDifferentialEvolutionProposal(bayesshipSampler *sampler, std::vector<std::vector<int>> blocks, std::vector<double> blockProb)
{
	this->blocks = std::vector<std::vector<int>>(blocks.size());
	this->blockProb = std::vector<double>(blocks.size());
	this->blockProbBoundaries = std::vector<double>(blocks.size());
	for(int i = 0 ; i<blocks.size(); i++){
		this->blockProb[i] = blockProb[i];
		this->blocks[i] = std::vector<int>(blocks[i].size());
		for(int j =0 ; j<blocks[i].size(); j++){
			this->blocks[i][j] = blocks[i][j];
		}
		
	}
	//Need the boundaries to do the random number generation easier. Storing for convenience and speed
	//For prob array {.3,.4,.3}, boundaries should be {.3,.7,1.}
	this->blockProbBoundaries[0] = blockProb[0];
	for(int i = 1 ; i<blocks.size(); i++){
		this->blockProbBoundaries[i] = 	this->blockProbBoundaries[i-1]+ this->blockProb[i];
	}
	this->sampler = sampler;
}

void blockDifferentialEvolutionProposal::propose(positionInfo *currentPosition, positionInfo *proposedPosition, int chainID,int stepID,double *MHRatioModifications)
{
	samplerData *data = sampler->getActiveData();
	int currentStep = data->currentStepID[chainID];

	proposedPosition->updatePosition(currentPosition);


	double beta = gsl_rng_uniform(sampler->rvec[chainID]);
	int blockID = 0 ;
	for(int i = 0 ; i<blocks.size(); i++){
		if (beta < blockProbBoundaries[i]){
			blockID = i;
			break;
		}
	}

	/*Not doing RJ version yet..*/
	if(currentStep < 2 ){
		return;
	}
	positionInfo *historyPosition1=nullptr;
	positionInfo *historyPosition2=nullptr;

	if(sampler->burnPeriod || !sampler->burnData){
	//if(true){
		int id1, id2=-1 ;
		id1 = (int) (gsl_rng_uniform(sampler->rvec[chainID])*currentStep);
		do{
			id2 = (int) (gsl_rng_uniform(sampler->rvec[chainID])*currentStep);
		}while(id1 ==id2)	;
		historyPosition1 = data->positions[chainID][id1];
		historyPosition2 = data->positions[chainID][id2];
	}
	else{
		int burnIterations = sampler->burnData->currentStepID[chainID];
		int id1, id2=-1 ;
		id1 = (int)(gsl_rng_uniform(sampler->rvec[chainID])*(currentStep+burnIterations));
		do{
			id2 = (int)(gsl_rng_uniform(sampler->rvec[chainID])*(currentStep+burnIterations));
		}while(id1 ==id2)	;
		if(id1 >= burnIterations){
			historyPosition1 = data->positions[chainID][id1-burnIterations];
		}
		else {
			historyPosition1 = sampler->burnData->positions[chainID][id1];

		}
		if(id2 >= burnIterations){
			historyPosition2 = data->positions[chainID][id2-burnIterations];
		}
		else {
			historyPosition2 = sampler->burnData->positions[chainID][id2];

		}

	}

	//double stepWidth = 1;
	double  alpha = gsl_rng_uniform(sampler->rvec[chainID]);
	double gamma = 1;
	if( alpha < .9){
		//gamma = gsl_ran_gaussian(sampler->rvec[chainID],2.38/std::sqrt(2.*blocks[blockID].size()));
		//gamma = gsl_ran_gaussian(sampler->rvec[chainID],2.38/std::sqrt(2.*sampler->maxDim));
		gamma = gsl_ran_gaussian(sampler->rvec[chainID],1)*2.38/std::sqrt(2.*sampler->maxDim);
	}
	for(int i = 0 ; i<blocks[blockID].size(); i++){
		int paramID = blocks[blockID][i];
		proposedPosition->parameters[paramID] +=
			gamma*(historyPosition1->parameters[paramID]-historyPosition2->parameters[paramID]);
	}

	//std::cout<<blockID<<", ";
	//for(int i = 0 ; i<proposedPosition->dimension; i++){
	//	std::cout<<std::setprecision(5)<<proposedPosition->parameters[i] -currentPosition->parameters[i]<<", ";
	//}
	//std::cout<<std::endl;

}


}
