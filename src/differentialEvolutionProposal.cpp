#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <gsl/gsl_randist.h>

/*! \file 
 *
 * # Source file for the differential evolution  proposal template
 */

namespace bayesship{
differentialEvolutionProposal::differentialEvolutionProposal(bayesshipSampler *sampler)
{
	this->sampler = sampler;
}

void differentialEvolutionProposal::propose(positionInfo *currentPosition, positionInfo *proposedPosition, int chainID,int stepID,double *MHRatioModifications)
{
	samplerData *data = sampler->getActiveData();
	int currentStep = data->currentStepID[chainID];

	proposedPosition->updatePosition(currentPosition);

	/*Not doing RJ version yet..*/
	if(currentStep < 2 ){
	//if(true){
		return;
	}
	positionInfo *historyPosition1=nullptr;
	positionInfo *historyPosition2=nullptr;
	int internalDim = sampler->maxDim;
	if(sampler->RJ && sampler->minDim !=0 ){
		internalDim = sampler->minDim;
	}
	else if(sampler->RJ){
		std::cout<<"DIFFERENTIAL EVOLUTION DOESN'T WORK WITH RJ YET"<<std::endl;
		return;
	}

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
	double stepWidth = 2.38/std::sqrt(2.*internalDim);
	double alpha = gsl_ran_gaussian(sampler->rvec[chainID],stepWidth);
	for(int i = 0 ; i<internalDim; i++){
		proposedPosition->parameters[i] +=
			alpha*(historyPosition1->parameters[i]-historyPosition2->parameters[i]);
	}

}


}
