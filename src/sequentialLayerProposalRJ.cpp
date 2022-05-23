#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <iostream>

namespace bayesship{

/*! \file 
 *
 * # Source file for the sequential RJ  proposal template
 */


/*! \brief Proposal for RJMCMC that sequentially adds or subtracts dimensions between minDim and maxDim
 *
 * Probability to add a dimension: alpha
 *
 * Probability to remove a dimension: 1 - alpha
 *
 * If you have a maximum dimension of 10 and minimum dimension of 5, an example of a series of proposals would be:
 *
 * 	Start with 5 dimensions;
 *
 * 	Propose a 6th (accept);
 *
 * 	Propose a 7th (reject);
 *
 * 	propose removing a dimension (move back to 5) (reject);
 *
 * 	Propose a 7th (accept);
 *
 * 	etc
 *
 * If priorRanges is provided, it proposes a uniform number in that range. Otherwise, uses a uniform number between (0,1)
 */
void sequentialLayerRJProposal::propose(positionInfo *currentStep, positionInfo *proposedStep, int chainID,int stepID,double *MHRatioModifications)
{ 
	*MHRatioModifications = 0;

	proposedStep->updatePosition(currentStep);

	int activeDims = 0;
	
	for(int i =0 ; i<sampler->maxDim; i++){
		activeDims += currentStep->status[i];
	}

	double prob = gsl_rng_uniform(sampler->rvec[chainID]);

	int lastID = activeDims - 1;// ID of the last active dimension

	//##################################################
	//create
	if(prob < alpha ){
		if( activeDims < sampler->maxDim){
			proposedStep->status[lastID+1] = 1;
			if(sampler->priorRanges){
				proposedStep->parameters[lastID+1] = gsl_rng_uniform(sampler->rvec[chainID])*(sampler->priorRanges[lastID+1][1]-sampler->priorRanges[lastID + 1][0])+sampler->priorRanges[lastID + 1][0];
				//*MHRatioModifications -=std::log(1./(sampler->priorRanges[lastID+1][1]-sampler->priorRanges[lastID+1][0])) ;
				*MHRatioModifications -=std::log(1./(sampler->priorRanges[lastID+1][1]-sampler->priorRanges[lastID+1][0])) ;
			}
			else{
				proposedStep->parameters[lastID+1] = gsl_rng_uniform(sampler->rvec[chainID]);
			}
			*MHRatioModifications+=std::log((1.-alpha)/(alpha));
			//proposedStep->parameters[P+1] = gsl_rng_uniform(h->r);
			//*MHRatioModifications-=std::log( 1./(20.)  );
		}

	}
	//Kill 
	else{
		if( activeDims >sampler->minDim){
			proposedStep->status[lastID] = 0;
			proposedStep->parameters[lastID] = 0;
			if(sampler->priorRanges){
				*MHRatioModifications +=std::log(1./(sampler->priorRanges[lastID][1]-sampler->priorRanges[lastID][0])) ;
			}
			*MHRatioModifications+=std::log(alpha/(1.-alpha));
			//*MHRatioModifications+=std::log(1./(20.));
			//if(P-1 == 1){
			//	*MHRatioModifications+=std::log(1./(.5));
			//}
		}
	}
	//Create 

	return ;
}

sequentialLayerRJProposal::sequentialLayerRJProposal(bayesshipSampler *sampler,double alpha )
{
	this->alpha = alpha;
	this->sampler = sampler;
	return;
	
}

}
