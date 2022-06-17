#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <iostream>

namespace bayesship{

/*! \file 
 *
 * # Source file for the random RJ  proposal template
 */


/*! \brief Proposal for RJMCMC that randomly adds or subtracts dimensions between minDim and maxDim
 *
 * Probability to add a dimension: alpha
 *
 * Probability to remove a dimension: 1 - alpha
 *
 * If you have a maximum dimension of 10 and minimum dimension of 5, an example of a series of proposals would be:
 *
 * 	Start with 5 dimensions;
 *
 * 	Propose a random 6th (accept);
 *
 * 	Propose a random 7th (reject);
 *
 * 	propose removing a random dimension (move back to 5) (reject);
 *
 * 	Propose a random 7th (accept);
 *
 * 	etc
 *
 * If priorRanges is provided, it proposes a uniform number in that range. Otherwise, uses a uniform number between (0,1)
 */
void randomLayerRJProposal::propose(positionInfo *currentStep, positionInfo *proposedStep, int chainID,int stepID,double *MHRatioModifications)
{ 
	*MHRatioModifications = 0;

	proposedStep->updatePosition(currentStep);

	int activeDims = 0;
	
	for(int i =0 ; i<sampler->maxDim; i++){
		activeDims += currentStep->status[i];
	}

	double prob = gsl_rng_uniform(sampler->rvec[chainID]);

	//##################################################
	//create
	if(prob < alpha ){
		if( activeDims < sampler->maxDim){
				
			//Pick random, inactive dimension
			int id = (sampler->maxDim -activeDims)*gsl_rng_uniform(sampler->rvec[chainID]);
			int ct=0;
			for(int i = 0 ; i<sampler->maxDim; i++){
				if (!proposedStep->status[i] )
				{
					if(ct == id){
						id = i;
						break;
					}
					ct+=1;	
				}
			}

			proposedStep->status[id] = 1;
			if(sampler->priorRanges){
				proposedStep->parameters[id] = gsl_rng_uniform(sampler->rvec[chainID])*(sampler->priorRanges[id][1]-sampler->priorRanges[id][0])+sampler->priorRanges[id][0];
				//*MHRatioModifications -=std::log(1./(sampler->priorRanges[lastID+1][1]-sampler->priorRanges[lastID+1][0])) ;
				*MHRatioModifications -=std::log(1./(sampler->priorRanges[id][1]-sampler->priorRanges[id][0])) ;
			}
			else{
				proposedStep->parameters[id] = gsl_rng_uniform(sampler->rvec[chainID]);
			}
			*MHRatioModifications+=std::log((1.-alpha)/(alpha));
			//proposedStep->parameters[P+1] = gsl_rng_uniform(h->r);
			//*MHRatioModifications-=std::log( 1./(20.)  );
		}

	}
	//Kill 
	else{
		if( activeDims >sampler->minDim){

			//Pick random, active dimension
			int id = (activeDims-sampler->minDim)*gsl_rng_uniform(sampler->rvec[chainID]);
			int ct=0;
			for(int i = sampler->minDim ; i<sampler->maxDim; i++){
				if (proposedStep->status[i] )
				{
					if(ct == id){
						id = i;
						break;
					}
					ct+=1;	
				}
			}
			proposedStep->status[id] = 0;
			proposedStep->parameters[id] = 0;
			if(sampler->priorRanges){
				*MHRatioModifications +=std::log(1./(sampler->priorRanges[id][1]-sampler->priorRanges[id][0])) ;
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

randomLayerRJProposal::randomLayerRJProposal(bayesshipSampler *sampler,double alpha )
{
	this->alpha = alpha;
	this->sampler = sampler;
	return;
	
}

}
