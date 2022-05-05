#include "bayesship/standardPriors.h"
#include "bayesship/dataUtilities.h"
#include "bayesship/bayesshipSampler.h"
#include <math.h>


/*! \file 
 *
 * Source file for the uniform prior template
 */

namespace bayesship{

double uniformPrior(positionInfo *position, int chainID, bayesshipSampler *sampler, void *userparameters)
{
	double lp = 0;
	if(sampler->RJ){
		for(int i= 0; i< sampler->maxDim; i++){
			if(position->status[i] == 1){
				if(position->parameters[i]<sampler->priorRanges[i][0] || position->parameters[i]>sampler->priorRanges[i][1] ){
					return limitInf;
				}
				else{
					lp-= std::log(sampler->priorRanges[i][1] - sampler->priorRanges[i][0]);//Minus because 1/range
				}
			}
		}
			
	}
	else{
		for(int i= 0; i< sampler->maxDim; i++){
			if(position->parameters[i]<sampler->priorRanges[i][0] || position->parameters[i]>sampler->priorRanges[i][1] ){
				return limitInf;
			}
			else{
				lp-= std::log(sampler->priorRanges[i][1] - sampler->priorRanges[i][0]);//Minus because 1/range
			}
		}
		
	}
	//return lp;
	return 1;
}
}
