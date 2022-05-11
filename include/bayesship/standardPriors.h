#ifndef STANDARDPRIORS_H
#define STANDARDPRIORS_H

#include "bayesship/bayesshipSampler.h"
#include "bayesship/dataUtilities.h"
namespace bayesship{

/*! \file 
 *
 * # Header file for the general priors
 */

class uniformPrior: public probabilityFn
{
public:
	bayesshipSampler *sampler;
	double eval(positionInfo *position, int chainID);
};

//double uniformPrior(positionInfo *position, int chainID, bayesshipSampler *sampler, void *userParameters);

}

#endif
