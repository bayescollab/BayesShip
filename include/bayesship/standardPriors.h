#ifndef STANDARDPRIORS_H
#define STANDARDPRIORS_H

#include "bayesship/bayesshipSampler.h"
#include "bayesship/dataUtilities.h"
namespace bayesship{

/*! \file 
 *
 * # Header file for the general priors
 */

double uniformPrior(positionInfo *position, int chainID, bayesshipSampler *sampler, void *userParameters);

}

#endif
