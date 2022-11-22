#include <iostream>
#include <fstream>
#include <bayesship/bayesshipSampler.h>
#include <bayesship/utilities.h>
#include <bayesship/dataUtilities.h>
#include <bayesship/proposalFunctions.h>
#include <limits.h>
#include <math.h>

class bimodal2DPrior: public bayesship::probabilityFn
{
public:
	virtual double eval(bayesship::positionInfo *position, int chainID)
	{
		if(fabs(position->parameters[0]) > 4 ){
			return -std::numeric_limits<double>::infinity();
		}
		else if(fabs(position->parameters[1]) > 4 ){
			return -std::numeric_limits<double>::infinity();
		}
		return 0;

	}

};

class bimodal2DLikelihood: public bayesship::probabilityFn
{
public:
	virtual double eval(bayesship::positionInfo *position, int chainID)
	{
		//return 2;
		double x = position->parameters[0];
		double y = position->parameters[1];
		double power1 = -x*x - (9. + 4 * x * x + 8 *y)*(9. + 4 * x * x + 8 *y);
		double power2 = -8*x*x - 8*(y-2)*(y-2);
		return log( 16. / 3. / M_PI *( exp(power1) + .5* exp(power2)));

	}

};


int main(int argc, char *argv[])
{

	bimodal2DPrior *prior = new bimodal2DPrior();
	bimodal2DLikelihood *likelihood = new bimodal2DLikelihood();
	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(likelihood,prior);
	sampler->outputDir = "data/";
	sampler->outputFileMoniker = "multiModalLikelihoodTest";

	sampler->maxDim = 2;

	double **priorRange = new double*[sampler->maxDim];
	for(int i = 0 ; i<sampler->maxDim; i++){
		priorRange[i] = new double[2];
		priorRange[i][0] = -4;
		priorRange[i][1] = 4;
	}
	sampler->priorRanges = priorRange;
	//sampler->burnPriorIterations =1000;
	//sampler->priorIterations =10000;
	sampler->burnPriorIterations =1000;
	sampler->priorIterations =10000;
	sampler->writePriorData = true;

	sampler->coldOnlyStorage=true;

	sampler->threadPool = true;

	sampler->independentSamples = 1000;
	//sampler->independentSamples = 45*100;
	sampler->burnIterations = 1000;
	sampler->swapProb=.1;	
	sampler->ensembleN = 5;
	sampler->ensembleSize=4;
	sampler->threads = 4;
	sampler->restrictSwapTemperatures = false;
	sampler->randomizeSwapping = true;
	int chainN = sampler->ensembleSize*sampler->ensembleN;
	sampler->initialPosition = new bayesship::positionInfo(sampler->maxDim,false);
	for(int i = 0 ; i<sampler->maxDim; i++){
		sampler->initialPosition->parameters[i] = 0;
	}
	
	sampler->sample();

	
	for(int i = 0 ; i<sampler->maxDim; i++){
		delete [] priorRange[i];
	}
	delete [] priorRange;
	
	delete likelihood;
	delete prior;
	delete sampler->initialPosition;
	delete sampler;

	return 0;
}

