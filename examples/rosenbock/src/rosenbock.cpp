#include <iostream>
#include <fstream>
#include <bayesship/bayesshipSampler.h>
#include <bayesship/utilities.h>
#include <bayesship/dataUtilities.h>
#include <bayesship/proposalFunctions.h>
#include <limits.h>
#include <math.h>


class uniformPrior: public bayesship::probabilityFn
{
public:
	int maxDim;
	double **maxBounds;
	virtual double eval(bayesship::positionInfo *position, int chainID)
	{
		for(int i = 0 ; i<maxDim; i++)
		{
			if(position->parameters[i] < maxBounds[i][0] || position->parameters[i] > maxBounds[i][1]  ){
				return -std::numeric_limits<double>::infinity();
			}
		}
		return 0;

	}
};

class rosenbockLikelihood: public bayesship::probabilityFn
{
public:
	double a;
	double mu;
	int n1;
	int n2;
	int n;
	double *b;
	virtual double eval(bayesship::positionInfo *position, int chainID)
	{
		double *c = position->parameters;
		double LL = 0;
		LL-= a * bayesship::powInt(c[0] - mu,2)	;
		for(int j = 0 ; j<n2; j++){
			for(int i = 1 ; i<n1; i++){
				if(i == 1){
					LL -= b[(j)*(n1-1) + i]*bayesship::powInt(c[(j)*(n1-1) + i] - c[0]*c[0],2);
				}
				else{
					LL -= b[(j)*(n1-1) + i]*bayesship::powInt(c[(j)*(n1-1) + i] - c[(j)*(n1-1) + i-1]*c[(j)*(n1-1) + i-1],2);

				}
			}
		}
		return LL;

	}
};

int main(int argc, char *argv[])
{

	//Likelihood parameters
	double a = 1./20.;
	//int n1 = 3;
	//int n2 = 2;
	//int n1 = 5;
	//int n2 = 3;
	int n1 = 3;
	int n2 = 2;
	int n = (n1-1)*n2 + 1;
	double b[n];
	for(int i = 0 ; i<n ; i++){
		b[i]=100./20.;
	}
	double mu = 1.;
	double out_param[5+n];
	out_param[0]=a;
	out_param[1]=mu;
	out_param[2]=n;
	out_param[3]=n1;
	out_param[4]=n2;
	for(int i = 0 ; i<n; i++){
		out_param[5+i]=b[i];
	}
	bayesship::writeCSVFile("data/rosenbock_parameters.csv",out_param,6+n);




	rosenbockLikelihood *rl = new rosenbockLikelihood();
	rl->a = a;
	rl->b = &b[0];
	rl->n1 = n1;
	rl->n2 = n2;
	rl->n = n;
	uniformPrior *up = new uniformPrior();
	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(rl,up);

	sampler->maxDim = n;
	sampler->coldOnlyStorage = true;

	double **priorRange = new double*[sampler->maxDim];
	for(int i = 0 ; i<n; i++){
		priorRange[i] = new double[2];
		priorRange[i][0] = -1e3;
		priorRange[i][1] = 1e3;
	}
	sampler->priorRanges = priorRange;
	up->maxDim=n;
	up->maxBounds=priorRange;
	//sampler->priorIterations = 10000;
	sampler->priorIterations = 10000;
	sampler->burnPriorIterations = 10000;
	sampler->writePriorData = true;
	sampler->ignoreExistingCheckpoint=true;
	//sampler->priorIterations = 0;


	sampler->threadPool = true;


	sampler->outputDir = "data/";
	sampler->outputFileMoniker = "rosenbock";
	sampler->independentSamples = 500;
	sampler->burnIterations = 50000;
	//sampler->ensembleN = 1;
	sampler->ensembleN = 5;
	sampler->ensembleSize=50;
	//sampler->swapProb=.05;
	sampler->swapProb=.5;
	sampler->isolateEnsembles=false;
	sampler->restrictSwapTemperatures=false;
	sampler->threads = 14;
	int chainN = sampler->ensembleSize*sampler->ensembleN;
	sampler->initialPosition = new bayesship::positionInfo(sampler->maxDim,false);
	for(int i = 0 ; i<sampler->maxDim; i++){
		sampler->initialPosition->parameters[i] = 0;
	}
	
	sampler->sample();
	std::cout<<"Done Sampling"<<std::endl;
	double ave_ac;
	bayesship::mean_list(sampler->data->maxACs,sampler->ensembleN, &ave_ac);

	
	delete rl;
	delete up;
	delete sampler->initialPosition;
	delete sampler;
	delete [] priorRange[0];
	delete [] priorRange;

	return 0;
}
