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
	virtual double eval(bayesship::positionInfo *position, int chainID)
	{
		if(fabs(position->parameters[0]) > 10 ){
			return -std::numeric_limits<double>::infinity();
		}
		return 0;

	}
};

class gaussianLikelihood: public bayesship::probabilityFn
{
public:
	virtual double eval(bayesship::positionInfo *position, int chainID)
	{
		return -.5 * (position->parameters[0]*position->parameters[0]);

	}
};

int main(int argc, char *argv[])
{
	//bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(gaussianLikelihood,uniformPrior);
	gaussianLikelihood *gl = new gaussianLikelihood();
	uniformPrior *up = new uniformPrior();
	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(gl,up);

	sampler->maxDim = 1;
	sampler->coldOnlyStorage = true;

	double **priorRange = new double*[sampler->maxDim];
	priorRange[0] = new double[2];
	priorRange[0][0] = -10;
	priorRange[0][1] = 10;
	sampler->priorRanges = priorRange;
	//sampler->priorIterations = 10000;
	sampler->priorIterations = 10000;
	sampler->burnPriorIterations = 1000;
	sampler->writePriorData = true;
	sampler->ignoreExistingCheckpoint=false;
	//sampler->priorIterations = 0;


	sampler->threadPool = true;


	sampler->outputDir = "data/";
	sampler->outputFileMoniker = "gaussianLikelihoodTest";
	//sampler->iterations = 1000;
	sampler->independentSamples = 3000;
	//sampler->independentSamples = 500;
	//sampler->independentSamples = 50;
	sampler->burnIterations = 5000;
	//sampler->burnIterations = 10;
	//sampler->burnIterations = 0;
	sampler->ensembleN = 4;
	//sampler->ensembleN = 1;
	sampler->ensembleSize=5;
	sampler->threads = 4;
	int chainN = sampler->ensembleSize*sampler->ensembleN;
	sampler->initialPosition = new bayesship::positionInfo(sampler->maxDim,false);
	for(int i = 0 ; i<sampler->maxDim; i++){
		sampler->initialPosition->parameters[i] = 0;
	}
	
	sampler->sample();
	std::cout<<"Done Sampling"<<std::endl;
	double ave_ac;
	bayesship::mean_list(sampler->data->maxACs,sampler->ensembleN, &ave_ac);

	//std::ofstream outFile;
	//outFile.open("data/gaussianLikelihoodTestData.csv");
	//for(int j = 0 ; j<sampler->ensembleN; j++){
	//	//std::cout<<sampler->getBeta(sampler->chainIndex(j,0))<<std::endl;;
	//	for(int i = 0 ; i<sampler->data->iterations; i+=(int)ave_ac){
	//		outFile<<sampler->data->positions[sampler->chainIndex(j,0)][i]->parameters[0]<<" "<<std::endl;	
	//	}
	//}
	//outFile.close();

	//outFile.open("data/gaussianLikelihoodTestDataHot.csv");
	//for(int j = 0 ; j<sampler->ensembleN; j++){
	//	//std::cout<<sampler->betas[(j)+(sampler->ensembleSize-1)*(sampler->ensembleN)]<<std::endl;
	//	for(int i = 0 ; i<sampler->data->iterations; i+=(int)ave_ac){
	//		outFile<<sampler->data->positions[sampler->chainIndex(j,sampler->ensembleSize-1)][i]->parameters[0]<<" "<<std::endl;	
	//	}
	//}
	//outFile.close();
	
	//sampler->data->writeStatFile("data/gaussianLikelihoodTestStat.txt");

	//sampler->priorData->create_data_dump(true, true, sampler->outputDir+sampler->outputFileMoniker+"_prior.hdf5");
	
	delete gl;
	delete up;
	delete sampler->initialPosition;
	delete sampler;
	delete [] priorRange[0];
	delete [] priorRange;

	return 0;
}
