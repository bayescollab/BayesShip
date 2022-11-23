#include <iostream>
#include <bayesship/bayesshipSampler.h>
#include <bayesship/utilities.h>
#include <bayesship/dataUtilities.h>
#include <bayesship/proposalFunctions.h>

#include <omp.h>

void RT_ERROR_MSG();
int GMMDrawTesting(int argc, char *argv[]);
int main(int argc, char *argv[])
{
	std::cout<<"GMM testing"<<std::endl;
	if(argc < 2){
		RT_ERROR_MSG();
		return 1;
	}
	int runtimeOpt = std::stoi(argv[1]);
	if(runtimeOpt == 0){
		std::cout<<"GMM Draw Testing -- Must run python/create_GMM_test_data.py first"<<std::endl;
		return GMMDrawTesting(argc, argv);
	}

	return 0;
}

//#####################################################
//#####################################################
class toy_L : public bayesship::probabilityFn
{
public:
	virtual double eval(bayesship::positionInfo *pos, int chainID){return 0;}
};
class toy_P : public bayesship::probabilityFn
{
public:
	virtual double eval(bayesship::positionInfo *pos, int chainID){return 0;}
};
//double toy_L(bayesship::positionInfo *pos, int chainID, bayesship::bayesshipSampler *sampler, void *p)
//{
//
//	return 0;
//}
//double toy_P(bayesship::positionInfo *pos, int chainID, bayesship::bayesshipSampler *sampler, void *p)
//{
//
//	return 0;
//}
int GMMDrawTesting(int argc, char *argv[])
{

	toy_L *ll = new toy_L();
	toy_P *lp = new toy_P();
	//bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(toy_L, toy_P);
	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(ll, lp);
	sampler->ensembleN = 1;
	sampler->ensembleSize = 3;
	//sampler->maxDim = 9;
	sampler->maxDim = 3;

	int length = 10000;
	//int length = 5827;
	int lengthOut = 100000;
	double **data =new double*[length];
	double **dataOut =new double*[lengthOut];
	for (int i = 0; i<length; i++){
		data[i] = new double[sampler->maxDim];
	}
	for (int i = 0; i<lengthOut; i++){
		dataOut[i] = new double[sampler->maxDim];
	}

	bayesship::readCSVFile("data/GMM_testing_true_data.csv",data,length,sampler->maxDim);



	sampler->iterations = length+1;

	int NG = 10;
	int km_iter = 10;
	int em_iter = 10;
	double var_floor = 1e-10;

	std::vector<std::vector<int>> blocks = {{0,1,2}, {1,2}} ;
	std::vector<double> blockProb = {.5,.5};
	bayesship::proposal *proposal = new bayesship::GMMProposal(sampler->ensembleN*sampler->ensembleSize, sampler->maxDim, sampler, blocks,blockProb,NG, km_iter, em_iter,var_floor,false,1000);
	double prob = 1;
	bayesship::proposalData *pf = new bayesship::proposalData(sampler->ensembleN*sampler->ensembleSize,1,&proposal,  &prob );
	//bayesship::proposalFnData *pf = new bayesship::proposalFnData(sampler->ensembleN*sampler->ensembleSize, sampler->maxDim, false);

	sampler->proposalFns = pf;
	
	sampler->allocateMemory();

	sampler->data = new bayesship::samplerData(sampler->maxDim, sampler->ensembleN, sampler->ensembleSize, sampler->iterations, 1, false, sampler->betas);
	sampler->burnData = new bayesship::samplerData(sampler->maxDim, sampler->ensembleN, sampler->ensembleSize, sampler->iterations, 1, false, sampler->betas);
	sampler->setActiveData(sampler->data);


	for(int i = 0 ; i<length; i++){
		for (int j = 0 ;j<sampler->maxDim ; j++){
			sampler->data->positions[0][i]->parameters[j] = data[i][j];
			sampler->burnData->positions[0][i]->parameters[j] = data[i][j];
		}
	}

	sampler->data->currentStepID[0] = length-2;

	double MHRatioCorrections = 0;

	
	double start = omp_get_wtime();
	for(int i = 0 ; i<lengthOut; i++){
		//sampler->proposalFns->proposals[0](sampler->data, 0, 0, sampler, &MHRatioCorrections);
		sampler->proposalFns->proposals[0]->propose(sampler->data->positions[0][length-2], sampler->data->positions[0][length-1],0, 0,  &MHRatioCorrections);
		sampler->data->currentStepID[0] = length-2;
		for(int j = 0 ; j<sampler->maxDim; j++){
			dataOut[i][j] = sampler->data->positions[0][length-1]->parameters[j];
		}
	}
	std::cout<<"TIME: "<<omp_get_wtime() - start<<std::endl;

	bayesship::writeCSVFile("data/GMM_testing_resampled.csv",dataOut, lengthOut, sampler->maxDim);

	//Cleanup
	delete proposal;
	delete pf;
	sampler->~bayesshipSampler();
	for (int i = 0; i<length; i++){
		delete [] data[i];
	}
	delete [] data;
	for (int i = 0; i<lengthOut; i++){
		delete [] dataOut[i];
	}
	delete [] dataOut;
	delete ll;
	delete lp;
	delete sampler;

	
	
	return 0;
}
//#####################################################
//#####################################################

void RT_ERROR_MSG()
{
	std::cout<<"ERROR -- incorrect arguments"<<std::endl;
	std::cout<<"Please supply a test number:"<<std::endl;
	std::cout<<"0 -- GMM Draw Testing"<<std::endl;
	return;

}