#include <iostream>
#include <bayesship/bayesshipSampler.h>
#include <bayesship/utilities.h>
#include <bayesship/dataUtilities.h>
#include <bayesship/proposalFunctions.h>

#include <omp.h>

void RT_ERROR_MSG();
int KDEDrawTesting(int argc, char *argv[]);
int main(int argc, char *argv[])
{
	std::cout<<"KDE testing"<<std::endl;
	if(argc < 2){
		RT_ERROR_MSG();
		return 1;
	}
	int runtimeOpt = std::stoi(argv[1]);
	if(runtimeOpt == 0){
		std::cout<<"KDE Draw Testing -- Must run python/create_KDE_test_data.py first"<<std::endl;
		return KDEDrawTesting(argc, argv);
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
int KDEDrawTesting(int argc, char *argv[])
{
	//if(!std::filesystem::exists("data/KDE_test_data.csv")){
	//	std::cout<<"ERROR -- Must run python/create_KDE_test_data.py first"<<std::endl;
	//	return 1;
	//}

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
	int lengthOut = 8000;
	double **data =new double*[length];
	double **dataOut =new double*[lengthOut];
	for (int i = 0; i<length; i++){
		data[i] = new double[sampler->maxDim];
	}
	for (int i = 0; i<lengthOut; i++){
		dataOut[i] = new double[sampler->maxDim];
	}

	bayesship::readCSVFile("data/KDE_testing_true_data.csv",data,length,sampler->maxDim);



	sampler->iterations = length+1;


	float prob = 1;
	bayesship::KDEProposalVariables *kdepv = new bayesship::KDEProposalVariables(sampler->ensembleN * sampler->ensembleSize,sampler->maxDim, false, 1000, 1000);

	bayesship::proposalFn *proposalFnArray = new bayesship::proposalFn[1];

	proposalFnArray[0] = bayesship::KDEProposal;

	void **varArray = new void *[1];

	varArray[0] = (void *)kdepv;
	
	bayesship::proposalFnData *pf = new bayesship::proposalFnData(sampler->ensembleN*sampler->ensembleSize,1,proposalFnArray, varArray,  &prob );
	//bayesship::proposalFnData *pf = new bayesship::proposalFnData(sampler->ensembleN*sampler->ensembleSize, sampler->maxDim, false);

	sampler->proposalFns = pf;
	
	sampler->allocateMemory();

	sampler->data = new bayesship::samplerData(sampler->maxDim, sampler->ensembleN, sampler->ensembleSize, sampler->iterations, 1, false, sampler->betas);


	for(int i = 0 ; i<length; i++){
		for (int j = 0 ;j<sampler->maxDim ; j++){
			sampler->data->positions[0][i]->parameters[j] = data[i][j];
		}
	}

	sampler->data->currentStepID[0] = length-1;

	double MHRatioCorrections = 0;

	
	double start = omp_get_wtime();
	for(int i = 0 ; i<lengthOut; i++){
		sampler->proposalFns->proposalFnArray[0](sampler->data, 0, 0, sampler, &MHRatioCorrections);
		for(int j = 0 ; j<sampler->maxDim; j++){
			dataOut[i][j] = sampler->data->positions[0][length]->parameters[j];
		}
	}
	std::cout<<"TIME: "<<omp_get_wtime() - start<<std::endl;

	bayesship::writeCSVFile("data/KDE_testing_resampled.csv",dataOut, lengthOut, sampler->maxDim);

	//Cleanup
	delete kdepv;
	delete [] proposalFnArray;
	delete [] varArray;
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

	
	
	return 0;
}
//#####################################################
//#####################################################

void RT_ERROR_MSG()
{
	std::cout<<"ERROR -- incorrect arguments"<<std::endl;
	std::cout<<"Please supply a test number:"<<std::endl;
	std::cout<<"0 -- KDE Draw Testing"<<std::endl;
	return;

}
