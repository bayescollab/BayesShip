#include <iostream>
#include <fstream>
#include <bayesship/bayesshipSampler.h>
#include <bayesship/utilities.h>
#include <bayesship/dataUtilities.h>
#include <bayesship/proposalFunctions.h>
#include <limits.h>
#include <math.h>


void RT_ERROR_MSG();
//##############################################################
//##############################################################
int gaussianLikelihoodTest(int argc,char *argv[]);
double uniformPrior(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters);
double gaussianLikelihood(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters);
//##############################################################
//##############################################################

//##############################################################
//##############################################################
int multiModalLikelihoodTest(int argc,char *argv[]);
double uniformPriorMultiModal(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters);
double multiModalLikelihood(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters);
//##############################################################
//##############################################################

//##############################################################
//##############################################################
int validate_evidence(int argc, char *argv[]);
int validate_evidence_single(int argc, char *argv[]);
double Chebyshev_fn(int P, double *coeff, double x);
double transdimensional_likelihood(
	bayesship::positionInfo *param, 
	int chainID,
	bayesship::bayesshipSampler *sampler,
	void * parameters);
double transdimensional_likelihood_single(
	bayesship::positionInfo *param, 
	int chainID,
	bayesship::bayesshipSampler *sampler,
	void * parameters);
double transdimensional_prior(
	bayesship::positionInfo *param, 
	int chainID,
	bayesship::bayesshipSampler *sampler,
	void * parameters);
double transdimensional_prior_single(
	bayesship::positionInfo *param, 
	int chainID,
	bayesship::bayesshipSampler *sampler,
	void * parameters);
//double transdimensional_likelihood_fixed(
//	double *param, 
//	mcmc_data_interface *interface, 
//	void * parameters);
//double transdimensional_prior_fixed(
//	double *param, 
//	mcmc_data_interface *interface, 
//	void * parameters);
void transdimensional_RJprop(
	bayesship::samplerData *data,
	int chainID,
	int stepID, 
	bayesship::bayesshipSampler *sampler,
	double *MH_corrections);
struct trans_helper
{
	int N;
	double dt;
	double * data;
	gsl_rng *r;
};
//##############################################################
//##############################################################

int main(int argc, char *argv[])
{
	std::cout<<"Standard Test Suite"<<std::endl;	
	if(argc < 2){
		RT_ERROR_MSG();
		return 1;
	}
	int runtimeOpt = std::stoi(argv[1]);
	if(runtimeOpt == 0){
		std::cout<<"Gaussian Likelihood with uniform prior"<<std::endl;
		return gaussianLikelihoodTest(argc, argv);
	}
	else if(runtimeOpt == 1){
		std::cout<<"2D Multi modal distribution with uniform prior"<<std::endl;
		return multiModalLikelihoodTest(argc, argv);
	}
	else if(runtimeOpt == 2){
		std::cout<<"Transdimensional Testing -- Chebyshev function"<<std::endl;
		return validate_evidence(argc, argv);
	}
	else if(runtimeOpt == 3){
		std::cout<<"Transdimensional Testing on single dimension -- Chebyshev function"<<std::endl;
		return validate_evidence_single(argc, argv);
	}
	return 0;

}

//##############################################################
//##############################################################

double Chebyshev_fn(int P, double *coeff, double x)
{
	double sum = 0 ;
	for (int i = 0 ; i<P; i++){
		sum += coeff[i] * std::cos( i * std::acos(x));
	}
	return sum;
}
double transdimensional_likelihood_single(
	bayesship::positionInfo *param, 
	int chainid,
	bayesship::bayesshipSampler *sampler,
	void * parameters)
{
	//return 1;
	//sigma first, always there
	//start at 1 so there's always at least one coeff
	trans_helper *h = (trans_helper *)parameters;
	double reconst_signal[h->N];
	double dn =2. / ( h->N-1); 
	for(int i = 0 ; i < h->N; i++){
		reconst_signal[i] = Chebyshev_fn(sampler->maxDim-1, &(param->parameters[1]), -1 + i *dn );
	}
	
	double ll = 0;
	for (int i = 0 ; i<h->N; i++){
		ll -= bayesship::powInt((h->data[i] - reconst_signal[i]),2) ;
	}
	ll /= ( 2. * param->parameters[0]*param->parameters[0]);
	ll-= (h->N / 2.)*std::log(2. * M_PI * param->parameters[0]*param->parameters[0]);
	
	return ll;
}
double transdimensional_prior_single(
	bayesship::positionInfo *param, 
	int chainid,
	bayesship::bayesshipSampler *sampler,
	void * parameters)
{
	double a = -std::numeric_limits<double>::infinity();
	double prior=1;
	if (param->parameters[0] < .01|| param->parameters[0] > 10){ return a;}
	prior*= 1. /(10-.01); 	
	for(int i = 1 ; i<sampler->maxDim; i++){
		if (param->parameters[i] < -10|| param->parameters[i] > 10){ return a;}
		prior*= 1. /(20); 	
	}
	return std::log(prior);
	//return 1;
}

double transdimensional_likelihood(
	bayesship::positionInfo *param, 
	int chainid,
	bayesship::bayesshipSampler *sampler,
	void * parameters)
{
	//return 1;
	//sigma first, always there
	//start at 1 so there's always at least one coeff
	int p = 1;
	for(int i = 2 ; i<sampler->maxDim; i++){
		if(param->status[i] == 0){
			break;
		}
		p+=1;
	}
	trans_helper *h = (trans_helper *)parameters;
	double reconst_signal[h->N];
	double dn =2. / ( h->N-1); 
	for(int i = 0 ; i < h->N; i++){
		reconst_signal[i] = Chebyshev_fn(p, &(param->parameters[1]), -1 + i *dn );
	}
	
	double ll = 0;
	for (int i = 0 ; i<h->N; i++){
		ll -= bayesship::powInt((h->data[i] - reconst_signal[i]),2) ;
	}
	ll /= ( 2. * param->parameters[0]*param->parameters[0]);
	ll-= (h->N / 2.)*std::log(2. * M_PI * param->parameters[0]*param->parameters[0]);
	
	return ll;
}
double transdimensional_prior(
	bayesship::positionInfo *param, 
	int chainid,
	bayesship::bayesshipSampler *sampler,
	void * parameters)
{
	double a = -std::numeric_limits<double>::infinity();
	double prior=1;
	if (param->parameters[0] < .01|| param->parameters[0] > 10){ return a;}
	prior*= 1. /(10-.01); 	
	for(int i = 1 ; i<sampler->maxDim; i++){
		if(param->status[i] !=0){
			if (param->parameters[i] < -10|| param->parameters[i] > 10){ return a;}
			prior*= 1. /(20); 	
		}
	}
	return std::log(prior);
	//return 1;
}
//double transdimensional_likelihood_fixed(
//	double *param, 
//	mcmc_data_interface *interface, 
//	void * parameters)
//{
//	int temp_status[interface->max_dim];
//	for(int i = 0 ; i<interface->max_dim; i++){
//		temp_status[i] = 1;
//	}
//	return transdimensional_likelihood(param, temp_status, 0, interface, parameters);
//
//}
//double transdimensional_prior_fixed(
//	double *param, 
//	mcmc_data_interface *interface, 
//	void * parameters)
//{
//	int temp_status[interface->max_dim];
//	for(int i = 0 ; i<interface->max_dim; i++){
//		temp_status[i] = 1;
//	}
//	return transdimensional_prior(param, temp_status, 0, interface, parameters);
//}
//void transdimensional_RJprop(
//	bayesship::samplerData *data,
//	int chainID,
//	int stepID, 
//	bayesship::bayesshipSampler *sampler,
//	double *MH_corrections
//	)
//{ 
//	*MH_corrections = 0;
//	bayesship::positionInfo *currentStep = data->positions[chainID][data->currentStepID[chainID]];
//	bayesship::positionInfo *proposedStep = data->positions[chainID][data->currentStepID[chainID]+1];
//
//	proposedStep->updatePosition(currentStep);
//
//	trans_helper **hvec = (trans_helper **)sampler->proposalFns->proposalFnVariables[stepID];
//	trans_helper *h = hvec[chainID];
//
//	int P = 0;
//	for(int i = 0 ; i<sampler->maxDim; i++){
//		P += currentStep->status[i];
//	}
//	P -=1;//For sigma
//
//	double alpha = gsl_rng_uniform(h->r);
//	//##################################################
//	//Kill 
//	if(alpha < .5 ){
//		if( P >1){
//			proposedStep->status[P] = 0;
//			proposedStep->parameters[P] = 0;
//			//*MH_corrections+=std::log(1./(20.));
//			//if(P-1 == 1){
//			//	*MH_corrections+=std::log(1./(.5));
//			//}
//		}
//	}
//	//Create 
//	else if(alpha >= .5 ){
//		if( P < sampler->maxDim-1){
//			proposedStep->status[P+1] = 1;
//			proposedStep->parameters[P+1] = gsl_rng_uniform(h->r)*20-10;
//			//proposedStep->parameters[P+1] = gsl_rng_uniform(h->r);
//			//*MH_corrections-=std::log( 1./(20.)  );
//		}
//
//	}
//
//
//	
//	//##################################################
//	////Kill 
//	//if(alpha < .5 && P >1){
//	//	proposedStep->status[P] = 0;
//	//	proposedStep->parameters[P] = 0;
//	//	//*MH_corrections+=std::log(1./(20.));
//	//	if(P-1 == 1){
//	//		*MH_corrections+=std::log(1./(.5));
//	//	}
//
//	//}
//	////Create 
//	//else if(alpha >= .5 && P <(sampler->maxDim-1)){
//	//	proposedStep->status[P+1] = 1;
//	//	//proposedStep->parameters[P+1] = gsl_rng_uniform(h->r)*20-10;
//	//	proposedStep->parameters[P+1] = gsl_rng_uniform(h->r);
//	//	//*MH_corrections+=std::log(1./ ( 1./(20.) ) );
//	//	if(P+1 == sampler->maxDim-1){
//	//		*MH_corrections+=std::log(1./(.5));
//	//	}
//
//	//}
//	////Create 
//	//if(P == 1){
//	//	proposedStep->status[P+1] = 1;
//	//	//proposedStep->parameters[P+1] = gsl_rng_uniform(h->r)*20-10;
//	//	proposedStep->parameters[P+1] = gsl_rng_uniform(h->r);
//	//	//*MH_corrections+=std::log(1./ (1./(20.) ) );//ratio of new prior
//	//	*MH_corrections+=std::log(.5/1.);//Odds of proposing a creation next jump
//
//	//}
//	////Kill 
//	//if(P == sampler->maxDim-1){
//	//	proposedStep->status[P] = 0;
//	//	proposedStep->parameters[P] = 0;
//	//	//*MH_corrections+=std::log(1./(20.));//ratio of uniform random number
//	//	*MH_corrections+=std::log(.5/1.);//Odds of proposing a creation next jump
//
//	//}
//	return ;
//}
int validate_evidence(int argc, char *argv[])
{
	std::string beta = "2";
	std::string M = "3";
	std::string sigma = "1";
	std::string t = "100";
	std::string dtstr = "1";
	std::string dirname = "data/transdimensionalChebyshev/"+beta+"_"+M+"_"+sigma+"_"+t+"_"+dtstr+"/";
	std::string data_file("full_data_transdimensional.csv");
	int N = 100;
	double dt = 1;

	double *data=new double[N] ;
	bayesship::readCSVFile(dirname+data_file, data);


	//######################################################################
	//######################################################################
	//######################################################################
	//######################################################################
	//######################################################################
	//######################################################################


	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(transdimensional_likelihood,transdimensional_prior);

	sampler->RJ = true;

	sampler->maxDim = 10;
	sampler->minDim = 2;
	//sampler->coldOnlyStorage = true;


	double **priorRange = new double*[sampler->maxDim];
	for(int i = 0 ; i<sampler->maxDim; i++){
		priorRange[i] = new double[2];
		priorRange[i][0] = -10;
		priorRange[i][1] = 10;
	}
	priorRange[0][0] = .01;
	priorRange[0][1] = 10;
	sampler->priorRanges = priorRange;
	//sampler->priorRanges = nullptr;
	sampler->burnPriorIterations = 5000;
	//sampler->burnPriorIterations = 500;
	sampler->priorIterations = 5000;
	//sampler->priorIterations = 500;
	sampler->writePriorData=true;
	sampler->ignoreExistingCheckpoint=true;

	sampler->outputDir = dirname;
	sampler->outputFileMoniker = "transdimensionalChebyshev";
	sampler->iterations = 50000;
	//sampler->iterations = 500;
	sampler->burnIterations = 30000;
	//sampler->burnIterations = 100;
	//sampler->burnIterations = 0;
	sampler->ensembleN = 5;
	sampler->ensembleSize=10;
	int chain_N = sampler->ensembleN*sampler->ensembleSize;
	sampler->threads = 10;
	int chainN = sampler->ensembleSize*sampler->ensembleN;
	sampler->initialPosition = new bayesship::positionInfo(sampler->maxDim,true);
	for(int i = 0 ; i<sampler->maxDim; i++){
		sampler->initialPosition->parameters[i] = .5;
		sampler->initialPosition->status[i] = 1;
		sampler->initialPosition->modelID = 0;
	}



	trans_helper **helpers = new trans_helper*[chain_N];
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	for(int i = 0 ; i<chain_N; i++){
		helpers[i] = new trans_helper;
		helpers[i]->dt = dt;
		helpers[i]->N = N;
		helpers[i]->data = data;
		helpers[i]->r = gsl_rng_alloc(T);
	}

	int proposalFnN = 4;
	//bayesship::proposalFn proposalFnArray[4] = {bayesship::gaussianProposal, bayesship::differentialEvolutionProposal, bayesship::KDEProposal, transdimensional_RJprop};
	bayesship::proposalFn proposalFnArray[4] = {bayesship::gaussianProposal, bayesship::differentialEvolutionProposal, bayesship::KDEProposal, bayesship::sequentialLayerRJProposal};
	float proposalFnProb[4] = {.5,.0,.0,.5};
	bayesship::gaussianProposalVariables *gpv = new bayesship::gaussianProposalVariables(chainN, sampler->maxDim);
	bayesship::KDEProposalVariables *kdepv = new bayesship::KDEProposalVariables(chainN, sampler->maxDim);
	bayesship::sequentialLayerRJProposalVariables *slpv = new bayesship::sequentialLayerRJProposalVariables(0.5);
	void *proposalFnVariables[4] = { (void *)gpv,(void *)nullptr,(void *)kdepv,(void *) slpv};
	sampler->userParameters = (void**)helpers;
	sampler->proposalFns = new bayesship::proposalFnData(chain_N,proposalFnN,proposalFnArray, proposalFnVariables,proposalFnProb);
	
	sampler->sample();

	//sampler->data->writeStatFile("data/transdimensionalChebyshevStat.txt");
	//sampler->priorData->writeStatFile("data/transdimensionalChebyshevPriorStat.txt");

	//sampler->priorData->create_data_dump(true, true, sampler->outputDir+sampler->outputFileMoniker+"_prior.hdf5");
	//sampler->data->create_data_dump(true, true, sampler->outputDir+sampler->outputFileMoniker+"_output.hdf5");
	
	delete sampler->initialPosition;
	for(int i = 0 ; i<sampler->maxDim; i++){
		delete [] priorRange[i];
	}
	
	delete [] priorRange;
	delete sampler->proposalFns;
	delete gpv;
	delete kdepv;
	delete sampler;

	delete [] data;


	for(int i = 0 ; i<chain_N; i++){
		gsl_rng_free(helpers[i]->r);
		delete helpers[i];
	}
	delete [] helpers;
	return 0;
}

int validate_evidence_single(int argc, char *argv[])
{
	std::string beta = "2";
	std::string M = "3";
	std::string sigma = "1";
	std::string t = "100";
	std::string dtstr = "1";
	std::string dirname = "data/transdimensionalChebyshev/"+beta+"_"+M+"_"+sigma+"_"+t+"_"+dtstr+"/";
	std::string data_file("full_data_transdimensional.csv");
	int N = 100;
	double dt = 1;

	double *data=new double[N] ;
	bayesship::readCSVFile(dirname+data_file, data);


	//######################################################################
	//######################################################################
	//######################################################################
	//######################################################################
	//######################################################################
	//######################################################################


	//bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(transdimensional_likelihood,transdimensional_prior);
	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(transdimensional_likelihood_single,transdimensional_prior_single);

	sampler->RJ = false;

	sampler->maxDim = 3;

	double **priorRange = new double*[sampler->maxDim];
	for(int i = 0 ; i<sampler->maxDim; i++){
		priorRange[i] = new double[2];
		priorRange[i][0] = -10;
		priorRange[i][1] = 10;
	}
	priorRange[0][0] = .01;
	priorRange[0][1] = 10;
	sampler->priorRanges = priorRange;
	sampler->burnPriorIterations = 5000;
	//sampler->burnPriorIterations = 500;
	sampler->priorIterations = 5000;
	//sampler->priorIterations = 500;
	sampler->writePriorData=true;
	sampler->ignoreExistingCheckpoint=true;

	sampler->outputDir = dirname;
	sampler->outputFileMoniker = "transdimensionalChebyshevSingle_"+std::to_string(sampler->maxDim-1);
	//sampler->iterations = 50000;
	//sampler->iterations = 500;
	sampler->independentSamples = 1000;
	sampler->burnIterations = 15000;
	//sampler->burnIterations = 100;
	//sampler->burnIterations = 0;
	sampler->ensembleN = 5;
	sampler->ensembleSize=10;
	int chain_N = sampler->ensembleN*sampler->ensembleSize;
	sampler->threads = 8;
	int chainN = sampler->ensembleSize*sampler->ensembleN;
	sampler->initialPosition = new bayesship::positionInfo(sampler->maxDim,false);
	for(int i = 0 ; i<sampler->maxDim; i++){
		sampler->initialPosition->parameters[i] = .5;
		//sampler->initialPosition->status[i] = 1;
		//sampler->initialPosition->modelID = 0;
	}



	trans_helper **helpers = new trans_helper*[chain_N];
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	for(int i = 0 ; i<chain_N; i++){
		helpers[i] = new trans_helper;
		helpers[i]->dt = dt;
		helpers[i]->N = N;
		helpers[i]->data = data;
		helpers[i]->r = gsl_rng_alloc(T);
	}

	int proposalFnN = 3;
	//bayesship::proposalFn proposalFnArray[4] = {bayesship::gaussianProposal, bayesship::differentialEvolutionProposal, bayesship::KDEProposal, transdimensional_RJprop};
	bayesship::proposalFn proposalFnArray[4] = {bayesship::gaussianProposal, bayesship::differentialEvolutionProposal, bayesship::KDEProposal};
	float proposalFnProb[3] = {.3,.7,.0};
	bayesship::gaussianProposalVariables *gpv = new bayesship::gaussianProposalVariables(chainN, sampler->maxDim);
	bayesship::KDEProposalVariables *kdepv = new bayesship::KDEProposalVariables(chainN, sampler->maxDim);
	void *proposalFnVariables[3] = { (void *)gpv,(void *)nullptr,(void *)kdepv};
	sampler->userParameters = (void**)helpers;
	sampler->proposalFns = new bayesship::proposalFnData(chain_N,proposalFnN,proposalFnArray, proposalFnVariables,proposalFnProb);

	//#############################################################
	//bayesship::positionInfo *paramTest = new bayesship::positionInfo(sampler->maxDim,false);
	//paramTest->parameters[0]  = .32;	
	//paramTest->parameters[1]  = -3.2;	
	//paramTest->parameters[2]  = 2.7;	
	//paramTest->parameters[3]  = 3.3;	
	//std::cout<<"Test likelihood: "<<transdimensional_likelihood_single(paramTest, 0, sampler, sampler->userParameters[0])<<std::endl;
	//delete paramTest;
	//exit(1);
	//#############################################################
	
	sampler->sample();
	double ave_ac;
	bayesship::mean_list(sampler->data->maxACs,sampler->ensembleN, &ave_ac);

	//sampler->data->writeStatFile("data/transdimensionalChebyshevStat.txt");
	//sampler->priorData->writeStatFile("data/transdimensionalChebyshevPriorStat.txt");

	//sampler->priorData->create_data_dump(true, true, sampler->outputDir+sampler->outputFileMoniker+"_prior.hdf5");
	//sampler->data->create_data_dump(true, true, sampler->outputDir+sampler->outputFileMoniker+"_output.hdf5");
	
	delete sampler->initialPosition;
	for(int i = 0 ; i<sampler->maxDim; i++){
		delete [] priorRange[i];
	}
	
	delete [] priorRange;
	delete sampler->proposalFns;
	delete gpv;
	delete kdepv;
	delete sampler;

	delete [] data;


	for(int i = 0 ; i<chain_N; i++){
		gsl_rng_free(helpers[i]->r);
		delete helpers[i];
	}
	delete [] helpers;
	return 0;
}
//##############################################################
//##############################################################

double uniformPrior(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters){

	if(fabs(position->parameters[0]) > 10 ){
		return -std::numeric_limits<double>::infinity();
	}
	return 0;
}

double gaussianLikelihood(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters){
	return -.5 * (position->parameters[0]*position->parameters[0]);
}
/*! \brief Test to establish basic functionality quickly.
 *
 * Simply a gaussian likelihood with a flat prior, centered at 0 and with unit variance
 */
int gaussianLikelihoodTest(int argc,char *argv[])
{
	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(gaussianLikelihood,uniformPrior);

	sampler->maxDim = 1;
	sampler->coldOnlyStorage = true;

	double **priorRange = new double*[sampler->maxDim];
	priorRange[0] = new double[2];
	priorRange[0][0] = -10;
	priorRange[0][1] = 10;
	sampler->priorRanges = priorRange;
	//sampler->priorIterations = 10000;
	sampler->priorIterations = 0;
	sampler->burnPriorIterations = 0;
	sampler->writePriorData = true;
	//sampler->priorIterations = 0;


	sampler->threadPool = false;


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
	sampler->threads = 8;
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
	
	delete sampler->initialPosition;
	delete sampler;
	delete [] priorRange[0];
	delete [] priorRange;

	return 0;
}
//##############################################################
//##############################################################

double uniformPriorMultiModal(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters){

	if(fabs(position->parameters[0]) > 4 ){
		return -std::numeric_limits<double>::infinity();
	}
	else if(fabs(position->parameters[1]) > 4 ){
		return -std::numeric_limits<double>::infinity();
	}
	//return -.5 * bayesship::pow_int(position->parameters[0],2)/1. -.5 * bayesship::pow_int(1-position->parameters[1],2)/1.;
	return 0;
}

double multiModalLikelihood(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler, void *userParameters){
	//return 2;
	double x = position->parameters[0];
	double y = position->parameters[1];
	double power1 = -x*x - (9. + 4 * x * x + 8 *y)*(9. + 4 * x * x + 8 *y);
	double power2 = -8*x*x - 8*(y-2)*(y-2);
	return log( 16. / 3. / M_PI *( exp(power1) + .5* exp(power2)));
}
/*! \brief Test to establish basic functionality quickly.
 *
 * Simply a student t likelihood with a flat prior
 */
int multiModalLikelihoodTest(int argc,char *argv[])
{
	bayesship::bayesshipSampler *sampler = new bayesship::bayesshipSampler(multiModalLikelihood,uniformPriorMultiModal);
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
	sampler->burnPriorIterations =0;
	sampler->priorIterations =0;
	sampler->writePriorData = true;

	sampler->coldOnlyStorage=true;

	sampler->threadPool = true;

	//sampler->iterations = 50000;
	sampler->independentSamples = 1000;
	sampler->burnIterations = 10000;
	//sampler->independentSamples = 100;
	//sampler->burnIterations = 100;
	sampler->swapProb=.1;	
	//sampler->burnIterations = 250;
	//sampler->independentSamples = 100;
	sampler->ensembleN = 5;
	sampler->ensembleSize=4;
	sampler->threads = 10;
	sampler->randomizeSwapping = true;
	int chainN = sampler->ensembleSize*sampler->ensembleN;
	sampler->initialPosition = new bayesship::positionInfo(sampler->maxDim,false);
	for(int i = 0 ; i<sampler->maxDim; i++){
		sampler->initialPosition->parameters[i] = 0;
	}
	
	sampler->sample();

	//std::cout<<"KDE bandwidths, stepNumber : "<<std::endl;
	//for(int i =0 ; i<sampler->getChainN(); i++){
	//	bayesship::KDEProposalVariables *temp = (bayesship::KDEProposalVariables  *)(sampler->proposalFns->proposalFnVariables[2]);
	//	std::cout<<temp->bandwidth[i]<<" "<<temp->stepNumber[i]<<std::endl;
	//}
	//std::cout<<std::endl;
	double ave_ac;
	bayesship::mean_list(sampler->data->maxACs,sampler->ensembleN, &ave_ac);

	//std::ofstream outFile;
	//outFile.open("data/multiModalLikelihoodTestData.csv");
	//for(int j = 0 ; j<sampler->ensembleN; j++){
	//	int chainIndex = sampler->chainIndex(j,0);
	//	for(int i = 0 ; i<sampler->data->iterations; i+=(int)ave_ac){
	//		outFile<<sampler->data->positions[chainIndex][i]->parameters[0]<<", "<<sampler->data->positions[chainIndex][i]->parameters[1]<<" "<<std::endl;	
	//	}
	//}
	//outFile.close();
	//outFile.open("data/multiModalLikelihoodTestDataHot.csv");
	//for(int j = 0 ; j<sampler->ensembleN; j++){
	//	int chainIndex = sampler->chainIndex(j,sampler->ensembleSize-1);
	//	for(int i = 0 ; i<sampler->data->iterations; i+=(int)ave_ac){
	//		outFile<<sampler->data->positions[chainIndex][i]->parameters[0]<<", "<<sampler->data->positions[chainIndex][i]->parameters[1]<<" "<<std::endl;	
	//	}
	//}
	//outFile.close();
	
	//sampler->data->writeStatFile("data/multiModalLikelihoodTestStat.txt");

	//sampler->priorData->create_data_dump(true, true, sampler->outputDir+sampler->outputFileMoniker+"_prior.hdf5");
	
	for(int i = 0 ; i<sampler->maxDim; i++){
		delete [] priorRange[i];
	}
	delete [] priorRange;
	
	delete sampler->initialPosition;
	delete sampler;

	return 0;
}

void RT_ERROR_MSG()
{
	std::cout<<"ERROR -- incorrect arguments"<<std::endl;
	std::cout<<"Please supply a test number:"<<std::endl;
	std::cout<<"0 -- Gaussian likelihood with a flat prior"<<std::endl;
	std::cout<<"1 -- 2D multimodal distribution with a flat prior"<<std::endl;
	std::cout<<"2 -- Transdimensional -- Chebyshev"<<std::endl;
	return;

}
