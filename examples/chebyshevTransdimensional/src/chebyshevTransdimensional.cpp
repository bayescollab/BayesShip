#include <iostream>
#include <fstream>
#include <bayesship/bayesshipSampler.h>
#include <bayesship/utilities.h>
#include <bayesship/dataUtilities.h>
#include <bayesship/proposalFunctions.h>
#include <limits.h>
#include <math.h>

void RT_ERROR_MSG();

int validate_evidence(int argc, char *argv[]);
int validate_evidence_single(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	std::cout<<"Standard Test Suite"<<std::endl;	
	if(argc < 2){
		RT_ERROR_MSG();
		return 1;
	}
	int runtimeOpt = std::stoi(argv[1]);
	if(runtimeOpt == 0){
		std::cout<<"Transdimensional Version -- Chebyshev function"<<std::endl;
		return validate_evidence(argc, argv);
	}
	else if(runtimeOpt == 1){
		std::cout<<"Fixed dimension Version -- Chebyshev function"<<std::endl;
		return validate_evidence_single(argc, argv);
	}
	return 0;

}
//##############################################################
//##############################################################


struct trans_helper
{
	int N;
	double dt;
	double * data;
	gsl_rng *r;
};

double Chebyshev_fn(int P, double *coeff, double x)
{
	double sum = 0 ;
	for (int i = 0 ; i<P; i++){
		sum += coeff[i] * std::cos( i * std::acos(x));
	}
	return sum;
}

class chebyshevLikelihoodSingle: public bayesship::probabilityFn
{
public:
	trans_helper *h;
	bayesship::bayesshipSampler *sampler;
	virtual double eval(bayesship::positionInfo *param, int chainid)
	{
		//return 1;
		//sigma first, always there
		//start at 1 so there's always at least one coeff
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
};

class chebyshevPriorSingle: public bayesship::probabilityFn
{
public:
	bayesship::bayesshipSampler *sampler;
	virtual double eval(bayesship::positionInfo *param, int chainid)
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
};
int validate_evidence_single(int argc, char *argv[])
{
	std::string beta = "2";
	std::string M = "3";
	std::string sigma = "1";
	std::string t = "100";
	std::string dtstr = "1";
	std::string dirname = "data/"+beta+"_"+M+"_"+sigma+"_"+t+"_"+dtstr+"/";
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


	trans_helper *h = new trans_helper;
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	h->dt = dt;
	h->N = N;
	h->data = data;
	h->r = gsl_rng_alloc(T);


	bayesship::bayesshipSampler *sampler;
	
	chebyshevLikelihoodSingle *likelihood = new chebyshevLikelihoodSingle();
	likelihood->h = h;
	chebyshevPriorSingle *prior = new chebyshevPriorSingle();
	
	sampler = new bayesship::bayesshipSampler(likelihood,prior);
	likelihood->sampler = sampler;
	prior->sampler = sampler;


	sampler->RJ = false;

	sampler->maxDim = 4;

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




	int proposalN = 2;
	//bayesship::proposalFn proposalFnArray[4] = {bayesship::gaussianProposal, bayesship::differentialEvolutionProposal, bayesship::KDEProposal, transdimensional_RJprop};
	
	bayesship::proposal **proposals = new bayesship::proposal*[2];
	
	proposals[0] = 	new bayesship::gaussianProposal(chainN, sampler->maxDim, sampler );
	proposals[1] = 	new bayesship::differentialEvolutionProposal( sampler );
		
	double proposalFnProb[2] = {.3,.7};

	sampler->proposalFns = new bayesship::proposalData(chain_N,proposalN,proposals, proposalFnProb);

	//#############################################################
	bayesship::positionInfo *paramTest = new bayesship::positionInfo(sampler->maxDim,false);
	paramTest->parameters[0]  = .32;	
	paramTest->parameters[1]  = -3.2;	
	paramTest->parameters[2]  = 2.7;	
	paramTest->parameters[3]  = 3.3;	
	double ll = likelihood->eval(paramTest, 0);
	std::cout<<"Test likelihood: "<<ll<<std::endl;
	delete paramTest;
	//exit(1);
	//#############################################################
	
	sampler->sample();
	double ave_ac;
	bayesship::mean_list(sampler->data->maxACs,sampler->ensembleN, &ave_ac);

	
	delete sampler->initialPosition;
	for(int i = 0 ; i<sampler->maxDim; i++){
		delete [] priorRange[i];
	}
	
	delete [] priorRange;
	delete proposals[0];
	delete proposals[1];
	delete [] proposals;
	delete sampler->proposalFns;
	delete likelihood;
	delete prior;
	delete sampler;

	delete [] data;


	gsl_rng_free(h->r);
	delete h;
	return 0;
}


//########################################################################
class chebyshevLikelihood: public bayesship::probabilityFn
{
public:
	trans_helper *h;
	bayesship::bayesshipSampler *sampler;
	virtual double eval(bayesship::positionInfo *param, int chainid)
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
};

class chebyshevPrior: public bayesship::probabilityFn
{
public:
	bayesship::bayesshipSampler *sampler;
	virtual double eval(bayesship::positionInfo *param, int chainid)
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
};

int validate_evidence(int argc, char *argv[])
{
	std::string beta = "2";
	std::string M = "3";
	std::string sigma = "1";
	std::string t = "100";
	std::string dtstr = "1";
	std::string dirname = "data/"+beta+"_"+M+"_"+sigma+"_"+t+"_"+dtstr+"/";
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

	trans_helper *h= new trans_helper;
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	h->dt = dt;
	h->N = N;
	h->data = data;
	h->r = gsl_rng_alloc(T);


	bayesship::bayesshipSampler *sampler;
	
	chebyshevLikelihood *likelihood = new chebyshevLikelihood();
	likelihood->h = h;
	chebyshevPrior *prior = new chebyshevPrior();
	

	sampler = new bayesship::bayesshipSampler(likelihood,prior);
	likelihood->sampler = sampler;
	prior->sampler = sampler;


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
	sampler->burnPriorIterations = 10000;
	sampler->priorIterations = 10000;
	sampler->writePriorData=true;
	sampler->ignoreExistingCheckpoint=true;

	sampler->outputDir = dirname;
	sampler->outputFileMoniker = "transdimensionalChebyshev";
	sampler->iterations = 50000;
	sampler->batchSize = 1000;
	sampler->burnIterations = 20000;
	sampler->ensembleN = 5;
	sampler->ensembleSize=10;
	int chain_N = sampler->ensembleN*sampler->ensembleSize;
	sampler->threads = 4;
	int chainN = sampler->ensembleSize*sampler->ensembleN;
	sampler->initialPosition = new bayesship::positionInfo(sampler->maxDim,true);
	for(int i = 0 ; i<sampler->maxDim; i++){
		sampler->initialPosition->parameters[i] = .5;
		sampler->initialPosition->status[i] = 1;
		sampler->initialPosition->modelID = 0;
	}




	int proposalN = 3;
	//bayesship::proposalFn proposalFnArray[4] = {bayesship::gaussianProposal, bayesship::differentialEvolutionProposal, bayesship::KDEProposal, transdimensional_RJprop};
	
	bayesship::proposal **proposals = new bayesship::proposal*[3];
	
	proposals[0] = 	new bayesship::gaussianProposal(chainN, sampler->maxDim, sampler );
	proposals[1] = 	new bayesship::differentialEvolutionProposal( sampler );
	proposals[2] = 	new bayesship::sequentialLayerRJProposal( sampler, .5 );
		
	double proposalFnProb[3] = {.3,.4,.3};

	sampler->proposalFns = new bayesship::proposalData(chain_N,proposalN,proposals, proposalFnProb);
	
	sampler->sample();

	
	delete sampler->initialPosition;
	for(int i = 0 ; i<sampler->maxDim; i++){
		delete [] priorRange[i];
	}
	
	delete [] priorRange;
	delete sampler->proposalFns;
	delete proposals[0];
	delete proposals[1];
	delete proposals[2];
	delete [] proposals;
	delete likelihood;
	delete prior;
	delete sampler;

	delete [] data;


	gsl_rng_free(h->r);
	delete h;
	return 0;
}
//##############################################################
//##############################################################
void RT_ERROR_MSG()
{
	std::cout<<"ERROR -- incorrect arguments"<<std::endl;
	std::cout<<"Please supply a test number:"<<std::endl;
	std::cout<<"0 -- Chebyshev -- Transdimensional"<<std::endl;
	std::cout<<"1 -- Chebyshev -- Single Dimension"<<std::endl;
	return;

}
