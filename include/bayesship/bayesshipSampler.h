#ifndef BAYESSHIPSAMPLER_H
#define BAYESSHIPSAMPLER_H

#include "bayesship/dataUtilities.h"
#include <string>
#include <iostream>
#include <functional>
#include <gsl/gsl_rng.h>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <bayesship/ThreadPool.h>


namespace bayesship{

const double limitInf = - std::numeric_limits<double>::infinity();

/*! \file 
 * # bayesshipSampler Header File
 * 
 * Header file for a majority of the BayesShip sampler routines.
 * 
 */


/*Must forwardd declare the calss to use it in the declarations of the functions below*/
class bayesshipSampler;



/*! \brief Structure to package swap ``jobs'' for sampling
 *
 * Packages up a job to queue up a chain for swapping
 *
 * See ThreadPool.h to see how this relates to parallel processing, and see the bayesShip.cpp for the implementation of this structure
 */
struct swapJob
{
	/*! Chain ID to be swapped*/
	int chainID;
	/*! samplerData for the sampler storing current data for chainID*/
	samplerData *data;
	/*! The sampler being currently run*/
	bayesshipSampler *sampler;

};

/*! \brief Structure to package sample jobs for sampling
 *
 * Packages up a job to perform Metropolis-Hastings sampling
 * 
 * See ThreadPool.h to see how this relates to parallel processing, and see the bayesShip.cpp for the implementation of this structure
 *
 * */
struct sampleJob
{
	/*! The sampler being currently run*/
	bayesshipSampler *sampler;
	/*! samplerData for the sampler storing current data for chainID*/
	samplerData *data;
	/*! Chain ID to be stepped*/
	int chainID;
	/*! Pool for swapping -- The sampler periodically passes the chain on to be swapped after steping with Metropolis-Hastings*/
	ThreadPoolPair<swapJob> *swapPool;

};


/*! \brief Wrapper function to perform stepMH for sampleJob job and with thread threadID
 *
 * Simply calls stepMH for the chain associated with sampleJob job using thread threadID.
 * 
 * stepMH performs a step in parameter space using the Metropolis-Hastings algorithm.
 */
void sampleThreadedFunctionNoSwap(int threadID, sampleJob job);
void sampleThreadedFunction(int threadID, sampleJob job);
void swapThreadedFunction(int threadID, swapJob job1, swapJob job2);
bool swapPairFunction( swapJob job1, swapJob job2);


/*! \brief Proposal function typedef 
 *
 * Specifies the form of proposal functions
 * 
 * All user defined proposal functions must be caste-able to this type
 * */
typedef void(*proposalFn)(
	samplerData *data,
	int chainID,/**< ID of the chain being utilized*/
	int stepID,/**< Proposal ID for extracting proposal variables */
	bayesshipSampler *sampler,/**< Sampler object that is currently sampling*/
	double *MHRatioModifications/**< Modifications to the MH ration (like asymmetric proposals*/
	);





class probabilityFn
{
public:
	probabilityFn(){};
	~probabilityFn(){};
	virtual double eval(positionInfo *position, int chainID) { std::cout<<"OOPS"<<std::endl;return 0;}
};






/*! \brief Likelihood function typedef 
 *
 * Specifies the form of likelihood functions
 * 
 * All user defined likelihood functions must be caste-able to this type
 * */
//typedef double(*likelihoodFn)(
//	positionInfo *position,/**< positionInfo for current point in parameter space*/
//	int chainID,/**< Proposal ID for extracting proposal variables */
//	bayesshipSampler *sampler,/**< Sampler object that is currently sampling*/
//	void *userParameters /**< User specific parameters that can be passed to the sampler through its member variables. These are then passed on to the likelihood function*/
//	);
/*! \brief Prior function typedef 
 *
 * Specifies the form of prior functions
 * 
 * All user defined likelihood functions must be caste-able to this type
 * */
//typedef double(*priorFn)(
//	positionInfo *position,/**< positionInfo for current point in parameter space*/
//	int chainID,/**< Proposal ID for extracting proposal variables */
//	bayesshipSampler *sampler,/**< Sampler object that is currently sampling*/
//	void *userParameters/**< User specific parameters that can be passed to the sampler through its member variables. These are then passed on to the likelihood function*/
//	);



/*! \brief proposalFn write checkpoint function typedef 
 *
 * Specifies the form of proposal function write checkpoint functions.
 *
 * These functions are called when writing out checkpoint files. It's up to the user whether and how these files should be written.
 *
 * These functions are only necessary if the proposal saves data or optimizes itself in some way. If not necessary, (void *) can be passed instead.
 *
 * The write and read checkpoint functions *should be compatible*. That is, the file naming should be consistent between the two, in such a way the files being written out can be read in without external information.
 * 
 * */
typedef void(*proposalFnWriteCheckpoint)(
	void *proposalFnVariables,
	bayesshipSampler *sampler
	);

/*! \brief proposalFn read checkpoint function typedef 
 *
 * Specifies the form of proposal function read checkpoint functions.
 *
 * These functions are called when reading in checkpoint files. It's up to the user whether and how these files should be written.
 *
 * These functions are only necessary if the proposal saves data or optimizes itself in some way. If not necessary, (void *) can be passed instead.
 *
 * The write and read checkpoint functions *should be compatible*. That is, the file naming should be consistent between the two, in such a way the files being written out can be read in without external information.
 * 
 * */
typedef void(*proposalFnLoadCheckpoint)(
	void *proposalFnVariables,
	bayesshipSampler *sampler
	);


/*! \brief Class containing all the information about proposal functions for the bayesshipSampler object
 *
 * Used internally to keep the information tidy inside the sampler object 
 *
 * Copies everything by value
 *
 * Implementations that want to use custom proposals will need to declare and populate one of these structures, then pass it to the bayesshipSampler object
 */
class proposalFnData
{
public:
	/*! Number of proposal functions*/
	int proposalFnN;
	/*! Array of proposal function pointers */
	proposalFn *proposalFnArray=nullptr;
	/*! Probability of each proposalFn*/
	float **proposalFnProb=nullptr;
	/*! Any necessary variables/data for each proposal Fn*/
	void **proposalFnVariables=nullptr;
	/*! writeCheckpointFns for all proposals. If not necessary for a specific proposal, pass (void *)*/
	proposalFnWriteCheckpoint *writeCheckpointFns=nullptr;
	/*! readCheckpointFns for all proposals. If not necessary for a specific proposal, pass (void *)*/
	proposalFnLoadCheckpoint *loadCheckpointFns=nullptr;
	
	/*! \brief Constructor for the class 
 *
 * 		Manual assignment.
 */
	proposalFnData(
		int chainN,/**< Number of chains being used*/
		int proposalFnN,/**< Number of proposal functions */
		proposalFn *proposalFnArray,/**< array of proposalFn objects of length proposalFnN*/
		void **proposalFnVariables,/**< Array of auxiliary parameters for each proposal. If not needed, pass (void *)*/
		float *proposalFnProbFixed=nullptr ,/**< Probabilities of each step, float length proposalFnN -- if populated, all chains (regardless of temperature) have the same probabilities*/
		float **proposalFnProb=nullptr ,/**< Probabilities of each step for each chain, 2d float shape [chainN][proposalFnN] -- if populated, chains *do not* have the same probabilities necessarily*/
		proposalFnWriteCheckpoint *writeCheckpointFns=nullptr,/**< array of proposalWriteCheckpoint functions. If not needed, pass (void *)*/
		proposalFnLoadCheckpoint *loadCheckpointFns=nullptr/**< array of proposalReadCheckpoint functions. If not needed, pass (void *)*/
		);
	/*! \brief Constructor for the class 
 *
 * 		Default values will use the standard proposals 
 */
	proposalFnData(
		int chainN,/**< number of chains being used*/
		int maxDim,/**< Maximum number of dimensions*/
		bool RJ/**< Bool for RJ or regular MCMC*/
		);
	/*! \brief Destructor to free data*/
	~proposalFnData();

private:
	/*! Number of chains in the ensemble*/
	int chainN;
	/*! Maximum dimension of the parameter space*/	
	int maxDim;
	/*! Memory flag for internal memory allocation (so memory can be appropriately freed)*/
	bool internalMemorySet=false;
	
	
};




class bayesshipSampler
{
public:
	/* Run Specific Parameters -- Set every run regardless of checkpoint */

	/*! The raw number of steps to take -- Either this parameter or independentSamples is used*/
	int iterations=100;
	/*! Estimates the number of independent samples to draw and continues until independentSamples is reached -- Either this parameter or iterations is used*/
	int independentSamples=0;
	/*! Number of raw samples to draw before calculating autocorrelation and writing checkpoint for independentSamples*/
	int batchSize=0;
	/*! Samples to burn in with -- first burnIterations for each chain will not obey strict MCMC requirements*/
	int burnIterations = 0;
	/*! Samples to take from prior */
	int priorIterations = 0;
	/*! Samples to burn in with for the prior-- first burnPriorIterations for each chain will not obey strict MCMC requirements*/
	int burnPriorIterations = 0;
	/*! Boolean flag to write out prior data*/
	bool writePriorData = true;
	/*! Whether to store only the cold chains or the full ensemble -- Full ensemble produces much larger files*/
	bool coldOnlyStorage=true;
	/*! Number of threads to launch*/
	int threads=1;
	/*! Class containing all the information about the proposal functions used in the sampling*/
	proposalFnData *proposalFns=nullptr;
	/*! Output Destinations*/
	std::string outputDir="";
	/*! Output files base name*/
	std::string outputFileMoniker="BayesShip";
	/*! Likelihood function*/
	//likelihoodFn likelihood;
	probabilityFn *likelihood;
	/*! Prior function*/
	//likelihoodFn prior;
	probabilityFn *prior;
	/*! User Parameters -- These parameters are passed into the likelihood function and the prior function -- shape should be (void *)[chainN]*/
	void **userParameters=nullptr;
	/*! If a checkpoint file exists in the output directory, ignore it (true) or load it (false)*/
	bool ignoreExistingCheckpoint=false;


	/* Meta Data -- read in from Checkpoint File*/
	/*! Run variable dimension RJMCMC (true) or regular, fixed dimension MCMC (false)*/
	bool RJ=false;
	/*! Whether to use a pool of threads (true) or OpenMP (false)*/
	bool threadPool=false;
	/*! Maximum dimension of the parameter space*/
	int maxDim=1;
	/*! Minimum dimension of the parameter space -- useful to increase efficiency for RJ with nested models -- leave at 0 as default (not nested models)*/
	int minDim=0;
	/*! Number of chains in each Ensemble of temperatures*/
	int ensembleSize=5;
	/*! Number of ensembles to run in parallel*/
	int ensembleN=2;
	/*! Beta schedule for a single ensemble (beta_i = 1/temp_i) starting with 1 and moving to 0 (ie, from temperature 1 to temperature infinity)  -- shape [ensembleSize]*/
	double *betaSchedule = nullptr;	
	/*! prior ranges for sampling the prior -- size = [maxDim][2] -- [min,max]*/
	double **priorRanges =nullptr;

	/*! Seed for random number generation*/
	double seed = 1;
	/*! Beta parameters for each chain = 1/T -- shape [chainN]*/
	double *betas=nullptr;	
	double swapProb = .2;
	/*! Whether to average the temperature dynamics during evolution (true), or after (false)*/
	bool averageDynamics = true;
	bool randomizeSwapping = true;
	/************************************/
	/*! If specified, the initial position for each chain will be assigned in order -- shape [chainN]*/
	positionInfo **initialPositionEnsemble=nullptr;
	/*! If the ensemble position is not specified, and the single position is specified, the initial position is assigned to every chain in the ensemble*/
	positionInfo *initialPosition=nullptr;
	/************************************/
	

	// non-user-facing members (should probably be private but that's annoying
	/* Re-initialized between runs*/
	samplerData *data=nullptr;
	samplerData *priorData=nullptr;
	samplerData *burnData=nullptr;

	bool *waitingSample=nullptr;
	bool *referenceStatus=nullptr;
	std::mutex *waitingMutexes=nullptr;
	std::mutex statusMutex;
	bool burnPeriod = false;
	bool adjustTemps = false;
	/* Parameters for burn in temperature adjustment*/
	double t0;
	double nu=100;
	double *A=nullptr;
	gsl_rng **rvec=nullptr;
		

	///*! All positions for sampler (ptrs) -- shape [chainN][iterations]*/
	//positionInfo ***positions=nullptr;
	///*! Array storing the current position for each sampler in the positions array -- shape [chainN]*/
	//int *currentStepID =nullptr;
	///*! Array containing all the likelihood values for each position in positions -- shape [chainN][iterations]*/
	//double **likelihoodVals=nullptr;
	///*! Array containing all the prior values for each position in positions -- shape [chainN][iterations]*/
	//double **priorVals=nullptr;
	///*! Array counting the rejected number of steps for each proposal type for each chain -- shape [chainN][proposalFnN]*/
	//int **rejectN=nullptr;
	///*! Array counting the successful number of steps for each proposal type for each chain -- shape [chainN][proposalFnN]*/
	//int **successN=nullptr;

	//int **swapRejects=nullptr;

	//int **swapAccepts=nullptr;



	//###################################
	//bayesshipSampler(int chainN=10, double *betas=(double*)nullptr, int maxDim=1, int proposalFnN=0,proposalFn *proposalFnArray=nullptr,float *proposalFnProb=nullptr, void **proposalVariables=nullptr);
	bool checkStatus();
	void allocateMemory();
	void deallocateMemory();
	//bayesshipSampler(likelihoodFn likelihood, priorFn prior);
	bayesshipSampler(probabilityFn *likelihood, probabilityFn *prior);
	~bayesshipSampler();
	void sample();
	void sampleLoop(int iterations,samplerData *data);
	void stepMH(int chainID,samplerData *data);
	int getChainN();
	double getBeta(int chainID);

	void setInitialPositionIdentical(positionInfo *position);

	void assignInitialPosition(samplerData *data);

	void chainSwap(int chainID1, int chainID2,samplerData *data);
	void adjustTemperatures(int t);
	int chainIndex(int ensemble, int betaN);
	int betaN(int chainID);
	
	/*! Write checkpoint file for sampler -- stores data in json to completely reconstruct Sampler object*/
	void writeCheckpoint(samplerData *data);
	/*! Load checkpoint file for sampler -- Loads data and allocates memory for sampler
 * 		A sampler object with the following must be first declared, then this function can be called:
 * 			Likelihood function
 * 			Prior function
 * 			ProposalFns
 * 			Outdir and outdir moniker (identical to previous run)
 * 			new number of steps (burn,prior, independent, etc)
 * 		All else will be initialized 
 * 		*/
	void loadCheckpoint();
	//NLOHMANN_DEFINE_TYPE_INTRUSIVE(bayesshipSampler,chainN)
private:
	/*! Number of chains total = ensembleSize*ensembleN*/
	int chainN;
	/*! Temperature parameters beta_i = 1/T_i*/
	bool internalBetaScheduleFlag = false;
	bool internalProposalFnsFlag = false;
	bool internalUserParameters=false;
	/*! iterations + burnSamples*/
	bool internalPriorRanges=false;
	bool internalInitialPositionEnsemble=false;

};

void to_json(nlohmann::json& j, const bayesshipSampler& s);
//void from_json(nlohmann::json& j, bayesshipSampler& s);

}
#endif
