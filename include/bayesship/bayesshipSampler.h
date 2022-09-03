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


/*Must forwardd declare the class to use it in the declarations of the functions below*/
class bayesshipSampler;
//class proposal;

class proposal
{
public:
	proposal(){return;};
	virtual ~proposal(){return;};
	virtual void propose(positionInfo *current, positionInfo *proposed, int chainID,int stepID,double *MHRatioModifications)
	{
		proposed->updatePosition(current);
		return;	
	};
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
	virtual void writeCheckpoint(std::string outputDirectory, std::string runMoniker )
	{
		return ;
	};
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
	virtual void loadCheckpoint( std::string inputDirectory, std::string runMoniker)
	{
		return ;
	};
	/*! \brief proposalFn write stat file 
	 *
	 * Writes out statistics about the proposal functions, if needed. This goes beyond what's already reported in the general stat file
	 * 
	 * */
	virtual void writeStatFile( std::string inputDirectory, std::string runMoniker)
	{
		return ;
	};


};





class probabilityFn
{
public:
	probabilityFn(){};
	virtual ~probabilityFn(){};
	virtual double eval(positionInfo *position, int chainID) { return 0;}
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




/*! \brief Class containing all the information about proposal functions for the bayesshipSampler object
 *
 * Used internally to keep the information tidy inside the sampler object 
 *
 * Copies everything by value
 *
 * Implementations that want to use custom proposals will need to declare and populate one of these structures, then pass it to the bayesshipSampler object
 */
class proposalData
{
public:
	/*! Number of proposal functions*/
	int proposalN;
	/*! Array of proposal function pointers */
	proposal **proposals=nullptr;
	/*! Probability of each proposal*/
	double **proposalProb=nullptr;
	
	/*! \brief Constructor for the class 
 *
 * 		Manual assignment.
 */
	proposalData(
		int chainN,/**< Number of chains being used*/
		int proposalN,/**< Number of proposal functions */
		proposal **proposals,/**< array of proposalFn objects of length proposalFnN*/
		double *proposalProbFixed=nullptr ,/**< Probabilities of each step, float length proposalFnN -- if populated, all chains (regardless of temperature) have the same probabilities*/
		double **proposalProb=nullptr /**< Probabilities of each step for each chain, 2d float shape [chainN][proposalFnN] -- if populated, chains *do not* have the same probabilities necessarily*/
		);
	/*! \brief Constructor for the class 
 *
 * 		Default values will use the standard proposals 
 */
	proposalData(
		int chainN,/**< number of chains being used*/
		int maxDim,/**< Maximum number of dimensions*/
		bayesshipSampler *sampler,/**< Sampler associated with this data*/
		bool RJ/**< Bool for RJ or regular MCMC*/
		);
	/*! \brief Destructor to free data*/
	~proposalData();

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
	proposalData *proposalFns=nullptr;
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

	/*! Whether to allow for swapping between ensembles in burn perior*/
	bool isolateEnsembles=false;
	/*! Whether to allow for swapping between ensembles in collection*/
	bool isolateEnsemblesBurn=false;


	/* Meta Data -- read in from Checkpoint File*/
	/*! Run variable dimension RJMCMC (true) or regular, fixed dimension MCMC (false)*/
	bool RJ=false;
	/*! Whether to use a pool of threads (true) or OpenMP (false)
 * TODO This should be renamed. The sampler currently DOES use a thread pool no matter what, but only uses it for *swapping* if
 * this is true. Otherwise, it swaps in order.
 * */
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

	
	samplerData *activeData=nullptr;

	// non-user-facing members (should probably be private but that's annoying
	/* Re-initialized between runs*/
	samplerData *data=nullptr;
	samplerData *priorData=nullptr;
	samplerData *burnData=nullptr;

	bool *waitingSample=nullptr;
	bool *referenceStatus=nullptr;
	std::mutex *waitingMutexes=nullptr;
	std::mutex *statusMutex=nullptr;
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

	//void setInitialPositionIdentical(positionInfo *position);

	void assignInitialPosition(samplerData *data);

	void chainSwap(int chainID1, int chainID2,samplerData *data);
	void adjustTemperatures(int t);
	int chainIndex(int ensemble, int betaN);
	int betaN(int chainID);
	int ensembleID(int chainID);
	samplerData *getActiveData();
	void setActiveData( samplerData *newData);

	
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
	bool getCurrentIsolateEnsemblesInternal();
private:
	/*! Whether to allow for swapping between ensembles INTERNAL USE ONLY*/
	bool isolateEnsemblesInternal=false;
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

//void to_json(nlohmann::json& j, const bayesshipSampler& s);
//void from_json(nlohmann::json& j, bayesshipSampler& s);

}
#endif
