#ifndef PROPOSALFUNCTIONS_H
#define PROPOSALFUNCTIONS_H
#include "bayesship/bayesshipSampler.h"
#include "bayesship/dataUtilities.h"
#include <vector>
#include <string>

/*! \file 
 *
 * # Header file for general proposal functions
 */

#ifdef _MLPACK
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#endif

namespace bayesship{

//class proposalFn
//{
//public:
//	proposalFn();
//	~proposalFn();
//	virtual void propose(positionInfo *current, positionInfo *proposed, int chainID)
//	{
//		proposed.updatePosition(current);
//		return;	
//	};
//
//};

/*! Generic framework for setting up a class for a proposal function and its associated data
 */
class genericProposalVariables
{};


//###################################################################
//###################################################################

typedef void(*FisherCalculation)(
	positionInfo *pos,
	bayesshipSampler *sampler,
	double **fisher,
	void *parameters
	);

/*! Gaussian proposal*/
void FisherProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler,double *MHRatioModifications);

/*! Class for storing data associated with the gaussian proposal
 */
class FisherProposalVariables : public genericProposalVariables
{
public:
	/*! Number of chains*/
	int chainN;
	/*! Maximum dimension of the space*/
	int maxDim;
	/*! Array of step widths (standard deviations) of Gaussian proposals for each dimension for each chain*/
	double ***Fisher=nullptr;
	double **FisherEigenVals = nullptr;
	double ***FisherEigenVecs = nullptr;
	
	int *FisherAttemptsSinceLastUpdate=nullptr;
	int updateFreq = 200;

	void **parameters=nullptr;
	
	FisherCalculation fisherCalc;

	/*! Constructor function*/
	FisherProposalVariables(
		int chainN, /**< Number of chains in the ensemble*/
		int maxDim, /**< Maximum dimension of the space*/
		FisherCalculation fisherCalc,
		void ** parameters,
		int updateFreq=200
		);
	~FisherProposalVariables();

};

//###################################################################
//###################################################################


//###################################################################
//###################################################################
//Checkpoint functions to read and write out gaussianProposalVariables object
void gaussianProposalWriteCheckpoint( void *var,bayesshipSampler *sampler);
void gaussianProposalLoadCheckpoint( void *var,bayesshipSampler *sampler);

/*! Gaussian proposal*/
void gaussianProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler,double *MHRatioModifications);

/*! Class for storing data associated with the gaussian proposal
 */
class gaussianProposalVariables : public genericProposalVariables
{
public:
	/*! Vector of random number generators*/
	gsl_rng **r=nullptr;
	/*! Number of chains*/
	int chainN;
	/*! Maximum dimension of the space*/
	int maxDim;
	/*! Array of step widths (standard deviations) of Gaussian proposals for each dimension for each chain*/
	double **gaussWidths=nullptr;
	/*! Track last step ID for updating widths*/
	int *previousDimID=nullptr;
	/*! Track what the last acceptance was for updating widths*/
	int *previousAccepts=nullptr;

	/*! Constructor function*/
	gaussianProposalVariables(
		int chainN, /**< Number of chains in the ensemble*/
		int maxDim, /**< Maximum dimension of the space*/
		int seed=1 /**< Seed to use for initiating random numbers*/
		);
	~gaussianProposalVariables();

};

//###################################################################
//###################################################################

/*KDE proposal*/
void KDEProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler,double *MHRatioModifications);

/*! Class for storing data associated with the KDE proposal
 */
class KDEProposalVariables : public genericProposalVariables
{
public:
	/*! Vector of random number generators*/
	gsl_rng **r=nullptr;
	/*! Number of chains*/
	int chainN;
	/*! Maximum dimension of the space*/
	int maxDim;
	/*! Current step number in storage*/
	int *stepNumber=nullptr;
	/*! Current size of the storage for the KDE structure*/
	int *currentStorageSize=nullptr;
	/*! Last ID that was harvested from a samplerData structure*/
	int *lastUpdatePositionID=nullptr;
	/*! Continuously updated variances of the different distributions*/
	double ***runningCov=nullptr;
	double ***runningCovCholeskyDecomp=nullptr;
	double **runningSTD=nullptr;
	/*! Continuously updated means of the different distributions*/
	double **runningMean=nullptr;
	/*Bandwidths of the various kdes -- using Scott factor*/
	double *bandwidth = nullptr;
	/*! Pointer of the current samplerData object -- if this changes, we can restart counters (Does NOT erase old samples)*/
	samplerData **currentData = nullptr;

	/*! Previous samples stored for KDE usage*/
	/*! Could make this a new structure as a linked list to have variable size, but not now*/
	positionInfo ***storedSamples = nullptr;
	
	/*! The number of samples to skip between storing samples in storage*/
	int updateInterval;

	int *drawCt = nullptr;

	/*! Batch size to allocate memory for the storage of past samples*/
	int batchSize;

	bool RJ;

	bool useMLPack=false;

	/*! Batch size to use for each training of the KDE*/
	int KDETrainingBatchSize;
	/*! IDs used for the latest training*/
	std::vector<int> *trainingIDs=nullptr;

#ifdef _MLPACK
	mlpack::kde::KDE<mlpack::kernel::GaussianKernel,mlpack::metric::EuclideanDistance,arma::mat, mlpack::tree::KDTree> **kde=nullptr;
#endif
	
	/*! Constructor function*/
	KDEProposalVariables(
		int chainN, /**< Number of chains in the ensemble*/
		int maxDim, /**< Maximum dimension of the space*/
		bool RJ=false,
		int batchSize = 5000,/**< Batch size to allocate memory for data points */
		int KDETrainingBatchSize= 1000,/**< Batch size to use for KDE training/eval ; -1 means full*/
		int updateInterval = 5,/**< number of steps to take before storing a sample*/
		int seed=1 /**< Seed to use for initiating random numbers*/
		);
	~KDEProposalVariables();
	void updateCov( int chainID);
	//void updateVar( int chainID);
	int trainKDE(int chainID );
	int trainKDEMLPACK(int chainID );
	int trainKDECustom(int chainID );
	void updateStorageSize(int chainID);
	void reset(int chainID);
	double evalKDEMLPACK(positionInfo *position,int chainID); 
	double evalKDECustom(positionInfo *position,int chainID); 
	double evalKDE(positionInfo *position,int chainID); 
};


int KDEDraw(positionInfo *sampleLocation, double **cov, positionInfo *output);

//###################################################################
//###################################################################

/*differentialEvolution proposal*/
void differentialEvolutionProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler,double *MHRatioModifications);

//###################################################################
//###################################################################
/*Reversible Jump proposal -- layer on modifications sequentially with creation probability alpha and destruction probability 1-alpha*/

void sequentialLayerRJProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler,double *MHRatioModifications);

class sequentialLayerRJProposalVariables : public genericProposalVariables
{
public:
	sequentialLayerRJProposalVariables(double alpha=.5){
		this->alpha = alpha;
	}
	double alpha= .5;
};

}

#endif
