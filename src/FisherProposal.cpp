#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <armadillo>


/*! \file
 *
 * # Source file for the Fisher proposal template
 */
namespace bayesship{

FisherProposalVariables::FisherProposalVariables(int chainN, int maxDim, FisherCalculation fisherCalc, void **parameters, int updateFreq)
{

	this->parameters = parameters;
	this->chainN = chainN;
	this->maxDim = maxDim;
	this->fisherCalc = fisherCalc;
	this->updateFreq = updateFreq;

	if(!Fisher){
		Fisher = new double**[chainN];
		for(int i = 0 ; i<chainN; i++){
			Fisher[i] = new double*[maxDim];
			for(int j = 0 ; j<maxDim; j++){
				Fisher[i][j] = new double[maxDim];
			}
		}
	}
	if(!FisherEigenVals){
		FisherEigenVals = new double*[chainN];
		for(int i = 0 ; i<chainN; i++){
			FisherEigenVals[i] = new double[maxDim];
		}
	}
	if(!FisherEigenVecs){
		FisherEigenVecs = new double**[chainN];
		for(int i = 0 ; i<chainN; i++){
			FisherEigenVecs[i] = new double*[maxDim];
			for(int j = 0 ; j<maxDim; j++){
				FisherEigenVecs[i][j] = new double[maxDim];
			}
		}
	}
	if(!FisherAttemptsSinceLastUpdate){
		FisherAttemptsSinceLastUpdate = new int[chainN];
		for(int i = 0 ; i<chainN; i++){
			FisherAttemptsSinceLastUpdate[i] = this->updateFreq+1;
		}
	}

}
	
FisherProposalVariables::~FisherProposalVariables()
{
	if(Fisher){
		for(int i = 0 ; i<chainN; i++){
			for(int j = 0 ; j<maxDim; j++){
				delete [] Fisher[i][j];
			}
			delete [] Fisher[i];
		}
		delete [] Fisher;
	}
	if(FisherEigenVecs){
		for(int i = 0 ; i<chainN; i++){
			for(int j = 0 ; j<maxDim; j++){
				delete [] FisherEigenVecs[i][j];
			}
			delete [] FisherEigenVecs[i];
		}
		delete [] FisherEigenVecs;
	}
	if(FisherEigenVals){
		for(int i = 0 ; i<chainN; i++){
			delete [] FisherEigenVals[i];
		}
		delete [] FisherEigenVals;
	}
	if(FisherAttemptsSinceLastUpdate){
		delete [] FisherAttemptsSinceLastUpdate;
	}
}

void FisherProposal(samplerData *data, int chainID, int stepID, bayesshipSampler *sampler, double *MHRatioModification)
{
	positionInfo *currentPosition = data->positions[chainID][data->currentStepID[chainID]];
	positionInfo *proposedPosition = data->positions[chainID][data->currentStepID[chainID]+1];
	FisherProposalVariables *fpv = (FisherProposalVariables *)sampler->proposalFns->proposalFnVariables[stepID];	
	proposedPosition->updatePosition(currentPosition);

	if((fpv->FisherAttemptsSinceLastUpdate[chainID] >= fpv->updateFreq) && (sampler->burnPeriod)){
	//if((fpv->FisherAttemptsSinceLastUpdate[chainID] >= fpv->updateFreq) ){
		fpv->fisherCalc(currentPosition, sampler, fpv->Fisher[chainID], fpv->parameters[chainID]);
		//Update eigenvalues/eigenvectors
		arma::mat f;
		f.zeros(fpv->maxDim,fpv->maxDim);
		for(int i = 0 ; i<sampler->maxDim; i++){
			for(int j = 0 ; j<sampler->maxDim; j++){
				//std::cout<<fpv->Fisher[chainID][i][j]<<", ";
				f(i,j) = fpv->Fisher[chainID][i][j];
			}
			//std::cout<<std::endl;
		}
		arma::vec eigval;
		arma::mat eigenvec;
		bool success = eig_sym(eigval,eigenvec, f);
		if(!success){
			//std::cout<<"Failed Fisher"<<std::endl;
			return;	
		}
		for(int i = 0 ; i<sampler->maxDim; i++){
			for(int j = 0 ; j<sampler->maxDim; j++){
				fpv->FisherEigenVecs[chainID][i][j] = eigenvec(i,j);
			}
			fpv->FisherEigenVals[chainID][i] = eigval(i);
		}
		fpv->FisherAttemptsSinceLastUpdate[chainID] = 0;

		
	}

	int randDim = (int)(gsl_rng_uniform(sampler->rvec[chainID])*sampler->maxDim);	
	//Adding 1e-10 to soften the effect of broadening gaussian step based on Temperature dependence -- beta==0 is problematic
	double scaling = std::abs(fpv->FisherEigenVals[chainID][randDim]); 
	if(scaling <10. ){scaling = 10.;}
	scaling *=(sampler->betas[chainID]+1e-5);
	double randGauss = gsl_ran_gaussian(sampler->rvec[chainID], 1./std::sqrt(scaling));	
	for(int i = 0 ; i<sampler->maxDim; i++){
		proposedPosition->parameters[i] += randGauss * (fpv->FisherEigenVecs[chainID][randDim][i]);
	}
	fpv->FisherAttemptsSinceLastUpdate[chainID] ++;
	
	return;
}

}
