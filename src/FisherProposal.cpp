#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <armadillo>


/*! \file
 *
 * # Source file for the Fisher proposal template
 */
namespace bayesship{

fisherProposal::fisherProposal(int chainN, int maxDim, FisherCalculation fisherCalc, void **parameters, int updateFreq, bayesshipSampler *sampler)
{

	this->parameters = parameters;
	this->chainN = chainN;
	this->maxDim = maxDim;
	this->fisherCalc = fisherCalc;
	this->updateFreq = updateFreq;
	this->sampler = sampler;

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
	
fisherProposal::~fisherProposal()
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

void fisherProposal::propose(positionInfo *currentPosition, positionInfo *proposedPosition,int chainID,int stepID,  double *MHRatioModification)
{
	proposedPosition->updatePosition(currentPosition);

	if((FisherAttemptsSinceLastUpdate[chainID] >= updateFreq) && (sampler->burnPeriod)){
	//if((FisherAttemptsSinceLastUpdate[chainID] >= updateFreq) ){
		fisherCalc(currentPosition,  Fisher[chainID], parameters[chainID]);
		//Update eigenvalues/eigenvectors
		arma::mat f;
		f.zeros(maxDim,maxDim);
		for(int i = 0 ; i<sampler->maxDim; i++){
			for(int j = 0 ; j<sampler->maxDim; j++){
				//std::cout<<Fisher[chainID][i][j]<<", ";
				f(i,j) = Fisher[chainID][i][j];
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
		for(int i = 0 ; i<maxDim; i++){
			for(int j = 0 ; j<maxDim; j++){
				FisherEigenVecs[chainID][i][j] = eigenvec(i,j);
			}
			FisherEigenVals[chainID][i] = eigval(i);
		}
		FisherAttemptsSinceLastUpdate[chainID] = 0;

		
	}

	int randDim = (int)(gsl_rng_uniform(sampler->rvec[chainID])*maxDim);	
	//Adding 1e-10 to soften the effect of broadening gaussian step based on Temperature dependence -- beta==0 is problematic
	double scaling = std::abs(FisherEigenVals[chainID][randDim]); 
	if(scaling <10. ){scaling = 10.;}
	scaling *=(sampler->betas[chainID]+1e-5);
	double randGauss = gsl_ran_gaussian(sampler->rvec[chainID], 1./std::sqrt(scaling));	
	for(int i = 0 ; i<maxDim; i++){
		proposedPosition->parameters[i] += randGauss * (FisherEigenVecs[chainID][randDim][i]);
	}
	FisherAttemptsSinceLastUpdate[chainID] ++;
	
	return;
}

}
