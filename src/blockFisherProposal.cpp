#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <armadillo>


/*! \file
 *
 * # Source file for the Gibbs block Fisher proposal template
 *
 * This version of the Fisher proposal only focuses on a block of parameter space. This can be more manageable for larger parameters spaces. It accepts multiple chunks of parameters, and randomly picks a block according to the probability vector blockProb.
 *
 */
namespace bayesship{

blockFisherProposal::blockFisherProposal(int chainN, int maxDim, blockFisherCalculation fisherCalc, void **parameters, int updateFreq, bayesshipSampler *sampler, std::vector<std::vector<int>> blocks, std::vector<double> blockProb)
{
	this->parameters = parameters;
	this->chainN = chainN;
	this->maxDim = maxDim;
	this->fisherCalc = fisherCalc;
	this->updateFreq = updateFreq;
	this->sampler = sampler;
	this->blocks = std::vector<std::vector<int>>(blocks.size());
	this->blockProb = std::vector<double>(blocks.size());
	this->blockProbBoundaries = std::vector<double>(blocks.size());
	for(int i = 0 ; i<blocks.size(); i++){
		this->blockProb[i] = blockProb[i];
		this->blocks[i] = std::vector<int>(blocks[i].size());
		for(int j =0 ; j<blocks[i].size(); j++){
			this->blocks[i][j] = blocks[i][j];
		}
		
	}
	//Need the boundaries to do the random number generation easier. Storing for convenience and speed
	//For prob array {.3,.4,.3}, boundaries should be {.3,.7,1.}
	this->blockProbBoundaries[0] = blockProb[0];
	for(int i = 1 ; i<blocks.size(); i++){
		this->blockProbBoundaries[i] = 	this->blockProbBoundaries[i-1]+ this->blockProb[i];
	}

	if(!Fisher){
		Fisher = new double***[chainN];
		for(int i = 0 ; i<chainN; i++){
			Fisher[i] = new double**[blocks.size()];
			for(int j = 0 ; j<blocks.size(); j++){
				Fisher[i][j] = new double*[blocks[j].size()];
				for(int k =0 ; k<blocks[j].size(); k++){
					Fisher[i][j][k] = new double[blocks[j].size()];
				}
			}
		}
	}
	if(!FisherEigenVals){
		FisherEigenVals = new double**[chainN];
		for(int i = 0 ; i<chainN; i++){
			FisherEigenVals[i] = new double*[blocks.size()];
			for(int j = 0 ; j<blocks.size();j++){
				FisherEigenVals[i][j] = new double[blocks[j].size()];
			}
		}
	}
	if(!FisherEigenVecs){
		FisherEigenVecs = new double***[chainN];
		for(int i = 0 ; i<chainN; i++){
			FisherEigenVecs[i] = new double**[blocks.size()];
			for(int j = 0 ; j<blocks.size(); j++){
				FisherEigenVecs[i][j] = new double*[blocks[j].size()];
				for(int k = 0 ; k<blocks[j].size(); k++){
					FisherEigenVecs[i][j][k] = new double[blocks[j].size()];
			
				}
		
			}
		}
	}
	if(!FisherAttemptsSinceLastUpdate){
		FisherAttemptsSinceLastUpdate = new int*[chainN];
		for(int i = 0 ; i<chainN; i++){
			FisherAttemptsSinceLastUpdate[i] = new int[blocks.size()];
			for(int j = 0 ; j<blocks.size(); j++){
				FisherAttemptsSinceLastUpdate[i][j] = this->updateFreq+1;
			}
		}
	}
	if(!noFisher){ 
		noFisher = new bool*[chainN];

		for(int i = 0 ; i<chainN; i++){
			noFisher[i] = new bool[blocks.size()];
			for(int j = 0 ; j<blocks.size();j ++){
				noFisher[i][j] = true;
			}
		}
	
	}


}
	
blockFisherProposal::~blockFisherProposal()
{
	if(Fisher){
		for(int i = 0 ; i<chainN; i++){
			for(int j = 0 ; j<blocks.size(); j++){
				for(int k = 0 ; k<blocks[j].size(); k++){
					delete [] Fisher[i][j][k];
				}
				delete [] Fisher[i][j];
			}
			delete [] Fisher[i];
		}
		delete [] Fisher;
		Fisher = nullptr;
	}
	if(FisherEigenVecs){
		for(int i = 0 ; i<chainN; i++){
			for(int j = 0 ; j<blocks.size(); j++){
				for(int k = 0 ; k<blocks[j].size(); k++){
					delete [] FisherEigenVecs[i][j][k];
				}
				delete [] FisherEigenVecs[i][j];
			}
			delete [] FisherEigenVecs[i];
		}
		delete [] FisherEigenVecs;
		FisherEigenVecs = nullptr;
	}
	if(FisherEigenVals){
		for(int i = 0 ; i<chainN; i++){
			for(int j = 0 ; j<blocks.size();j++){
				delete [] FisherEigenVals[i][j];
			}
			delete [] FisherEigenVals[i];
		}
		delete [] FisherEigenVals;
		FisherEigenVals = nullptr;
	}
	if(FisherAttemptsSinceLastUpdate){
		for(int i = 0 ; i<chainN; i++){
			delete [] FisherAttemptsSinceLastUpdate[i];
		}
		delete [] FisherAttemptsSinceLastUpdate;
		FisherAttemptsSinceLastUpdate = nullptr;
	}
	if(noFisher){
		for(int i = 0 ; i<chainN; i++){
			delete [] noFisher[i];
		}
		delete [] noFisher;
		noFisher=nullptr;
	}
}

void blockFisherProposal::propose(positionInfo *currentPosition, positionInfo *proposedPosition,int chainID,int stepID,  double *MHRatioModification)
{
	proposedPosition->updatePosition(currentPosition);
	
	//Pick random block
	//int alpha = (int)(gsl_rng_uniform(sampler->rvec[chainID])*blocks.size());
	double beta = gsl_rng_uniform(sampler->rvec[chainID]);
	int alpha = 0 ;
	for(int i = 0 ; i<blocks.size(); i++){
		if (beta < blockProbBoundaries[i]){
			alpha = i;
			break;
		}
	}

	//std::cout<<"Updating: ";
	//for(int i = 0 ; i<blocks[alpha].size();i++){
	//	std::cout<<blocks[alpha][i]<<", ";
	//}
	//std::cout<<std::endl;

	if(	
		((FisherAttemptsSinceLastUpdate[chainID][alpha] >= updateFreq) && (sampler->burnPeriod)) 
		||
		noFisher[chainID][alpha]
	){
		fisherCalc(currentPosition,  Fisher[chainID][alpha], blocks[alpha],parameters[chainID]);
		//Update eigenvalues/eigenvectors
		arma::mat f;
		f.zeros(blocks[alpha].size(),blocks[alpha].size());
		for(int i = 0 ; i<blocks[alpha].size(); i++){
			for(int j = 0 ; j<blocks[alpha].size(); j++){
				//std::cout<<Fisher[chainID][i][j]<<", ";
				f(i,j) = Fisher[chainID][alpha][i][j];
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
		noFisher[chainID][alpha]= false;
		for(int i = 0 ; i<blocks[alpha].size(); i++){
			for(int j = 0 ; j<blocks[alpha].size(); j++){
				FisherEigenVecs[chainID][alpha][i][j] = eigenvec(i,j);
			}
			FisherEigenVals[chainID][alpha][i] = eigval(i);
		}
		FisherAttemptsSinceLastUpdate[chainID][alpha] = 0;

		
	}

	int randDim = (int)(gsl_rng_uniform(sampler->rvec[chainID])*blocks[alpha].size());	
	//Adding 1e-10 to soften the effect of broadening gaussian step based on Temperature dependence -- beta==0 is problematic
	double scaling = std::abs(FisherEigenVals[chainID][alpha][randDim]); 
	if(scaling <10. ){scaling = 10.;}
	scaling *=(sampler->betas[chainID]+1e-5);
	double randGauss = gsl_ran_gaussian(sampler->rvec[chainID], 1./std::sqrt(scaling));	
	if(proposedPosition->RJ){
		for(int i = 0 ; i<blocks[alpha].size(); i++){
			if(proposedPosition->status[blocks[alpha][i]]){
				proposedPosition->parameters[blocks[alpha][i]] += randGauss * (FisherEigenVecs[chainID][alpha][randDim][i]);
				//std::cout<<blocks[alpha][i]<<"|"<<randGauss * (FisherEigenVecs[chainID][alpha][randDim][i])<<", ";
			}
		}
	}
	else{
		//std::cout<<"Fisher:"<<std::endl;
		for(int i = 0 ; i<blocks[alpha].size(); i++){
			proposedPosition->parameters[blocks[alpha][i]] += randGauss * (FisherEigenVecs[chainID][alpha][randDim][i]);
			//std::cout<<blocks[alpha][i]<<"|"<<1./std::sqrt(scaling) * (FisherEigenVecs[chainID][alpha][randDim][i])<<"\n";
		}
	}
	//std::cout<<std::endl;
	FisherAttemptsSinceLastUpdate[chainID][alpha] ++;
	
	return;
}

}
