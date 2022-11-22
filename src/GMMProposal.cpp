#include "bayesship/proposalFunctions.h"
#include <armadillo>

namespace bayesship{

GMMProposal::GMMProposal(int chainN, int maxDim, bayesshipSampler *sampler, int gaussians, int km_iter, int em_iter, double var_floor, bool RJ, int updateInterval)
{
	
	this->chainN = chainN;
	this->maxDim = maxDim;
	this->sampler = sampler;
	this->gaussians = gaussians;
	this->km_iter = km_iter;
	this->em_iter = em_iter;
	this->var_floor = var_floor;
	this->RJ = RJ;
	this->updateInterval = updateInterval;
	this->primed = new bool[chainN];
	this->stepNumber = new int[chainN];
	for(int i = 0 ; i<chainN; i++){
		this->primed[i] = false;
		this->stepNumber[i] = 0;
	}

	this->models = new arma::gmm_full[chainN];

	this->currentData = new samplerData*[chainN];
	for(int i = 0 ; i<chainN; i++){
		currentData[i] = nullptr;
	}
		

	return;
}

GMMProposal::~GMMProposal()
{
	delete [] this->models;
	delete [] this->primed;
	delete [] this->currentData;
	return;
}

bool GMMProposal::train(int chainID)
{
	int samples  = stepNumber[chainID];
	arma::mat data(maxDim, samples, arma::fill::zeros);
	for(int i = 0 ; i<samples; i++){
		for(int j = 0 ; j<maxDim; j++){
			data.at(j, i) = sampler->activeData->positions[chainID][i]->parameters[j];
		}
	}
	bool status = models[chainID].learn(data, gaussians, arma::maha_dist, arma::random_subset, km_iter, em_iter, var_floor, false);
	if(status){
		primed[chainID] = true;
	}
	return status;
}

void GMMProposal::propose(positionInfo *current, positionInfo *proposed, int chainID,int stepID,double *MHRatioModifications)
{
	proposed->updatePosition(current);
	stepNumber[chainID] = sampler->activeData->currentStepID[chainID];
	if(!currentData[chainID]){
		primed[chainID] = false;
		currentData[chainID] = sampler->activeData;
	}
	else if(currentData[chainID] != sampler->activeData){
		primed[chainID] = false;
		currentData[chainID] = sampler->activeData;
	}
	bool status=true;
	if(stepNumber[chainID]%updateInterval== 0 && stepNumber[chainID] !=0 ){
		status = train(chainID);
	}
	if(!primed[chainID] ){
		return;
	}
	arma::vec v = models[chainID].generate();	
	double probProposed = models[chainID].log_p(v);
	//std::cout<<"DONE"<<std::endl;
	for(int i = 0 ; i<maxDim ; i++){
		proposed->parameters[i] = v.at(i);
	}
	for(int i = 0 ; i<maxDim ; i++){
		v.at(i) = current->parameters[i];
	}
	double probCurrent = models[chainID].log_p(v);
	*MHRatioModifications = probCurrent-probProposed;
	return;
}

};
