#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <armadillo>
#include <nlohmann/json.hpp>
#include <fstream>


/*! \file
 *
 * # Source file for the Fisher proposal template
 */
namespace bayesship{

/*! Same scheme as blockFisherProposal: the Fisher state is kept once in <moniker>_fisher/ (it only changes during
 * burn-in), and each checkpoint generation gets a reference file with the size and hash of the Fisher file it goes with.*/
static std::string fisherDirectory(bayesshipSampler *sampler)
{
	return sampler->outputDir+sampler->outputFileMoniker+"_fisher";
}

void fisherProposal::writeCheckpoint(std::string outputDirectory, std::string runMoniker)
{
	std::string fisherFile = fisherDirectory(sampler)+"/fisherProposal.json";
	if(fisherChanged){
		std::cout<<"Writing Fisher state to "<<fisherFile<<std::endl;
		nlohmann::json j = fisherState();
		if(makeDirectory(fisherDirectory(sampler)) && atomicWriteFile(fisherFile, j.dump())){
			fisherChanged = false;
		}
		else{
			std::cout<<"WARNING: could not write "<<fisherFile<<" -- will retry with the next checkpoint"<<std::endl;
		}
	}

	uint64_t hash, size;
	if(!fileHashFNV1a(fisherFile, &hash, &size)){
		/*Nothing computed yet (e.g. no burn-in), so nothing to refer to*/
		return;
	}
	nlohmann::json ref;
	ref["file"] = "fisherProposal.json";
	ref["size"] = size;
	ref["hash"] = hash;
	std::ofstream fileOut(outputDirectory+runMoniker + "_checkpoint_fisherProposalRef.json");
	fileOut << ref;
}

nlohmann::json fisherProposal::fisherState()
{
	nlohmann::json j;
	j["chainN"] = chainN;
	j["maxDim"] = maxDim;
	for(int i = 0 ; i<chainN; i++){
		nlohmann::json entry;
		entry["noFisher"] = noFisher[i];
		entry["attempts"] = FisherAttemptsSinceLastUpdate[i];
		/*Matrices are uninitialized until first computed*/
		if(!noFisher[i]){
			std::vector<double> fisher, vecs;
			for(int k = 0 ; k<maxDim; k++){
				fisher.insert(fisher.end(), Fisher[i][k], Fisher[i][k]+maxDim);
				vecs.insert(vecs.end(), FisherEigenVecs[i][k], FisherEigenVecs[i][k]+maxDim);
			}
			entry["Fisher"] = fisher;
			entry["eigenVals"] = std::vector<double>(FisherEigenVals[i], FisherEigenVals[i]+maxDim);
			entry["eigenVecs"] = vecs;
		}
		j["Chains"].push_back(entry);
	}
	return j;
}

void fisherProposal::loadCheckpoint(std::string inputDirectory, std::string runMoniker)
{
	std::string refFile(inputDirectory+runMoniker + "_checkpoint_fisherProposalRef.json");
	if(!checkDirExist(refFile)){
		/*Checkpoint from before the Fisher state was stored, or taken before any was computed*/
		return;
	}
	nlohmann::json ref;
	std::ifstream refIn(refFile);
	refIn>>ref;
	std::string inputFile = fisherDirectory(sampler)+"/"+ref.at("file").get<std::string>();
	uint64_t hash, size;
	if(!fileHashFNV1a(inputFile, &hash, &size) || size != ref.at("size").get<uint64_t>() || hash != ref.at("hash").get<uint64_t>()){
		std::cout<<"WARNING: "<<inputFile<<" is missing or does not match this checkpoint -- the Fisher matrices will be recomputed"<<std::endl;
		return;
	}
	nlohmann::json j;
	std::ifstream fileIn(inputFile);
	fileIn>>j;
	if(j.at("chainN").get<int>() != chainN || j.at("maxDim").get<int>() != maxDim){
		std::cout<<"WARNING: "<<inputFile<<" does not match the current chains/dimension -- the Fisher matrices will be recomputed"<<std::endl;
		return;
	}
	std::cout<<"Loading Fisher state from "<<inputFile<<std::endl;
	for(int i = 0 ; i<chainN; i++){
		const nlohmann::json &entry = j["Chains"][i];
		noFisher[i] = entry.at("noFisher").get<bool>();
		FisherAttemptsSinceLastUpdate[i] = entry.at("attempts").get<int>();
		if(noFisher[i]){
			continue;
		}
		std::vector<double> fisher = entry.at("Fisher").get<std::vector<double>>();
		std::vector<double> vals = entry.at("eigenVals").get<std::vector<double>>();
		std::vector<double> vecs = entry.at("eigenVecs").get<std::vector<double>>();
		for(int k = 0 ; k<maxDim; k++){
			for(int l = 0 ; l<maxDim; l++){
				Fisher[i][k][l] = fisher[k*maxDim+l];
				FisherEigenVecs[i][k][l] = vecs[k*maxDim+l];
			}
			FisherEigenVals[i][k] = vals[k];
		}
	}
	/*What is in memory now is exactly what is on disk*/
	fisherChanged = false;
}

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
	if(!noFisher){ 
		noFisher = new bool[chainN];

		for(int i = 0 ; i<chainN; i++){
			noFisher[i] = true;
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
		Fisher = nullptr;
	}
	if(FisherEigenVecs){
		for(int i = 0 ; i<chainN; i++){
			for(int j = 0 ; j<maxDim; j++){
				delete [] FisherEigenVecs[i][j];
			}
			delete [] FisherEigenVecs[i];
		}
		delete [] FisherEigenVecs;
		FisherEigenVecs = nullptr;
	}
	if(FisherEigenVals){
		for(int i = 0 ; i<chainN; i++){
			delete [] FisherEigenVals[i];
		}
		delete [] FisherEigenVals;
		FisherEigenVals = nullptr;
	}
	if(FisherAttemptsSinceLastUpdate){
		delete [] FisherAttemptsSinceLastUpdate;
		FisherAttemptsSinceLastUpdate = nullptr;
	}
	if(noFisher){
		delete [] noFisher;
		noFisher=nullptr;
	}
}

void fisherProposal::propose(positionInfo *currentPosition, positionInfo *proposedPosition,int chainID,int stepID,  double *MHRatioModification)
{
	proposedPosition->updatePosition(currentPosition);

	//if((FisherAttemptsSinceLastUpdate[chainID] >= updateFreq) && (sampler->burnPeriod)){
	if(	
		((FisherAttemptsSinceLastUpdate[chainID] >= updateFreq) && (sampler->burnPeriod)) 
		||
		noFisher[chainID]
	){
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
		noFisher[chainID]= false;
		fisherChanged = true;
		for(int i = 0 ; i<maxDim; i++){
			for(int j = 0 ; j<maxDim; j++){
				/*Armadillo returns eigenvectors as columns; store eigenvector j as row j, the way propose() reads it*/
				FisherEigenVecs[chainID][j][i] = eigenvec(i,j);
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
	if(proposedPosition->RJ){
		for(int i = 0 ; i<maxDim; i++){
			if(proposedPosition->status[i] == 1){
				proposedPosition->parameters[i] += randGauss * (FisherEigenVecs[chainID][randDim][i]);
			}
		}
	}
	else{
		for(int i = 0 ; i<maxDim; i++){
			proposedPosition->parameters[i] += randGauss * (FisherEigenVecs[chainID][randDim][i]);
		}
	}
	FisherAttemptsSinceLastUpdate[chainID] ++;
	
	return;
}

}
