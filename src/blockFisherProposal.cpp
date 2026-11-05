#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"
#include <armadillo>
#include <nlohmann/json.hpp>
#include <fstream>


/*! \file
 *
 * # Source file for the Gibbs block Fisher proposal template
 *
 * This version of the Fisher proposal only focuses on a block of parameter space. This can be more manageable for larger parameters spaces. It accepts multiple chunks of parameters, and randomly picks a block according to the probability vector blockProb.
 *
 */
namespace bayesship{

/*! The Fisher matrices are only recomputed during burn-in, so the ones in use while sampling cannot be rebuilt
 * when a run resumes -- store them (with their eigen-decompositions) so a resumed run proposes exactly as before.
 *
 * Since they stop changing after burn-in, they are kept once in their own folder, <moniker>_fisher/, and rewritten
 * only when a matrix was computed since the last write. Each checkpoint generation (outputDirectory here) gets a
 * small reference file with the size and hash of the Fisher file it goes with.*/
static std::string blockFisherDirectory(bayesshipSampler *sampler)
{
	return sampler->outputDir+sampler->outputFileMoniker+"_fisher";
}

void blockFisherProposal::writeCheckpoint(std::string outputDirectory, std::string runMoniker)
{
	std::string fisherFile = blockFisherDirectory(sampler)+"/blockFisherProposal.json";
	if(fisherChanged){
		std::cout<<"Writing block Fisher state to "<<fisherFile<<std::endl;
		nlohmann::json j = fisherState();
		if(makeDirectory(blockFisherDirectory(sampler)) && atomicWriteFile(fisherFile, j.dump())){
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
	ref["file"] = "blockFisherProposal.json";
	ref["size"] = size;
	ref["hash"] = hash;
	std::ofstream fileOut(outputDirectory+runMoniker + "_checkpoint_blockFisherProposalRef.json");
	fileOut << ref;
}

nlohmann::json blockFisherProposal::fisherState()
{
	nlohmann::json j;
	std::vector<int> blockSizes;
	for(size_t b = 0 ; b<blocks.size(); b++){
		blockSizes.push_back(blocks[b].size());
	}
	j["chainN"] = chainN;
	j["blockSizes"] = blockSizes;
	for(int i = 0 ; i<chainN; i++){
		nlohmann::json chain = nlohmann::json::array();
		for(size_t b = 0 ; b<blocks.size(); b++){
			int n = blocks[b].size();
			nlohmann::json entry;
			entry["noFisher"] = noFisher[i][b];
			entry["attempts"] = FisherAttemptsSinceLastUpdate[i][b];
			/*Matrices are uninitialized until first computed*/
			if(!noFisher[i][b]){
				std::vector<double> fisher, vecs;
				for(int k = 0 ; k<n; k++){
					fisher.insert(fisher.end(), Fisher[i][b][k], Fisher[i][b][k]+n);
					vecs.insert(vecs.end(), FisherEigenVecs[i][b][k], FisherEigenVecs[i][b][k]+n);
				}
				entry["Fisher"] = fisher;
				entry["eigenVals"] = std::vector<double>(FisherEigenVals[i][b], FisherEigenVals[i][b]+n);
				entry["eigenVecs"] = vecs;
			}
			chain.push_back(entry);
		}
		j["Chains"].push_back(chain);
	}
	return j;
}

void blockFisherProposal::loadCheckpoint(std::string inputDirectory, std::string runMoniker)
{
	std::string refFile(inputDirectory+runMoniker + "_checkpoint_blockFisherProposalRef.json");
	if(!checkDirExist(refFile)){
		/*Checkpoint from before the Fisher state was stored, or taken before any was computed*/
		return;
	}
	nlohmann::json ref;
	std::ifstream refIn(refFile);
	refIn>>ref;
	std::string inputFile = blockFisherDirectory(sampler)+"/"+ref.at("file").get<std::string>();
	uint64_t hash, size;
	if(!fileHashFNV1a(inputFile, &hash, &size) || size != ref.at("size").get<uint64_t>() || hash != ref.at("hash").get<uint64_t>()){
		std::cout<<"WARNING: "<<inputFile<<" is missing or does not match this checkpoint -- the Fisher matrices will be recomputed"<<std::endl;
		return;
	}
	nlohmann::json j;
	std::ifstream fileIn(inputFile);
	fileIn>>j;

	std::vector<int> blockSizes;
	for(size_t b = 0 ; b<blocks.size(); b++){
		blockSizes.push_back(blocks[b].size());
	}
	if(j.at("chainN").get<int>() != chainN || j.at("blockSizes").get<std::vector<int>>() != blockSizes){
		std::cout<<"WARNING: "<<inputFile<<" does not match the current chains/blocks -- the Fisher matrices will be recomputed"<<std::endl;
		return;
	}
	std::cout<<"Loading block Fisher checkpoint from "<<inputFile<<std::endl;
	for(int i = 0 ; i<chainN; i++){
		for(size_t b = 0 ; b<blocks.size(); b++){
			const nlohmann::json &entry = j["Chains"][i][b];
			int n = blocks[b].size();
			noFisher[i][b] = entry.at("noFisher").get<bool>();
			FisherAttemptsSinceLastUpdate[i][b] = entry.at("attempts").get<int>();
			if(noFisher[i][b]){
				continue;
			}
			std::vector<double> fisher = entry.at("Fisher").get<std::vector<double>>();
			std::vector<double> vals = entry.at("eigenVals").get<std::vector<double>>();
			std::vector<double> vecs = entry.at("eigenVecs").get<std::vector<double>>();
			for(int k = 0 ; k<n; k++){
				for(int l = 0 ; l<n; l++){
					Fisher[i][b][k][l] = fisher[k*n+l];
					FisherEigenVecs[i][b][k][l] = vecs[k*n+l];
				}
				FisherEigenVals[i][b][k] = vals[k];
			}
		}
	}
	/*What is in memory now is exactly what is on disk*/
	fisherChanged = false;
}

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
			
			std::cout<<"Failed Fisher"<<std::endl;
			return;	
		}
		noFisher[chainID][alpha]= false;
		fisherChanged = true;
		for(int i = 0 ; i<blocks[alpha].size(); i++){
			for(int j = 0 ; j<blocks[alpha].size(); j++){
				/*Armadillo returns eigenvectors as columns; store eigenvector j as row j, the way propose() reads it*/
				FisherEigenVecs[chainID][alpha][j][i] = eigenvec(i,j);
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
			//std::cout<<blocks[alpha][i]<<"|"<<randGauss * (FisherEigenVecs[chainID][alpha][randDim][i])<<"\n";
		}
	}
	//std::cout<<std::endl;
	FisherAttemptsSinceLastUpdate[chainID][alpha] ++;


	//Testing 
	//####################################################
		
	//####################################################
	
	return;
}

}
