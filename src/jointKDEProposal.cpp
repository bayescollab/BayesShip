#include "bayesship/proposalFunctions.h"
#include "bayesship/utilities.h"

namespace bayesship{

jointKDEProposal::jointKDEProposal(int ensembleSize, int , int threads)
{
	this->additionalThreads = threads;
	this->ensembleSize  = ensembleSize;
}
jointKDEProposal::~jointKDEProposal()
{
	return;
}

void jointKDEProposal::propose(positionInfo *current, positionInfo *proposed, int chainID,int stepID,double *MHRatioModifications)
{
	return;
}

void jointKDEProposal::writeCheckpoint(std::string outputDirectory , std::string runMoniker)
{
	return;
}
void jointKDEProposal::loadCheckpoint( std::string inputDirectory, std::string runMoniker)
{
	return;
}

}
