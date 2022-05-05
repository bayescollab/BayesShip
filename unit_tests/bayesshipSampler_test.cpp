#include <bayesship/bayesshipSampler.h>


#include <math.h>
#include <gtest/gtest.h>
#include <limits.h>

double prior(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler,void *userParameters){
	return 15.;
}
double likelihood(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler,void *userParameters){
	return 10.;
}

/*! Test likelihood for Gaussian distribution*/
double likelihoodGaussian(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler,void *userParameters){
	double x = position->parameters[0];
	double mean = 0;
	double sigma = 1;
	return -.5 * pow(x-mean,2)/sigma/sigma;
}
/*! Test likelihood that guarantees rejection*/
double likelihoodRejection(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler,void *userParameters){
	return - std::numeric_limits<double>::infinity();
}
double priorGaussian(bayesship::positionInfo *position, int chainID, bayesship::bayesshipSampler *sampler,void *userParameters){
	double x = position->parameters[0];
	double mean = 0;
	double sigma = 1;
	return -.5 * pow(x-mean,2)/sigma/sigma;
}

void MHAlgorithmTestProposalAccept(bayesship::samplerData *data, int chainID, int stepID, bayesship::bayesshipSampler *sampler, double *MHratioCorrection)
{

	/*Accept*/
	data->positions[chainID][data->currentStepID[chainID]+1]->parameters[0]=0;
}
void MHAlgorithmTestProposalReject(bayesship::samplerData *data, int chainID, int stepID, bayesship::bayesshipSampler *sampler, double *MHratioCorrection)
{
	/*Reject*/
	data->positions[chainID][data->currentStepID[chainID]+1]->parameters[0]=20;
}

//###############################################################
class bayesshipSamplerInitialPositionTesting : public testing::Test{
	protected:
	void SetUp() override{
		betas = new double[ensembleN*ensembleSize];
		for(int i = 0   ;i<ensembleN*ensembleSize; i++){
			betas[i] = 1;
		}

		sampler = new bayesship::bayesshipSampler(likelihoodGaussian,prior);
		sampler->maxDim=maxDim;
		sampler->ensembleN = ensembleN;
		sampler->ensembleSize = ensembleSize;
		sampler->allocateMemory();
		data= new bayesship::samplerData(maxDim, ensembleN,ensembleSize, 1, 1, false,betas);
	
		ensemblePosition = new bayesship::positionInfo*[ensembleN*ensembleSize];
		for(int i =0 ; i<ensembleN*ensembleSize; i++){
			ensemblePosition[i] = new bayesship::positionInfo(maxDim, false);
			for(int j = 0 ; j<maxDim; j++){
				ensemblePosition[i]->parameters[j] = i*j;

			}
		}
		singlePosition = new bayesship::positionInfo(maxDim, false);
		for(int j = 0 ; j<maxDim; j++){
			singlePosition->parameters[j] = j;

		}
	}
	void TearDown() override{
		delete sampler;
		for(int i =0 ; i<ensembleN*ensembleSize; i++){
			delete ensemblePosition[i];	
		}
		delete [] ensemblePosition;
		delete singlePosition;
		delete data;
		delete [] betas;
	}
	/*defaults*/
	double *betas = nullptr;
	int ensembleN = 2;
	int ensembleSize = 5;
	time_t start_time_;
	int maxDim = 4;
	bayesship::bayesshipSampler *sampler;;
	bayesship::positionInfo **ensemblePosition=nullptr;
	bayesship::positionInfo *singlePosition=nullptr;
	bayesship::samplerData *data;

};
/*! Test initial position assignment Ensemble style
 *
 */
TEST_F(bayesshipSamplerInitialPositionTesting, initialEnsemblePosition) {
	sampler->initialPositionEnsemble = ensemblePosition;
	sampler->assignInitialPosition(data);
	for(int i = 0 ; i<ensembleN*ensembleSize; i++){
		for(int j = 0 ; j<maxDim; j++){
			EXPECT_EQ(data->positions[i][0]->parameters[j],ensemblePosition[i]->parameters[j]);
		}
	}

	
}

/*! Test initial position assignment single position style
 *
 */
TEST_F(bayesshipSamplerInitialPositionTesting, initialSinglePosition) {
	sampler->initialPosition = singlePosition;
	sampler->assignInitialPosition(data);
	for(int i = 0 ; i<ensembleN*ensembleSize; i++){
		for(int j = 0 ; j<maxDim; j++){
			EXPECT_EQ(data->positions[i][0]->parameters[j],singlePosition->parameters[j]);
		}
	}

	
}
//###############################################################

//###############################################################
class bayesshipSamplerMHAlgorithms : public testing::Test{
	protected:
	void SetUp() override{
		bayesship::proposalFn propArray[1] = {MHAlgorithmTestProposalAccept};
		float prob = 1;
		tempPropVariables = new void *[1];	
		propFnData = new bayesship::proposalFnData(1, propArray, &prob,tempPropVariables);
		sampler = new bayesship::bayesshipSampler(likelihoodGaussian,prior);
		sampler->proposalFns = propFnData;
		sampler->maxDim=maxDim;
		sampler->iterations = iterations;
		sampler->ensembleN = ensembleN;
		sampler->ensembleSize = ensembleSize;
		sampler->allocateMemory();
		betas = new double[ensembleN*ensembleSize];
		for(int i =0 ; i<ensembleN*ensembleSize; i++){
			betas[i]=1;
		}
		data = new bayesship::samplerData(maxDim, ensembleN,ensembleSize, 3, 1,false,betas);
	}
	void TearDown() override{
		delete sampler;
		delete propFnData;
		delete [] tempPropVariables;
		delete data;
		delete [] betas;
	}
	double *betas=nullptr;
	/*defaults*/
	void **tempPropVariables = nullptr;
	bayesship::proposalFnData *propFnData=nullptr;
	int ensembleN = 2;
	int ensembleSize = 5;
	int iterations = 10;
	time_t start_time_;
	bool threadPool=false;
	int maxDim = 1;
	bayesship::bayesshipSampler *sampler;
	bayesship::samplerData *data=nullptr;

};
/*! Test basic functionality of stepMH 
 *
 * Verify that it rejects steps it should and accepts steps it should. 
 *
 * To guarantee predictable behavior, we have to swap the likelihood and proposal functions out so we can control the environment 
 */
TEST_F(bayesshipSamplerMHAlgorithms, stepMHBasic) {

	/*Set initial Position*/
	data->positions[0][0]->parameters[0] = 2;
	/*Set initial step ID*/
	data->currentStepID[0] = 0;
	/*Set initial likelihood*/
	data->likelihoodVals[0][0] = sampler->likelihood(data->positions[0][0],0, sampler,(void *)nullptr);
	/*Set initial prior*/
	data->priorVals[0][0] = sampler->prior(data->positions[0][0],0, sampler,(void*)nullptr);
	
	/*Step once -- should be accepted*/
	sampler->stepMH(0,data);

	/*Check acceptance*/
	EXPECT_EQ(data->successN[0][0],1);
	EXPECT_EQ(data->rejectN[0][0],0);

	/*Change proposal function to something that should be rejected*/
	sampler->proposalFns->proposalFnArray[0] = MHAlgorithmTestProposalReject;
	sampler->likelihood=likelihoodRejection;
	sampler->stepMH(0,data);

	/*Check acceptance and rejection*/
	EXPECT_EQ(data->successN[0][0],1);
	EXPECT_EQ(data->rejectN[0][0],1);

	/*Check to see if position is update*/
	EXPECT_EQ(data->positions[0][2]->parameters[0],0);
	EXPECT_EQ(data->currentStepID[0],2);
	
}
//###############################################################

//###############################################################
class bayesshipSamplerTestNonDefault : public testing::Test{
	protected:
	void SetUp() override{
		betaSchedule = new double[ensembleSize];
		betaSchedule[0] = 1;
		for(int i = 1 ; i<ensembleSize; i++){
			betaSchedule[i] = betaSchedule[i-1]*((double)(ensembleSize- i)/ensembleSize);
		}
		sampler = new bayesship::bayesshipSampler(likelihood,prior);
		sampler->betaSchedule = betaSchedule;
		sampler->ensembleSize=ensembleSize;
		sampler->ensembleN=ensembleN;
		sampler->maxDim = maxDim;
		sampler->threads = threads;
		sampler->threadPool = threadPool;
		sampler->allocateMemory();
	}
	void TearDown() override{
		delete [] betaSchedule;
		betaSchedule = nullptr;
		delete sampler;
	}
	double *betaSchedule=nullptr;
	time_t start_time_;
	int ensembleSize=8;
	int ensembleN=3;
	bool threadPool=true;
	double *betas = nullptr;
	int threads = 5;
	int maxDim = 4;
	bayesship::bayesshipSampler *sampler;;

};
// Demonstrate some basic assertions.
TEST_F(bayesshipSamplerTestNonDefault, INITIALIZATION) {
	EXPECT_EQ(sampler->getChainN(), ensembleSize*ensembleN);
	for(int i = 0 ; i<ensembleSize; i++){
		for(int j = 0 ; j<ensembleN; j++){
			EXPECT_EQ(sampler->getBeta(i*ensembleN + j), betaSchedule[i]);
		}
	}
	EXPECT_EQ(sampler->threadPool,threadPool);
	EXPECT_EQ(sampler->threads,threads);
	EXPECT_EQ(sampler->maxDim,maxDim);
	EXPECT_EQ(sampler->proposalFns->proposalFnN,bayesship::standardProposalFnN);
	
	/*Check basic functionality of calling probability distributions*/
	bayesship::positionInfo *pos=nullptr ;
	EXPECT_EQ(sampler->likelihood(pos, 1,sampler,(void*)nullptr),  likelihood(pos,1,sampler,(void*)nullptr));
	EXPECT_EQ(sampler->prior(pos, 1,sampler,(void*)nullptr),  prior(pos,1,sampler,(void*)nullptr));

}
//###############################################################

//#######################################################################
class bayesshipSamplerTestDefault : public testing::Test{
	protected:
	void SetUp() override{
		betaSchedule = new double[ensembleSize];
		betaSchedule[0] = 1;
		betaSchedule[ensembleSize - 1] = 0;
		double deltaBeta = std::pow( (1e-10) , 1./ensembleSize);
		for(int i = 1 ; i<ensembleSize-1; i++){
			betaSchedule[i] = betaSchedule[i-1]*deltaBeta;
		}
		sampler = new bayesship::bayesshipSampler(likelihood,prior);
		sampler->allocateMemory();
	}
	void TearDown() override{
		delete [] betaSchedule;
		betaSchedule = nullptr;
		delete sampler;
	}
	/*defaults*/
	double *betaSchedule=nullptr;
	time_t start_time_;
	int ensembleSize=5;
	int ensembleN=2;
	bool threadPool=false;
	int threads = 1;
	int maxDim = 1;
	bayesship::bayesshipSampler *sampler;

};
// Demonstrate some basic assertions.
TEST_F(bayesshipSamplerTestDefault, INITIALIZATION) {
	EXPECT_EQ(sampler->getChainN(), ensembleSize*ensembleN);
	for(int i = 0 ; i<ensembleSize; i++){
		for(int j = 0 ; j<ensembleN; j++){
			EXPECT_EQ(sampler->getBeta(i*ensembleN + j), betaSchedule[i]);
		}
	}
	EXPECT_EQ(sampler->threadPool,threadPool);
	EXPECT_EQ(sampler->threads,threads);
	EXPECT_EQ(sampler->maxDim,maxDim);
	EXPECT_EQ(sampler->proposalFns->proposalFnN,bayesship::standardProposalFnN);

}
//#######################################################################

class samplerDataTestDefault : public testing::Test{
	protected:
	void SetUp() override{
		betas = new double[chainN];
		for(int i = 0 ; i<chainN; i++){
			betas[i] = 1;
		}
		data = new bayesship::samplerData(maxDim, ensembleN,ensembleSize, iterations,proposalFnN,RJ,betas);
		for(int k = 0 ; k< chainN; k++){
			data->currentStepID[k] = iterations-1;
			for(int i = 0 ; i<iterations; i++){
				for(int j = 0 ; j<maxDim; j++){
					data->positions[k][i]->parameters[j] = k*i*j;
				}
				data->likelihoodVals[k][i] = k*i;
				data->priorVals[k][i] = -1*k*i;
			}
		}
	}
	void TearDown() override{
		delete data;
		delete [] betas;
	}
	/*defaults*/
	double *betas = nullptr;
	bayesship::samplerData *data;
	int maxDim = 5;
	int ensembleN = 4;
	int ensembleSize = 5;
	int chainN = 20;
	int iterations = 100;
	int proposalFnN = 2;
	bool RJ = false;

};
TEST_F(samplerDataTestDefault, EXTENDSIZE) {
	data->extendSize(100);
	for(int k = 0 ; k< chainN; k++){
		for(int i = iterations ; i<iterations+100; i++){
			for(int j = 0 ; j<maxDim; j++){
				data->positions[k][i]->parameters[j] = k*i*j;
			}
			data->likelihoodVals[k][i] = k*i;
			data->priorVals[k][i] = -1*k*i;
		}
	}
	
	for(int k = 0 ; k< chainN; k++){
		for(int i = 0 ; i<iterations; i++){
			for(int j = 0 ; j<maxDim; j++){
				EXPECT_EQ(data->positions[k][i]->parameters[j] , k*i*j);
			}
			EXPECT_EQ(data->likelihoodVals[k][i] , k*i);
			EXPECT_EQ(data->priorVals[k][i] , -1*k*i);
		}
	}
		
}
class samplerDataTestPRIMITIVE : public testing::Test{
	protected:
	void SetUp() override{
		betas = new double[chainN];
		for(int i = 0 ; i<chainN; i++){
			betas[i] = 1;
		}
		data = new bayesship::samplerData(maxDim, ensembleN,ensembleSize, iterations,proposalFnN,RJ,betas);
		for(int k = 0 ; k< chainN; k++){
			data->currentStepID[k] = iterations-1;
			for(int i = 0 ; i<iterations; i++){
				for(int j = 0 ; j<maxDim; j++){
					data->positions[k][i]->parameters[j] = k*i*j;
				}
			}
		}
	}
	void TearDown() override{
		delete data;
		delete [] betas;
	}
	/*defaults*/
	double *betas=nullptr;
	bayesship::samplerData *data;
	int maxDim = 5;
	int ensembleN = 4;
	int ensembleSize = 5;
	int chainN = 20;
	int iterations = 100;
	int proposalFnN = 2;
	bool RJ = false;

};
TEST_F(samplerDataTestPRIMITIVE, PRIMITIVECONVERSION) {
	double ***pointer = data->convertToPrimitivePointer();
	for(int k = 0 ; k< chainN; k++){
		for(int i = 0 ; i<iterations; i++){
			for(int j = 0 ; j<maxDim; j++){
				EXPECT_EQ(data->positions[k][i]->parameters[j] , k*i*j);
				EXPECT_EQ(pointer[k][i][j] , k*i*j);
			}
		}
	}
	data->deallocatePrimitivePointer(pointer);
}
//#######################################################################
/*Tests the updatePosition Function for the positionInfo class*/
TEST(positionInfoTest, updatePosition){

	int dim = 3;
	bayesship::positionInfo *pos1 = new bayesship::positionInfo(dim, true);
	bayesship::positionInfo *pos2 = new bayesship::positionInfo(dim, true);

	for(int i = 0 ; i<dim; i++){
		pos2->parameters[i] = i;
	}
	pos1->updatePosition(pos2);
	for(int i = 0 ; i<dim; i++){
		EXPECT_EQ(pos1->parameters[i],pos2->parameters[i]);	
	}

	pos2->modelID = 1;
	for(int i = 0 ; i<dim; i++){
		pos2->status[i] = 1;
	}

	pos1->updatePosition(pos2);

	for(int i = 0 ; i<dim; i++){
		EXPECT_EQ(pos1->parameters[i],pos2->parameters[i]);	
		EXPECT_EQ(pos1->status[i],pos2->status[i]);	
	}

	EXPECT_EQ(pos1->modelID,pos2->modelID);	
	delete pos1;
	delete pos2;
	
}
//#######################################################################

//#######################################################################
/*! Just a test to see if the function evaluates properly and frees memory properly*/
TEST(gaussianProposalTest,functionEvalCheck){
	double *betas = new double[1];
	for(int i = 0 ; i<1; i++){
		betas[i] = 1;
	}
		
	bayesship::samplerData *data = new bayesship::samplerData(3, 1,1, 2, 1, true,betas);
	/*Create old and new position objects*/
	//positionInfo *pos1 = new positionInfo(3,true);
	//positionInfo *pos2 = new positionInfo(3,true);

	/*Assign starting position*/
	for(int i = 0  ; i<3 ; i++){
		data->positions[0][0]->parameters[i]= 1;
		data->positions[0][0]->status[i]= 1;
	}
	/*Mimic RJ environment and set the status of the last variable to 0*/
	data->positions[0][0]->status[2]= 1;

	/*New sampler object with standard inputs (doesn't really matter)*/
	bayesship::bayesshipSampler *sampler=new bayesship::bayesshipSampler(likelihood,prior);
	/*Update relevant options (dimension, proposal fn data)*/
	sampler->maxDim = 3;
	/*Create new proposal fn object*/
	bayesship::gaussianProposalVariables *gpv = new bayesship::gaussianProposalVariables(sampler->ensembleN*sampler->ensembleSize,3);
	
	bayesship::proposalFn propArray[1] = {bayesship::gaussianProposal};
	float proposalFnProb[1] = {1};
	void *proposalFnVariables[1] = {(void *)gpv};
	
	bayesship::proposalFnData *propData = new bayesship::proposalFnData(1, propArray,proposalFnProb, proposalFnVariables);
	
	sampler->proposalFns = propData;
	sampler->RJ = true;
	double MHRatioCorrection=0;
	/*Run the proposal*/
	gaussianProposal(data, 0, 0, sampler, &MHRatioCorrection);

	/*Check if the proposal ``succeeded'', meaning the new position is different than the old position if the dimension in question is ``on''*/
	double sum1=0;
	double sum2=0;
	for(int i = 0  ; i<3 ; i++){
		if(data->positions[0][0]->status[i] == 1){
			sum1+=data->positions[0][0]->parameters[i];
			sum2+=data->positions[0][1]->parameters[i];
		}
	}
	EXPECT_NE(sum1,sum2);

	/*Clean up*/
	delete data;
	delete sampler;
	delete propData;
	delete gpv;
	delete [] betas;
}
//#######################################################################
