#include "bayesship/dataUtilities.h"
#include "bayesship/autocorrelationUtilities.h"
#include "bayesship/utilities.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>

#ifdef _HDF5
#include <H5Cpp.h>
#endif

namespace bayesship{

/*! \file 
 *
 * File containing source code for the utilities related to information storage, io, writing data to disk, etc.
 */

//##########################################################
//##########################################################

/*!\brief Utility to read in data
 *
 * Takes filename, and assigns to output[rows][cols]
 *
 * File must be comma separated doubles
 */
void readCSVFile(std::string filename, /**< input filename, relative to execution directory*/
		double **output, /**<[out] array to store output, dimensions rowsXcols*/
		int rows, /**< first dimension*/
		int cols /**<second dimension*/
		)
{
	std::fstream file_in;
	file_in.open(filename, std::ios::in);
	std::string line, word;
	int i=0, j=0;
	double *temp = (double *)malloc(sizeof(double)*rows*cols);
	
	if(file_in){
		while(std::getline(file_in, line)){
			std::stringstream lineStream(line);
			std::string item;
			while(std::getline(lineStream,item, ',')){
				temp[i]=std::stod(item);	
				i+=1;	
			}	
		}	
	}
	else{std::cout<<"ERROR -- File "<<filename<<" not found"<<std::endl;exit(1);}
	for(i =0; i<rows;i++){
		for(j=0; j<cols;j++)
			output[i][j] = temp[cols*i + j];
	}
	free(temp);
}

/*!\brief Utility to read in data
 *
 * Takes filename, and assigns to output[rows][cols]
 *
 * File must be comma separated doubles
 *
 * integer version
 */
void readCSVFile(std::string filename, /**< input filename, relative to execution directory*/
		int **output, /**<[out] array to store output, dimensions rowsXcols*/
		int rows, /**< first dimension*/
		int cols /**<second dimension*/
		)
{
	std::fstream file_in;
	file_in.open(filename, std::ios::in);
	std::string line, word;
	int i=0, j=0;
	double *temp = (double *)malloc(sizeof(double)*rows*cols);
	
	if(file_in){
		while(std::getline(file_in, line)){
			std::stringstream lineStream(line);
			std::string item;
			while(std::getline(lineStream,item, ',')){
				temp[i]=std::stoi(item);	
				i+=1;	
			}	
		}	
	}
	else{std::cout<<"ERROR -- File "<<filename<<" not found"<<std::endl;exit(1);}
	for(i =0; i<rows;i++){
		for(j=0; j<cols;j++)
			output[i][j] = temp[cols*i + j];
	}
	free(temp);
}

/*!\brief Utility to read in data (single dimension vector) 
 *
 * Takes filename, and assigns to output[i*rows + cols]
 *
 * Output vector must be long enough, no check is done for the length
 *
 * File must be comma separated doubles
 */
void readCSVFile(std::string filename, /**< input filename, relative to execution directory*/
	double *output /**<[out] output array, assumed to have the proper length of total items*/
	)
{
	std::fstream file_in;
	file_in.open(filename, std::ios::in);
	std::string line, word, temp;
	int i =0;
	if(file_in){
		while(std::getline(file_in, line)){
			std::stringstream lineStream(line);
			std::string item;
			while(std::getline(lineStream,item, ',')){
				output[i]=std::stod(item);	
				i+=1;
			}	
		}	
	}
	else{std::cout<<"ERROR -- File "<<filename<<" not found"<<std::endl;exit(1);}
}
/*!\brief Utility to read in data (single dimension vector) 
 *
 * Takes filename, and assigns to output[i*rows + cols]
 *
 * Output vector must be long enough, no check is done for the length
 *
 * File must be comma separated doubles
 *
 * Int version
 */
void readCSVFile(std::string filename, /**< input filename, relative to execution directory*/
	int *output /**<[out] output array, assumed to have the proper length of total items*/
	)
{
	std::fstream file_in;
	file_in.open(filename, std::ios::in);
	std::string line, word, temp;
	int i =0;
	if(file_in){
		while(std::getline(file_in, line)){
			std::stringstream lineStream(line);
			std::string item;
			while(std::getline(lineStream,item, ',')){
				output[i]=std::stoi(item);	
				i+=1;
			}	
		}	
	}
	else{std::cout<<"ERROR -- File "<<filename<<" not found"<<std::endl;exit(1);}
}
/*! \brief Utility to write 2D array to file
 *
 * Grid of data, comma separated
 *
 * Grid has rows rows and cols columns
 */
void writeCSVFile(std::string filename, /**<Filename of output file, relative to execution directory*/
		double **input, /**< Input 2D array pointer array[rows][cols]*/
		int rows, /**< First dimension of array*/
		int cols /**< second dimension of array*/
		)
{
	
	std::ofstream out_file;
	out_file.open(filename);
	out_file.precision(15);
	if(out_file){
		for(int i =0; i<rows; i++){
			for(int j=0; j<cols;j++){
				if(j==cols-1)
					out_file<<input[i][j]<<std::endl;
				else
					out_file<<input[i][j]<<" , ";
			}
		}
		out_file.close();
	}
	else{
		std::cout<<"ERROR -- Could not open file"<<std::endl;
	}
}
/*! \brief Utility to write 2D array to file
 *
 * Grid of data, comma separated
 *
 * Grid has rows rows and cols columns
 *  
 * integer version
 */
void writeCSVFile(std::string filename, /**<Filename of output file, relative to execution directory*/
		int **input, /**< Input 2D array pointer array[rows][cols]*/
		int rows, /**< First dimension of array*/
		int cols /**< second dimension of array*/
		)
{
	
	std::ofstream out_file;
	out_file.open(filename);
	out_file.precision(15);
	if(out_file){
		for(int i =0; i<rows; i++){
			for(int j=0; j<cols;j++){
				if(j==cols-1)
					out_file<<input[i][j]<<std::endl;
				else
					out_file<<input[i][j]<<" , ";
			}
		}
		out_file.close();
	}
	else{
		std::cout<<"ERROR -- Could not open file"<<std::endl;
	}
}
/*! \brief Utility to write 1D array to file
 *
 * Single column of data
 */
void writeCSVFile(std::string filename, /**<Filename of output file, relative to execution directory*/
		double *input, /**< input 1D array pointer array[length]*/
		int length /**< length of array*/
		)
{
	std::ofstream out_file;
	out_file.open(filename);
	out_file.precision(15);
	if(out_file){
		for(int j =0; j<length; j++)
			out_file<<input[j]<<std::endl;
		out_file.close();
	}
	else{
		std::cout<<"ERROR -- Could not open file"<<std::endl;
	}
}
/*! \brief Utility to write 1D array to file
 *
 * Single column of data
 * 
 * integer version
 */
void writeCSVFile(std::string filename, /**<Filename of output file, relative to execution directory*/
		int *input, /**< input 1D array pointer array[length]*/
		int length /**< length of array*/
		)
{
	std::ofstream out_file;
	out_file.open(filename);
	out_file.precision(15);
	if(out_file){
		for(int j =0; j<length; j++)
			out_file<<input[j]<<std::endl;
		out_file.close();
	}
	else{
		std::cout<<"ERROR -- Could not open file"<<std::endl;
	}
}
//##########################################################
//##########################################################

void positionInfo::updatePosition(positionInfo *newPosition)
{
	if(newPosition->dimension != dimension){
		errorMessage("The dimensions do not match for two position objects!",2);
	}
	for(int i = 0 ; i<dimension; i++){
		parameters[i] = newPosition->parameters[i];
		
	}
	if( newPosition->status){
		for(int i = 0 ; i<dimension; i++){
			status[i] = newPosition->status[i];
		}
		modelID = newPosition->modelID;
	}
}

int positionInfo::countActiveDimensions()
{
	int activeDims = dimension;
		
	if(RJ){
		activeDims = 0;
		for(int i = 0 ; i<dimension; i++){
			activeDims += status[i];	
		}
	}
	return activeDims;
}

struct helper_params
{
	gsl_interp_accel *a;	
	gsl_spline *s;	
};
double integration_helper(double beta, void *params)
{
	helper_params *p = (helper_params *) params;
	double likelihood =  gsl_spline_eval(p->s, beta, p->a);
	return likelihood;
}

void samplerData::calculateEvidence()
{
	double integratedLikelihoods[this->ensembleSize];
	double betasLocal[this->ensembleSize];
	for(int i = 0 ; i<ensembleSize; i++){

		//For the integration, we have to reverse the order (ie, 0 -> 1 not 1 -> 0, which is how they're naturally stored)
		int index = ensembleSize -1 -i;

		integratedLikelihoods[index] = 0 ;
		double norm = 0 ;
		int chainIndex;
		for(int j = 0 ; j<ensembleN; j++){
			chainIndex = j + i*ensembleN;
			for(int k = 0 ;k<currentStepID[chainIndex]; k++){
				integratedLikelihoods[index]+=likelihoodVals[chainIndex][k];
			}
			norm+=currentStepID[chainIndex];
		}		
		integratedLikelihoods[index]/=norm;
		betasLocal[index] = betas[chainIndex];
	}
	//for(int i = 0 ; i<ensembleSize; i++){
	//	std::cout<<betasLocal[i]<<" "<<integratedLikelihoods[i]<<std::endl;
	//}


	gsl_interp_accel *acc = gsl_interp_accel_alloc();	
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, ensembleSize);
	gsl_spline_init(spline, betasLocal, integratedLikelihoods, ensembleSize);

	helper_params params;
	params.a = acc;
	params.s = spline;

	gsl_function F;
	F.function = &integration_helper;
	F.params = &params;

	gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

	double error;
	int errorcode = gsl_integration_qags(&F, betasLocal[0],betasLocal[ensembleSize-1],0,1e-7,1000,w, &evidence, &evidenceError);
	std::cout<<errorcode<<std::endl;

	gsl_integration_workspace_free(w);
	gsl_spline_free(spline);
	gsl_interp_accel_free(acc);
	calculatedEvidence = true;
	

}

void samplerData::writeStatFile(std::string filename)
{
	std::ofstream outFile;
	outFile.open(filename);
	outFile<<"Proposal Attempts [chain #][proposal #, Total steps]"<<std::endl;
	for(int i = 0 ; i<chainN; i++){
		int total = 0;
		for(int j = 0 ; j<proposalFnN; j++){
			int attempts = (successN[i][j]+rejectN[i][j]);
			total+=attempts;
			outFile<< attempts<<", ";
		}
		outFile<< total<<", ";
		outFile<<std::endl;
	}
	outFile<<std::endl;

	outFile<<"Proposal Acceptance Fractions [chain #][proposal #]"<<std::endl;
	double acceptStepAve[proposalFnN];
	for(int i = 0 ; i<proposalFnN; i++){
		acceptStepAve[i] = 0;
	}
	for(int i = 0 ; i<chainN; i++){
		for(int j = 0 ; j<proposalFnN; j++){
			double rate = (double)successN[i][j] / (successN[i][j]+rejectN[i][j]);
			outFile<< rate<<", ";
			acceptStepAve[j]+=rate;
		}
		outFile<<std::endl;
	}
	outFile<<std::endl;

	outFile<<"Proposal Acceptance Fractions Averages"<<std::endl;
	for(int i = 0 ; i<proposalFnN; i++){
		outFile<<acceptStepAve[i]/chainN<<", ";
	}
	outFile<<std::endl;

	outFile<<std::endl;
	outFile<<"Proposal Times [chain #][proposal #]"<<std::endl;
	double timeAve[proposalFnN];
	for(int i = 0 ; i<proposalFnN; i++){
		timeAve[i] = 0;
	}
	for(int i = 0 ; i<chainN; i++){
		for(int j = 0 ; j<proposalFnN; j++){
			outFile<< proposalTimes[i][j]<<", ";
			timeAve[j]+=proposalTimes[i][j];
		}
		outFile<<std::endl;
	}
	outFile<<std::endl;

	outFile<<"Proposal Time Averages"<<std::endl;
	for(int i = 0 ; i<proposalFnN; i++){
		outFile<<timeAve[i]/chainN<<", ";
	}
	outFile<<std::endl;
	outFile<<std::endl;

	outFile<<"Prior/Likelihood Times [chain #][Prior, Likelihood ]"<<std::endl;
	double timeAveL=0;
	double timeAveP=0;
	for(int i = 0 ; i<chainN; i++){
		outFile<< priorTimes[i]<<", ";
		outFile<< likelihoodTimes[i];
		timeAveL+=likelihoodTimes[i];
		timeAveP+=priorTimes[i];
		outFile<<std::endl;
	}
	outFile<<std::endl;

	outFile<<"Proposal/Likelihood Time Averages"<<std::endl;
	outFile<<timeAveP/chainN<<", "<<timeAveL/chainN;
	outFile<<std::endl;
	outFile<<std::endl;

	outFile<<"Swap Acceptance Fractions [chain #][chain #]"<<std::endl;
	double swapAve[chainN];
	double swapAveUp[chainN];
	double swapAveDown[chainN];
	int aveCounts[chainN];
	int aveCountsUp[chainN];
	int aveCountsDown[chainN];
	for(int i = 0 ; i<chainN; i++){
		swapAve[i] = 0;
		aveCounts[i] = 0;
		aveCountsUp[i] = 0;
		aveCountsDown[i] = 0;
		swapAveUp[i] = 0;
		swapAveDown[i] = 0;
	}
	bool skipAve = false;

	outFile<<"Beta_i/beta_j, ";
	for(int j = 0 ; j<chainN; j++){
		outFile<<betas[j]<<", ";
	}
	outFile<<std::endl;
	for(int i = 0 ; i<chainN; i++){
		outFile<<betas[i]<<", ";
		for(int j = 0 ; j<chainN; j++){
			double rate = 0;
			if(swapAccepts[i][j] +swapRejects[i][j] >0){
				rate = (double)swapAccepts[i][j] / (swapAccepts[i][j]+swapRejects[i][j]);
			}
			else{ skipAve = true;}
			outFile<< rate<<", ";
			if(!skipAve){
				swapAve[j]+=rate;
				aveCounts[j]+=1;
				if(i<j){
					swapAveDown[j]+=rate;
					aveCountsDown[j]+=1;
				}
				else if(i>j){
					swapAveUp[j]+=rate;
					aveCountsUp[j]+=1;
				}
			}
			else{
				skipAve = false;
			}
		}
		outFile<<std::endl;
	}
	outFile<<std::endl;

	int swapAttemptsVec[chainN];
	for(int i = 0 ; i<chainN; i++){
		int swapAttempts = 0;
		for(int j = 0 ; j<chainN; j++){
			swapAttempts += swapAccepts[i][j]+swapRejects[i][j];
		}
		swapAttemptsVec[i] = swapAttempts;
	}
 
	outFile<<"Beta ||| Swap Average ||| Swap Average (Down) ||| Swap Average (Up) ||| Swap Attempts"<<std::endl;
	for(int i = 0 ; i<chainN; i++){
		outFile<<this->betas[i]<<" ||| "<<swapAve[i]/aveCounts[i]<<" ||| "<<swapAveDown[i]/aveCountsDown[i]<<" ||| "<<swapAveUp[i]/aveCountsUp[i]<<" ||| "<< swapAttemptsVec[i]<<std::endl;;
	}
	outFile<<std::endl;

	//outFile<<"Swap Averages"<<std::endl;
	//for(int i = 0 ; i<chainN; i++){
	//	outFile<<swapAve[i]/aveCounts[i]<<", ";
	//}
	//outFile<<std::endl;


	//outFile<<"Swap Averages Down"<<std::endl;
	//for(int i = 0 ; i<chainN; i++){
	//	outFile<<swapAveDown[i]/aveCountsDown[i]<<", ";
	//}
	//outFile<<std::endl;

	//outFile<<"Swap Averages Up"<<std::endl;
	//for(int i = 0 ; i<chainN; i++){
	//	outFile<<swapAveUp[i]/aveCountsUp[i]<<", ";
	//}
	//outFile<<std::endl;

	//outFile<<"Swap Attempts"<<std::endl;
	//for(int i = 0 ; i<chainN; i++){
	//	int swapAttempts = 0;
	//	for(int j = 0 ; j<chainN; j++){
	//		swapAttempts += swapAccepts[i][j]+swapRejects[i][j];
	//	}
	//	outFile<<swapAttempts<<", ";
	//}
	//outFile<<std::endl;

	outFile.close();
}

void samplerData::extendSize(int additionalIterations)
{
	int newSize = additionalIterations+iterations;
	positionInfo ***tempPositions = new positionInfo**[chainN];
	for(int i = 0 ; i<chainN; i++){
		tempPositions[i] = new positionInfo*[newSize];
		for(int j = 0 ;j<newSize; j++){
			tempPositions[i][j] = new positionInfo(maxDim,RJ);
		}
	}
	//###########################
	for(int i = 0 ; i<chainN; i++){
		for(int j = 0 ;j<=currentStepID[i]; j++){
			tempPositions[i][j]->updatePosition(positions[i][j]);	
		}
	}
	//###########################
	for(int j = 0 ; j<chainN; j++){
		for(int i = 0 ; i<iterations; i++){
			delete positions[j][i];
		}
		delete [] positions[j];
	}
	delete [] positions;

	positions = tempPositions;	

	//###########################

	double **tempLL = new double*[chainN];
	double **tempLP = new double*[chainN];
	for(int i = 0 ; i<chainN; i++){
		tempLL[i] = new double[newSize];
		tempLP[i] = new double[newSize];
		for(int j = 0 ; j<=currentStepID[i];j++){
			tempLL[i][j]= likelihoodVals[i][j];
			tempLP[i][j]= priorVals[i][j];
		}
	}
	for(int i = 0 ; i<chainN; i++){
		delete [] likelihoodVals[i];
	}
	delete [] likelihoodVals;
	for(int i = 0 ; i<chainN; i++){
		delete [] priorVals[i];
	}
	delete [] priorVals;
	likelihoodVals = tempLL;
	priorVals = tempLP;

	iterations= newSize;
	return;
}

int samplerData::countIndependentSamples()
{
	int samples = 0 ;
	for(int i= 0 ; i<ensembleN; i++){
		if(maxACs[i] != 0){
			samples+= (currentStepID[i]+1)/(maxACs[i]);
		}
	}
	return samples/ensembleN;
}

void samplerData::updateACs(int threads){
	double ***acDataPointer = convertToPrimitivePointer();
	int ***ac = allocate_3D_array_int(ensembleN, maxDim,1);
	auto_corr_from_data_batch(acDataPointer,currentStepID[0],maxDim,ensembleN,ac, 1,.01,threads, true);
	for(int i = 0 ; i<ensembleN; i++){
		maxACs[i] = ac[i][0][0];
		for(int j = 0 ; j<maxDim; j++){
			acs[i][j] = ac[i][j][0];
			if(ac[i][j][0] > maxACs[i]){
				maxACs[i] = ac[i][j][0];
			}
		}

	}
	deallocate_3D_array(ac,ensembleN,maxDim);
	deallocatePrimitivePointer(acDataPointer);
	return;
}

void samplerData::updateBetas(double *betas)
{
	for(int i = 0 ; i<chainN; i++){
		this->betas[i] = betas[i];
	}
	return;
}

samplerData::samplerData(int maxDim, int ensembleN,int ensembleSize, int iterations , int proposalFnN, bool RJ, double *betas)
{
	this->maxDim = maxDim;
	this->chainN = ensembleN*ensembleSize;
	this->ensembleN = ensembleN;
	this->ensembleSize = ensembleSize;
	this->iterations = iterations;
	this->RJ = RJ;
	this->proposalFnN = proposalFnN;
	if(!this->proposalTimes){	
		this->proposalTimes = new double*[chainN];
		for(int j =0 ; j<chainN; j++){
			this->proposalTimes[j] = new double[proposalFnN];
			for(int i =0 ; i<proposalFnN; i++){
				this->proposalTimes[j][i] = 0;
			}
		}
		
	}
	if(!this->likelihoodTimes){	
		this->likelihoodTimes = new double[chainN];
		for(int j =0 ; j<chainN; j++){
			this->likelihoodTimes[j] = 0;
		}
		
	}
	if(!this->likelihoodEvals){	
		this->likelihoodEvals = new int[chainN];
		for(int j =0 ; j<chainN; j++){
			this->likelihoodEvals[j] = 0;
		}
		
	}
	if(!this->priorTimes){	
		this->priorTimes = new double[chainN];
		for(int j =0 ; j<chainN; j++){
			this->priorTimes[j] = 0;
		}
		
	}
	if(!currentStepID){
		currentStepID = new int[chainN];
		for(int i =0 ; i<chainN; i++){
			currentStepID[i] = 0;
		}
	}
	
	if(!positions){
		positions = new positionInfo**[chainN];
		for(int i = 0 ; i<chainN; i++){
			positions[i] = new positionInfo*[iterations];
			for(int j = 0 ;j<iterations; j++){
				positions[i][j] = new positionInfo(maxDim,RJ);
			}
		}
	}
	
	if(!likelihoodVals){
		likelihoodVals = new double*[chainN];		
		for(int i= 0 ; i<chainN; i++){
			likelihoodVals[i] = new double[iterations];
		}
	}
	if(!priorVals){
		priorVals = new double*[chainN];		
		for(int i= 0 ; i<chainN; i++){
			priorVals[i] = new double[iterations];
		}
	}

	if(!rejectN){
		rejectN = new int*[chainN];
		for(int i = 0 ; i<chainN; i++){
			rejectN[i] = new int[proposalFnN];
			for(int j = 0 ; j<proposalFnN; j++){
				rejectN[i][j] = 0;
			}
		}
	}
	if(!successN){
		successN = new int*[chainN];
		
		for(int i = 0 ; i<chainN; i++){
			successN[i] = new int[proposalFnN];
			for(int j = 0 ; j<proposalFnN; j++){
				successN[i][j] = 0;
			}
		}
	}

	if(!swapAccepts){
		swapAccepts = new int*[chainN];
		for(int i = 0 ; i<chainN ; i++){
			swapAccepts[i] = new int[chainN];
			for(int j = 0 ; j<chainN; j++){
				swapAccepts[i][j] = 0;
			}
		}
	}
	if(!swapRejects){
		swapRejects = new int*[chainN];
		for(int i = 0 ; i<chainN ; i++){
			swapRejects[i] = new int[chainN];
			for(int j = 0 ; j<chainN; j++){
				swapRejects[i][j] = 0;
			}
		}
	}
	if(!acs){
		acs = new int*[ensembleN];
		for(int i = 0 ; i<ensembleN; i++){
			acs[i] = new int[maxDim];
			for(int j = 0 ; j<maxDim; j++){
				acs[i][j] = 0;
			}
		}
	}
	if(!maxACs){
		maxACs = new int[ensembleN];
		for(int i = 0 ; i<ensembleN; i++){
			maxACs[i] = 0 ;
		}
	}
	trimLengths = new int[chainN];
	for(int i = 0 ;i<chainN; i++){
		trimLengths[i]=0;
	}

	if(!this->betas){
		this->betas = new double[chainN];
		for(int i = 0 ; i<chainN; i++){
			this->betas[i] = betas[i];	
		}
	}

}
samplerData::~samplerData()
{
	if(proposalTimes){
		for(int i = 0 ; i<chainN; i++){
			delete [] proposalTimes[i];
		}
		delete [] proposalTimes;
		proposalTimes = nullptr;
	}
	if(likelihoodTimes){
		delete [] likelihoodTimes;
		likelihoodTimes = nullptr;
	}
	if(likelihoodEvals){
		delete [] likelihoodEvals;
		likelihoodEvals = nullptr;
	}
	if(priorTimes){
		delete [] priorTimes;
		priorTimes = nullptr;
	}
	if(currentStepID){
		delete [] currentStepID;
		currentStepID=nullptr;
	}
	if(positions){
		for(int j = 0 ; j<chainN; j++){
			for(int i = 0 ; i<iterations; i++){
				delete positions[j][i];
			}
			delete [] positions[j];
		}
		delete [] positions;
		positions = nullptr;
	}
	if(likelihoodVals){
		for(int i = 0 ; i<chainN; i++){
			delete [] likelihoodVals[i];
		}
		delete [] likelihoodVals;
		likelihoodVals = nullptr;
	}
	if(priorVals){
		for(int i = 0 ; i<chainN; i++){
			delete [] priorVals[i];
		}
		delete [] priorVals;
		priorVals = nullptr;
	}

	if(rejectN){
		for(int i = 0 ; i<chainN; i++){
			delete [] rejectN[i];
		}
		delete [] rejectN;
		rejectN = nullptr;
	}
	if(successN){
		for(int i = 0 ; i<chainN; i++){
			delete [] successN[i];
		}
		delete [] successN;
		successN = nullptr;
	}

	if(swapAccepts){
		for(int i = 0 ; i< chainN ; i++){
			delete [] swapAccepts[i];
		}
		delete [] swapAccepts;
		swapAccepts = nullptr;
	}
	if(swapRejects){
		for(int i = 0 ; i< chainN ; i++){
			delete [] swapRejects[i];
		}
		delete [] swapRejects;
		swapRejects=nullptr;
	}
	if(acs){
		for(int i = 0 ; i<ensembleN; i++){
			delete [] acs[i];
		}
		delete [] acs;
		acs = nullptr;
	}
	if(maxACs){
		delete [] maxACs;
		maxACs = nullptr;
	}
	if(dump_files.size() != 0){
		for(size_t i = 0 ; i<dump_files.size(); i++){
			if(dump_files[i]->fileTrimLengths){
				delete [] dump_files[i]->fileTrimLengths;
				dump_files[i]->fileTrimLengths = NULL;
			}
			delete dump_files[i];
		}
	}
	if(trimLengths){
		delete [] trimLengths;
		trimLengths = nullptr;
	}
	if(betas){
		delete [] betas;
		betas = nullptr;
	}

}
void samplerData::set_trim(int trim){
	for(int i = 0 ; i<chainN; i++){
		trimLengths[i]=trim;
	}
}

#ifdef _HDF5
int samplerData::create_data_dump(bool cold_only, bool trim,std::string filename)
{
	int file_id = 0;
	bool found = false;
	if(dump_files.size() != 0){
		for(size_t i = 0 ; i<dump_file_names.size(); i++){
			if( filename == dump_file_names[i]){
				found = true;
				file_id = i;
			}
		}	
	}
	if(!found ){
		file_id = dump_files.size();	
		dump_file_struct *new_dump_file = new dump_file_struct;
		dump_files.push_back(new_dump_file);
		dump_files[file_id]->fileTrimLengths = new int[chainN];
		dump_file_names.push_back(filename);
	}
	dump_files[file_id]->coldOnly = cold_only;
	
	if(trim){
		dump_files[file_id]->trimmed = true;
		for(int i= 0 ; i<chainN; i++){
			dump_files[file_id]->fileTrimLengths[i]=trimLengths[i];
		}
	}
	else{
		dump_files[file_id]->trimmed = false;
	}
	try{
		std::string FILE_NAME(filename);
		int chains;
		int *ids=NULL;
		if(cold_only){
		
			ids =  new int[ensembleN];
			for(int i = 0  ; i<ensembleN; i++){
				ids[i]=i;
			}
			chains = ensembleN;
		}
		else{
			chains = chainN;
			ids = new int[chainN];
			for(int i = 0  ; i<chainN; i++){
				ids[i]=i;
			}
		}
		H5::H5File file(FILE_NAME,H5F_ACC_TRUNC);
		H5::Group output_group(file.createGroup("/MCMC_OUTPUT"));
		H5::Group output_LL_LP_group(file.createGroup("/MCMC_OUTPUT/LOGL_LOGP"));
		H5::Group status_group;
		H5::Group model_status_group;
		if(RJ){
			status_group = H5::Group(file.createGroup("/MCMC_OUTPUT/STATUS"));
			model_status_group = H5::Group(file.createGroup("/MCMC_OUTPUT/MODEL_STATUS"));
		}
		H5::Group meta_group(file.createGroup("/MCMC_METADATA"));
		double *temp_buffer=NULL;
		double *temp_ll_lp_buffer=NULL;
		int *temp_status_buffer=NULL;
		int *temp_model_status_buffer=NULL;
		H5::DataSpace *dataspace=NULL ;
		H5::DataSpace *dataspace_ll_lp=NULL ;
		H5::DataSpace *dataspace_status=NULL ;
		H5::DataSpace *dataspace_model_status=NULL ;
		H5::DataSet *dataset=NULL;
		H5::DataSet *dataset_ll_lp=NULL;
		H5::DataSet *dataset_status=NULL;
		H5::DataSet *dataset_model_status=NULL;
		H5::DSetCreatPropList *plist=NULL;
		H5::DSetCreatPropList *plist_ll_lp=NULL;
		H5::DSetCreatPropList *plist_status=NULL;
		H5::DSetCreatPropList *plist_model_status=NULL;
		hsize_t chunk_dims[2] = {(hsize_t)chunk_steps,(hsize_t)maxDim};	
		hsize_t chunk_dims_ll_lp[2] = {(hsize_t)chunk_steps,2};	
		hsize_t chunk_dims_status[2] = {(hsize_t)chunk_steps,(hsize_t)maxDim};	
		//hsize_t chunk_dims_status[2] = {(hsize_t)chunk_steps,2};	
		hsize_t chunk_dims_model_status[2] = {(hsize_t)chunk_steps,2};	
		hsize_t max_dims[2] = {H5S_UNLIMITED,H5S_UNLIMITED};
		for(int i = 0 ; i<chains; i++){
			//int RANK=2;
			hsize_t RANK=2;
			hsize_t dims[RANK];
			hsize_t dims_ll_lp[RANK];
			hsize_t dims_status[RANK];
			//hsize_t dims_model_status[RANK];
			if(trim){
				dims[0]= currentStepID[ids[i]]-trimLengths[ids[i]];
				dims_ll_lp[0]= currentStepID[ids[i]]-trimLengths[ids[i]];
				dims_status[0]= currentStepID[ids[i]]-trimLengths[ids[i]];
				//dims_model_status[0]= currentStepID[ids[i]]-trimLengths[ids[i]];
			}
			else{
				dims[0]= currentStepID[ids[i]];
				dims_ll_lp[0]= currentStepID[ids[i]];
				dims_status[0]= currentStepID[ids[i]];
				//dims_model_status[0]= currentStepID[ids[i]];
			}
			dims[1]= maxDim;
			dims_ll_lp[1]= 2;
			dims_status[1]= maxDim;

			if(chunk_steps>dims[0]){chunk_dims_ll_lp[0]=dims[0];chunk_dims[0] = dims[0];}
			else{chunk_dims_ll_lp[0]=chunk_steps;chunk_dims[0] = chunk_steps;}

			dataspace = new H5::DataSpace(RANK,dims,max_dims);
			dataspace_ll_lp = new H5::DataSpace(RANK,dims_ll_lp,max_dims);
			if(RJ){
				dataspace_status = new H5::DataSpace(RANK,dims_status,max_dims);
			}
	
			plist = new H5::DSetCreatPropList;
			plist->setChunk(2,chunk_dims);
			plist->setDeflate(6);

			plist_ll_lp = new H5::DSetCreatPropList;
			plist_ll_lp->setChunk(2,chunk_dims_ll_lp);
			plist_ll_lp->setDeflate(6);

			if(RJ){
				plist_status = new H5::DSetCreatPropList;
				plist_status->setChunk(2,chunk_dims_status);
				plist_status->setDeflate(6);

			}

			dataset = new H5::DataSet(
				output_group.createDataSet("CHAIN "+std::to_string(ids[i]),
					H5::PredType::NATIVE_DOUBLE,*dataspace,*plist)
				);
			dataset_ll_lp = new H5::DataSet(
				output_LL_LP_group.createDataSet("CHAIN "+std::to_string(ids[i]),
					H5::PredType::NATIVE_DOUBLE,*dataspace_ll_lp,*plist_ll_lp)
				);
			if(RJ){
				dataset_status = new H5::DataSet(
					status_group.createDataSet("CHAIN "+std::to_string(ids[i]),
						H5::PredType::NATIVE_INT,*dataspace_status,*plist_status)
					);

			}

			temp_buffer = new double[ int(dims[0]*dims[1]) ];
			temp_ll_lp_buffer = new double[ int(dims_ll_lp[0]*dims_ll_lp[1]) ];
			int beginning_id=0;
			if(trim){ beginning_id =trimLengths[ids[i]];}
			for(int j = 0 ; j<currentStepID[ids[i]] - beginning_id; j++){
				for(int k = 0 ; k<maxDim; k++){
					temp_buffer[j*maxDim +k] = positions[ids[i]][j+beginning_id]->parameters[k];	
				}
				temp_ll_lp_buffer[j*2]=likelihoodVals[ids[i]][j+beginning_id];
				temp_ll_lp_buffer[j*2+1]=priorVals[ids[i]][j+beginning_id];
			}
			dataset->write(temp_buffer, H5::PredType::NATIVE_DOUBLE);
			dataset_ll_lp->write(temp_ll_lp_buffer, H5::PredType::NATIVE_DOUBLE);
			if(RJ){
				temp_status_buffer = new int[ int(dims_status[0]*dims_status[1]) ];
				int beginning_id=0;
				if(trim){ beginning_id =trimLengths[ids[i]];}
				for(int j = 0 ; j<currentStepID[ids[i]] - beginning_id; j++){
					for(int k = 0 ; k<maxDim; k++){
						temp_status_buffer[j*maxDim +k] = positions[ids[i]][j+beginning_id]->status[k];	
					}
				}
				dataset_status->write(temp_status_buffer, H5::PredType::NATIVE_INT);

			}
			//Cleanup
			delete dataset;
			delete dataset_ll_lp;
			delete dataspace;
			delete dataspace_ll_lp;
			delete plist;
			delete plist_ll_lp;
			delete [] temp_buffer;
			delete [] temp_ll_lp_buffer;
			temp_buffer = NULL;
			temp_ll_lp_buffer = NULL;
			if(RJ){
				delete dataset_status;
				delete dataspace_status;
				delete plist_status;
				delete [] temp_status_buffer;
				temp_status_buffer = NULL;

			}
			
			//TODO -- this section can't be right.. 
			if(RJ){
				int RANK=2;
				hsize_t dims_model_status[RANK];
				if(trim){
					dims_model_status[0]= currentStepID[ids[i]]-trimLengths[ids[i]];
				}
				else{
					dims_model_status[0]= currentStepID[ids[i]];
				}
				dims_model_status[1]= 1;

				dataspace_model_status = new H5::DataSpace(RANK,dims_model_status,max_dims);
	

				plist_model_status = new H5::DSetCreatPropList;
				plist_model_status->setChunk(2,chunk_dims_model_status);
				plist_model_status->setDeflate(6);

				dataset_model_status = new H5::DataSet(
					model_status_group.createDataSet("CHAIN "+std::to_string(ids[i]),
					H5::PredType::NATIVE_INT,*dataspace_model_status,*plist_model_status)
					);

				temp_model_status_buffer = new int[ int(dims_model_status[0]*dims_model_status[1]) ];
				int beginning_id=0;
				if(trim){ beginning_id =trimLengths[ids[i]];}
				for(int j = 0 ; j<currentStepID[ids[i]] - beginning_id; j++){
					temp_model_status_buffer[j ] = positions[ids[i]][j+beginning_id]->modelID;	
				}
				dataset_model_status->write(temp_model_status_buffer, H5::PredType::NATIVE_INT);

				//Cleanup
				delete dataset_model_status;
				delete dataspace_model_status;
				delete plist_model_status;
				delete [] temp_model_status_buffer;
				temp_model_status_buffer = NULL;
			}
			
		}
		hsize_t dimsT[1];
		dimsT[0]= chainN;
		//#################################################
		dataspace = new H5::DataSpace(1,dimsT);
		dataset = new H5::DataSet(
			meta_group.createDataSet("CHAIN BETAS",
				H5::PredType::NATIVE_DOUBLE,*dataspace)
			);
		dataset->write(betas, H5::PredType::NATIVE_DOUBLE);	
		delete dataset;
		delete dataspace;
		//#################################################
		//if(integrated_likelihoods){
		//	hsize_t dimsIL[1];
		//	dimsIL[0]= ensemble_size;
		//	dataspace = new H5::DataSpace(1,dimsIL);
		//	dataset = new H5::DataSet(
		//		meta_group.createDataSet("INTEGRATED LIKELIHOODS",
		//			H5::PredType::NATIVE_DOUBLE,*dataspace)
		//		);
		//	dataset->write(integrated_likelihoods, H5::PredType::NATIVE_DOUBLE);	
		//	delete dataset;
		//	delete dataspace;
		//}
		//#################################################
		//if(integrated_likelihoods_terms){
		//	hsize_t dimsILT[1];
		//	dimsILT[0]= ensemble_size;
		//	dataspace = new H5::DataSpace(1,dimsILT);
		//	dataset = new H5::DataSet(
		//		meta_group.createDataSet("INTEGRATED LIKELIHOODS TERM NUMBER",
		//			H5::PredType::NATIVE_INT,*dataspace)
		//		);
		//	dataset->write(integrated_likelihoods_terms, H5::PredType::NATIVE_INT);	
		//	delete dataset;
		//	delete dataspace;
		//}
		//#################################################
		if(calculatedEvidence){
			hsize_t dimsE[1];
			dimsE[0]= 1;
			dataspace = new H5::DataSpace(1,dimsE);
			dataset = new H5::DataSet(
				meta_group.createDataSet("EVIDENCE",
					H5::PredType::NATIVE_DOUBLE,*dataspace)
				);
			dataset->write(&evidence, H5::PredType::NATIVE_DOUBLE);	
			delete dataset;
			delete dataspace;
		}
		//#################################################
		dataspace = new H5::DataSpace(1,dimsT);
		dataset = new H5::DataSet(
			meta_group.createDataSet("SUGGESTED TRIM LENGTHS",
				H5::PredType::NATIVE_INT,*dataspace)
			);
		dataset->write(trimLengths, H5::PredType::NATIVE_INT);	
		delete dataset;
		delete dataspace;
		//#################################################
		hsize_t dimsAC[2];
		dimsAC[0]= ensembleN;
		dimsAC[1]= maxDim;
		dataspace = new H5::DataSpace(2,dimsAC);

		int *int_temp_buffer=NULL;
		if(acs){
			int_temp_buffer = new int[ensembleN*maxDim];
			for(int i  = 0 ; i<ensembleN; i++){
				for(int j = 0 ; j<maxDim ; j++){
					int_temp_buffer[i*maxDim +j ] = acs[i][j];
				}
			}

			dataset = new H5::DataSet(
				meta_group.createDataSet("AC VALUES",
					H5::PredType::NATIVE_INT,*dataspace)
				);
			dataset->write(int_temp_buffer, H5::PredType::NATIVE_INT);	

			delete [] int_temp_buffer;
			int_temp_buffer =NULL;
			delete dataset;
			delete dataspace;
		}
		//#################################################
	
		//Cleanup
		output_LL_LP_group.close();
		output_group.close();
		if(RJ){
			status_group.close();
			model_status_group.close();
		}
		meta_group.close();
		delete [] ids;
		ids = NULL;

	}	
	catch( H5::FileIException error )
	{
		error.printErrorStack();
		return -1;
	}
	// catch failure caused by the DataSet operations
	catch( H5::DataSetIException error )
	{
		error.printErrorStack();
	   	return -1;
	}
	// catch failure caused by the DataSpace operations
	catch( H5::DataSpaceIException error )
	{
		error.printErrorStack();
	   	return -1;
	}
	// catch failure caused by the DataSpace operations
	catch( H5::DataTypeIException error )
	{
		error.printErrorStack();
	   	return -1;
	}
	return 0;

}

int samplerData::append_to_data_dump( std::string filename)
{
	int file_id = 0;
	bool found=false;
	for(size_t i = 0 ; i<dump_file_names.size(); i++){
		if( filename == dump_file_names[i]){
			found = true;
			file_id = i;
		}
	}	
	if(!found){
		std::cout<<"ERROR -- File doesn't exist"<<std::endl;
	}
	try{
		std::string FILE_NAME(filename);
		int chains;
		int *ids=NULL;
		if(dump_files[file_id]->coldOnly){
			ids =  new int[ensembleN];
			for(int i = 0  ; i<ensembleN; i++){
				ids[i]=i;
			}
			chains = ensembleN;
		}
		else{
			chains = chainN;
			ids = new int[chainN];
			for(int i = 0  ; i<chainN; i++){
				ids[i]=i;
			}
		}
		H5::H5File file(FILE_NAME,H5F_ACC_RDWR);
		H5::Group output_group(file.openGroup("/MCMC_OUTPUT"));
		H5::Group output_LL_LP_group(file.openGroup("/MCMC_OUTPUT/LOGL_LOGP"));
		H5::Group status_group;
		H5::Group model_status_group;
		if(RJ){
			status_group = H5::Group(file.openGroup("/MCMC_OUTPUT/STATUS"));
			model_status_group = H5::Group(file.openGroup("/MCMC_OUTPUT/MODEL_STATUS"));
		}
		H5::Group meta_group(file.openGroup("/MCMC_METADATA"));
		double *temp_buffer=NULL;
		double *temp_buffer_ll_lp=NULL;
		int *temp_buffer_status=NULL;
		int *temp_buffer_model_status=NULL;
		H5::DataSpace *dataspace=NULL ;
		H5::DataSpace *dataspace_ll_lp=NULL ;
		H5::DataSpace *dataspace_status=NULL ;
		H5::DataSpace *dataspace_model_status=NULL ;
		H5::DataSpace *dataspace_ext=NULL ;
		H5::DataSpace *dataspace_ext_ll_lp=NULL ;
		H5::DataSpace *dataspace_ext_status=NULL ;
		H5::DataSpace *dataspace_ext_model_status=NULL ;
		H5::DataSet *dataset=NULL;
		H5::DataSet *dataset_ll_lp=NULL;
		H5::DataSet *dataset_status=NULL;
		H5::DataSet *dataset_model_status=NULL;
		H5::DSetCreatPropList *plist=NULL;
		H5::DSetCreatPropList *plist_ll_lp=NULL;
		H5::DSetCreatPropList *plist_status=NULL;
		H5::DSetCreatPropList *plist_model_status=NULL;
		hsize_t chunk_dims[2] = {(hsize_t)chunk_steps,(hsize_t)maxDim};	
		hsize_t chunk_dims_ll_lp[2] = {(hsize_t)chunk_steps,2};	
		hsize_t chunk_dims_status[2] = {(hsize_t)chunk_steps,(hsize_t)maxDim};	
		//hsize_t chunk_dims_status[2] = {(hsize_t)chunk_steps,2};	
		hsize_t chunk_dims_model_status[2] = {(hsize_t)chunk_steps,1};	
		hsize_t max_dims[2] = {H5S_UNLIMITED,H5S_UNLIMITED};
		for(int i = 0 ; i<chains; i++){
			dataset = new H5::DataSet(output_group.openDataSet("CHAIN "+std::to_string(ids[i])));
			dataset_ll_lp = new H5::DataSet(output_LL_LP_group.openDataSet("CHAIN "+std::to_string(ids[i])));
			
			dataspace = new H5::DataSpace(dataset->getSpace());
			dataspace_ll_lp = new H5::DataSpace(dataset_ll_lp->getSpace());

			plist = new H5::DSetCreatPropList(dataset->getCreatePlist());
			plist_ll_lp = new H5::DSetCreatPropList(dataset_ll_lp->getCreatePlist());
			int RANK = dataspace->getSimpleExtentNdims();
			int RANK_ll_lp = dataspace_ll_lp->getSimpleExtentNdims();

			hsize_t base_dims[RANK];
			hsize_t base_dims_ll_lp[RANK_ll_lp];
			herr_t statusH5 = dataspace->getSimpleExtentDims(base_dims);
			statusH5 = dataspace_ll_lp->getSimpleExtentDims(base_dims_ll_lp);

			int RANK_chunked;
			int RANK_chunked_ll_lp;
			hsize_t base_chunk_dims[RANK];
			hsize_t base_chunk_dims_ll_lp[RANK_ll_lp];
			if(H5D_CHUNKED == plist->getLayout()){
				RANK_chunked= plist->getChunk(RANK,base_chunk_dims);
			}
			if(H5D_CHUNKED == plist_ll_lp->getLayout()){
				RANK_chunked_ll_lp= plist_ll_lp->getChunk(RANK_ll_lp,base_chunk_dims_ll_lp);
			}
			
			hsize_t new_size[RANK];
			hsize_t new_size_ll_lp[RANK];
			if(dump_files[file_id]->trimmed){
				new_size[0]= currentStepID[ids[i]]-dump_files[file_id]->fileTrimLengths[ids[i]];
				new_size_ll_lp[0]= currentStepID[ids[i]]-dump_files[file_id]->fileTrimLengths[ids[i]];
			}
			else{
				new_size[0]= currentStepID[ids[i]];
				new_size_ll_lp[0]= currentStepID[ids[i]];
			}
			new_size[1]= maxDim;
			new_size_ll_lp[1]= 2;
			dataset->extend(new_size);
			dataset_ll_lp->extend(new_size_ll_lp);

			delete dataspace;
			delete dataspace_ll_lp;
			dataspace = new H5::DataSpace(dataset->getSpace());
			dataspace_ll_lp = new H5::DataSpace(dataset_ll_lp->getSpace());
			
			hsize_t dimext[RANK];	
			hsize_t dimext_ll_lp[RANK];	
			dimext[0]=new_size[0]-base_dims[0];
			dimext[1]=maxDim;
			dimext_ll_lp[0]=new_size_ll_lp[0]-base_dims_ll_lp[0];
			dimext_ll_lp[1]=2;
			
			hsize_t offset[RANK];
			hsize_t offset_ll_lp[RANK];
			offset[0]=base_dims[0];	
			offset[1]=0;	
			offset_ll_lp[0]=base_dims_ll_lp[0];	
			offset_ll_lp[1]=0;	

			dataspace->selectHyperslab(H5S_SELECT_SET,dimext,offset);
			dataspace_ll_lp->selectHyperslab(H5S_SELECT_SET,dimext_ll_lp,offset_ll_lp);

			dataspace_ext = new H5::DataSpace(RANK, dimext,NULL);
			dataspace_ext_ll_lp = new H5::DataSpace(RANK_ll_lp, dimext_ll_lp,NULL);

			temp_buffer = new double[ dimext[0]*dimext[1] ];
			temp_buffer_ll_lp = new double[ dimext_ll_lp[0]*dimext_ll_lp[1] ];
			int beginning_id = 0 ; 
			if(dump_files[file_id]->trimmed){beginning_id = dump_files[file_id]->fileTrimLengths[ids[i]];}
			for(int j = base_dims[0] ; j<currentStepID[ids[i]]-beginning_id; j++){
				for(int k = 0 ; k<maxDim; k++){
					temp_buffer[(j-base_dims[0])*maxDim +k] = positions[ids[i]][j+beginning_id]->parameters[k];	
				}
				temp_buffer_ll_lp[(j-base_dims_ll_lp[0])*2 ] = likelihoodVals[ids[i]][j+beginning_id];	
				temp_buffer_ll_lp[(j-base_dims_ll_lp[0])*2+1 ] = priorVals[ids[i]][j+beginning_id];	
			}
			
			dataset->write(temp_buffer,H5::PredType::NATIVE_DOUBLE,*dataspace_ext, *dataspace);
			dataset_ll_lp->write(temp_buffer_ll_lp,H5::PredType::NATIVE_DOUBLE,*dataspace_ext_ll_lp, *dataspace_ll_lp);

			//Cleanup
			delete dataset;
			delete dataset_ll_lp;
			delete dataspace;
			delete dataspace_ll_lp;
			delete dataspace_ext;
			delete dataspace_ext_ll_lp;
			delete plist;
			delete plist_ll_lp;
			delete [] temp_buffer;
			delete [] temp_buffer_ll_lp;
			temp_buffer = NULL;
			temp_buffer_ll_lp = NULL;

			if(RJ){
				dataset_status = new H5::DataSet(status_group.openDataSet("CHAIN "+std::to_string(ids[i])));

				dataspace_status = new H5::DataSpace(dataset_status->getSpace());
				plist_status= new H5::DSetCreatPropList(dataset_status->getCreatePlist());
				int RANK_status = dataspace_status->getSimpleExtentNdims();
				hsize_t base_dims_status[RANK_status];
				herr_t statusH5 = dataspace_status->getSimpleExtentDims(base_dims_status);
				hsize_t base_chunk_dims_status[RANK_status];

				int RANK_chunked_status;
				if(H5D_CHUNKED == plist_status->getLayout()){
                                	RANK_chunked_status= plist_status->getChunk(RANK_status,base_chunk_dims_status);
                                }
				
				hsize_t new_size_status[RANK];
				if(dump_files[file_id]->trimmed){
					new_size_status[0]= currentStepID[ids[i]]-dump_files[file_id]->fileTrimLengths[ids[i]];
				}
				else{
					new_size_status[0]= currentStepID[ids[i]];
				}
				new_size_status[1]= maxDim;
				dataset_status->extend(new_size_status);

				delete dataspace_status;
				dataspace_status = new H5::DataSpace(dataset_status->getSpace());
				
				hsize_t dimext_status[RANK];	
				dimext_status[0]=new_size_status[0]-base_dims_status[0];
				dimext_status[1]=maxDim;
				
				hsize_t offset_status[RANK];
				offset_status[0]=base_dims_status[0];	
				offset_status[1]=0;	

				dataspace_status->selectHyperslab(H5S_SELECT_SET,dimext_status,offset_status);

				dataspace_ext_status = new H5::DataSpace(RANK_status, dimext_status,NULL);

				temp_buffer_status = new int[ dimext_status[0]*dimext_status[1] ];
				int beginning_id = 0 ; 
				if(dump_files[file_id]->trimmed){beginning_id = dump_files[file_id]->fileTrimLengths[ids[i]];}
				for(int j = base_dims_status[0] ; j<currentStepID[ids[i]]-beginning_id; j++){
					for(int k = 0 ; k<maxDim; k++){
						temp_buffer_status[(j-base_dims_status[0])*maxDim +k] = positions[ids[i]][j+beginning_id]->status[k];	
					}
				}
				
				dataset_status->write(temp_buffer_status,H5::PredType::NATIVE_INT,*dataspace_ext_status, *dataspace_status);
				//Cleanup
				delete dataset_status;
				delete dataspace_status;
				delete dataspace_ext_status;
				delete plist_status;
				delete [] temp_buffer_status;
				temp_buffer_status = NULL;

				{
				//TODO -- this section can't be right.. 
				dataset_model_status = new H5::DataSet(model_status_group.openDataSet("CHAIN "+std::to_string(ids[i])));

				dataspace_model_status = new H5::DataSpace(dataset_model_status->getSpace());
				plist_model_status= new H5::DSetCreatPropList(dataset_model_status->getCreatePlist());
				int RANK_model_status = dataspace_model_status->getSimpleExtentNdims();
				hsize_t base_dims_model_status[RANK_model_status];
				herr_t modelStatusH5 = dataspace_model_status->getSimpleExtentDims(base_dims_model_status);
				int RANK_chunked_model_status;
				hsize_t base_chunk_dims_model_status[RANK_model_status];
				if(H5D_CHUNKED == plist_model_status->getLayout()){
					RANK_chunked_model_status= plist_model_status->getChunk(RANK_model_status,base_chunk_dims_model_status);
				}
				
				hsize_t new_size_model_status[RANK];
				if(dump_files[file_id]->trimmed){
					new_size_model_status[0]= currentStepID[ids[i]]-dump_files[file_id]->fileTrimLengths[ids[i]];
				}
				else{
					new_size_model_status[0]= currentStepID[ids[i]];
				}
				new_size_model_status[1]= 1;
				dataset_model_status->extend(new_size_model_status);

				delete dataspace_model_status;
				dataspace_model_status = new H5::DataSpace(dataset_model_status->getSpace());
				
				hsize_t dimext_model_status[RANK];	
				dimext_model_status[0]=new_size_model_status[0]-base_dims_model_status[0];
				dimext_model_status[1]=1;
				
				hsize_t offset_model_status[RANK];
				offset_model_status[0]=base_dims_model_status[0];	
				offset_model_status[1]=0;	

				dataspace_model_status->selectHyperslab(H5S_SELECT_SET,dimext_model_status,offset_model_status);

				dataspace_ext_model_status = new H5::DataSpace(RANK_model_status, dimext_model_status,NULL);

				temp_buffer_model_status = new int[ dimext_model_status[0]*dimext_model_status[1] ];
				int beginning_id = 0 ; 
				if(dump_files[file_id]->trimmed){beginning_id = dump_files[file_id]->fileTrimLengths[ids[i]];}
				for(int j = base_dims_model_status[0] ; j<currentStepID[ids[i]]-beginning_id; j++){
					temp_buffer_model_status[(j-base_dims_model_status[0]) ] = positions[ids[i]][j+beginning_id]->modelID;	
				}
				
				dataset_model_status->write(temp_buffer_model_status,H5::PredType::NATIVE_INT,*dataspace_ext_model_status, *dataspace_model_status);
				//Cleanup
				delete dataset_model_status;
				delete dataspace_model_status;
				delete dataspace_ext_model_status;
				delete plist_model_status;
				delete [] temp_buffer_model_status;
				temp_buffer_model_status = NULL;
				}
				
			}
		}



		//#####################################################
		dataset = new H5::DataSet(meta_group.openDataSet("CHAIN BETAS"));
		dataset->write(betas, H5::PredType::NATIVE_DOUBLE);	
		delete dataset;
		////#####################################################
		//if(integrated_likelihoods){
		//	dataset = new H5::DataSet(meta_group.openDataSet("INTEGRATED LIKELIHOODS"));
		//	dataset->write(integrated_likelihoods, H5::PredType::NATIVE_DOUBLE);	
		//	delete dataset;
		//}
		////#####################################################
		//if(integrated_likelihoods_terms){
		//	dataset = new H5::DataSet(meta_group.openDataSet("INTEGRATED LIKELIHOODS TERM NUMBER"));
		//	dataset->write(integrated_likelihoods_terms, H5::PredType::NATIVE_INT);	
		//	delete dataset;
		//}
		////#####################################################
		if(calculatedEvidence){
			dataset = new H5::DataSet(meta_group.openDataSet("EVIDENCE"));
			dataset->write(&evidence, H5::PredType::NATIVE_DOUBLE);	
			delete dataset;
		}
		////#####################################################
		if(!dump_files[file_id]->trimmed ){
			
			dataset = new H5::DataSet(meta_group.openDataSet("SUGGESTED TRIM LENGTHS"));
		
			dataset->write(trimLengths, H5::PredType::NATIVE_INT);	
			delete dataset;
		}
		if(acs){
			int *int_temp_buffer = new int[ensembleN*maxDim];
			dataset = new H5::DataSet(meta_group.openDataSet("AC VALUES"));
			for(int i  = 0 ; i<ensembleN; i++){
				for(int j = 0 ; j<maxDim ; j++){
					int_temp_buffer[i*maxDim +j ] = acs[i][j];
				}
			}
			dataset->write(int_temp_buffer, H5::PredType::NATIVE_INT);	
			delete [] int_temp_buffer;
			int_temp_buffer = NULL;
			delete dataset;
		}
	
		//Cleanup
		output_group.close();
		output_LL_LP_group.close();
		if(RJ){
			status_group.close();
			model_status_group.close();
		}
		meta_group.close();
		if(ids){
			delete [] ids;
			ids = NULL;
		}

	}	
	catch( H5::FileIException error )
	{
		error.printErrorStack();
		return -1;
	}
	// catch failure caused by the DataSet operations
	catch( H5::DataSetIException error )
	{
		error.printErrorStack();
	   	return -1;
	}
	// catch failure caused by the DataSpace operations
	catch( H5::DataSpaceIException error )
	{
		error.printErrorStack();
	   	return -1;
	}
	// catch failure caused by the DataSpace operations
	catch( H5::DataTypeIException error )
	{
		error.printErrorStack();
	   	return -1;
	}
	return 0;

}
#endif


}
