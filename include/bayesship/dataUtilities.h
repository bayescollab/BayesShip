#ifndef DATAUTILITIES_H
#define DATAUTILITIES_H
#include <string>
#include <vector>

namespace bayesship{

/*! \file 
 *
 *  # Header file for routines related to data storage, io, writing data to disk, etc
 *
 *
 */


struct dump_file_struct{
	std::string filename;
	bool trimmed;
	bool coldOnly;
	int *fileTrimLengths=NULL;
};

void readCSVFile(std::string filename, double **output, int rows,int cols );
void readCSVFile(std::string filename, double *output);
void readCSVFile(std::string filename, int **output, int rows,int cols );
void readCSVFile(std::string filename, int *output);
void writeCSVFile(std::string filename, double **input, int rows, int cols );
void writeCSVFile(std::string filename, int **input, int rows,int cols );
void writeCSVFile(std::string filename, double *input,int length);
void writeCSVFile(std::string filename, int *input, int length );

/*! \brief Class to hold information about a position in parameter/model space
 *
 * Represents a single link in the (RJ)MCMC chain
 */
class positionInfo{
public:
	/*! Max dimension of the arrays*/
	int dimension;
	/*! Array holding the parameter space coordinates*/
	double *parameters=nullptr;
	/*! Array holding the model information for which parameters are ``on''*/
	int *status=nullptr;
	/*! Model ID to distinguish between perfectly nested models*/
	int modelID=0;
	/*! Function to update all the information for the current instance of positionInfo to a new set of values defined by a separate instance of positionInfo*/
	void updatePosition(positionInfo *newPosition);
	int countActiveDimensions();
	/*! Boolean for RJ or not*/
	bool RJ=false;
	
	positionInfo(int dimension, bool RJ=false)	
	{
		this->dimension = dimension;
		this->RJ = RJ;
		this->parameters = new double[dimension];
		if(RJ){
			this->status = new int[dimension];
		}
		
	}
	~positionInfo(){
		if(parameters){
			delete [] parameters;
			parameters = nullptr;
		}
		if(status){
			delete [] status;
			status = nullptr;
		}
	}
	void setParameter(double val, int id){
		parameters[id] = val;
	}
	double getParameter( int id){
		return parameters[id] ;
	}
};

/*! Class containing all the data about a sampling run 
 *
 * Includes the output chain positions, statistics about the proposal functions, and statistics about swapping
 *
 * Meant to modularize each run of the sampler, so the settings and optimizations persist, but the data isn't overwritten.
 */
class samplerData
{
public:
	int maxDim;
	int ensembleN;
	int ensembleSize;
	int chainN;
	int iterations;
	int proposalFnN;
	bool RJ;
	int *trimLengths=NULL;
	int chunk_steps = 1000;
	double *betas=nullptr;
	double **proposalTimes = nullptr;
	double *likelihoodTimes = nullptr;
	double *priorTimes = nullptr;
	int *likelihoodEvals = nullptr;
	bool calculatedEvidence=false;
	double evidence;
	double evidenceError;
	double *integratedLikelihoods=nullptr;
	//int *chain_lengths=NULL;


	/*! All positions for sampler (ptrs) -- shape [chainN][iterations]*/
	positionInfo ***positions=nullptr;
	/*! Array storing the current position for each sampler in the positions array -- shape [chainN]*/
	int *currentStepID =nullptr;
	/*! Array containing all the likelihood values for each position in positions -- shape [chainN][iterations]*/
	double **likelihoodVals=nullptr;
	/*! Array containing all the prior values for each position in positions -- shape [chainN][iterations]*/
	double **priorVals=nullptr;
	/*! Array counting the rejected number of steps for each proposal type for each chain -- shape [chainN][proposalFnN]*/
	int **rejectN=nullptr;
	/*! Array counting the successful number of steps for each proposal type for each chain -- shape [chainN][proposalFnN]*/
	int **successN=nullptr;
	/*! Array counting the failed number of swaps between chains-- shape [chainN][chainN]*/
	int **swapRejects=nullptr;
	/*! Array counting the successful number of swaps between chains-- shape [chainN][chainN]*/
	int **swapAccepts=nullptr;

	/*! Array containing autocorrelation lengths for each chain and dimension -- shape [chainN][dimension]*/
	int **acs=nullptr;
	/*! Array containing max autocorrelation lengths for each chain (maxed over dimension) -- shape [chainN]*/
	int *maxACs = nullptr;

	samplerData(int maxDim, int ensembleN, int ensembleSize, int iterations, int proposalFnN,bool RJ ,double *betas);
	~samplerData();
	void writeStatFile(std::string filename);
	void updateACs(int threads);
	int countIndependentSamples();
	void extendSize(int additionalIterations);
	double *** convertToPrimitivePointer();
	void deallocatePrimitivePointer(double ***newPointer);
	int create_data_dump(bool cold_only,bool trim,std::string filename);
	int append_to_data_dump(std::string filename);
	void set_trim(int trim);
	void updateBetas(double *betas);
	void calculateEvidence();

private:
	int *file_trim_lengths =NULL;
	bool trimmed_file=false;
	std::vector<dump_file_struct *> dump_files;
	std::vector<std::string> dump_file_names;
};
}
#endif
