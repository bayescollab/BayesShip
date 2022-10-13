#include <iostream>
#include <math.h>
#include <bayesship/ThreadPool.h>
#include <unistd.h>
#include <cmath>

void RT_ERROR_MSG();

int standardThreadPoolTest(int argc, char*argv[]);
int standardThreadPoolPairTest(int argc, char*argv[]);

int main(int argc, char*argv[])
{
	std::cout<<"Thread Pool Testing"<<std::endl;
	if(argc < 2){
		RT_ERROR_MSG();
		return 1;
	}
	int runtimeOpt = std::stoi(argv[1]);
	if(runtimeOpt == 0){
		standardThreadPoolTest(argc, argv);	
	}
	else if(runtimeOpt == 1){
		standardThreadPoolPairTest(argc, argv);	
	}
	return 0;
}

struct jobStruct
{
	int jobID;
	double gridpoint;
	double *val;
	double *prodVal;
	double *sJobVal;
	double *pJobVal;
	bool primary;
};

void job(int threadID, jobStruct job)
{
	*(job.val) = std::pow(job.gridpoint,2);

	return;
}

void jobMatched(int threadID, jobStruct job1, jobStruct job2)
{
	jobStruct pJob;
	jobStruct sJob;
	if(job1.primary){pJob = job1; sJob = job2;}
	else if(job2.primary){pJob = job2; sJob = job1;}
	*(pJob.val) = sJob.gridpoint * pJob.gridpoint;
	*(pJob.pJobVal) =  pJob.gridpoint;
	*(pJob.sJobVal) =  sJob.gridpoint;
		
	return;
}

bool match( jobStruct job1, jobStruct job2)
{
	if(job1.primary && !job2.primary){return true;}
	if(!job1.primary && job2.primary){return true;}
	return false;
}
int standardThreadPoolPairTest(int argc, char*argv[])
{
	int length = 200;
	int xgrid[length];
	int xgrid2[length];
	double vals[length];
	double vals1[length];
	double vals2[length];
	for(int i = 0 ; i<length; i++){
		xgrid[i] = 2*i+1;
		xgrid2[i] = -2*i-1;
	}

	bayesship::ThreadPoolPair<jobStruct> *pool = new bayesship::ThreadPoolPair<jobStruct>(4, jobMatched,match, true);
	for (int i = 0 ; i<length; i++){
		//usleep(1000);
		jobStruct job;	
		job.gridpoint = xgrid[i];	
		job.val = &(vals[i]);	
		job.jobID = i;
		job.primary = true;
		job.pJobVal = &(vals1[i]);
		job.sJobVal = &(vals2[i]);
		pool->enqueue(job);
		
	}
	for (int i = length-1 ; i>=0; i--){
		//usleep(1000);
		jobStruct job;	
		job.gridpoint = xgrid2[i];	
		//job.val = &(vals[i]);	
		job.jobID = i;
		job.primary = false;
		pool->enqueue(job);
		
	}
	pool->startPool();
	pool->stopPool();

	for(int i = 0 ; i <length; i+=(int)length/100){
		//std::cout<<xgrid[i]<<" "<<xgrid[i]*xgrid[i]-vals[i]<<std::endl;
		std::cout<<vals1[i]<<" "<<vals2[i]<<" "<<vals[i]<<" "<<vals[i]/vals2[i]-vals1[i]<<std::endl;
	}
	delete pool;

	return 0;
}

int standardThreadPoolTest(int argc, char*argv[])
{
	int length = 200;
	double xgrid[length];
	double vals[length];
	for(int i = 0 ; i<length; i++){
		xgrid[i] = i;
	}

	bayesship::ThreadPool<jobStruct> *pool = new bayesship::ThreadPool<jobStruct>(4, job, true);
	//pool->startPool();
	for (int i = 0 ; i<length; i++){
		//usleep(1000);
		jobStruct job;	
		job.gridpoint = xgrid[i];	
		job.val = &(vals[i]);	
		job.jobID = i;
		pool->enqueue(job);
		
	}
	pool->stopPool();

	for(int i = 0 ; i <length; i+=(int)length/100){
		std::cout<<xgrid[i]<<" "<<xgrid[i]*xgrid[i]-vals[i]<<std::endl;
	}
	delete pool;

	return 0;
}

void RT_ERROR_MSG()
{
	std::cout<<"ERROR -- incorrect arguments"<<std::endl;
	std::cout<<"Please supply a test number:"<<std::endl;
	std::cout<<"0 -- Standard Thread Pool Test"<<std::endl;
	std::cout<<"1 -- Standard Thread Pool Pair Test"<<std::endl;
	return;

}
