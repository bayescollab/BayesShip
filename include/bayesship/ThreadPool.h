
#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <iostream>
#include <functional>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <gsl/gsl_rng.h>

namespace bayesship{

/*! \file 
 *
 * Header file (declarations and definitions because of template functions) for the implementation of a generic thread pool
 */

/*! 
 * \brief Default comparator for priority_queue in ThreadPool -- no comparison
 *
 * First in first out, not sorting
 */
template<class jobtype>
class DefaultComp
{
public:
	bool operator()(jobtype j, jobtype k)
	{
		return false;
	}
};


/*! \brief Class for creating a pool of threads to asynchronously distribute work
 *
 * Template parameters: 
 *
 * jobtype defines a structure or class that represents a job or task 
 *
 * comparator defines how to compare jobs for sorting the list
 *
 * Default options correspond to jobs being defined by an integer job_id, and no sorting of the list (first in first out)
 */
template<class jobtype=int, class comparator=DefaultComp<jobtype>>
class ThreadPool
{
public: 

	/*! \brief Constructor -- starts thread pool running
	 */
	explicit ThreadPool(
		std::size_t numThreads, /**< Number of threads to launch*/
		std::function<void(int, jobtype)> work_fn,/**< Function to call for each threads*/
		bool initiatePool=true /**< Automatically begin the thread pool -- if false, the user must start the thread pool manually*/
	)
	{
		work_fn_internal = work_fn;
		numThreads_internal = numThreads;

		if(initiatePool){
			start(numThreads);
		}
	}
	
	/*! \brief Destructor -- stops threads
	 */
	~ThreadPool()
	{
		if(activePool()){
			stop();	
		}
		if(r){
			gsl_rng_free(r);
		}
	}
	
	/*! \brief Places jobs in queue to wait for scheduling
	 *
	 * job_id is sorted if a comparator is provided
	 */
	void enqueue(
		jobtype job_id/**< Job ID to put in the workk queue*/
	)
	{
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			//tasks.push_back(std::move(job_id));
			tasks.push_back(job_id);
			EventVar.notify_one();
		}

	}
	/*!\brief Get the number of threads being used by the thread pool
	 */
	int get_num_threads()
	{
		return numThreads_internal;
	}
	
	bool activePool()
	{
		bool active = false;
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			if ( !tasks.empty()  ){
				active = true;
			}
			else if(stopping){
				active = true;
			}
		}
			
		return active;
	}
	/*! \brief Get the current length of the job queue
	 */
	int get_queue_length()
	{
		int size ; 
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			size =  tasks.size();
		}
		return size;
	}
	
	/*! \brief Sets internal flag to fully randomize jobs
 	* 
 	* Overrides the FIFO model to randomly select a job from the queue  
 	* 
 	* This can be useful because queueing jobs is not always as random as one may think
 	*/
	void randomizeJobs()
	{

		randomizeJobsFlag = true;

		if(!r){
			gsl_rng_env_setup();

			T = gsl_rng_default;
  			r = gsl_rng_alloc (T);
		}
	}

	/*! \brief initiates the thread pool -- only necessary if initiatePool was set to false in constructor*/
	void startPool(){
		start(numThreads_internal);
	}
	void stopPool(){
		stop();
	}

	/*! \brief Returns the job at index ''i'' in the tasks array*/
	jobtype taskFetch(int i){
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			return std::move(tasks.at(i));
		}
	}
	
private:
	/*! Internal flag  to randomize the selection of the jobs from the queue*/
	bool randomizeJobsFlag = false;

	/*! Vector for thread ids*/
	std::vector<std::thread> Threads;

	/*! Condition Variable*/
	std::condition_variable EventVar;

	/*! Lock to prevent memory races*/
	std::mutex EventMutex;

	/*! Boolean stop condition*/
	bool stopping=false;

	/*! Number of threads in pool*/
	int numThreads_internal;

	/*! Function for each thread to perform -- Takes arguments thread_id and jobtype job*/
	std::function<void(int, jobtype)> work_fn_internal;

	/*! Queue of jobs*/
	//std::priority_queue<jobtype,std::vector<jobtype>, comparator> tasksQueue;
	std::vector<jobtype> tasks;
	const gsl_rng_type * T;
  	gsl_rng * r=NULL;
	
	/*! \brief Starts thread pool -- launches each thread, which continually check for work
	 */
	void start(std::size_t numThreads)
	{
		for(auto i =0u; i<numThreads; i++)
		{
			Threads.emplace_back([=]{
				while(true)
				{
					jobtype j;
					{
						std::unique_lock<std::mutex> lock{EventMutex};

						EventVar.wait(lock,[=]{return stopping || !tasks.empty(); });
						
						if (stopping && tasks.empty())
							break;	

						if(randomizeJobsFlag && tasks.size()>1){

							int randomID = (int)(gsl_rng_uniform(r)*tasks.size());
							j = std::move(tasks.at(randomID));
							tasks.erase(tasks.begin() + randomID );
							//j = std::move(tasks.top());
							//tasks.pop();
						}
						else{	
							j = std::move(tasks.front());
							tasks.erase(tasks.begin() );
							//j = std::move(tasks.top());
							//tasks.pop();
						}
					}
					work_fn_internal(i, j);
					
				}
			});
		}
	}
	/*! \brief Stops thread pool 
	 *
	 * Waits for all threads to end and joins them
	 *
	 * Finishes the thread pool before ending
	 */
	void stop() noexcept
	{
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			stopping = true;
			EventVar.notify_all();
		}
		
		
		
		for(auto &thread: Threads){
			thread.join();
		}
		Threads.clear();

		{
			std::unique_lock<std::mutex> lock{EventMutex};
			stopping = false;
		}
	}
};


/*! \brief Class for creating a pool of threads to asynchronously distribute work in groups of variable size
 *
 * Template parameters: 
 *
 * jobtype defines a structure or class that represents a job or task 
 *
 * comparator defines how to compare jobs for sorting the list
 *
 * Default options correspond to jobs being defined by an integer job_id, and no sorting of the list (first in first out)
 */
template <class jobtype=int, class comparator=DefaultComp<jobtype>>
class ThreadPoolPair
{
public:
	explicit ThreadPoolPair(std::size_t numThreads,
		std::function<void(int,jobtype,jobtype)> work_fn,
		std::function<bool(jobtype,jobtype)> pair_match,
		bool initiatePool=true)
	{
		this->work_fn_internal = work_fn;
		this->pair_match = pair_match;
		this->numThreads_internal = numThreads;

		if(initiatePool){
			this->start(numThreads);
		}
	}
	
	/*! \brief Destructor -- stops threads
	 */
	~ThreadPoolPair()
	{
		if(this->activePool()){
			this->stop();	
		}
		if(this->r){
			gsl_rng_free(this->r);
		}
	}
	/*! \brief Places jobs in queue to wait for scheduling
	 *
	 * job_id is sorted if a comparator is provided
	 */
	void enqueue(
		jobtype job_id/**< Job ID to put in the workk queue*/
	)
	{
		{
			//tasks.push_back(std::move(job_id));
			std::unique_lock<std::mutex> lock{this->pairMutex};

			bool notify = false;
			if(prePaired.empty()){
				prePaired.emplace_back(job_id);
			}
			else{
				//std::vector<int>::iterator ptr;
				bool found = false;
				jobtype job2;
				for(size_t j =0; j<prePaired.size(); j++){
					if(pair_match(job_id,prePaired.at(j))){
						job2 = prePaired.at(j);
						prePaired.erase(prePaired.begin() + j);
						found = true;
						break;
					}
				}
				if(found){
					jobPair pair;
					pair.job1 = job_id;	
					pair.job2 = job2;	
					std::unique_lock<std::mutex> lock{this->EventMutex};
					tasks.push_back(pair);
					notify=true;
				}
				else{
					prePaired.emplace_back(job_id);
				}
			}
			if(notify){
				std::unique_lock<std::mutex> lock{this->EventMutex};
				this->EventVar.notify_one();
			}
		}

	}

	void flushPairQueue()
	{
		int size;
		{
			std::unique_lock<std::mutex> lock{this->pairMutex};
			size = prePaired.size();
		}
		jobtype j;
		for(int i = 0 ; i<size; i++){
			{
				std::unique_lock<std::mutex> lock{this->pairMutex};
				j = prePaired.front();
				prePaired.erase(prePaired.begin());
			}	
			this->enqueue(j);
		}
	}

	/*!\brief Get the number of threads being used by the thread pool
	 */
	int get_num_threads()
	{
		return numThreads_internal;
	}
	
	bool activePool()
	{
		bool active = false;
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			if ( !tasks.empty()  ){
				active = true;
			}
			else if(stopping){
				active = true;
			}
		}
			
		return active;
	}
	/*! \brief Get the current length of the job queue
	 */
	int get_queue_length()
	{
		int size ; 
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			size =  tasks.size();
		}
		return size;
	}
	
	/*! \brief Sets internal flag to fully randomize jobs
 	* 
 	* Overrides the FIFO model to randomly select a job from the queue  
 	* 
 	* This can be useful because queueing jobs is not always as random as one may think
 	*/
	void randomizeJobs()
	{

		randomizeJobsFlag = true;

		if(!r){
			gsl_rng_env_setup();

			T = gsl_rng_default;
  			r = gsl_rng_alloc (T);
		}
	}

	/*! \brief initiates the thread pool -- only necessary if initiatePool was set to false in constructor*/
	void startPool(){
		start(numThreads_internal);
	}
	void stopPool(){
		stop();
	}

	/*! \brief Returns the job at index ''i'' in the tasks array*/
	jobtype taskFetch(int i){
		{
			std::unique_lock<std::mutex> lock{EventMutex};
			return std::move(tasks.at(i));
		}
	}
	
private:
	
	struct jobPair
	{
		jobtype job1;	
		jobtype job2;	
	};
	std::function<bool(jobtype, jobtype)> pair_match;
	std::function<void(int i , jobtype, jobtype)> work_fn_internal;

	std::vector<jobPair> tasks;
	std::vector<jobtype> prePaired;

	/*! Lock to prevent memory races*/
	std::mutex pairMutex;

	/*! Internal flag  to randomize the selection of the jobs from the queue*/
	bool randomizeJobsFlag = false;

	/*! Vector for thread ids*/
	std::vector<std::thread> Threads;

	/*! Condition Variable*/
	std::condition_variable EventVar;

	/*! Lock to prevent memory races*/
	std::mutex EventMutex;

	/*! Boolean stop condition*/
	bool stopping=false;

	/*! Number of threads in pool*/
	int numThreads_internal;


	const gsl_rng_type * T;
  	gsl_rng * r=NULL;




	/*! \brief Starts thread pool -- launches each thread, which continually check for work
	 */
	void start(std::size_t numThreads)
	{
		for(auto i =0u; i<numThreads; i++)
		{
			this->Threads.emplace_back([=]{
				while(true)
				{
					jobtype j,k;
					{
						std::unique_lock<std::mutex> lock{this->EventMutex};

						this->EventVar.wait(lock,[=]{return this->stopping || !this->tasks.empty(); });
						
						if (this->stopping && this->tasks.empty())
							break;	

						if(this->randomizeJobsFlag && this->tasks.size()>1){

							int randomID = (int)(gsl_rng_uniform(this->r)*this->tasks.size());
							j = tasks.at(randomID).job1;
							k = tasks.at(randomID).job2;
							this->tasks.erase(this->tasks.begin() + randomID );
							//j = std::move(tasks.top());
							//tasks.pop();
						}
						else{	
							j = std::move(this->tasks.front()).job1;
							k = std::move(this->tasks.front()).job2;
							this->tasks.erase(this->tasks.begin() );
							//j = std::move(tasks.top());
							//tasks.pop();
						}
					}
					this->work_fn_internal(i, j,k);
					
				}
			});
		}
	}
	/*! \brief Stops thread pool 
	 *
	 * Waits for all threads to end and joins them
	 *
	 * Finishes the thread pool before ending
	 */
	void stop() noexcept
	{
		this->flushPairQueue();
		{
			std::unique_lock<std::mutex> lock{this->EventMutex};
			this->stopping = true;
			this->EventVar.notify_all();
		}
		
		
		
		for(auto &thread: this->Threads){
			thread.join();
		}
		this->Threads.clear();

		{
			std::unique_lock<std::mutex> lock{this->EventMutex};
			this->stopping = false;
		}
	}
};

}
#endif
