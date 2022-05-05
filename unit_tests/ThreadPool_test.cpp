#include <bayesship/ThreadPool.h>
#include <time.h>
#include <math.h>


#include <gtest/gtest.h>

namespace{

struct testPoolFnStruct
{
	/*! Job ID*/
	int ID ;
	/*! Output [thread, ID]*/
	double *output;
};

void testPoolFn(int thread, testPoolFnStruct jobStruct)
{
	usleep((int)1e3);
	jobStruct.output[0] = thread;
	jobStruct.output[1] = jobStruct.ID;
}

class ThreadPoolTest : public testing::Test{
	protected:
	void SetUp() override{
		testPool = new bayesship::ThreadPool<testPoolFnStruct>(4,testPoolFn,false);
	}
	void TearDown() override{
		//testPool->~ThreadPool();
		delete testPool;
	}
	time_t start_time_;
	bayesship::ThreadPool<testPoolFnStruct> *testPool;
};
class ThreadPoolTestRandom : public testing::Test{
	protected:
	void SetUp() override{
		testPool = new bayesship::ThreadPool<testPoolFnStruct>(4,testPoolFn,false);
		testPool->randomizeJobs();
	}
	void TearDown() override{
		//testPool->~ThreadPool();
		delete testPool;
	}
	time_t start_time_;
	bayesship::ThreadPool<testPoolFnStruct> *testPool;
};


//Test
/*
TEST_F(ThreadPoolTest,Time)
{
	start_time_ = time(nullptr);	
	sleep(1);	
	const time_t end_time = time(nullptr);
	std::cout<<"TIME: "<<end_time - start_time_<<std::endl;
}
*/

TEST_F(ThreadPoolTestRandom,PoolTestReuse)
{
	/*Number of jobs to use for testing*/
	int iterations = 100;
	/*Create array of jobs*/
	testPoolFnStruct *jobs = new testPoolFnStruct[iterations];
	for( int i = 0 ;i<iterations; i++){
		jobs[i].ID = i;
		jobs[i].output = new double[2];
		jobs[i].output[0] = -1;
		jobs[i].output[1] = -1;

	}
	for(int i = 0 ; i<2; i++){
		/*Queue jobs */
		for( int i = 0 ;i<iterations; i++){
			testPool->enqueue(jobs[i]);
		}

		/*Ensure Queueing properly -- ID should be in ascending order*/
		for( int i = 0 ;i<iterations; i++){
			EXPECT_EQ(testPool->taskFetch(i).ID, i);
		}
		
		/*Job length should be 100*/
  		EXPECT_EQ(testPool->get_queue_length(), iterations);

		/*Start pool*/
		testPool->startPool();
		testPool->stopPool();


		/* Should have a queue length of 0 now*/
  		EXPECT_EQ(testPool->get_queue_length(), 0);

		/*Ensure the function executed properly*/
		for( int i = 0 ;i<iterations; i++){
  			EXPECT_EQ(jobs[i].output[1], i);
		}
		
	}
	/*Cleanup*/
	for( int i = 0 ;i<iterations; i++){
		delete [] jobs[i].output;
	}
	delete [] jobs;
}

TEST_F(ThreadPoolTest,PoolTestReuse)
{
	/*Number of jobs to use for testing*/
	int iterations = 100;
	/*Create array of jobs*/
	testPoolFnStruct *jobs = new testPoolFnStruct[iterations];
	for( int i = 0 ;i<iterations; i++){
		jobs[i].ID = i;
		jobs[i].output = new double[2];
		jobs[i].output[0] = -1;
		jobs[i].output[1] = -1;

	}
	for(int i = 0 ; i<2; i++){
		/*Queue jobs */
		for( int i = 0 ;i<iterations; i++){
			testPool->enqueue(jobs[i]);
		}

		/*Ensure Queueing properly -- ID should be in ascending order*/
		for( int i = 0 ;i<iterations; i++){
			EXPECT_EQ(testPool->taskFetch(i).ID, i);
		}
		
		/*Job length should be 100*/
  		EXPECT_EQ(testPool->get_queue_length(), iterations);

		/*Start pool*/
		testPool->startPool();
		testPool->stopPool();


		/* Should have a queue length of 0 now*/
  		EXPECT_EQ(testPool->get_queue_length(), 0);

		/*Ensure the function executed properly*/
		for( int i = 0 ;i<iterations; i++){
  			EXPECT_EQ(jobs[i].output[1], i);
		}
		
	}
	/*Cleanup*/
	for( int i = 0 ;i<iterations; i++){
		delete [] jobs[i].output;
	}
	delete [] jobs;
}

TEST_F(ThreadPoolTest,PoolTestConcurrentQueuing)
{
	/*Number of jobs to use for testing*/
	int iterations = 100;
	/*Create array of jobs*/
	testPoolFnStruct *jobs = new testPoolFnStruct[iterations];
	for( int i = 0 ;i<iterations; i++){
		jobs[i].ID = i;
		jobs[i].output = new double[2];
		jobs[i].output[0] = -1;
		jobs[i].output[1] = -1;

	}
	/*Queue jobs */
	for( int i = 0 ;i<iterations; i++){
		testPool->enqueue(jobs[i]);
	}

	/*Ensure Queueing properly -- ID should be in ascending order*/
	for( int i = 0 ;i<iterations; i++){
		EXPECT_EQ(testPool->taskFetch(i).ID, i);
	}
	
	/*Job length should be 100*/
  	EXPECT_EQ(testPool->get_queue_length(), iterations);

	/*Start pool*/
	testPool->startPool();
	/*Queue jobs while running*/
	for(int i = 0 ; i<3; i++){
		for( int i = 0 ;i<iterations; i++){
			testPool->enqueue(jobs[i]);
		}
	}
	testPool->stopPool();

	/* Should have a queue length of 0 now*/
  	EXPECT_EQ(testPool->get_queue_length(), 0);

	/*Ensure the function executed properly*/
	for( int i = 0 ;i<iterations; i++){
  		EXPECT_EQ(jobs[i].output[1], i);
	}
	
	/*Cleanup*/
	for( int i = 0 ;i<iterations; i++){
		delete [] jobs[i].output;
	}
	delete [] jobs;
}

TEST_F(ThreadPoolTest,PoolTestFIFO)
{
	/*Number of jobs to use for testing*/
	int iterations = 100;
	/*Create array of jobs*/
	testPoolFnStruct *jobs = new testPoolFnStruct[iterations];
	for( int i = 0 ;i<iterations; i++){
		jobs[i].ID = i;
		jobs[i].output = new double[2];
		jobs[i].output[0] = -1;
		jobs[i].output[1] = -1;

	}
	/*Queue jobs */
	for( int i = 0 ;i<iterations; i++){
		testPool->enqueue(jobs[i]);
	}

	/*Ensure Queueing properly -- ID should be in ascending order*/
	for( int i = 0 ;i<iterations; i++){
		EXPECT_EQ(testPool->taskFetch(i).ID, i);
	}
	
	/*Job length should be 100*/
  	EXPECT_EQ(testPool->get_queue_length(), iterations);

	/*Start pool*/
	testPool->startPool();
	testPool->stopPool();

	/* Should have a queue length of 0 now*/
  	EXPECT_EQ(testPool->get_queue_length(), 0);

	/*Ensure the function executed properly*/
	for( int i = 0 ;i<iterations; i++){
  		EXPECT_EQ(jobs[i].output[1], i);
	}
	
	/*Cleanup*/
	for( int i = 0 ;i<iterations; i++){
		delete [] jobs[i].output;
	}
	delete [] jobs;
}

// Demonstrate some basic assertions.
TEST_F(ThreadPoolTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  EXPECT_STREQ("hello", "hello");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

}
