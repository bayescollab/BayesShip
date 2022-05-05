
#include "bayesship/utilities.h"

#include <fftw3.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
/*! \file 
 *
 * #Source file for general utilities
 */

namespace bayesship{
//#########################################################
//#########################################################
//General math operations
//#########################################################
//#########################################################

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

/*! \brief routine to print the progress of a process to the terminal as a progress bar
 *
 * Call everytime you want the progress printed
 */
void printProgress (double percentage)
{
    	int val = (int) (percentage * 100);
    	int lpad = (int) (percentage * PBWIDTH);
    	int rpad = PBWIDTH - lpad;
    	printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    	fflush (stdout);
}


bool checkDirExist(std::string fname){
	if( FILE *file = fopen(fname.c_str(),"r")){
		return true;	
	}
	return false;	
}


/*! \brief Local utility to dot a matrix and a vector: M.A = O
 *
 * Matrix of dim [m][n]
 *
 * A of dim [n]
 *
 * B of dim [m]
 */
void matrixDot(double **M, double *A, double *O, int m, int n )
{ 
	for(int i = 0 ; i < m; i++){
		double sum = 0 ;
		for(int j = 0 ; j<n; j++){
			sum+= M[i][j]*A[j];
		}
		O[i] = sum;
	}
	return;
}


/*! \brief Local power function, specifically for integer powers
 *
 * Much faster than the std version, because this is only for integer powers
 */
double powInt(double base, int power)
{
	if (power == 0) return 1.;
	double prod = 1;
	int pow = std::abs(power);
	for (int i = 0; i< pow;i++){
		prod = prod * base;
	}
	if (power>0)
		return prod;
	else
		return 1./prod;
}

int mvn_sample(
	int samples, 
	double *mean,
	double **cov, 
	int dim, 
	double **output /**< [out] Size [samples][dim]*/
	)
{
	const gsl_rng_type *T;
	gsl_rng * r ;
	gsl_rng_env_setup();
	T  = gsl_rng_default;
	r= gsl_rng_alloc(T);	
	int status = mvn_sample(samples, mean, cov, dim , r, output);	
	gsl_rng_free(r);
	return status;
}

/*! \brief Samples from a multivariate-gaussian with covariance cov and mean mean, with dimension dim and puts the output in output
 * Taken from blogpost: https://juanitorduz.github.io/multivariate_normal/
 */
int mvn_sample(
	int samples, 
	double *mean,
	double **cov, 
	int dim, 
	gsl_rng *r,
	double **output /**< [out] Size [samples][dim]*/
	)
{
	gsl_matrix *matrix = gsl_matrix_alloc(dim,dim);
	for(int i = 0 ;i<dim ; i++){
		for (int j = 0 ; j<dim ; j++){
			gsl_matrix_set(matrix, i, j , cov[i][j]);
		}
	}



	int status = gsl_linalg_cholesky_decomp1(matrix);
	if(status == 0 ){

		double **newmat = new double*[dim];
		for(int i = 0 ; i<dim ; i++){
			newmat[i]=new double[dim];
			for(int j = 0 ; j < dim; j++){
				if(j<=i){
					newmat[i][j] = gsl_matrix_get(matrix,i,j);
				}
				else{
					newmat[i][j] = 0;
				}
			}
		}
		//double test = gsl_matrix_get(matrix, row, column);
		
		double *random_nums = new double[samples*dim];
		for (int i = 0 ; i<samples*dim ; i++){
			random_nums[i] = gsl_ran_gaussian(r, 1);
		}
		
		for(int i = 0 ; i<samples; i++){
			for(int j =0  ; j< dim ; j ++){
				output[i][j] = mean[j];
				for(int k = 0 ; k<dim ; k++){
					output[i][j]+=newmat[j][k]*random_nums[i*dim +k ];
				}
			}
			
		}

		for(int i = 0 ; i<dim ; i++){
			delete [] newmat[i];
		}
		delete [] newmat;
		delete [] random_nums;
	}

	gsl_matrix_free(matrix);
	
	return status;
}



/*! \brief Local power function, specifically for integer powers
 *
 * Much faster than the std version, because this is only for integer powers
 */
template<class U>
U pow_int(U base, int power)
{
	if (power == 0) return 1.;
	U prod = 1;
	int pow = std::abs(power);
	for (int i = 0; i< pow;i++){
		prod = prod * base;
	}
	if (power>0)
		return prod;
	else
		return 1./prod;
}
template double pow_int<double>(double , int);
template int pow_int<int>(int , int);

/*! \brief Calculates the mean of an array
 *
 */
template<class T, class U>
void mean_list(T *list, int length, U *result)
{
	U m = 0;
	for(int i = 0 ; i<length; i++){
		m +=list[i];
	}
	m/=length;
	*result=m;
}
template void mean_list<double,double>(double *,int,double*);
template void mean_list<int,double>(int *,int,double*);

/*! \brief Calculates the variance of an array
 *
 */
template<class T, class U>
void variance_list(T *list, int length,U *result)
{
	U m;
	mean_list(list,length,&m);
	U variance=0;
	for(int i = 0; i<length ;i++){
		variance += pow_int(list[i] - m, 2);
	}
	variance/=length;
	*result=m;
}
template void variance_list<double,double>(double *,int,double*);
template void variance_list<int,double>(int *,int,double*);
//#########################################################
//#########################################################

//#########################################################
//#########################################################
//Memory management
//#########################################################
//#########################################################
/*! \brief Allocates memory for array size [dim1][dim2]
 *
 * Uses C++ new operator with double
 */
double**  allocate_2D_array(int dim1, int dim2)
{
	double **array = new double*[dim1];
	for (int i = 0; i<dim1; i ++)
	{       
	        array[i] = new double[dim2];
	}
	return array;

}
/*! \brief Allocates memory for array size [dim1][dim2]
 *
 * Uses C++ new operator with int
 */
int ** allocate_2D_array_int(int dim1, int dim2)
{
	int **array = new int*[dim1];
	for (int i = 0; i<dim1; i ++)
	{       
	        array[i] = new int[dim2];
	}
	return array;

}
/*! \brief Allocates memory for array size [dim1][dim2][dim3]
 *
 * Uses C++ new operator with double
 */
double*** allocate_3D_array(int dim1, int dim2,int dim3)
{
	double ***array = new double**[dim1];
	for (int i = 0; i<dim1; i ++)
	{       
	        array[i] = new double*[dim2];
		for (int j = 0; j<dim2; j ++){
	        	array[i][j] = new double[dim3];
		}
	}
	return array;

}
/*! \brief Allocates memory for array size [dim1][dim2][dim3]
 *
 * Uses C++ new operator with int
 */
int*** allocate_3D_array_int(int dim1, int dim2,int dim3)
{
	int ***array = new int**[dim1];
	for (int i = 0; i<dim1; i ++)
	{       
	        array[i] = new int*[dim2];
		for (int j = 0; j<dim2; j ++){
	        	array[i][j] = new int[dim3];
		}
	}
	return array;

}

/*! \brief deallocates 2D array allocated with internal function 
 *
 * deallocates array using delete operator for size double*[dim1]
 */
void deallocate_2D_array(double **arr, int dim1)
{
	for (int i = 0 ; i<dim1; i++){
		delete [] arr[i];
	}
	delete[] arr;
}
/*! \brief deallocates 2D array allocated with internal function 
 *
 * deallocates array using delete operator for size double*[dim1]
 */
void deallocate_2D_array(int **arr, int dim1)
{
	for (int i = 0 ; i<dim1; i++){
		delete [] arr[i];
	}
	delete[] arr;
}
/*! \brief deallocates 3D array allocated with internal function 
 *
 * deallocates array using delete operator for size double*[dim1][dim2]
 */
void deallocate_3D_array(double ***arr, int dim1, int dim2)
{
	for (int i = 0 ; i<dim1; i++){
		for (int j = 0 ; j<dim2; j++){
			delete [] arr[i][j];
		}
		delete [] arr[i];
	}
	delete[] arr;
}
/*! \brief deallocates 3D array allocated with internal function 
 *
 * deallocates array using delete operator for size double*[dim1][dim2]
 */
void deallocate_3D_array(int ***arr, int dim1, int dim2)
{
	for (int i = 0 ; i<dim1; i++){
		for (int j = 0 ; j<dim2; j++){
			delete [] arr[i][j];
		}
		delete [] arr[i];
	}
	delete[] arr;
}
//#########################################################
//#########################################################



//#########################################################
//#########################################################
//FFTW utilities
//#########################################################
//#########################################################
/*! \brief Allocate memory for FFTW3 methods used in a lot of inner products
 * input is a locally defined structure that houses all the pertinent data
 */
void allocate_FFTW_mem_forward(fftw_outline *plan, int length)
{
	plan->in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);	
	plan->out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);	
	plan->p = fftw_plan_dft_1d(length, plan->in, plan->out,FFTW_FORWARD, FFTW_MEASURE);
}
/*! \brief Allocate memory for FFTW3 methods used in a lot of inner products --INVERSE
 * input is a locally defined structure that houses all the pertinent data
 */
void allocate_FFTW_mem_reverse(fftw_outline *plan, int length)
{
	plan->in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);	
	plan->out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);	
	plan->p = fftw_plan_dft_1d(length, plan->in, plan->out,FFTW_BACKWARD, FFTW_MEASURE);
}
/*!\brief deallocates the memory used for FFTW routines
 */
void deallocate_FFTW_mem(fftw_outline *plan)
{
	fftw_destroy_plan(plan->p);
	fftw_free(plan->in);
	fftw_free(plan->out);
	//fftw_cleanup();
}

//#########################################################
//#########################################################
/*! \brief Print an error message and return an error code
 *
 * Codes:
 * 
 * 1 -- Input error 
 *
 * */
void errorMessage(std::string message, int code)
{
	std::cout<<message<<std::endl;
	exit(code);
	return;
}
}
