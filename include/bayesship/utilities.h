#ifndef UTILITIES_H
#define UTILITIES_H

#include <fftw3.h>
#include <string>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

namespace bayesship{

/*! \file 
 *
 * # Header file for general utility functions
 */


bool checkDirExist(std::string fname);

void printProgress (double percentage);

void errorMessage(std::string message, int code);

void matrixDot(double **M, double *A, double *O ,int m , int n);

double powInt(double base, int power);

//#################################################################
//#################################################################
int mvn_sample(int samples, double *mean, double **cov, int dim, double **output );
int mvn_sample(int samples, double *mean, double **cov, int dim, gsl_rng *r,double **output );
//#################################################################
//#################################################################

//#################################################################
//#################################################################
double ** allocate_2D_array(int dim1, int dim2);
int ** allocate_2D_array_int(int dim1, int dim2);
double *** allocate_3D_array(int dim1, int dim2, int dim3);
int *** allocate_3D_array_int(int dim1, int dim2, int dim3);
void deallocate_2D_array(double **, int dim1);
void deallocate_2D_array(int **, int dim1);
void deallocate_3D_array(double ***, int dim1, int dim2);
void deallocate_3D_array(int ***, int dim1, int dim2);
//#################################################################
//#################################################################

//#################################################################
//#################################################################
struct fftw_outline
{
	fftw_complex *in, *out;
	fftw_plan p;
};

void allocate_FFTW_mem_forward(fftw_outline *plan,int length);
void allocate_FFTW_mem_reverse(fftw_outline *plan,int length);
void deallocate_FFTW_mem(fftw_outline *plan);
//#################################################################
//#################################################################

//#################################################################
//#################################################################
template<class T, class U>
void variance_list(T *list, int length,U *result);
template<class T,class U>
void mean_list(T *list, int length, U *result);
template<class U>
U pow_int(U base, int power);
//#################################################################
//#################################################################
void printTime(double time, std::string message);

/// @brief Compute a geometrically-spaced temperature ladder.
///
/// The endpoints of the ladder are fixed to 1 and 0. The interior points are assigned as
/// betas[i] = betas[i-1] / C,
/// where C = Tmax^{1 / (length-2)} so that beta[length-2] = 1/Tmax.
/// @param betas   [out] Pre-allocated array of [length].
/// @param length  Number of temperatures, including the endpoints.
/// @param Tmax    Maximum finite temperature (before T=∞).
void geometric_beta_schedule(double* betas, int length, double Tmax);
}
#endif
