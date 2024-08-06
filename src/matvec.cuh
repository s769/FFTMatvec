#ifndef __matvec_h__
#define __matvec_h__

#include "shared.cuh"
#include <cutranspose.h>


void setup_new(Complex **, const double *const, const unsigned int, const unsigned int, const unsigned int);//, const bool, const bool, const unsigned int, double **, double **, cufftHandle *, cufftHandle *, cufftHandle *, cufftHandle *, double **, double **, Complex **, Complex **, Complex **);
void setup(Complex **, const double *const, const unsigned int, const unsigned int, const unsigned int);//, const bool, const bool, const unsigned int, double **, double **, cufftHandle *, cufftHandle *, cufftHandle *, cufftHandle *, double **, double **, Complex **, Complex **, Complex **);
void fft_matvec(double *const, double *const, const Complex *const, const unsigned int, const unsigned int, const unsigned int, const bool, const bool, const unsigned int, cufftHandle, cufftHandle, double * const, Complex *const, Complex *const, Complex *const, cudaStream_t);
void fft_matvec_new(double *const r, double *const d_in, const Complex *const d_mat_freq, const unsigned int size, const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool unpad, const unsigned int device, cufftHandle forward_plan, cufftHandle inverse_plan, double *const d_out, Complex *const d_freq, Complex *const d_red_freq, Complex * const, Complex * const, cudaStream_t s, cublasHandle_t cublasHandle);



void init_comms(MPI_Comm, MPI_Comm *, MPI_Comm *, Comm_t *, Comm_t *, cudaStream_t *, int *, int, int, bool);
void compute_matvec(double *, double *, const double *, Complex *, const unsigned int, const unsigned int, const unsigned int, const bool, const bool, const unsigned int, double, Comm_t, Comm_t, cudaStream_t, const unsigned int, double * const, cufftHandle, cufftHandle, cufftHandle, cufftHandle, double * const, double * const, Complex * const, Complex * const, Complex * const, Complex *const, Complex *const, Complex * const, Complex * const, Complex * const, Complex * const, cublasHandle_t, bool, Complex * = NULL);
void cleanup(double *, double *, cufftHandle, cufftHandle, cufftHandle, cufftHandle, double *, double *, Complex *, Complex *, Complex *, const unsigned int, const bool);






#endif