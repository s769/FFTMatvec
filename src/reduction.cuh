#ifndef __reduction_h__
#define __reduction_h__

#include "shared.cuh"

static __global__ void SVR4Kernel(double2 *const, const unsigned int, const unsigned int, const unsigned int);
static __global__ void SVR5Kernel(double2 *const, const double2 *const, const unsigned int, const unsigned int, const unsigned int, const bool);
static __global__ void SVR5Kernel(double *const, const double *const, const unsigned int, const unsigned int, const unsigned int, const bool);
static __global__ void SVR4KernelConj(double2 *const, const unsigned int, const unsigned int, const unsigned int);
static __global__ void SVR5KernelConj(double2 *const, const double2 *const, const unsigned int, const unsigned int, const unsigned int, const bool);
static __global__ void SVR5KernelConj(double *const, const double *const, const unsigned int, const unsigned int, const unsigned int, const bool);

static __global__ void ComplexSVR4Kernel(Complex *const, const unsigned int, const unsigned int, const unsigned int);
static __global__ void ComplexSVR5Kernel(Complex *const, Complex *const, const unsigned int, const unsigned int, const unsigned int);
static __global__ void ComplexSVR4KernelConj(Complex *const, const unsigned int, const unsigned int, const unsigned int);
static __global__ void ComplexSVR5KernelConj(Complex *const, Complex *const, const unsigned int, const unsigned int, const unsigned int);



void Reduction(double *const, double *const, const unsigned int, const unsigned int, const unsigned int, const bool, const bool, const unsigned int, cudaStream_t);
void ComplexReduction(Complex *const, Complex *const, const unsigned int, const unsigned int, const unsigned int, const bool, const unsigned int, cudaStream_t);



#endif