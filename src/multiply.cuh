#ifndef __multiply_h__
#define __multiply_h__

#include "shared.cuh"

static __global__ void ComplexPointwiseMulAndScale(cufftDoubleComplex *const, const cufftDoubleComplex *const,
                                                   const cufftDoubleComplex *const, const unsigned int);
static __global__ void ComplexConjPointwiseMulAndScale(cufftDoubleComplex *const, const cufftDoubleComplex *const,
                                                       const cufftDoubleComplex *const, const unsigned int);

static __global__ void ScaleKernel(double2 *const, const unsigned int, const double);

void multiplyCoefficient(Complex *const, const Complex *const,
                         const Complex *const, const unsigned int,
                         const unsigned int, const unsigned int, const bool, cudaStream_t);
void ScaleVector(double *const, const unsigned int, const unsigned int,
                 const double, cudaStream_t);

void ScaleMatrix(Complex *const, const unsigned int, unsigned int, const unsigned int, const double);
static __global__ void ScaleMatKernel(Complex *const, const unsigned int, const unsigned int, const unsigned int, const double);

#endif