#ifndef __utils_h__
#define __utils_h__

#include "shared.cuh"

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

uint64_t getHostHash(const char *);
void getHostName(char *, int);
unsigned long long dtime_usec(unsigned long long);


__global__ void PadVectorKernel(const double2 * const, double2 * const, const unsigned int, const unsigned int);
__global__ void PadVectorKernel(const double * const, double * const, const unsigned int, const unsigned int);
void PadVector(const double * const, double * const, const unsigned int, const unsigned int, cudaStream_t);

__global__ void UnpadVectorKernel(const double2 * const, double2 * const, const unsigned int, const unsigned int);
__global__ void UnpadVectorKernel(const double * const, double * const, const unsigned int, const unsigned int);
__global__ void RepadVectorKernel(const double2 * const, double2 * const, const unsigned int, const unsigned int);
void UnpadRepadVector(const double * const, double * const, const unsigned int, const unsigned int, const bool, cudaStream_t);

__global__ void transposeNoBankConflicts(Complex *, const Complex *, const unsigned int, const unsigned int);

__global__ void createIdentityKernel(double * const, int, int);

void transpose2d(Complex *, const Complex *, const unsigned int, const unsigned int, cudaStream_t);

void createIdentity(double * const, int, int, cudaStream_t);

void printVec(double * vec, int len, int unpad_size);
void printVecMPI(double * vec, int len, int unpad_size, int rank, int world_size);


#endif