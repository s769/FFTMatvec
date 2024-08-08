#ifndef __UTILS_H__
#define __UTILS_H__

#include "shared.hpp"


uint64_t getHostHash(const char* string);
void getHostName(char* hostname, int maxlen);



__global__ void PadVectorKernel(const double2* const d_in, double2* const d_pad,
    const unsigned int num_cols, const unsigned int size);
__global__ void PadVectorKernel(const double* const d_in, double* const d_pad,
    const unsigned int num_cols, const unsigned int size);
    
__global__ void UnpadVectorKernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_cols, const unsigned int size);
__global__ void UnpadVectorKernel(const double* const d_in, double* const d_unpad,
    const unsigned int num_cols, const unsigned int size);
__global__ void RepadVectorKernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_cols, const unsigned int size);
void PadVector(const double* const d_in, double* const d_pad, const unsigned int num_cols,
    const unsigned int size, cudaStream_t s);
void UnpadRepadVector(const double* const d_in, double* const d_out, const unsigned int num_cols,
    const unsigned int size, const bool unpad, cudaStream_t s);


void printVec(double * vec, int len, int unpad_size,std::string name = "Vector");
void printVecComplex(Complex * vec, int len, int unpad_size,std::string name = "Vector");
void printVecMPI(double * vec, int len, int unpad_size, int rank, int world_size, std::string name = "Vector");


#endif

