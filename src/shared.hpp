
/**
 * @file shared.hpp
 * @brief Header file containing shared definitions and includes.
 *
 * This file contains various definitions and includes that are commonly used across multiple source files.
 * It includes standard C and C++ libraries, CUDA libraries, OpenMP, MPI, NCCL, and other necessary headers.
 * It also defines various constants, types, and helper functions for error checking.
 *
 * @note This file should be included in all source files that require these shared definitions and includes.
 */
#ifndef __SHARED_H__
#define __SHARED_H__

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#include <omp.h>


#include <cublas_v2.h>


#define MAX_BLOCK_SIZE 1024
#define TIME_MPI 1
#define ERR_CHK 1
#define FFT_64 0
#define ROW_SETUP 1

#if FFT_64
typedef long long int fft_int_t;
#else
typedef int fft_int_t;
#endif


#include <string>
#include <vector>


#include <mpi.h>
#include <unistd.h>
#include <nccl.h>

#if TIME_MPI
#include "profiler.hpp"
#include <array>
#endif

typedef double2 Complex;


#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
/**
 * @brief Checks the CUDA error code and prints an error message if an error occurred.
 *
 * @param code The CUDA error code to check.
 * @param file The name of the file where the error occurred.
 * @param line The line number where the error occurred.
 * @param abort Flag indicating whether to abort the program if an error occurred (default: true).
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            getchar();
            exit(code);
        }
    }
}

/**
 * @brief Returns the string representation of a CUDA FFT error code.
 *
 * @param error The CUDA FFT error code.
 * @return The string representation of the error code.
 */
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

/**
 * @brief Returns the string representation of a CUDA error code.
 *
 * This function takes a `cublasStatus_t` error code as input and returns the corresponding string representation.
 *
 * @param error The CUDA error code to convert.
 * @return The string representation of the CUDA error code.
 */
static const char *_cudaGetErrorEnum2(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}


#define cufftSafeCall(err) __cufftSafeCall(err, __FILE__, __LINE__)
/**
 * @brief Safely calls the cufft function and checks for errors.
 *
 * @param err The cufftResult error code.
 * @param file The file path where the function is called.
 * @param line The line number where the function is called.
 */
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if (CUFFT_SUCCESS != err)
    {
        fprintf(stderr, "CUFFT error in file '%s', line %d\n error %d: %s\nterminating!\n", __FILE__, __LINE__, err,
                _cudaGetErrorEnum(err));
        cudaDeviceReset();
        assert(0);
    }
}

typedef Complex data_t;

#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
/**
 * @brief Safely calls the cuBLAS function and checks for errors.
 * 
 * @param err The cuBLAS status code.
 * @param file The file path where the function is called.
 * @param line The line number where the function is called.
 */
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    // printf("CUBLAS status: %s\n", _cudaGetErrorEnum2(err));
    if (CUBLAS_STATUS_SUCCESS != err)
    {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\n error %d: %s\nterminating!\n", __FILE__, __LINE__, err,
                _cudaGetErrorEnum2(err));
        cudaDeviceReset();
        assert(0);
    }
}



#define MPICHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS)                        \
        {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        ncclResult_t r = cmd;                                  \
        if (r != ncclSuccess)                                  \
        {                                                      \
            printf("Failed, NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#if TIME_MPI
template <typename E, class T, std::size_t N>
class enum_array : public std::array<T, N>
{
public:
    T &operator[](E e)
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }

    const T &operator[](E e) const
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }

    T &operator[](int e)
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }

    const T &operator[](int e) const
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }
};

enum class ProfilerTimesFull : unsigned int
{
    COMM_INIT = 0,
    SETUP,
    FULL,
};


enum class ProfilerTimes : unsigned int
{
    BROADCAST = 0,
    PAD,
    FFT,
    TRANS1,
    SBGEMV,
    TRANS2,
    IFFT,
    UNPAD,
    NCCLC,
    TOT,
};



extern enum_array<ProfilerTimesFull, profiler_t, 3> t_list;
extern enum_array<ProfilerTimes, profiler_t, 10> t_list_f;
extern enum_array<ProfilerTimes, profiler_t, 10> t_list_fs;


#endif

#endif