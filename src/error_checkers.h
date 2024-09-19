#ifndef __ERROR_CHECKERS_H__
#define __ERROR_CHECKERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cutensor.h>
#include <assert.h>



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

static const char *_cudaGetErrorEnum3(cutensorStatus_t error)
{
    switch (error)
    {
    case CUTENSOR_STATUS_SUCCESS:
        return "CUTENSOR_STATUS_SUCCESS";

    case CUTENSOR_STATUS_NOT_INITIALIZED:
        return "CUTENSOR_STATUS_NOT_INITIALIZED";

    case CUTENSOR_STATUS_ALLOC_FAILED:
        return "CUTENSOR_STATUS_ALLOC_FAILED";

    case CUTENSOR_STATUS_INVALID_VALUE:
        return "CUTENSOR_STATUS_INVALID_VALUE";

    case CUTENSOR_STATUS_ARCH_MISMATCH:
        return "CUTENSOR_STATUS_ARCH_MISMATCH";

    case CUTENSOR_STATUS_MAPPING_ERROR:
        return "CUTENSOR_STATUS_MAPPING_ERROR";

    case CUTENSOR_STATUS_EXECUTION_FAILED:
        return "CUTENSOR_STATUS_EXECUTION_FAILED";

    case CUTENSOR_STATUS_INTERNAL_ERROR:
        return "CUTENSOR_STATUS_INTERNAL_ERROR";

    case CUTENSOR_STATUS_NOT_SUPPORTED:
        return "CUTENSOR_STATUS_NOT_SUPPORTED";

    case CUTENSOR_STATUS_LICENSE_ERROR:
        return "CUTENSOR_STATUS_LICENSE_ERROR";

    case CUTENSOR_STATUS_CUBLAS_ERROR:
        return "CUTENSOR_STATUS_CUBLAS_ERROR";
    
    case CUTENSOR_STATUS_CUDA_ERROR:
        return "CUTENSOR_STATUS_CUDA_ERROR";
    
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
        return "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE";

    case CUTENSOR_STATUS_INSUFFICIENT_DRIVER:
        return "CUTENSOR_STATUS_INSUFFICIENT_DRIVER";
    
    case CUTENSOR_STATUS_IO_ERROR:
        return "CUTENSOR_STATUS_IO_ERROR";

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


#define cutensorSafeCall(err) __cutensorSafeCall(err, __FILE__, __LINE__)
/**
 * @brief Safely calls the cuTENSOR function and checks for errors.
 * 
 * @param err The cuTENSOR status code.
 * @param file The file path where the function is called.
 * @param line The line number where the function is called.
 */ 
inline void __cutensorSafeCall(cutensorStatus_t err, const char *file, const int line)
{
    if (CUTENSOR_STATUS_SUCCESS != err)
    {
        fprintf(stderr, "cuTENSOR error in file '%s', line %d\n error %d: %s\nterminating!\n", __FILE__, __LINE__, err,
                _cudaGetErrorEnum3(err));
        cudaDeviceReset();
        assert(0);
    }
}




#endif // __ERROR_CHECKERS_H__  