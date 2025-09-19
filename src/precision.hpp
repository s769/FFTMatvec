#ifndef __PRECISION_HPP__
#define __PRECISION_HPP__

#include "shared.hpp"
#if CUTENSOR_AVAILABLE
#include <cutensor.h>
#endif
#undef Complex 

/**
 * @brief Enumeration for precision types.
 */
enum class Precision
{
    SINGLE,
    DOUBLE,
};

/** @brief Configuration for the precision of different components of the matvec operation. */
struct MatvecPrecisionConfig
{
    Precision broadcast_and_pad = Precision::DOUBLE;
    Precision fft = Precision::DOUBLE;
    Precision sbgemv = Precision::DOUBLE;
    Precision ifft = Precision::DOUBLE;
    Precision unpad_and_reduce = Precision::DOUBLE;
};

template <Precision P>
struct TypeTraits;

// Specialization for SINGLE precision
template <>
struct TypeTraits<Precision::SINGLE>
{
    using Real = float;
    using Complex = ComplexF;
    static constexpr cufftType_t cufft_fwd_type = CUFFT_R2C; // Single-to-Complex
    static constexpr cufftType_t cufft_inv_type = CUFFT_C2R; // Complex-to-Single
    static constexpr cudaDataType_t cuda_data_type = CUDA_R_32F;
#if CUTENSOR_AVAILABLE
    static cutensorDataType_t cutensor_type() { return CUTENSOR_C_32F; }
    static cutensorComputeDescriptor_t compute_desc() { return CUTENSOR_COMPUTE_DESC_32F; }
#endif

    /**
     * @brief Explicit wrapper for cublasCgemvStridedBatched.
     */
    static cublasStatus_t blasSBgemv(cublasHandle_t handle, cublasOperation_t trans,
                                       int m, int n, const Complex *alpha,
                                       const Complex *A, int lda, long long int strideA,
                                       const Complex *x, int incx, long long int stridex,
                                       const Complex *beta, Complex *y, int incy,
                                       long long int stridey, int batch_count)
    {
#if !INDICES_64_BIT
        return cublasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                         x, incx, stridex, beta, y, incy, stridey,
                                         batch_count);
#else
        return cublasCgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA,
                                            x, incx, stridex, beta, y, incy, stridey,
                                            batch_count);
#endif
    }

    /**
     * @brief Explicit wrapper for cublasCgeam (Matrix-Matrix Transpose/Addition).
     */
    static cublasStatus_t blasGeam(cublasHandle_t handle, cublasOperation_t transa,
                                     cublasOperation_t transb, int m, int n,
                                     const Complex *alpha, const Complex *A, int lda,
                                     const Complex *beta, const Complex *B, int ldb,
                                     Complex *C, int ldc)
    {
#if !INDICES_64_BIT
        return cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
        return cublasCgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
    }



    static __host__ __device__ inline Complex one() { return make_cuComplex(1.0f, 0.0f); }
    static __host__ __device__ inline Complex zero() { return make_cuComplex(0.0f, 0.0f); }
};

// Specialization for DOUBLE precision
template <>
struct TypeTraits<Precision::DOUBLE>
{
    using Real = double;
    using Complex = ComplexD;
    static constexpr cufftType_t cufft_fwd_type = CUFFT_D2Z; // Double-to-Double Complex
    static constexpr cufftType_t cufft_inv_type = CUFFT_Z2D; // Double Complex-to-Double
    static constexpr cudaDataType_t cuda_data_type = CUDA_R_64F;
#if CUTENSOR_AVAILABLE
    static cutensorDataType_t cutensor_type() { return CUTENSOR_C_64F; }
    static cutensorComputeDescriptor_t compute_desc() { return CUTENSOR_COMPUTE_DESC_64F; }
#endif

    /**
     * @brief Explicit wrapper for cublasZgemvStridedBatched.
     */
    static cublasStatus_t blasSBgemv(cublasHandle_t handle, cublasOperation_t trans,
                                       int m, int n, const Complex *alpha,
                                       const Complex *A, int lda, long long int strideA,
                                       const Complex *x, int incx, long long int stridex,
                                       const Complex *beta, Complex *y, int incy,
                                       long long int stridey, int batch_count)
    {
#if !INDICES_64_BIT
        return cublasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA,
                                         x, incx, stridex, beta, y, incy, stridey,
                                         batch_count);
#else
        return cublasZgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA,
                                            x, incx, stridex, beta, y, incy, stridey,
                                            batch_count);
#endif
    }

    /**
     * @brief Explicit wrapper for cublasZgeam (Matrix-Matrix Transpose/Addition).
     */
    static cublasStatus_t blasGeam(cublasHandle_t handle, cublasOperation_t transa,
                                     cublasOperation_t transb, int m, int n,
                                     const Complex *alpha, const Complex *A, int lda,
                                     const Complex *beta, const Complex *B, int ldb,
                                     Complex *C, int ldc)
    {
#if !INDICES_64_BIT
        return cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
        return cublasZgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
#endif
    }



    static __host__ __device__ inline Complex one() { return make_cuDoubleComplex(1.0, 0.0); }
    static __host__ __device__ inline Complex zero() { return make_cuDoubleComplex(0.0, 0.0); }
};

#endif // __PRECISION_HPP__