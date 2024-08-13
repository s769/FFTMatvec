/**
 * @file matvec.hpp
 * @brief Header file for the Matvec namespace.
 * 
 * This file contains the declarations of functions and variables related to the Matvec namespace.
 * The Matvec namespace provides functions for matrix-vector operations.
 */

#ifndef __matvec_h__
#define __matvec_h__

#include "shared.hpp"
#include "utils.hpp"
#include <cutranspose.h>

/**
 * @namespace Matvec
 * @brief Namespace containing functions for matrix-vector operations.
 */
namespace Matvec {

    /**
     * @brief Set up the matrix and perform necessary initialization.
     * 
     * This function sets up the matrix and performs necessary initialization for matrix-vector operations.
     * 
     * @param d_mat_freq Pointer to the matrix in device memory.
     * @param h_mat Pointer to the matrix in host memory.
     * @param block_size The block size for matrix operations.
     * @param num_cols The number of columns in the matrix.
     * @param num_rows The number of rows in the matrix.
     * @param cublasHandle The handle for the cuBLAS library.
     */
    void setup(Complex** d_mat_freq, const double* const h_mat, const unsigned int block_size,
        const unsigned int num_cols, const unsigned int num_rows, cublasHandle_t cublasHandle);

    /**
     * @brief Perform local matrix-vector multiplication.
     * 
     * This function performs local matrix-vector multiplication using the provided matrix and vectors.
     * 
     * @param out_vec Pointer to the output vector.
     * @param in_vec Pointer to the input vector.
     * @param d_mat_freq Pointer to the matrix in device memory.
     * @param size The size of the vectors.
     * @param num_cols The number of columns in the matrix.
     * @param num_rows The number of rows in the matrix.
     * @param conjugate Flag indicating whether to perform conjugate multiplication.
     * @param unpad Flag indicating whether to unpad the output vector.
     * @param device The device ID.
     * @param forward_plan The forward FFT plan.
     * @param inverse_plan The inverse FFT plan.
     * @param out_vec_pad Pointer to the padded output vector.
     * @param in_vec_freq Pointer to the input vector in frequency domain.
     * @param out_vec_freq_tosi Pointer to the output vector in frequency domain (transpose of input, scaled).
     * @param in_vec_freq_tosi Pointer to the input vector in frequency domain (transpose of input, scaled).
     * @param out_vec_freq Pointer to the output vector in frequency domain.
     * @param s The CUDA stream.
     * @param cublasHandle The handle for the cuBLAS library.
     */
    void local_matvec(double* const out_vec, double* const in_vec, const Complex* const d_mat_freq,
        const unsigned int size, const unsigned int num_cols, const unsigned int num_rows,
        const bool conjugate, const bool unpad, const unsigned int device, cufftHandle forward_plan,
        cufftHandle inverse_plan, double* const out_vec_pad, Complex* const in_vec_freq,
        Complex* const out_vec_freq_tosi, Complex* const in_vec_freq_tosi, Complex* const out_vec_freq,
        cudaStream_t s, cublasHandle_t cublasHandle);

    /**
     * @brief Perform matrix-vector multiplication.
     * 
     * This function performs matrix-vector multiplication using the provided matrix and vectors.
     * 
     * @param out_vec Pointer to the output vector.
     * @param in_vec Pointer to the input vector.
     * @param mat_freq_tosi Pointer to the matrix in frequency domain (transpose of input, scaled).
     * @param block_size The block size for matrix operations.
     * @param num_cols The number of columns in the matrix.
     * @param num_rows The number of rows in the matrix.
     * @param conjugate Flag indicating whether to perform conjugate multiplication.
     * @param full Flag indicating whether to perform full matrix-vector multiplication.
     * @param device The device ID.
     * @param scale The scaling factor.
     * @param nccl_row_comm The NCCL communicator for row-wise communication.
     * @param nccl_col_comm The NCCL communicator for column-wise communication.
     * @param s The CUDA stream.
     * @param in_vec_pad Pointer to the padded input vector.
     * @param forward_plan The forward FFT plan.
     * @param inverse_plan The inverse FFT plan.
     * @param forward_plan_conj The forward FFT plan for conjugate multiplication.
     * @param inverse_plan_conj The inverse FFT plan for conjugate multiplication.
     * @param out_vec_pad Pointer to the padded output vector.
     * @param in_vec_freq Pointer to the input vector in frequency domain.
     * @param out_vec_freq_tosi Pointer to the output vector in frequency domain (transpose of input, scaled).
     * @param in_vec_freq_tosi Pointer to the input vector in frequency domain (transpose of input, scaled).
     * @param out_vec_freq Pointer to the output vector in frequency domain.
     * @param cublasHandle The handle for the cuBLAS library.
     * @param mat_freq_tosi_other Pointer to the matrix in frequency domain (transpose of input, scaled) on other devices.
     * @param res_pad Pointer to the padded result vector.
     */
    void compute_matvec(double* out_vec, double* in_vec, Complex* mat_freq_tosi, const unsigned int block_size,
        const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool full,
        const unsigned int device, ncclComm_t nccl_row_comm, ncclComm_t nccl_col_comm,
        cudaStream_t s, double* const in_vec_pad, cufftHandle forward_plan, cufftHandle inverse_plan,
        cufftHandle forward_plan_conj, cufftHandle inverse_plan_conj, double* const out_vec_pad,
        Complex* const in_vec_freq, Complex* const out_vec_freq_tosi, Complex* const in_vec_freq_tosi,
        Complex* const out_vec_freq, cublasHandle_t cublasHandle, Complex* mat_freq_tosi_other,
        double* const res_pad);

}

#endif