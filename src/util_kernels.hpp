#ifndef __UTIL_KERNELS_H__
#define __UTIL_KERNELS_H__

#include "error_checkers.h"
#include "precision.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BLOCK_SIZE 1024

/**
 * @namespace UtilKernels
 * @brief Namespace containing utility CUDA kernel wrappers.
 */
namespace UtilKernels {
/**
 * @brief Casts a vector from one type to another.
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param size Size of the input and output vectors.
 * @param s The CUDA stream to use for the operation.
 * @tparam T_in Input type.
 * @tparam T_out Output type.
 */
template <typename T_in, typename T_out>
void cast_vector(const T_in *const d_in, T_out *const d_out,
                 const unsigned int size, cudaStream_t s);

/**
 * @brief Pads each block of a vector to twice the length with zeros.
 *
 * This function takes an input vector `d_in` and pads each block of the vector
 * to twice the length with zeros. The padded vector is stored in the output
 * vector `d_pad`. The number of columns in each block is specified by
 * `num_cols`. The total size of the vector is specified by `size`. The padding
 * operation is performed asynchronously on the CUDA stream `s`.
 *
 * @param d_in      Pointer to the input vector.
 * @param d_pad     Pointer to the output padded vector.
 * @param num_blocks  Number of blocks in the vector.
 * @param padded_size Padded size of each block.
 * @param s         CUDA stream for asynchronous execution.
 * @tparam T_in   Data type of the input and output vectors (real).
 * @tparam T_out  Data type of the input and output vectors (real).
 */
template <typename T_in, typename T_out>
void pad_vector(const T_in *const d_in, T_out *const d_pad,
                const unsigned int num_blocks, const unsigned int padded_size,
                cudaStream_t s);

/**
 * @brief Unpads or repads a vector.
 *
 * This function either unpads each block of the vector back to the original
 * length or resets the second half of each block to zeros.
 *
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param num_blocks Number of blocks in the vector.
 * @param padded_size Padded size of each block.
 * @param unpad Flag indicating whether to unpad or repad the vector. If true,
 * the vector will be unpadded. If false, the second half of each block will be
 * reset to zeros.
 * @param s The CUDA stream to use for the operation.
 * @tparam T_in Data type of the input and output vectors (real).
 * @tparam T_out Data type of the input and output vectors (real).
 */
template <typename T_in, typename T_out>
void unpad_repad_vector(const T_in *const d_in, T_out *const d_out,
                        const unsigned int num_blocks,
                        const unsigned int padded_size, const bool unpad,
                        cudaStream_t s);
/**
 * @brief Swaps the axes of a matrix and using cutranspose.
 * @param d_in Pointer to the input matrix.
 * @param d_out Pointer to the output matrix.
 * @param num_cols Number of columns in the input matrix.
 * @param num_rows Number of rows in the input matrix.
 * @param block_size Block size of input matrix.
 * @param s The CUDA stream to use for the operation.
 * @tparam T_complex Data type of the input and output matrices.
 *
 */
template <typename T_complex>
void swap_axes_cutranspose(const T_complex *d_in, T_complex *d_out,
                           const unsigned int num_cols,
                           const unsigned int num_rows,
                           const unsigned int block_size, cudaStream_t s);
/**
 * @brief Extends a vector by padding each block with zeros.
 *
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param num_blocks Number of blocks in the vector.
 * @param current_block_size Current size of each block.
 * @param new_block_size New size of each block after padding.
 * @param s The CUDA stream to use for the operation.
 */
void extend_vector(const double *d_in, double *d_out, size_t num_blocks,
                   size_t current_block_size, size_t new_block_size,
                   cudaStream_t s);
/**
 * @brief Shrinks a vector by removing padding.
 *
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param num_blocks Number of blocks in the vector.
 * @param current_block_size Current size of each block.
 * @param new_block_size New size of each block after padding.
 * @param s The CUDA stream to use for the operation.
 */
void shrink_vector(const double *d_in, double *d_out, size_t num_blocks,
                   size_t current_block_size, size_t new_block_size,
                   cudaStream_t s);

/**
 * @brief Computes the element-wise (Hadamard) product of two vectors: d_out[i]
 * = d_in1[i] * d_in2[i]
 * @param d_in1 Pointer to the first input vector.
 * @param d_in2 Pointer to the second input vector.
 * @param d_out Pointer to the output vector.
 * @param size Total number of elements in the vectors.
 * @param s The CUDA stream to use for the operation.
 */
void elementwise_multiply(const double *d_in1, const double *d_in2,
                          double *d_out, size_t size, cudaStream_t s);

/**
 * @brief Computes the element-wise (Hadamard) quotient of two vectors: d_out[i]
 * = d_in1[i] / d_in2[i]
 * @param d_in1 Pointer to the first input vector.
 * @param d_in2 Pointer to the second input vector.
 * @param d_out Pointer to the output vector.
 * @param size Total number of elements in the vectors.
 * @param s The CUDA stream to use for the operation.
 */
void elementwise_divide(const double *d_in1, const double *d_in2, double *d_out,
                        size_t size, cudaStream_t s);

/**
 * @brief Computes the element-wise inverse of a vector: d_out[i] = 1.0 /
 * d_in[i]
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param size Total number of elements in the vectors.
 * @param s The CUDA stream to use for the operation.
 */
void elementwise_inverse(const double *d_in, double *d_out, size_t size,
                         cudaStream_t s);

/**
 * @brief Computes the element-wise power of a vector: d_out[i] = pow(d_in[i],
 * exponent)
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param exponent The scalar power to raise each element to.
 * @param size Total number of elements in the vectors.
 * @param s The CUDA stream to use for the operation.
 */
void elementwise_power(const double *d_in, double *d_out, double exponent,
                       size_t size, cudaStream_t s);

/**
 * @brief Adds a scalar value to each element of a vector: d_out[i] = d_in[i] +
 * scalar
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param scalar The scalar value to add.
 * @param size Total number of elements in the vectors.
 * @param s The CUDA stream to use for the operation.
 */
void add_scalar(const double *d_in, double *d_out, double scalar, size_t size,
                cudaStream_t s);

/**
 * @brief Computes the fused element-wise multiply-add of three vectors:
 * d_out[i] = d_x[i] * d_y[i] + d_z[i]
 * @param d_x Pointer to the first input vector (multiplicand).
 * @param d_y Pointer to the second input vector (multiplier).
 * @param d_z Pointer to the third input vector (addend).
 * @param d_out Pointer to the output vector.
 * @param size Total number of elements in the vectors.
 * @param s The CUDA stream to use for the operation.
 */
void elementwise_multiply_add(const double *d_x, const double *d_y,
                              const double *d_z, double *d_out, size_t size,
                              cudaStream_t s);

} // namespace UtilKernels

#endif // __UTIL_KERNELS_H__