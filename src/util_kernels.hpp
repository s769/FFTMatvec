#ifndef __UTIL_KERNELS_H__
#define __UTIL_KERNELS_H__

#include <cuda_runtime.h>
#include "error_checkers.h"
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include "precision.hpp"

#define MAX_BLOCK_SIZE 1024


/**
 * @namespace UtilKernels
 * @brief Namespace containing utility CUDA kernel wrappers.
 */
namespace UtilKernels
{
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
    void cast_vector(const T_in *const d_in, T_out *const d_out, const unsigned int size, cudaStream_t s);


    /**
     * @brief Pads each block of a vector to twice the length with zeros.
     *
     * This function takes an input vector `d_in` and pads each block of the vector to twice the length
     * with zeros. The padded vector is stored in the output vector `d_pad`. The number of columns in
     * each block is specified by `num_cols`. The total size of the vector is specified by `size`.
     * The padding operation is performed asynchronously on the CUDA stream `s`.
     *
     * @param d_in      Pointer to the input vector.
     * @param d_pad     Pointer to the output padded vector.
     * @param num_blocks  Number of blocks in the vector.
     * @param padded_size Padded size of each block.
     * @param s         CUDA stream for asynchronous execution.
     * @tparam T_real   Data type of the input and output vectors.
     */
    template <typename T_real>
    void pad_vector(const T_real *const d_in, T_real *const d_pad, const unsigned int num_blocks,
                    const unsigned int padded_size, cudaStream_t s);

    /**
     * @brief Unpads or repads a vector.
     *
     * This function either unpads each block of the vector back to the original length or resets the
     * second half of each block to zeros.
     *
     * @param d_in Pointer to the input vector.
     * @param d_out Pointer to the output vector.
     * @param num_blocks Number of blocks in the vector.
     * @param padded_size Padded size of each block.
     * @param unpad Flag indicating whether to unpad or repad the vector. If true, the vector will be
     * unpadded. If false, the second half of each block will be reset to zeros.
     * @param s The CUDA stream to use for the operation.
     * @tparam T_real Data type of the input and output vectors.
     */
    template <typename T_real>
    void unpad_repad_vector(const T_real *const d_in, T_real *const d_out, const unsigned int num_blocks,
                            const unsigned int padded_size, const bool unpad, cudaStream_t s);
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
    void swap_axes_cutranspose(const T_complex *const d_in, T_complex *const d_out, const unsigned int num_cols, const unsigned int num_rows, const unsigned int block_size, cudaStream_t s);

} // namespace UtilKernels

#endif // __UTIL_KERNELS_H__