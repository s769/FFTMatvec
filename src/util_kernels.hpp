#ifndef __UTIL_KERNELS_H__
#define __UTIL_KERNELS_H__

#include <cuda_runtime.h>
#include "error_checkers.h"
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#define MAX_BLOCK_SIZE 1024


/**
 * @namespace UtilKernels
 * @brief Namespace containing utility CUDA kernel wrappers.
 */
namespace UtilKernels {
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
 * @param num_cols  Number of columns in each block.
 * @param size      Total size of the vector.
 * @param s         CUDA stream for asynchronous execution.
 */
void pad_vector(const double* const d_in, double* const d_pad, const unsigned int num_cols,
    const unsigned int size, cudaStream_t s);

/**
 * @brief Unpads or repads a vector.
 *
 * This function either unpads each block of the vector back to the original length or resets the
 * second half of each block to zeros.
 *
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param num_cols The number of columns in the vector.
 * @param size The size of the vector.
 * @param unpad Flag indicating whether to unpad or repad the vector. If true, the vector will be
 * unpadded. If false, the second half of each block will be reset to zeros.
 * @param s The CUDA stream to use for the operation.
 */
void unpad_repad_vector(const double* const d_in, double* const d_out, const unsigned int num_cols,
    const unsigned int size, const bool unpad, cudaStream_t s);

} // namespace UtilKernels









#endif // __UTIL_KERNELS_H__