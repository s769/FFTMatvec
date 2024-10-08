#include "util_kernels.hpp"


__global__ void pad_vector_kernel(const double2* const d_in, double2* const d_pad,
    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size; j += blockDim.x) {
        if (j < padded_size / 2)
            d_pad[(size_t)blockIdx.x * padded_size + j] = d_in[(size_t)blockIdx.x * padded_size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * padded_size + j] = { 0, 0 };
    }
}

__global__ void pad_vector_kernel(const double* const d_in, double* const d_pad,
    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size; j += blockDim.x) {
        if (j < padded_size / 2)
            d_pad[(size_t)blockIdx.x * padded_size + j] = d_in[(size_t)blockIdx.x * padded_size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * padded_size + j] = 0;
    }
}

__global__ void unpad_vector_kernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * padded_size / 2 + j] = d_in[(size_t)blockIdx.x * padded_size + j];
    }
}
__global__ void unpad_vector_kernel(const double* const d_in, double* const d_unpad,
    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * padded_size / 2 + j] = d_in[(size_t)blockIdx.x * padded_size + j];
    }
}

__global__ void repad_vector_kernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size; j += blockDim.x) {
        if (j < padded_size / 2)
            d_unpad[(size_t)blockIdx.x * padded_size + j] = d_in[(size_t)blockIdx.x * padded_size + j];
        else if (padded_size % 2 == 1 && j == padded_size / 2)
            d_unpad[(size_t)blockIdx.x * padded_size + j] = { d_in[(size_t)blockIdx.x * padded_size + j].x, 0 };
        else
            d_unpad[(size_t)blockIdx.x * padded_size + j] = { 0, 0 };
    }
}

void UtilKernels::pad_vector(const double* const d_in, double* const d_pad, const unsigned int num_blocks,
    const unsigned int padded_size, cudaStream_t s)
{
    if (padded_size % 4 == 0)
        pad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_pad), num_blocks,
            padded_size / 2);
    else
        pad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
            d_in, d_pad, num_blocks, padded_size);
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void UtilKernels::unpad_repad_vector(const double* const d_in, double* const d_out,
    const unsigned int num_blocks, const unsigned int padded_size, const bool unpad, cudaStream_t s)
{
    if (unpad) {
        if (padded_size % 4 == 0)
            unpad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
                reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_blocks,
                padded_size / 2);
        else
            unpad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
                d_in, d_out, num_blocks, padded_size);
    } else {
        repad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_blocks,
            padded_size / 2);
    }
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}