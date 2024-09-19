#include "util_kernels.hpp"


__global__ void pad_vector_kernel(const double2* const d_in, double2* const d_pad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size; j += blockDim.x) {
        if (j < size / 2)
            d_pad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * size + j] = { 0, 0 };
    }
}

__global__ void pad_vector_kernel(const double* const d_in, double* const d_pad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size; j += blockDim.x) {
        if (j < size / 2)
            d_pad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * size + j] = 0;
    }
}

__global__ void unpad_vector_kernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
    }
}
__global__ void unpad_vector_kernel(const double* const d_in, double* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
    }
}

__global__ void repad_vector_kernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size; j += blockDim.x) {
        if (j < size / 2)
            d_unpad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size + j];
        else if (size % 2 == 1 && j == size / 2)
            d_unpad[(size_t)blockIdx.x * size + j] = { d_in[(size_t)blockIdx.x * size + j].x, 0 };
        else
            d_unpad[(size_t)blockIdx.x * size + j] = { 0, 0 };
    }
}

void UtilKernels::pad_vector(const double* const d_in, double* const d_pad, const unsigned int num_cols,
    const unsigned int size, cudaStream_t s)
{
    if (size % 4 == 0)
        pad_vector_kernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_pad), num_cols,
            size / 2);
    else
        pad_vector_kernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
            d_in, d_pad, num_cols, size);
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void UtilKernels::unpad_repad_vector(const double* const d_in, double* const d_out,
    const unsigned int num_cols, const unsigned int size, const bool unpad, cudaStream_t s)
{
    if (unpad) {
        if (size % 4 == 0)
            unpad_vector_kernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
                reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_cols,
                size / 2);
        else
            unpad_vector_kernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
                d_in, d_out, num_cols, size);
    } else {
        repad_vector_kernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_cols,
            size / 2);
    }
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}