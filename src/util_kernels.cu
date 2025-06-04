#include "util_kernels.hpp"
#define MAX_GRID_DIM 65535
typedef struct
{
    int y, z;
} grid_factors_t;

__global__ void pad_vector_kernel(const double2 *const d_in, double2 *const d_pad,
                                  const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size; j += blockDim.x)
    {
        if (j < padded_size / 2)
            d_pad[(size_t)blockIdx.x * padded_size + j] = d_in[(size_t)blockIdx.x * padded_size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * padded_size + j] = {0, 0};
    }
}

__global__ void pad_vector_kernel(const double *const d_in, double *const d_pad,
                                  const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size; j += blockDim.x)
    {
        if (j < padded_size / 2)
            d_pad[(size_t)blockIdx.x * padded_size + j] = d_in[(size_t)blockIdx.x * padded_size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * padded_size + j] = 0;
    }
}

__global__ void unpad_vector_kernel(const double2 *const d_in, double2 *const d_unpad,
                                    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size / 2; j += blockDim.x)
    {
        d_unpad[(size_t)blockIdx.x * padded_size / 2 + j] = d_in[(size_t)blockIdx.x * padded_size + j];
    }
}
__global__ void unpad_vector_kernel(const double *const d_in, double *const d_unpad,
                                    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size / 2; j += blockDim.x)
    {
        d_unpad[(size_t)blockIdx.x * padded_size / 2 + j] = d_in[(size_t)blockIdx.x * padded_size + j];
    }
}

__global__ void repad_vector_kernel(const double2 *const d_in, double2 *const d_unpad,
                                    const unsigned int num_blocks, const unsigned int padded_size)
{
    int t = threadIdx.x;
    for (int j = t; j < padded_size; j += blockDim.x)
    {
        if (j < padded_size / 2)
            d_unpad[(size_t)blockIdx.x * padded_size + j] = d_in[(size_t)blockIdx.x * padded_size + j];
        else if (padded_size % 2 == 1 && j == padded_size / 2)
            d_unpad[(size_t)blockIdx.x * padded_size + j] = {d_in[(size_t)blockIdx.x * padded_size + j].x, 0};
        else
            d_unpad[(size_t)blockIdx.x * padded_size + j] = {0, 0};
    }
}

template <int TILE_SIZE, int EPT>
__global__ void swap_axes_kernel(
    Complex *out,
    const Complex *in,
    int np0,
    int np1,
    int np2,
    int fold_y,
    int fold_z)
{
    __shared__ Complex tile[TILE_SIZE][TILE_SIZE + 1];

    size_t logical_block_x = blockIdx.x;
    size_t extra_y = 0, extra_z = 0;
    if (fold_y > 1)
    {
        extra_y = logical_block_x % fold_y;
        logical_block_x /= fold_y;
    }
    if (fold_z > 1)
    {
        extra_z = logical_block_x % fold_z;
        logical_block_x /= fold_z;
    }
    size_t bx = logical_block_x;
    size_t by = blockIdx.y + extra_y * gridDim.y;
    size_t bz = blockIdx.z + extra_z * gridDim.z;

    size_t lx = threadIdx.x, ly = threadIdx.y;
    size_t y = bz;

    // Input: Each thread loads EPT elements along z_in
#pragma unroll
    for (int e = 0; e < EPT; ++e)
    {
        size_t z_in = ly + e * (TILE_SIZE / EPT) + TILE_SIZE * by;
        size_t x_in = lx + TILE_SIZE * bx;
        size_t ind_in = x_in + (y + z_in * (size_t)np1) * (size_t)np0;
        if (x_in < (size_t)np0 && z_in < (size_t)np2 && y < (size_t)np1)
        {
            tile[lx][ly + e * (TILE_SIZE / EPT)] = in[ind_in];
        }
    }

    __syncthreads();

    // Output: Each thread writes EPT elements along x_out
#pragma unroll
    for (int e = 0; e < EPT; ++e)
    {
        size_t x_out = ly + e * (TILE_SIZE / EPT) + TILE_SIZE * bx;
        size_t z_out = lx + TILE_SIZE * by;
        size_t ind_out = z_out + (y + x_out * (size_t)np1) * (size_t)np2;
        if (z_out < (size_t)np2 && x_out < (size_t)np0 && y < (size_t)np1)
        {
            out[ind_out] = tile[ly + e * (TILE_SIZE / EPT)][lx];
        }
    }
}

void UtilKernels::pad_vector(const double *const d_in, double *const d_pad, const unsigned int num_blocks,
                             const unsigned int padded_size, cudaStream_t s)
{
    if (padded_size % 4 == 0)
        pad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2 *>(d_in), reinterpret_cast<double2 *>(d_pad), num_blocks,
            padded_size / 2);
    else
        pad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
            d_in, d_pad, num_blocks, padded_size);
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void UtilKernels::unpad_repad_vector(const double *const d_in, double *const d_out,
                                     const unsigned int num_blocks, const unsigned int padded_size, const bool unpad, cudaStream_t s)
{
    if (unpad)
    {
        if (padded_size % 4 == 0)
            unpad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
                reinterpret_cast<const double2 *>(d_in), reinterpret_cast<double2 *>(d_out), num_blocks,
                padded_size / 2);
        else
            unpad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
                d_in, d_out, num_blocks, padded_size);
    }
    else
    {
        repad_vector_kernel<<<num_blocks, std::min((int)(padded_size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2 *>(d_in), reinterpret_cast<double2 *>(d_out), num_blocks,
            padded_size / 2);
    }
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}

/**
 * @brief Set the kernel launch parameters for swap_axes with cutranspose.
 * @param size The size of the input tensor.
 * @param d2 The last index of the permutation.
 * @param block_dims The block size to use (output).
 * @param grid_dims The grid size to use (output).
 * @param elements_per_thread The number of elements to process per thread.
 * @param tile_size The tile size to use.
 * @param fold_factors The factors by which the grid size is folded (output).
 */
static void set_grid_dims(const int *size,
                          int d2,
                          dim3 *block_dims,
                          dim3 *grid_dims,
                          int elements_per_thread,
                          int tile_size,
                          grid_factors_t *fold_factors)
{
    block_dims->x = tile_size;
    block_dims->y = tile_size / elements_per_thread;
    block_dims->z = 1;

    int nblocks_x = (size[0] + tile_size - 1) / tile_size;
    if (d2 == 0)
        d2 = 1;
    int nblocks_y = (size[d2] + tile_size - 1) / tile_size;
    int nblocks_z = size[(d2 == 1) ? 2 : 1];

    int fold_y = 1, fold_z = 1;
    if (nblocks_y > MAX_GRID_DIM)
    {
        fold_y = (nblocks_y + MAX_GRID_DIM - 1) / MAX_GRID_DIM;
        nblocks_x *= fold_y;
        nblocks_y = MAX_GRID_DIM;
    }
    if (nblocks_z > MAX_GRID_DIM)
    {
        fold_z = (nblocks_z + MAX_GRID_DIM - 1) / MAX_GRID_DIM;
        nblocks_x *= fold_z;
        nblocks_z = MAX_GRID_DIM;
    }
    grid_dims->x = nblocks_x;
    grid_dims->y = nblocks_y;
    grid_dims->z = nblocks_z;

    if (fold_factors)
    {
        fold_factors->y = fold_y;
        fold_factors->z = fold_z;
    }
}

void UtilKernels::swap_axes_cutranspose(const Complex *const d_in, Complex *const d_out, const unsigned int num_cols, const unsigned int num_rows, const unsigned int block_size, cudaStream_t s)
{
    int sz[3] = {(int)block_size, (int)num_cols, (int)num_rows};
    constexpr int EPT = 2;
    constexpr int TILE_SIZE = 32;
    dim3 block_dims, grid_dims;
    grid_factors_t fold_factors = {1, 1};

    set_grid_dims(sz, 2, &block_dims, &grid_dims, EPT, TILE_SIZE, &fold_factors);


    

    swap_axes_kernel<TILE_SIZE, EPT><<<grid_dims, block_dims, 0, s>>>(d_out, d_in, sz[0], sz[1], sz[2], fold_factors.y, fold_factors.z);



    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}