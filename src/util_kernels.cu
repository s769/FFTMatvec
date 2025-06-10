#include "util_kernels.hpp"
#define MAX_GRID_DIM 65535
#include <type_traits> // Required for std::is_floating_point

typedef struct
{
    int y, z;
} grid_factors_t;
template<typename T_real>
__global__ void pad_vector_kernel(const T_real* d_in, T_real* d_pad, const unsigned int padded_size)
{
    const size_t unpadded_size = padded_size / 2;
    const T_real* block_in_base = d_in + (size_t)blockIdx.x * unpadded_size;
    T_real* block_pad_base = d_pad + (size_t)blockIdx.x * padded_size;

    for (size_t j = threadIdx.x; j < padded_size; j += blockDim.x) {
        if (j < unpadded_size) {
            block_pad_base[j] = block_in_base[j];
        } else {
            block_pad_base[j] = T_real(0.0);
        }
    }
}

template<typename T_real>
__global__ void unpad_vector_kernel(const T_real* d_in, T_real* d_unpad, const unsigned int padded_size)
{
    const size_t unpadded_size = padded_size / 2;
    const T_real* block_in_base = d_in + (size_t)blockIdx.x * padded_size;
    T_real* block_unpad_base = d_unpad + (size_t)blockIdx.x * unpadded_size;

    const size_t num_elements_to_copy = unpadded_size;
    for (size_t j = threadIdx.x; j < num_elements_to_copy; j += blockDim.x) {
        block_unpad_base[j] = block_in_base[j];
    }
}

template<typename T_real>
__global__ void repad_vector_kernel(const T_real* d_in, T_real* d_out, const unsigned int padded_size)
{
    const size_t unpadded_size = padded_size / 2;
    const T_real* block_in_base = d_in + (size_t)blockIdx.x * unpadded_size;
    T_real* block_out_base = d_out + (size_t)blockIdx.x * padded_size;

    for (size_t j = threadIdx.x; j < padded_size; j += blockDim.x) {
        if (j < unpadded_size) {
            block_out_base[j] = block_in_base[j];
        } else {
            block_out_base[j] = T_real(0.0);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && (padded_size % 2 == 1)) {
        size_t nyquist_real_idx = padded_size / 2;
        if (nyquist_real_idx + 1 < padded_size) {
            block_out_base[nyquist_real_idx + 1] = T_real(0.0);
        }
    }
}


//============================================================================//
//                      HOST LAUNCHER IMPLEMENTATIONS                         //
//============================================================================//

template<typename T_real>
void UtilKernels::pad_vector(const T_real *const d_in, T_real *const d_pad, const unsigned int num_blocks,
                             const unsigned int padded_size, cudaStream_t s)
{
    pad_vector_kernel<T_real><<<num_blocks, MAX_BLOCK_SIZE, 0, s>>>(d_in, d_pad, padded_size);
    gpuErrchk(cudaPeekAtLastError());
}

template<typename T_real>
void unpad_vector(const T_real *const d_in, T_real *const d_unpad, const unsigned int num_blocks,
                               const unsigned int padded_size, cudaStream_t s)
{
    unpad_vector_kernel<T_real><<<num_blocks, MAX_BLOCK_SIZE, 0, s>>>(d_in, d_unpad, padded_size);
    gpuErrchk(cudaPeekAtLastError());
}

template<typename T_real>
void repad_vector(const T_real* const d_in, T_real* const d_repad,
                               const unsigned int num_blocks, const unsigned int padded_size, cudaStream_t s)
{
    repad_vector_kernel<T_real><<<num_blocks, MAX_BLOCK_SIZE, 0, s>>>(d_in, d_repad, padded_size);
    gpuErrchk(cudaPeekAtLastError());
}

template<typename T_real>
void UtilKernels::unpad_repad_vector(const T_real* const d_in, T_real* const d_out,
                                     const unsigned int num_blocks, const unsigned int padded_size,
                                     const bool unpad, cudaStream_t s)
{
    if (unpad)
    {
        // Call the dedicated unpad function/kernel
        unpad_vector<T_real>(d_in, d_out, num_blocks, padded_size, s);
    }
    else // It's a repad operation
    {
        // Call the dedicated repad function/kernel
        repad_vector<T_real>(d_in, d_out, num_blocks, padded_size, s);
    }
}


// --- Explicit Host Launcher Instantiations ---
template void UtilKernels::unpad_repad_vector<float>(const float*, float*, unsigned int, unsigned int, bool, cudaStream_t);
template void UtilKernels::unpad_repad_vector<double>(const double*, double*, unsigned int, unsigned int, bool, cudaStream_t);

template void UtilKernels::pad_vector<float>(const float*, float*, unsigned int, unsigned int, cudaStream_t);
template void UtilKernels::pad_vector<double>(const double*, double*, unsigned int, unsigned int, cudaStream_t);

template <typename T_complex, int TILE_SIZE, int EPT>
__global__ void swap_axes_kernel(
    T_complex *out,
    const T_complex *in,
    int np0,
    int np1,
    int np2,
    int fold_y,
    int fold_z)
{
    // The shared memory tile now uses the templated complex type
    __shared__ T_complex tile[TILE_SIZE][TILE_SIZE + 1];

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

//============================================================================//
//                      HOST LAUNCHER IMPLEMENTATION                          //
//============================================================================//

// This helper function does not depend on the data type and can remain unchanged.
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

// The host function is now templated on T_complex
template<typename T_complex>
void UtilKernels::swap_axes_cutranspose(const T_complex *const d_in, T_complex *const d_out, const unsigned int num_cols, const unsigned int num_rows, const unsigned int block_size, cudaStream_t s)
{
    int sz[3] = {(int)block_size, (int)num_cols, (int)num_rows};
    
    // These template parameters are hard-coded here, matching your original design.
    // If you need to change them, you must also update the explicit instantiations above.
    constexpr int EPT = 2;
    constexpr int TILE_SIZE = 32;
    
    dim3 block_dims, grid_dims;
    grid_factors_t fold_factors = {1, 1};

    set_grid_dims(sz, 2, &block_dims, &grid_dims, EPT, TILE_SIZE, &fold_factors);

    // Call the correctly templated version of the kernel
    swap_axes_kernel<T_complex, TILE_SIZE, EPT><<<grid_dims, block_dims, 0, s>>>(
        d_out, d_in, sz[0], sz[1], sz[2], fold_factors.y, fold_factors.z);

    // gpuErrchk(cudaPeekAtLastError());
    // #if ERR_CHK
    // gpuErrchk(cudaDeviceSynchronize());
    // #endif
}

// Explicit instantiations for the host launcher function.
template void UtilKernels::swap_axes_cutranspose<ComplexF>(const ComplexF*, ComplexF*, unsigned int, unsigned int, unsigned int, cudaStream_t);
template void UtilKernels::swap_axes_cutranspose<ComplexD>(const ComplexD*, ComplexD*, unsigned int, unsigned int, unsigned int, cudaStream_t);


//============================================================================//
//                  TEMPLATED KERNEL IMPLEMENTATION                           //
//============================================================================//


// --- __device__ Helper Functions for Casting ---

// This helper is chosen by the compiler when T is a primitive float or double.
// The third argument (std::true_type) is selected when std::is_floating_point<T_in>::value is true.
template <typename T_in, typename T_out>
__device__ __forceinline__ void perform_cast(const T_in& in, T_out& out, std::true_type /* is_floating_point */)
{
    out = in;
}

// This helper is chosen by the compiler when T is a struct (like ComplexF/D).
// The third argument (std::false_type) is selected when std::is_floating_point<T_in>::value is false.
template <typename T_in, typename T_out>
__device__ __forceinline__ void perform_cast(const T_in& in, T_out& out, std::false_type /* is_not_floating_point */)
{
    // Your proposed logic: assign member-wise. This works for all complex conversions.
    out.x = in.x;
    out.y = in.y;
}


/**
 * @brief A single, generic CUDA kernel to cast elements from one type to another.
 * It uses compile-time dispatch to handle real and complex types differently.
 */
template <typename T_in, typename T_out>
__global__ void cast_kernel(const T_in *d_in, T_out *d_out, const unsigned int size)
{
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += gridDim.x * blockDim.x)
    {
        // Call the appropriate helper function. The compiler will resolve this at
        // compile time, generating either a direct assignment or a member-wise assignment.
        // There is zero runtime overhead.
        perform_cast(d_in[idx], d_out[idx], typename std::is_floating_point<T_in>::type());
    }
}

//============================================================================//
//                      HOST LAUNCHER IMPLEMENTATION                          //
//============================================================================//

/**
 * @brief Host launcher for the cast_vector kernel.
 * This is the implementation of your declared function.
 */
template <typename T_in, typename T_out>
void UtilKernels::cast_vector(const T_in *const d_in, T_out *const d_out, const unsigned int size, cudaStream_t s)
{
    if (size == 0) {
        return; // Nothing to do
    }

    // Standard CUDA launch configuration
    constexpr int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    // Launch the templated kernel
    cast_kernel<T_in, T_out><<<grid_size, block_size, 0, s>>>(d_in, d_out, size);

    gpuErrchk(cudaPeekAtLastError());
    #if ERR_CHK
        gpuErrchk(cudaDeviceSynchronize());
    #endif
}


//============================================================================//
//                 EXPLICIT TEMPLATE INSTANTIATIONS                           //
//============================================================================//
// This is a CRITICAL step to prevent linker errors. It tells the compiler to
// generate concrete code for these specific template combinations.

// Case 1: Single-precision float to double-precision float
template void UtilKernels::cast_vector<float, double>(const float*, double*, const unsigned int, cudaStream_t);

// Case 2: Double-precision float to single-precision float
template void UtilKernels::cast_vector<double, float>(const double*, float*, const unsigned int, cudaStream_t);

// Case 3: Single-precision complex to double-precision complex
template void UtilKernels::cast_vector<ComplexF, ComplexD>(const ComplexF*, ComplexD*, const unsigned int, cudaStream_t);

// Case 4: Double-precision complex to single-precision complex
template void UtilKernels::cast_vector<ComplexD, ComplexF>(const ComplexD*, ComplexF*, const unsigned int, cudaStream_t);


