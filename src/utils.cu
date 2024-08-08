#include "utils.cuh"




/**
 * This function hashes the given string using DJB2a. This hash is
 * suitable for use with the HashMap class.
 *
 * @param string The string to hash.
 * @return The hash of the string.
 */
uint64_t getHostHash(const char* string)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

/**
 * Returns the hostname of the machine.
 *
 * @param hostname The hostname to return.
 * @param maxlen The maximum length of the hostname.
 */

void getHostName(char* hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

void PadVector(const double* const d_in, double* const d_pad, const unsigned int num_cols,
    const unsigned int size, cudaStream_t s)
{
    if (size % 4 == 0)
        PadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_pad), num_cols,
            size / 2);
    else
        PadVectorKernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
            d_in, d_pad, num_cols, size);
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}

__global__ void PadVectorKernel(const double2* const d_in, double2* const d_pad,
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

__global__ void PadVectorKernel(const double* const d_in, double* const d_pad,
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

__global__ void UnpadVectorKernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
    }
}
__global__ void UnpadVectorKernel(const double* const d_in, double* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
    }
}

__global__ void RepadVectorKernel(const double2* const d_in, double2* const d_unpad,
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

void UnpadRepadVector(const double* const d_in, double* const d_out, const unsigned int num_cols,
    const unsigned int size, const bool unpad, cudaStream_t s)
{
    if (unpad) {
        if (size % 4 == 0)
            UnpadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
                reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_cols,
                size / 2);
        else
            UnpadVectorKernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
                d_in, d_out, num_cols, size);
    } else {
        RepadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_cols,
            size / 2);
    }
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}



void printVec(double* vec, int len, int unpad_size, std::string name)
{
    double* h_vec;
    h_vec = (double*)malloc(len * unpad_size * sizeof(double));
    gpuErrchk(cudaMemcpy(h_vec, vec, len * unpad_size * sizeof(double), cudaMemcpyDeviceToHost));
    printf("%s:\n", name.c_str());

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < unpad_size; j++) {
            printf("block: %d, t: %d, val: %f\n", i, j, h_vec[i * unpad_size + j]);
        }
        printf("\n");
    }
    free(h_vec);
}

void printVecMPI(double* vec, int len, int unpad_size, int rank, int world_size, std::string name)
{
    if (rank == 0)
    {
        printf("%s:\n", name.c_str());
    }
    for (int r = 0; r < world_size; r++) {
        if (rank == r) {
            printf("Rank: %d\n", r);
            printVec(vec, len, unpad_size);
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
}

void printVecComplex(Complex* vec, int len, int unpad_size, std::string name)
{

    Complex* h_vec;
    h_vec = (Complex*)malloc(len * unpad_size * sizeof(Complex));
    gpuErrchk(cudaMemcpy(h_vec, vec, len * unpad_size * sizeof(Complex), cudaMemcpyDeviceToHost));

    printf("%s:\n", name.c_str());

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < unpad_size; j++) {
            printf("block: %d, t: %d, val: %f + %f i\n", i, j, h_vec[i * unpad_size + j].x,
                h_vec[i * unpad_size + j].y);
        }
        printf("\n");
    }
    free(h_vec);
}
