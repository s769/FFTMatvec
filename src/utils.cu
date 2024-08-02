#include "utils.cuh"

// __global__ void PrintReal(double *d_in, const unsigned int num_cols, const unsigned int size, const unsigned int num_rows)
// {
//   for (int k = 0; k < num_rows; k++)
//   {
//     for (int i = 0; i < num_cols; i++)
//     {
//       for (int j = 0; j < size; j++)
//       {
//         printf("row %d, vec %d, j %d, %f \n", k, i, j, d_in[(size_t)k * num_cols * size + (size_t)i * size + j]);
//       }
//     }
//   }
// }

// __global__ void PrintComplex(Complex *d_in, const unsigned int num_cols, const unsigned int size, const unsigned int num_rows)
// {
//   for (int k = 0; k < num_rows; k++)
//   {
//     for (int i = 0; i < num_cols; i++)
//     {
//       for (int j = 0; j < size; j++)
//       {
//         printf("row %d, vec %d, j %d, %f %f\n", k, i, j, d_in[(size_t)k * num_cols * size + (size_t)i * size + j].x, d_in[(size_t)k * num_cols * size + (size_t)i * size + j].y);
//       }
//     }
//   }
// }

// /**
//  * This function returns the time in microseconds since the UNIX epoch.
//  * If the start argument is not 0, the time returned is the difference between
//  * the current time and the start time. This function is used to time code segments.
//  * The start argument is used to time multiple code segments in a single run.
//  *
//  * @param start The start time.
//  * @return The time in microseconds since the UNIX epoch.
//  */

unsigned long long dtime_usec(unsigned long long start = 0)
{

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

/**
 * This function hashes the given string using DJB2a. This hash is
 * suitable for use with the HashMap class.
 *
 * @param string The string to hash.
 * @return The hash of the string.
 */
uint64_t getHostHash(const char *string)
{
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++)
  {
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

void getHostName(char *hostname, int maxlen)
{
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++)
  {
    if (hostname[i] == '.')
    {
      hostname[i] = '\0';
      return;
    }
  }
}

void PadVector(const double *const d_in, double *const d_pad, const unsigned int num_cols, const unsigned int size, cudaStream_t s)
{
  if (size % 4 == 0)
    PadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(reinterpret_cast<const double2 *>(d_in), reinterpret_cast<double2 *>(d_pad), num_cols, size / 2);
  else
    PadVectorKernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(d_in, d_pad, num_cols, size);
  gpuErrchk(cudaPeekAtLastError());
  #if ERR_CHK

  gpuErrchk(cudaDeviceSynchronize());
  #endif
}

__global__ void PadVectorKernel(const double2 *const d_in, double2 *const d_pad, const unsigned int num_cols, const unsigned int size)
{
  int t = threadIdx.x;
  for (int j = t; j < size; j += blockDim.x)
  {
    if (j < size / 2)
      d_pad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size / 2 + j];
    else
      d_pad[(size_t)blockIdx.x * size + j] = {0, 0};
  }
}

__global__ void PadVectorKernel(const double *const d_in, double *const d_pad, const unsigned int num_cols, const unsigned int size)
{
  int t = threadIdx.x;
  for (int j = t; j < size; j += blockDim.x)
  {
    if (j < size / 2)
      d_pad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size / 2 + j];
    else
      d_pad[(size_t)blockIdx.x * size + j] = 0;
  }
}

__global__ void UnpadVectorKernel(const double2 *const d_in, double2 *const d_unpad, const unsigned int num_cols, const unsigned int size)
{
  int t = threadIdx.x;
  for (int j = t; j < size / 2; j += blockDim.x)
  {
    d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
  }
}
__global__ void UnpadVectorKernel(const double *const d_in, double *const d_unpad, const unsigned int num_cols, const unsigned int size)
{
  int t = threadIdx.x;
  for (int j = t; j < size / 2; j += blockDim.x)
  {
    d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
  }
}

__global__ void RepadVectorKernel(const double2 *const d_in, double2 *const d_unpad, const unsigned int num_cols, const unsigned int size)
{
  int t = threadIdx.x;
  for (int j = t; j < size; j += blockDim.x)
  {
    if (j < size / 2)
      d_unpad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size + j];
    else if (size % 2 == 1 && j == size / 2)
      d_unpad[(size_t)blockIdx.x * size + j] = {d_in[(size_t)blockIdx.x * size + j].x, 0};
    else
      d_unpad[(size_t)blockIdx.x * size + j] = {0, 0};
  }
}

void UnpadRepadVector(const double *const d_in, double *const d_out, const unsigned int num_cols, const unsigned int size, const bool unpad, cudaStream_t s)
{
  if (unpad)
  {
    if (size % 4 == 0)
      UnpadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(reinterpret_cast<const double2 *>(d_in), reinterpret_cast<double2 *>(d_out), num_cols, size / 2);
    else
      UnpadVectorKernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(d_in, d_out, num_cols, size);
  }
  else
  {
    RepadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(reinterpret_cast<const double2 *>(d_in), reinterpret_cast<double2 *>(d_out), num_cols, size / 2);
  }
  gpuErrchk(cudaPeekAtLastError());
  #if ERR_CHK

  gpuErrchk(cudaDeviceSynchronize());
  #endif
}



__global__ void transposeNoBankConflicts(Complex *odata, const Complex *idata, const unsigned int width, const unsigned int height)
{
    __shared__ Complex block[TILE_DIM][TILE_DIM + 1];

    // read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
    unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if ((xIndex < width) && (yIndex + j < height))
        {
            block[threadIdx.y + j][threadIdx.x] = idata[(size_t)(yIndex + j) * width + xIndex];
        }
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (odata) in linear order
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if ((xIndex < height) && (yIndex + j < width))
        {
            odata[(size_t)(yIndex + j) * height + xIndex] = block[threadIdx.x][threadIdx.y + j];
        }
    }
}


void transpose2d(Complex *odata, const Complex *idata, const unsigned int width, const unsigned int height, cudaStream_t s)
{
    if (width == 1 || height == 1)
    {
      if (s != NULL)
        cudaMemcpyAsync(odata, idata, width * height * sizeof(Complex), cudaMemcpyDeviceToDevice, s);
      else
        cudaMemcpy(odata, idata, width * height * sizeof(Complex), cudaMemcpyDeviceToDevice);
    }


    dim3 blocks((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);
    dim3 threads(TILE_DIM, BLOCK_ROWS, 1);
    if (s != NULL)
      transposeNoBankConflicts<<<blocks, threads, 0, s>>>(odata, idata, width, height);
    else
      transposeNoBankConflicts<<<blocks, threads>>>(odata, idata, width, height);

  gpuErrchk(cudaPeekAtLastError());
  #if ERR_CHK

  gpuErrchk(cudaDeviceSynchronize());
  #endif

}


__global__ void createIdentityKernel(double * const d_in, int num_r, int num_c)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;

  if (x < num_r && y < num_c)
  {
    size_t ind = (size_t)num_r*y + x;
    if (y < num_r)
    {
      if (x == y)
        d_in[ind] = 1.0;
      else
        d_in[ind] = 0.0;
    }
    else
      d_in[ind] = 0.0;
    }

}

void createIdentity(double * const d_in, int num_r, int num_c, cudaStream_t s)
{
  dim3 threads(32, 32, 1);
  dim3 blocks((num_r + threads.x - 1) / threads.x, (num_c + threads.y - 1) / threads.y, 1);
  createIdentityKernel<<<blocks, threads, 0, s>>>(d_in, num_r, num_c);
  gpuErrchk(cudaPeekAtLastError());
  #if ERR_CHK

  gpuErrchk(cudaDeviceSynchronize());
  #endif
}

void printVec(double * vec, int len, int unpad_size)
{
  double * h_vec;
  h_vec = (double *)malloc(len * unpad_size * sizeof(double));
  gpuErrchk(cudaMemcpy(h_vec, vec, len * unpad_size * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i = 0; i < len; i++)
  {
    for (int j = 0; j < unpad_size; j++)
    {
      printf("block: %d, t: %d, val: %f\n", i, j, h_vec[i * unpad_size + j]);
    }
    printf("\n");
  }
  free(h_vec);
}