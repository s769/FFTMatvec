#include "reduction.cuh"

/**
 * This function computes an out-of-place strided vector reduction. The input vector is
 * assumed to consist of num_cols vectors of size size. The reduction is performed on each
 * row of the matrix. The result is stored in the output vector. The output vector is assumed
 * to be of size num_rows * size. The unpad argument is used to indicate whether the output
 * will be unpadded (for returning answer) or remain zero-padded (for subsequent matvecs).
 * If size is not a multiple of 4, the algorithm is slightly less efficient.
 *
 * @param d_out The output vector.
 * @param d_in The input vector.
 * @param num_cols The number of columns in the input vector.
 * @param size The size of each vector in the input vector - 2*N_t (padded).
 * @param num_rows The number of rows in the input vector.
 * @param unpad Whether to unpad the output vector.
 * @param device The device to use.
 *
 * @return void.
 */
void Reduction(double *const d_out, double *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int num_rows, const bool unpad, const bool conjugate, const unsigned int device, cudaStream_t s)
{
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  gpuErrchk(cudaGetDeviceProperties(&deviceProp, device));
  if (conjugate)
  {
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SVR4KernelConj, MAX_BLOCK_SIZE, device));
  }
  else
  {
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SVR4Kernel, MAX_BLOCK_SIZE, device));
  }

  const unsigned int MaxNumBlocks = numBlocksPerSm * deviceProp.multiProcessorCount;
  const unsigned int num_threads = std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE);
  dim3 dimBlock(num_threads, 1, 1);

  dim3 dimGrid(num_rows, std::min(MaxNumBlocks, num_cols), 1);

  if (!conjugate)
  {
    if (num_cols > MaxNumBlocks)
    {
      SVR4Kernel<<<dimGrid, dimBlock, 0, s>>>(reinterpret_cast<double2 *>(d_in), num_cols, size / 2, num_rows);
    #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    if (size % 4 != 0 && unpad)
      SVR5Kernel<<<num_rows, std::min((int)size / 2, MAX_BLOCK_SIZE), 0, s>>>(d_out, d_in, num_cols, size / 2, dimGrid.y, unpad);
    else
      SVR5Kernel<<<num_rows, dimBlock, 0, s>>>(reinterpret_cast<double2 *>(d_out), reinterpret_cast<double2 *>(d_in), num_cols, size / 2, dimGrid.y, unpad);
  #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }
  else
  {
    if (num_cols > MaxNumBlocks)
    {
      SVR4KernelConj<<<dimGrid, dimBlock, 0, s>>>(reinterpret_cast<double2 *>(d_in), num_cols, size / 2, num_rows);
    #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    if (size % 4 != 0 && unpad)
      SVR5KernelConj<<<num_rows, std::min((int)size / 2, MAX_BLOCK_SIZE), 0, s>>>(d_out, d_in, num_rows, size / 2, dimGrid.y, unpad);
    else
      SVR5KernelConj<<<num_rows, dimBlock, 0, s>>>(reinterpret_cast<double2 *>(d_out), reinterpret_cast<double2 *>(d_in), num_rows, size / 2, dimGrid.y, unpad);

  #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }
}

/**
 * First step of the strided vector reduction.
 * Each thread block in a row of the grid reduces a row of the matrix down to <num_blocks_in_grid_row> vectors.
 * @param d_in The input matrix
 * @param num_cols The number of columns in the input matrix
 * @param size The size of the input vector
 * @param num_rows The number of rows in the input matrix
 *
 * @return void
 */
static __global__ void SVR4Kernel(double2 *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int num_rows)
{
  int t = threadIdx.x;
  double2 val, tmp;
  size_t row_start;
  for (int j = t; j < (size + 1) / 2; j += blockDim.x)
  {
    val = {0, 0};
    row_start = (size_t)blockIdx.x * num_cols * size;

    for (int i = blockIdx.y; i < num_cols; i += gridDim.y)
    {
      tmp = d_in[row_start + (size_t)i * size + j];
      val.x += tmp.x;
      val.y += tmp.y;
    }
    d_in[row_start + (size_t)blockIdx.y * size + j] = {val.x, val.y};
  }
}

static __global__ void SVR4KernelConj(double2 *const d_in, const unsigned int num_rows, const unsigned int size, const unsigned int num_cols)
{
  int t = threadIdx.x;
  double2 val, tmp;
  size_t col_start;
  for (int j = t; j < (size + 1) / 2; j += blockDim.x)
  {
    val = {0, 0};
    col_start = (size_t)blockIdx.x * size;

    for (int i = blockIdx.y; i < num_rows; i += gridDim.y)
    {
      tmp = d_in[col_start + (size_t)i * num_cols * size + j];
      val.x += tmp.x;
      val.y += tmp.y;
    }
    d_in[col_start + (size_t)blockIdx.y * num_cols * size + j] = {val.x, val.y};
  }
}

/**
 * Second step of the strided vector reduction.
 * Each block reduces a row of the matrix down to a single vector.
 * @param d_out The output vector
 * @param d_in The input matrix
 * @param num_cols The number of columns in the input matrix
 * @param size The size of the input vector
 * @param sum_up_to The number of vectors to sum up in each row
 * @param unpad Whether to unpad the output vector
 *
 * @return void
 *
 */
static __global__ void SVR5Kernel(double2 *const d_out, const double2 *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int sum_up_to, const bool unpad)
{
  int t = threadIdx.x;
  double2 val, tmp;
  size_t row_start = (size_t)blockIdx.x * num_cols * size;
  for (int j = t; j < (size + 1) / 2; j += blockDim.x)
  {
    if (j < size / 2)
    {
      val = d_in[row_start + j];
      for (int i = 1; i < sum_up_to; i++)
      {
        tmp = d_in[row_start + (size_t)i * size + j];
        val.x += tmp.x;
        val.y += tmp.y;
      }
    }
    else if (j == size / 2 && size % 2 == 1)
    {
      val = d_in[row_start + j];
      for (int i = 1; i < sum_up_to; i++)
      {
        tmp = d_in[row_start + (size_t)i * size + j];
        val.x += tmp.x;
      }
    }
    if (unpad)
      d_out[(size_t)blockIdx.x * (size / 2) + j] = {val.x, val.y};
    else
    {
      if (j < size / 2)
        d_out[(size_t)blockIdx.x * size + j] = {val.x, val.y};
      else if (j == size / 2 && size % 2 == 1)
        d_out[(size_t)blockIdx.x * size + j] = {val.x, 0};
    }
  }
}

static __global__ void SVR5KernelConj(double2 *const d_out, const double2 *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int sum_up_to, const bool unpad)
{
  int t = threadIdx.x;
  double2 val, tmp;
  size_t col_start = (size_t)blockIdx.x * size;
  for (int j = t; j < (size + 1) / 2; j += blockDim.x)
  {
    if (j < size / 2)
    {
      val = d_in[col_start + j];
      for (int i = 1; i < sum_up_to; i++)
      {
        tmp = d_in[col_start + (size_t)i * num_cols * size + j];
        val.x += tmp.x;
        val.y += tmp.y;
      }
    }
    else if (j == size / 2 && size % 2 == 1)
    {
      val = d_in[col_start + j];
      for (int i = 1; i < sum_up_to; i++)
      {
        tmp = d_in[col_start + (size_t)i * num_cols * size + j];
        val.x += tmp.x;
      }
    }
    if (unpad)
      d_out[(size_t)blockIdx.x * (size / 2) + j] = {val.x, val.y};
    else
    {
      if (j < size / 2)
        d_out[(size_t)blockIdx.x * size + j] = {val.x, val.y};
      else if (j == size / 2 && size % 2 == 1)
        d_out[(size_t)blockIdx.x * size + j] = {val.x, 0};
    }
  }
}

/**
 * Second step of the strided vector reduction. Pure double version. Used if size is not a multiple of 4 and unpad is true.
 * Each block reduces a row of the matrix down to a single vector.
 * @param d_out The output vector
 * @param d_in The input matrix
 * @param num_cols The number of columns in the input matrix
 * @param size The size of the input vector
 * @param sum_up_to The number of vectors to sum up in each row
 * @param unpad Whether to unpad the output vector
 *
 */
static __global__ void SVR5Kernel(double *const d_out, const double *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int sum_up_to, const bool unpad)
{
  int t = threadIdx.x;
  double val;
  size_t row_start = (size_t)blockIdx.x * num_cols * size * 2;
  for (int j = t; j < size; j += blockDim.x)
  {
    val = d_in[row_start + j];
    for (int i = 1; i < sum_up_to; i++)
      val += d_in[row_start + (size_t)i * 2 * size + j];
    size_t ind = (unpad) ? (size_t)blockIdx.x * size + j : (size_t)blockIdx.x * size * 2 + j;
    d_out[ind] = val;
  }
}

static __global__ void SVR5KernelConj(double *const d_out, const double *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int sum_up_to, const bool unpad)
{
  int t = threadIdx.x;
  double val;
  size_t col_start = (size_t)blockIdx.x * size * 2;
  for (int j = t; j < size; j += blockDim.x)
  {
    val = d_in[col_start + j];
    for (int i = 1; i < sum_up_to; i++)
      val += d_in[col_start + (size_t)i * num_cols * 2 * size + j];
    size_t ind = (unpad) ? (size_t)blockIdx.x * size + j : (size_t)blockIdx.x * size * 2 + j;
    d_out[ind] = val;
  }
}

//////////////// Trying FFT Space Reduction ////////////////

void ComplexReduction(Complex *const d_out, Complex *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int num_rows, const bool conjugate, const unsigned int device, cudaStream_t s)
{
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  gpuErrchk(cudaGetDeviceProperties(&deviceProp, device));
  if (conjugate)
  {
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, ComplexSVR4KernelConj, MAX_BLOCK_SIZE, device));
  }
  else
  {
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, ComplexSVR4Kernel, MAX_BLOCK_SIZE, device));
  }

  const unsigned int MaxNumBlocks = numBlocksPerSm * deviceProp.multiProcessorCount;
  const unsigned int num_threads = std::min((int)size, MAX_BLOCK_SIZE);
  dim3 dimBlock(num_threads, 1, 1);

  dim3 dimGrid(num_rows, std::min(MaxNumBlocks, num_cols), 1);

  if (!conjugate)
  {
    if (num_cols > MaxNumBlocks)
    {
      ComplexSVR4Kernel<<<dimGrid, dimBlock, 0, s>>>(d_in, num_cols, size, num_rows);
    #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    ComplexSVR5Kernel<<<num_rows, dimBlock, 0, s>>>(d_out, d_in, num_cols, size, dimGrid.y);
  #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }
  else
  {
    if (num_cols > MaxNumBlocks)
    {
      ComplexSVR4KernelConj<<<dimGrid, dimBlock, 0, s>>>(d_in, num_cols, size, num_rows);
    #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    ComplexSVR5KernelConj<<<num_rows, dimBlock, 0, s>>>(d_out, d_in, num_rows, size, dimGrid.y);
  #if ERR_CHK
  gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }
}

static __global__ void ComplexSVR4Kernel(Complex *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int num_rows)
{
  int t = threadIdx.x;
  Complex val, tmp;
  size_t row_start;
  for (int j = t; j < size; j += blockDim.x)
  {
    val = {0, 0};
    row_start = (size_t)blockIdx.x * num_cols * size;

    for (int i = blockIdx.y; i < num_cols; i += gridDim.y)
    {
      tmp = d_in[row_start + (size_t)i * size + j];
      val.x += tmp.x;
      val.y += tmp.y;
    }
    d_in[row_start + (size_t)blockIdx.y * size + j] = {val.x, val.y};
  }
}

static __global__ void ComplexSVR4KernelConj(Complex *const d_in, const unsigned int num_rows, const unsigned int size, const unsigned int num_cols)
{
  int t = threadIdx.x;
  Complex val, tmp;
  size_t col_start;
  for (int j = t; j < size; j += blockDim.x)
  {
    val = {0, 0};
    col_start = (size_t)blockIdx.x * size;

    for (int i = blockIdx.y; i < num_rows; i += gridDim.y)
    {
      tmp = d_in[col_start + (size_t)i * num_cols * size + j];
      val.x += tmp.x;
      val.y += tmp.y;
    }
    d_in[col_start + (size_t)blockIdx.y * num_cols * size + j] = {val.x, val.y};
  }
}

static __global__ void ComplexSVR5Kernel(Complex *const d_out, Complex *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int sum_up_to)
{
  int t = threadIdx.x;
  Complex val, tmp;
  size_t row_start = (size_t)blockIdx.x * num_cols * size;
  for (int j = t; j < size; j += blockDim.x)
  {
    val = d_in[row_start + j];
    for (int i = 1; i < sum_up_to; i++)
    {
      tmp = d_in[row_start + (size_t)i * size + j];
      // printf("blockIdx.x: %d, threadIdx.x: %d, i: %d, j: %d, val.x: %f, val.y: %f, tmp.x: %f, tmp.y: %f\n", blockIdx.x, threadIdx.x, i, j, val.x, val.y, tmp.x, tmp.y);
      val.x += tmp.x;
      val.y += tmp.y;
    }
    d_out[(size_t)blockIdx.x * size + j] = {val.x, val.y};
  }
}

static __global__ void ComplexSVR5KernelConj(Complex *const d_out, Complex *const d_in, const unsigned int num_cols, const unsigned int size, const unsigned int sum_up_to)
{
  int t = threadIdx.x;
  Complex val, tmp;
  size_t col_start = (size_t)blockIdx.x * size;
  for (int j = t; j < size; j += blockDim.x)
  {
    val = d_in[col_start + j];
    for (int i = 1; i < sum_up_to; i++)
    {
      tmp = d_in[col_start + (size_t)i * num_cols * size + j];
      val.x += tmp.x;
      val.y += tmp.y;
    }

    d_out[(size_t)blockIdx.x * size + j] = {val.x, val.y};
  }
}