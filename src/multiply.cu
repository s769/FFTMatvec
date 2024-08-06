#include "multiply.cuh"

/**
 * Computes the element-wise product of two complex vectors and scales the result by 1/size.
 * @param d_out The output vector
 * @param d_mat The input matrix
 * @param d_vec The input vector
 * @param size The size of the input vector
 * @param num_cols The number of columns in the input matrix
 * @param num_rows The number of rows in the input matrix
 * @param conjugate Whether to conjugate the input matrix (for F* matvec)
 *
 * @return void
 *
 */
void multiplyCoefficient(Complex *const d_out, const Complex *const d_mat,
                         const Complex *const d_vec, const unsigned int size,
                         const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, cudaStream_t s)
{
  // Launch the ComplexPointwiseMulAndScale<<< >>> kernel
  dim3 dimBlock(std::min((int)size, MAX_BLOCK_SIZE));
  dim3 dimGrid(num_cols, num_rows, 1);
  if (conjugate)
    ComplexConjPointwiseMulAndScale<<<dimGrid, dimBlock, 0, s>>>(
        (cufftDoubleComplex *)d_out,
        (cufftDoubleComplex *)d_mat,
        (cufftDoubleComplex *)d_vec,
        (size / 2 + 1));
  else
    ComplexPointwiseMulAndScale<<<dimGrid, dimBlock, 0, s>>>(
        (cufftDoubleComplex *)d_out,
        (cufftDoubleComplex *)d_mat,
        (cufftDoubleComplex *)d_vec,
        (size / 2 + 1));
  gpuErrchk(cudaPeekAtLastError());
  #if ERR_CHK

  gpuErrchk(cudaDeviceSynchronize());
  #endif
}

/**
 * Scale a vector by a constant.
 * @param d_in The input vector
 * @param size The size of the input vector
 * @param num_rows The number of rows in the input matrix
 * @param scale The constant to scale by
 *
 * @return void
 */
void ScaleVector(double *const d_in, const unsigned int size, const unsigned int num_rows,
                 const double scale, cudaStream_t s)
{
  ScaleKernel<<<num_rows, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(reinterpret_cast<double2 *>(d_in), size / 2, scale);
  gpuErrchk(cudaPeekAtLastError());
  #if ERR_CHK

  gpuErrchk(cudaDeviceSynchronize());
  #endif
}

/**
 * Kernel to scale a vector by a constant.
 * @param d_in The input vector (scaled in place)
 * @param size The size of the input vector
 * @param scale The constant to scale by
 *
 * @return void
 */
static __global__ void ScaleKernel(double2 *const d_in, const unsigned int size, const double scale)
{
  double2 tmp;
  size_t ind;
  for (int i = threadIdx.x; i < (size + 1) / 2; i += blockDim.x)
  {
    ind = (size_t)blockIdx.x * size + i;
    tmp = d_in[ind];
    d_in[ind] = {tmp.x * scale, tmp.y * scale};
  }
}


static __global__ void ScaleMatKernel(Complex * const d_in, const unsigned int size, const unsigned int num_cols, const unsigned int num_rows, const double scale)
{
  size_t ind;
  Complex tmp;
  for (int i = threadIdx.x; i < size; i += blockDim.x)
  {
    ind = (size_t)blockIdx.y * gridDim.x * size + (size_t)blockIdx.x * size + i;
    tmp = d_in[ind];
    d_in[ind] = {tmp.x * scale, tmp.y * scale};
  }
}

void ScaleMatrix(Complex * const d_in, const unsigned int size, unsigned int num_cols, const unsigned int num_rows, const double scale)
{
  ScaleMatKernel<<<dim3(num_cols, num_rows, 1), std::min((int)size, MAX_BLOCK_SIZE)>>>(d_in, size, num_cols, num_rows, scale);
  gpuErrchk(cudaPeekAtLastError());
  #if ERR_CHK

  gpuErrchk(cudaDeviceSynchronize());
  #endif
}



/**
 * Kernel to compute the element-wise product of two complex vectors and scale the result by 1/size.
 * @param out The output vector
 * @param mat The input matrix
 * @param vec The input vector
 * @param size The size of the input vector
 * @param scale The constant to scale by
 *
 * @return void
 */
static __global__ void ComplexPointwiseMulAndScale(cufftDoubleComplex *const out, const cufftDoubleComplex *const mat,
                                                   const cufftDoubleComplex *const vec, const unsigned int size)
{
  Complex tmp1, tmp2;
  size_t ind;
  for (int i = threadIdx.x; i < size; i += blockDim.x)
  {
    ind = (size_t)blockIdx.y * gridDim.x * size + (size_t)blockIdx.x * size + i;
    tmp1 = mat[ind];
    tmp2 = vec[blockIdx.x * size + i];
    // printf("blockIdx.x %i blockIdx.y %i, i %i, tmp1.x %f, tmp1.y %f, tmp2.x %f, tmp2.y %f \n", blockIdx.x, blockIdx.y, i, tmp1.x, tmp1.y, tmp2.x, tmp2.y);
    out[ind] = {(tmp1.x * tmp2.x - tmp1.y * tmp2.y), (tmp1.x * tmp2.y + tmp1.y * tmp2.x)};
  }
}

/**
 * Kernel to compute the element-wise product of two complex vectors and scale the result by 1/size.
 * This version conjugates the first vector (used for F* matvec).
 * @param out The output vector
 * @param mat The input matrix
 * @param vec The input vector
 * @param size The size of the input vector
 * @param scale The constant to scale by
 *
 * @return void
 */
static __global__ void ComplexConjPointwiseMulAndScale(cufftDoubleComplex *const out, const cufftDoubleComplex *const mat,
                                                       const cufftDoubleComplex *const vec, const unsigned int size)
{
  Complex tmp1, tmp2;
  size_t ind;
  for (int i = threadIdx.x; i < size; i += blockDim.x)
  {
    ind = (size_t)blockIdx.x * size + (size_t)blockIdx.y * gridDim.x * size + i;
    tmp1 = mat[ind];
    tmp2 = vec[blockIdx.y * size + i];
    out[ind] = {(tmp1.x * tmp2.x + tmp1.y * tmp2.y), (tmp1.x * tmp2.y - tmp1.y * tmp2.x)};
  }
}