#include "matvec.cuh"
#include "utils.cuh"
#include "multiply.cuh"
#include "reduction.cuh"

#if TIME_MPI
enum_array<ProfilerTimesFull, profiler_t, 21> t_list;
enum_array<ProfilerTimes, profiler_t, 9> t_list_f;
enum_array<ProfilerTimes, profiler_t, 9> t_list_fs;
enum_array<ProfilerTimesNew, profiler_t, 10> t_list_f_new;
enum_array<ProfilerTimesNew, profiler_t, 10> t_list_fs_new;
#endif

void setup_new(Complex **d_mat_freq, const double *const h_mat, const unsigned int size, const unsigned int num_cols, const unsigned int num_rows)
{

  double *d_mat;
  cufftHandle forward_plan_mat;
  const size_t mat_len = (size_t)size * num_cols * num_rows * sizeof(double);

  fft_int_t n[1] = {(fft_int_t)size};
  int rank = 1;

  fft_int_t idist = size;
  fft_int_t odist = (size / 2 + 1);

  fft_int_t inembed[] = {0};
  fft_int_t onembed[] = {0};

  fft_int_t istride = 1;
  fft_int_t ostride = 1;

  // unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
  // unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;
#if !FFT_64
#if !ROW_SETUP
  cufftSafeCall(cufftPlanMany(&forward_plan_mat, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols * num_rows));
#else
  cufftSafeCall(cufftPlanMany(&forward_plan_mat, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols));
#endif
#else
  size_t ws = 0;
  cufftSafeCall(cufftCreate(&forward_plan_mat));
  cufftSafeCall(cufftMakePlanMany64(forward_plan_mat, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols * num_rows, &ws));
#endif
  gpuErrchk(cudaMalloc((void **)&d_mat, mat_len));
  gpuErrchk(cudaMemcpy(d_mat, h_mat, mat_len, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void **)d_mat_freq, (size_t)(size / 2 + 1) * num_cols * num_rows * sizeof(Complex)));

#if !ROW_SETUP
  cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat, *d_mat_freq));
#else
  for (int i = 0; i < num_rows; i++)
  {
    cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat + (size_t)i * size * num_cols, *d_mat_freq + (size_t)i * num_cols * (size / 2 + 1)));
  }
#endif

  cufftSafeCall(cufftDestroy(forward_plan_mat));
  gpuErrchk(cudaFree(d_mat));

  ScaleMatrix(*d_mat_freq, size / 2 + 1, num_cols, num_rows, 1.0 / size);

  Complex *d_mat_freq_trans;
  gpuErrchk(cudaMalloc((void **)&d_mat_freq_trans, sizeof(Complex) * (size_t)(size / 2 + 1) * num_cols * num_rows));
  if (num_cols > 1 && num_rows > 1)
  {
    int sz[3] = {(int)(size / 2 + 1), (int)num_cols, (int)num_rows};
    int perm[3] = {2, 1, 0};
    int elements_per_thread = 4;

    if (cut_transpose3d(d_mat_freq_trans,
                        *d_mat_freq,
                        sz,
                        perm,
                        elements_per_thread) < 0)
    {
      fprintf(stderr, "Error while performing transpose.\n");
      exit(1);
    }
  }
  else
  {
    cublasHandle_t cublasHandle;
    cublasSafeCall(cublasCreate(&cublasHandle));
    cuDoubleComplex aa({1, 0});
    cuDoubleComplex bb({0, 0});
    if (num_rows == 1)
    {
      // transpose2d(d_mat_freq_trans, *d_mat_freq, (size / 2 + 1), num_cols, NULL);
#if !FFT_64
      cublasSafeCall(cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_cols, (size / 2 + 1), &aa, *d_mat_freq, (size / 2 + 1), &bb, NULL, num_cols, d_mat_freq_trans, num_cols));
#else
      cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_cols, (size / 2 + 1), &aa, *d_mat_freq, (size / 2 + 1), &bb, NULL, num_cols, d_mat_freq_trans, num_cols));
#endif
    }
    else
    {
      // transpose2d(d_mat_freq_trans, *d_mat_freq, (size / 2 + 1), num_rows, NULL);
#if !FFT_64
      cublasSafeCall(cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_rows, (size / 2 + 1), &aa, *d_mat_freq, (size / 2 + 1), &bb, NULL, num_rows, d_mat_freq_trans, num_rows));
#else
      cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_rows, (size / 2 + 1), &aa, *d_mat_freq, (size / 2 + 1), &bb, NULL, num_rows, d_mat_freq_trans, num_rows));
#endif
    }
    cublasSafeCall(cublasDestroy(cublasHandle));
  }

  gpuErrchk(cudaFree(*d_mat_freq));
  *d_mat_freq = d_mat_freq_trans;

  // printf("d_mat_freq \n");
  // PrintComplex<<<1,1>>>(d_mat_freq, num_cols, size/2+1, num_rows);
  // cudaDeviceSynchronize();
}

/**
 * This function computes the FFT of the input matrix and stores the result on the device.
 * The input matrix is assumed to be of size num_cols * 2 * size * num_rows.
 *
 * @param d_mat_freq The output FFT matrix (on device).
 * @param h_mat The input matrix (on host).
 * @param size The size of each vector in the input matrix.
 * @param num_cols The number of columns in the input matrix.
 * @param num_rows The number of rows in the input matrix.
 *
 * @return void.
 *
 */

// const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool full, const unsigned int color, const unsigned int color2, double ** d_in, double ** d_in2, double ** res, double ** res2, double ** d_in_pad, cufftHandle * forward_plan, cufftHandle * inverse_plan, cufftHandle * forward_plan_conj, cufftHandle * inverse_plan_conj, double ** d_out, double ** d_out_conj, Complex ** d_freq, Complex ** d_freq_conj, Complex ** d_out_freq
void setup(Complex **d_mat_freq, const double *const h_mat, const unsigned int size, const unsigned int num_cols, const unsigned int num_rows)
{

  double *d_mat;
  cufftHandle forward_plan_mat;
  const size_t mat_len = (size_t)size * num_cols * num_rows * sizeof(double);

  fft_int_t n[1] = {(fft_int_t)size};
  int rank = 1;

  fft_int_t idist = size;
  fft_int_t odist = (size / 2 + 1);

  fft_int_t inembed[] = {0};
  fft_int_t onembed[] = {0};

  fft_int_t istride = 1;
  fft_int_t ostride = 1;

  // unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
  // unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;
#if !FFT_64
#if !ROW_SETUP
  cufftSafeCall(cufftPlanMany(&forward_plan_mat, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols * num_rows));
#else
  cufftSafeCall(cufftPlanMany(&forward_plan_mat, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols));
#endif
#else
  size_t ws = 0;
  cufftSafeCall(cufftCreate(&forward_plan_mat));
  cufftSafeCall(cufftMakePlanMany64(forward_plan_mat, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols * num_rows, &ws));
#endif
  gpuErrchk(cudaMalloc((void **)&d_mat, mat_len));
  gpuErrchk(cudaMemcpy(d_mat, h_mat, mat_len, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void **)d_mat_freq, (size_t)(size / 2 + 1) * num_cols * num_rows * sizeof(Complex)));
#if !ROW_SETUP
  cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat, *d_mat_freq));
#else
  for (int i = 0; i < num_rows; i++)
  {
    cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat + (size_t)i * size * num_cols, *d_mat_freq + (size_t)i * num_cols * (size / 2 + 1)));
  }
#endif

  // cufftSafeCall(cufftPlanMany(forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_in_len));
  // cufftSafeCall(cufftPlanMany(inverse_plan, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_out_len));// num_cols * num_rows));

  // gpuErrchk(cudaMalloc((void **)d_freq, sizeof(Complex) * (size / 2 + 1) * vec_in_len));
  // gpuErrchk(cudaMalloc((void **)d_out_freq, (size_t)sizeof(Complex) * (size / 2 + 1) * num_cols * num_rows));
  // gpuErrchk(cudaMalloc((void **)d_out, (size_t)sizeof(double) * size * vec_out_len));// num_cols * num_rows));
  // gpuErrchk(cudaMalloc((void **)d_in_pad, vec_in_len * size * sizeof(double)));

  // if (full){
  //   cufftSafeCall(cufftPlanMany(forward_plan_conj, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_out_len));
  //   cufftSafeCall(cufftPlanMany(inverse_plan_conj, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_in_len));
  //   gpuErrchk(cudaMalloc((void **)d_freq_conj, sizeof(Complex) * (size / 2 + 1) * vec_out_len));
  //   gpuErrchk(cudaMalloc((void **)d_out_conj, (size_t)sizeof(double) * size * vec_in_len));
  // }

  // if (color != 0)
  // {
  //   gpuErrchk(cudaMalloc((void **)d_in, vec_in_len * size / 2 * sizeof(double)));
  // }

  // if (color2 != 0 && full)
  // {
  //   gpuErrchk(cudaMalloc((void **)d_in2, vec_out_len * size / 2 * sizeof(double)));
  // }

  // if (color2 != 0) // if full for now
  // {
  //   gpuErrchk(cudaMalloc((void **)res, vec_out_len * size / 2 * sizeof(double)));
  // }

  // if (color != 0 && full)
  // {
  //   gpuErrchk(cudaMalloc((void **)res2, vec_in_len * size / 2 * sizeof(double)));
  // }

  cufftSafeCall(cufftDestroy(forward_plan_mat));
  gpuErrchk(cudaFree(d_mat));

  ScaleMatrix(*d_mat_freq, size / 2 + 1, num_cols, num_rows, 1.0 / size);

  // printf("d_mat_freq \n");
  // PrintComplex<<<1,1>>>(d_mat_freq, num_cols, size/2+1, num_rows);
  // cudaDeviceSynchronize();
}

/**
 * This function computes the FFT matvec of the input matrix and vector and stores the result on the device.
 *
 * @param r The output vector (on device).
 * @param d_in The input vector (on device).
 * @param d_mat_freq The input FFT matrix (on device).
 * @param size The size of each vector in the input matrix.
 * @param num_cols The number of columns in the input matrix.
 * @param num_rows The number of rows in the input matrix.
 * @param conjugate Whether to conjugate the input matrix (for F* matvecs).
 * @param unpad Whether to unpad the output vector.
 *
 * @return void.
 */
void fft_matvec(double *const r, double *const d_in, const Complex *const d_mat_freq, const unsigned int size, const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool unpad, const unsigned int device, cufftHandle forward_plan, cufftHandle inverse_plan, double *const d_out, Complex *const d_freq, Complex *const d_out_freq, Complex *const d_red_freq, cudaStream_t s)
{
  // cufftHandle forward_plan, inverse_plan;
  // int n[1] = {(int)size};
  // int rank = 1;

  // int idist = size;
  // int odist = (size / 2 + 1);

  // int inembed[] = {0};
  // int onembed[] = {0};

  // int istride = 1;
  // int ostride = 1;

  unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
  unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;

  // double *d_out;

  // Complex *d_freq;
  // Complex *d_out_freq;
  // gpuErrchk(cudaMalloc((void **)&d_freq, sizeof(Complex) * (size / 2 + 1) * vec_in_len));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::FFTFS].start();
  // else
  //   t_list[ProfilerTimes::FFT].start();

  enum_array<ProfilerTimes, profiler_t, 9> *tl = (conjugate) ? &t_list_fs : &t_list_f;

  (*tl)[ProfilerTimes::FFT].start();
#endif

  // double * h_in;
  // h_in = (double *)malloc(sizeof(double) * size * vec_in_len);
  // gpuErrchk(cudaMemcpy(h_in, d_in, sizeof(double) * size * vec_in_len, cudaMemcpyDeviceToHost));

  // for (unsigned int i = 0; i < vec_in_len; i++)
  // {
  //   for (unsigned int j = 0; j < size; j++)
  //   {
  //     printf("%f ", h_in[i * size + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // cufftSafeCall(cufftPlanMany(&forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_in_len));
  cufftSafeCall(cufftExecD2Z(forward_plan, d_in, d_freq));
#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  // if(conjugate)
  //   t_list[ProfilerTimes::FFTFS].stop();
  // else
  //   t_list[ProfilerTimes::FFTF].stop();
  (*tl)[ProfilerTimes::FFT].stop();
#endif
  // Complex * h_freq;
  // h_freq = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * vec_in_len);
  // gpuErrchk(cudaMemcpy(h_freq, d_freq, sizeof(Complex) * (size / 2 + 1) * vec_in_len, cudaMemcpyDeviceToHost));

  // for (int i = 0; i < vec_in_len; i++)
  // {
  //   for (int j = 0; j < size / 2 + 1; j++)
  //   {
  //     printf("%f + %f i\t", h_freq[i * (size / 2 + 1) + j].x, h_freq[i * (size / 2 + 1) + j].y);
  //   }
  //   printf("\n");
  // }

  // Complex *h_freq;
  // h_freq = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * vec_in_len);
  // gpuErrchk(cudaMemcpy(h_freq, d_freq, sizeof(Complex) * (size / 2 + 1) * vec_in_len, cudaMemcpyDeviceToHost));

  // printf("d_freq \n");
  // for (int i = 0; i < vec_in_len; i++)
  // {
  //   for (int j = 0; j < size / 2 + 1; j++)
  //   {
  //     printf("block %d: size: %d, %f + %f i\n", i, j, h_freq[i * (size / 2 + 1) + j].x, h_freq[i * (size / 2 + 1) + j].y);
  //   }
  // }

  // Complex *h_mat_freq;
  // h_mat_freq = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * num_cols * num_rows);
  // gpuErrchk(cudaMemcpy(h_mat_freq, d_mat_freq, sizeof(Complex) * (size / 2 + 1) * num_cols * num_rows, cudaMemcpyDeviceToHost));

  // printf("d_mat_freq \n");
  // for (int i = 0; i < num_rows; i++)
  // {
  //   for (int j = 0; j < num_cols; j++)
  //   {
  //     for (int k = 0; k < (size / 2 + 1); k++)
  //     {
  //       printf("row %d: col: %d size: %d, %f + %f i\n", i, j, k, h_mat_freq[i * num_cols * (size / 2 + 1) + j * (size / 2 + 1) + k].x, h_mat_freq[i * num_cols * (size / 2 + 1) + j * (size / 2 + 1) + k].y);
  //     }
  //   }
  // }

  // cufftSafeCall(cufftDestroy(forward_plan));

  // printf("d_freq \n");
  // PrintComplex<<<1,1>>>(d_freq, num_cols, size/2+1, 1);
  // cudaDeviceSynchronize();
  // gpuErrchk(cudaMalloc((void **)&d_out_freq, (size_t)sizeof(Complex) * (size / 2 + 1) * num_cols * num_rows));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::EWPFS].start();
  // else
  //   t_list[ProfilerTimes::EWP].start();
  (*tl)[ProfilerTimes::EWP].start();
#endif
  multiplyCoefficient(d_out_freq, d_mat_freq, d_freq, size, num_cols, num_rows, conjugate, s);
#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  // if(conjugate)
  //   t_list[ProfilerTimes::EWPFS].stop();
  // else{
  //   t_list[ProfilerTimes::EWP].stop();
  // }
  (*tl)[ProfilerTimes::EWP].stop();
#endif
  // gpuErrchk(cudaFree(d_freq));

  // Complex *d_red_freq;
  // gpuErrchk(cudaMalloc((void **)&d_red_freq, (size_t)sizeof(Complex) * (size / 2 + 1) * vec_out_len));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::REDFS].start();
  // else
  //   t_list[ProfilerTimes::RED].start();

  (*tl)[ProfilerTimes::REDN].start();
#endif

  ComplexReduction(d_red_freq, d_out_freq, vec_in_len, size / 2 + 1, vec_out_len, conjugate, device, s);
#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::REDFS].stop();
  // else
  //   t_list[ProfilerTimes::RED].stop();

  (*tl)[ProfilerTimes::REDN].stop();
#endif
  // Complex *h_freq2;
  // h_freq2 = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * vec_out_len);
  // gpuErrchk(cudaMemcpy(h_freq2, d_red_freq, sizeof(Complex) * (size / 2 + 1) * vec_out_len, cudaMemcpyDeviceToHost));

  // printf("d_red_freq \n");
  // for (int i = 0; i < vec_out_len; i++)
  // {
  //   printf("block %d: \n", i);
  //   for (int j = 0; j < size / 2 + 1; j++)
  //   {
  //     printf("%d: %f + %f i\t", j, h_freq2[i * (size / 2 + 1) + j].x, h_freq2[i * (size / 2 + 1) + j].y);
  //   }
  //   printf("\n");
  // }

  // double * h_out_freq;
  // h_out_freq = (double *)malloc(sizeof(double) * (size / 2 + 1) * num_cols * num_rows);
  // gpuErrchk(cudaMemcpy(h_out_freq, d_out_freq, sizeof(double) * (size / 2 + 1) * num_cols * num_rows, cudaMemcpyDeviceToHost));
  // for (int i = 0; i < num_cols * num_rows; i++)
  // {
  //   for (int j = 0; j < size / 2 + 1; j++)
  //   {
  //     printf("%f ", h_out_freq[i * (size / 2 + 1) + j]);
  //   }
  //   printf("\n");
  // }
  // free(h_out_freq);

  // printf("d_out_freq \n");
  // PrintComplex<<<1,1>>>(d_out_freq, num_cols, size/2+1, num_rows);
  // cudaDeviceSynchronize();
  // gpuErrchk(cudaMalloc((void **)&d_out, (size_t)sizeof(double) * size * vec_out_len));// num_cols * num_rows));
#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::IFFTFS].start();
  // else
  //   t_list[ProfilerTimes::IFFT].start();

  (*tl)[ProfilerTimes::IFFT].start();
#endif
  // cufftSafeCall(cufftPlanMany(&inverse_plan, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_out_len));// num_cols * num_rows));
  cufftSafeCall(cufftExecZ2D(inverse_plan, d_red_freq, d_out));
#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  // if(conjugate)
  //   t_list[ProfilerTimes::IFFTFS].stop();
  // else
  //   t_list[ProfilerTimes::IFFT].stop();

  (*tl)[ProfilerTimes::IFFT].stop();
#endif
  // cufftSafeCall(cufftDestroy(inverse_plan));
  // gpuErrchk(cudaFree(d_red_freq));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::UNPADFS].start();
  // else
  //   t_list[ProfilerTimes::UNPAD].start();

  (*tl)[ProfilerTimes::UNPAD].start();
#endif

  // print d_out
  // double * h_out = (double *)malloc(sizeof(double) * size * vec_out_len);
  // gpuErrchk(cudaMemcpy(h_out, d_out, sizeof(double) * size * vec_out_len, cudaMemcpyDeviceToHost));

  // for(int i = 0; i < vec_out_len; i++){
  //     printf("block %d: \n", i);
  //       for (int t = 0; t< size; t++)
  //         printf("%f ", h_out[(size_t)(i * size + t)]);
  //       printf("\n");
  //   }
  // free(h_out);

  UnpadRepadVector(d_out, r, vec_out_len, size, unpad, s);

  // double *h_r = (double *)malloc(sizeof(double) * vec_out_len * size / 2);
  // gpuErrchk(cudaMemcpy(h_r, r, sizeof(double) * vec_out_len * size / 2, cudaMemcpyDeviceToHost));
  // for (int i = 0; i < vec_out_len; i++)
  // {
  //   for (int j = 0; j < size / 2; j++)
  //   {
  //     printf("unpad block: %d, %d %f \n",i, j, h_r[(size_t)i * size / 2 + j]);
  //   }
  //   printf("\n");
  // }

  // Reduction(r, d_out, vec_in_len, size, vec_out_len, unpad, conjugate, device);
  // gpuErrchk(cudaDeviceSynchronize());

  // double *h_r = (double *)malloc(sizeof(double) * vec_out_len * size / 2);
  // gpuErrchk(cudaMemcpy(h_r, r, sizeof(double) * vec_out_len * size / 2, cudaMemcpyDeviceToHost));
  // for (int i = 0; i < vec_out_len; i++)
  // {
  //   for (int j = 0; j < size / 2; j++)
  //   {
  //     printf("%f ", h_r[(size_t)i * size / 2 + j]);
  //   }
  //   printf("\n");
  // }
  // free(h_r);

#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());

  // if(conjugate)
  //   t_list[ProfilerTimes::UNPADFS].stop();
  // else
  //   t_list[ProfilerTimes::UNPAD].stop();

  (*tl)[ProfilerTimes::UNPAD].stop();
#endif
  // gpuErrchk(cudaFree(d_out));

  // for (int j = 0; j < vec_out_len; j++)
  //     gpuErrchk(cudaMemcpy(h_out + (size_t)j*size/2, d_out + (size_t)j*vec_in_len*size, sizeof(double) * size / 2, cudaMemcpyDeviceToHost));
}

void fft_matvec_new(double *const r, double *const d_in, const Complex *const d_mat_freq, const unsigned int size, const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool unpad, const unsigned int device, cufftHandle forward_plan, cufftHandle inverse_plan, double *const d_out, Complex *const d_freq, Complex *const d_red_freq, Complex *const d_freq_t, Complex *const d_red_freq_t, cudaStream_t s, cublasHandle_t cublasHandle)
{
  // cufftHandle forward_plan, inverse_plan;
  // int n[1] = {(int)size};
  // int rank = 1;

  // int idist = size;
  // int odist = (size / 2 + 1);

  // int inembed[] = {0};
  // int onembed[] = {0};

  // int istride = 1;
  // int ostride = 1;

  unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
  unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;

  // double *d_out;

  // Complex *d_freq;
  // Complex *d_out_freq;
  // gpuErrchk(cudaMalloc((void **)&d_freq, sizeof(Complex) * (size / 2 + 1) * vec_in_len));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::FFTFS].start();
  // else
  //   t_list[ProfilerTimes::FFT].start();

  enum_array<ProfilerTimesNew, profiler_t, 10> *tl = (conjugate) ? &t_list_fs_new : &t_list_f_new;

  (*tl)[ProfilerTimesNew::FFT].start();
#endif

  // double * h_in;
  // h_in = (double *)malloc(sizeof(double) * size * vec_in_len);
  // gpuErrchk(cudaMemcpy(h_in, d_in, sizeof(double) * size * vec_in_len, cudaMemcpyDeviceToHost));

  // for (unsigned int i = 0; i < vec_in_len; i++)
  // {
  //   for (unsigned int j = 0; j < size; j++)
  //   {
  //     printf("%f ", h_in[i * size + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // cufftSafeCall(cufftPlanMany(&forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_in_len));
  cufftSafeCall(cufftExecD2Z(forward_plan, d_in, d_freq));
#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  // if(conjugate)
  //   t_list[ProfilerTimes::FFTFS].stop();
  // else
  //   t_list[ProfilerTimes::FFTF].stop();
  (*tl)[ProfilerTimesNew::FFT].stop();
#endif
  // Complex *h_freq;
  // h_freq = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * vec_in_len);
  // gpuErrchk(cudaMemcpy(h_freq, d_freq, sizeof(Complex) * (size / 2 + 1) * vec_in_len, cudaMemcpyDeviceToHost));

  //   printf("d_freq \n");
  // Complex vec1 = {0,0}, vec2 = {0,0}, vec3 = {0,0};
  // for (int i = 0; i < vec_in_len; i++)
  // {
  //   for (int j = 0; j < size / 2 + 1; j++)
  //   {
  //     printf("block %d: size: %d, %f + %f i\n", i, j, h_freq[i * (size / 2 + 1) + j].x, h_freq[i * (size / 2 + 1) + j].y);
  //     if (j==0)
  //     {
  //       vec1.x += h_freq[i * (size / 2 + 1) + j].x;
  //       vec1.y += h_freq[i * (size / 2 + 1) + j].y;
  //     }
  //     if (j==1)
  //     {
  //       vec2.x += h_freq[i * (size / 2 + 1) + j].x;
  //       vec2.y += h_freq[i * (size / 2 + 1) + j].y;
  //     }
  //     if (j==2)
  //     {
  //       vec3.x += h_freq[i * (size / 2 + 1) + j].x;
  //       vec3.y += h_freq[i * (size / 2 + 1) + j].y;
  //     }
  //   }
  // }
  // printf("vec1: %f + %f i\n", vec1.x, vec1.y);
  // printf("vec2: %f + %f i\n", vec2.x, vec2.y);
  // printf("vec3: %f + %f i\n", vec3.x, vec3.y);

  // printf("d_freq \n");
  // for (int i = 0; i < vec_in_len * (size / 2 + 1); i++)
  // {
  //   printf("%d: %f + %f i\t", i, h_freq[i].x, h_freq[i].y);
  // }

  // cufftSafeCall(cufftDestroy(forward_plan));

  // printf("d_freq \n");
  // PrintComplex<<<1,1>>>(d_freq, num_cols, size/2+1, 1);
  // cudaDeviceSynchronize();
  // gpuErrchk(cudaMalloc((void **)&d_out_freq, (size_t)sizeof(Complex) * (size / 2 + 1) * num_cols * num_rows));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::EWPFS].start();
  // else
  //   t_list[ProfilerTimes::EWP].start();
  (*tl)[ProfilerTimesNew::TRANS1].start();
#endif

  cuDoubleComplex alpha({1, 0});
  cuDoubleComplex beta({0, 0});
  // multiplyCoefficient(d_out_freq, d_mat_freq, d_freq, size, num_cols, num_rows, conjugate, s);

  // transpose2d(d_freq_t, d_freq, (size / 2 + 1), vec_in_len, s);
#if !FFT_64
  cublasSafeCall(cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, vec_in_len, (size / 2 + 1), &alpha, d_freq, (size / 2 + 1), &beta, NULL, vec_in_len, d_freq_t, vec_in_len));
#else
  cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, vec_in_len, (size / 2 + 1), &alpha, d_freq, (size / 2 + 1), &beta, NULL, vec_in_len, d_freq_t, vec_in_len));
#endif

  // Complex *h_freq;
  // h_freq = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * vec_in_len);
  // gpuErrchk(cudaMemcpy(h_freq, d_freq_t, sizeof(Complex) * (size / 2 + 1) * vec_in_len, cudaMemcpyDeviceToHost));

  // printf("d_freq transpose \n");
  // for (int i = 0; i < (size / 2 + 1); i++)
  // {
  //   for(int j = 0; j < vec_in_len; j++){
  //     printf("size: %d block %d: %f + %f i\n", i, j, h_freq[i * vec_in_len + j].x, h_freq[i * vec_in_len + j].y);
  //   }
  // }

#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  // if(conjugate)
  //   t_list[ProfilerTimes::EWPFS].stop();
  // else{
  //   t_list[ProfilerTimes::EWP].stop();
  // }
  (*tl)[ProfilerTimesNew::TRANS1].stop();
#endif
  // gpuErrchk(cudaFree(d_freq));

  // Complex *d_red_freq;
  // gpuErrchk(cudaMalloc((void **)&d_red_freq, (size_t)sizeof(Complex) * (size / 2 + 1) * vec_out_len));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::REDFS].start();
  // else
  //   t_list[ProfilerTimes::RED].start();

  (*tl)[ProfilerTimesNew::SBGEMV].start();
#endif

  // Complex *h_mat_freq;
  // h_mat_freq = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * num_cols * num_rows);
  // gpuErrchk(cudaMemcpy(h_mat_freq, d_mat_freq, sizeof(Complex) * (size / 2 + 1) * num_cols * num_rows, cudaMemcpyDeviceToHost));

  // printf("d_mat_freq \n");
  // for (int i = 0; i < (size / 2 + 1); i++)
  // {
  //   for (int j = 0; j < num_rows; j++)
  //   {
  //     for (int k = 0; k < num_cols; k++)
  //     {
  //       printf("size %d: row: %d col: %d,  %f + %f i\n", i, j, k, h_mat_freq[i * num_cols * num_rows + j * num_cols + k].x, h_mat_freq[i * num_cols * num_rows + j * num_cols + k].y);
  //     }
  //   }
  // }

  // Complex sum, mat, vec;
  // for (int i = 0; i < (size / 2 + 1); i++)
  // {
  //   sum = {0,0};
  //   mat = {0,0};
  //   vec = {0,0};
  //   for (int j = 0; j < num_rows; j++)
  //   {
  //     for (int k = 0; k < num_cols; k++)
  //     {
  //       sum.x += h_mat_freq[i * num_cols * num_rows + j * num_cols + k].x * h_freq[i * vec_in_len + k].x - h_mat_freq[i * num_cols * num_rows + j * num_cols + k].y * h_freq[i * vec_in_len + k].y;
  //       sum.y += h_mat_freq[i * num_cols * num_rows + j * num_cols + k].x * h_freq[i * vec_in_len + k].y + h_mat_freq[i * num_cols * num_rows + j * num_cols + k].y * h_freq[i * vec_in_len + k].x;
  //       mat.x += h_mat_freq[i * num_cols * num_rows + j * num_cols + k].x;
  //       mat.y += h_mat_freq[i * num_cols * num_rows + j * num_cols + k].y;
  //       vec.x += h_freq[i * vec_in_len + k].x;
  //       vec.y += h_freq[i * vec_in_len + k].y;
  //     }
  //   }
  //   printf("size %d: %f + %f i\n", i, sum.x, sum.y);
  //   printf("size %d: mat %f + %f i\n", i, mat.x, mat.y);
  //   printf("size %d: vec %f + %f i\n", i, vec.x, vec.y);
  // }

  // ComplexReduction(d_red_freq, d_out_freq, vec_in_len, size / 2 + 1, vec_out_len, conjugate, device, s);

  cublasOperation_t transa = (conjugate) ? CUBLAS_OP_C : CUBLAS_OP_N;
  // printf("here");

  // printf("num_rows: %d\n, num_cols: %d\n, alpha: %f + %f i\n, num_rows: %d\n, num_rows * num_cols: %d\n, vec_in_len: %d\n, beta: %f + %f i\n, vec_out_len: %d\n, size/2+1: %d\n", num_rows, num_cols, alpha->x, alpha->y, num_rows, num_rows * num_cols, vec_in_len, beta->x, beta->y, vec_out_len, (size/2+1));
#if !FFT_64
  cublasSafeCall(cublasZgemvStridedBatched(cublasHandle, transa, num_rows, num_cols, &alpha, d_mat_freq, num_rows, (size_t)num_rows * num_cols, d_freq_t, 1, vec_in_len, &beta, d_red_freq, 1, vec_out_len, (size / 2 + 1)));
#else
  cublasSafeCall(cublasZgemvStridedBatched_64(cublasHandle, transa, num_rows, num_cols, &alpha, d_mat_freq, num_rows, (size_t)num_rows * num_cols, d_freq_t, 1, vec_in_len, &beta, d_red_freq, 1, vec_out_len, (size / 2 + 1)));
#endif

  // Complex *h_freq2;
  // h_freq2 = (Complex *)malloc(sizeof(Complex) * (size / 2 + 1) * vec_out_len);
  // gpuErrchk(cudaMemcpy(h_freq2, d_red_freq, sizeof(Complex) * (size / 2 + 1) * vec_out_len, cudaMemcpyDeviceToHost));

  // printf("d_red_freq \n");
  // for (int i = 0; i < (size / 2 + 1); i++)
  // {
  //   for (int j = 0; j < vec_out_len; j++)
  //   {
  //     printf("%d: %f + %f i\t", i, h_freq2[i * vec_out_len + j].x, h_freq2[i * vec_out_len + j].y);
  //   }
  // }

  // printf("\n");

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::REDFS].stop();
  // else
  //   t_list[ProfilerTimes::RED].stop();
  gpuErrchk(cudaDeviceSynchronize());

  (*tl)[ProfilerTimesNew::SBGEMV].stop();
#endif
  // double * h_out_freq;
  // h_out_freq = (double *)malloc(sizeof(double) * (size / 2 + 1) * num_cols * num_rows);
  // gpuErrchk(cudaMemcpy(h_out_freq, d_out_freq, sizeof(double) * (size / 2 + 1) * num_cols * num_rows, cudaMemcpyDeviceToHost));
  // for (int i = 0; i < num_cols * num_rows; i++)
  // {
  //   for (int j = 0; j < size / 2 + 1; j++)
  //   {
  //     printf("%f ", h_out_freq[i * (size / 2 + 1) + j]);
  //   }
  //   printf("\n");
  // }
  // free(h_out_freq);

  // printf("d_out_freq \n");
  // PrintComplex<<<1,1>>>(d_out_freq, num_cols, size/2+1, num_rows);
  // cudaDeviceSynchronize();
  // gpuErrchk(cudaMalloc((void **)&d_out, (size_t)sizeof(double) * size * vec_out_len));// num_cols * num_rows));

#if TIME_MPI
  (*tl)[ProfilerTimesNew::TRANS2].start();
#endif

  // transpose2d(d_red_freq_t, d_red_freq, vec_out_len, (size / 2 + 1), s);

#if !FFT_64
  cublasSafeCall(cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (size / 2 + 1), vec_out_len, &alpha, d_red_freq, vec_out_len, &beta, NULL, (size / 2 + 1), d_red_freq_t, (size / 2 + 1)));
#else
  cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (size / 2 + 1), vec_out_len, &alpha, d_red_freq, vec_out_len, &beta, NULL, (size / 2 + 1), d_red_freq_t, (size / 2 + 1)));
#endif

#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  (*tl)[ProfilerTimesNew::TRANS2].stop();
#endif

  // gpuErrchk(cudaMemcpy(h_freq2, d_red_freq, sizeof(Complex) * (size / 2 + 1) * vec_out_len, cudaMemcpyDeviceToHost));

  // printf("d_red_freq transpose \n");
  // for (int i=0; i<vec_out_len*(size/2+1); i++){
  //   printf("%d: %f + %f i\n", i, h_freq2[i].x, h_freq2[i].y);
  // }
#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::IFFTFS].start();
  // else
  //   t_list[ProfilerTimes::IFFT].start();

  (*tl)[ProfilerTimesNew::IFFT].start();
#endif
  // cufftSafeCall(cufftPlanMany(&inverse_plan, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_out_len));// num_cols * num_rows));
  cufftSafeCall(cufftExecZ2D(inverse_plan, d_red_freq_t, d_out));
#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  // if(conjugate)
  //   t_list[ProfilerTimes::IFFTFS].stop();
  // else
  //   t_list[ProfilerTimes::IFFT].stop();

  (*tl)[ProfilerTimesNew::IFFT].stop();
#endif
  // cufftSafeCall(cufftDestroy(inverse_plan));
  // gpuErrchk(cudaFree(d_red_freq));

#if TIME_MPI
  // if(conjugate)
  //   t_list[ProfilerTimes::UNPADFS].start();
  // else
  //   t_list[ProfilerTimes::UNPAD].start();

  (*tl)[ProfilerTimesNew::UNPAD].start();
#endif

  // print d_out
  // double * h_out = (double *)malloc(sizeof(double) * size * vec_out_len);
  // gpuErrchk(cudaMemcpy(h_out, d_out, sizeof(double) * size * vec_out_len, cudaMemcpyDeviceToHost));

  // for(int i = 0; i < vec_out_len; i++){
  //     printf("block %d: \n", i);
  //       for (int t = 0; t< size; t++)
  //         printf("%f ", h_out[(size_t)(i * size + t)]);
  //       printf("\n");
  //   }
  // free(h_out);

  UnpadRepadVector(d_out, r, vec_out_len, size, unpad, s);

  // double *h_r = (double *)malloc(sizeof(double) * vec_out_len * size / 2);
  // gpuErrchk(cudaMemcpy(h_r, r, sizeof(double) * vec_out_len * size / 2, cudaMemcpyDeviceToHost));
  // for (int i = 0; i < vec_out_len; i++)
  // {
  //   for (int j = 0; j < size / 2; j++)
  //   {
  //     printf("unpad block: %d, %d %f \n",i, j, h_r[(size_t)i * size / 2 + j]);
  //   }
  //   printf("\n");
  // }

  // Reduction(r, d_out, vec_in_len, size, vec_out_len, unpad, conjugate, device);
  // gpuErrchk(cudaDeviceSynchronize());

  // double *h_r = (double *)malloc(sizeof(double) * vec_out_len * size / 2);
  // gpuErrchk(cudaMemcpy(h_r, r, sizeof(double) * vec_out_len * size / 2, cudaMemcpyDeviceToHost));
  // for (int i = 0; i < vec_out_len; i++)
  // {
  //   for (int j = 0; j < size / 2; j++)
  //   {
  //     printf("%f ", h_r[(size_t)i * size / 2 + j]);
  //   }
  //   printf("\n");
  // }
  // free(h_r);

#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());

  // if(conjugate)
  //   t_list[ProfilerTimes::UNPADFS].stop();
  // else
  //   t_list[ProfilerTimes::UNPAD].stop();

  (*tl)[ProfilerTimesNew::UNPAD].stop();
#endif
  // gpuErrchk(cudaFree(d_out));

  // for (int j = 0; j < vec_out_len; j++)
  //     gpuErrchk(cudaMemcpy(h_out + (size_t)j*size/2, d_out + (size_t)j*vec_in_len*size, sizeof(double) * size / 2, cudaMemcpyDeviceToHost));
}

void init_comms(MPI_Comm world_comm, MPI_Comm *row_comm, MPI_Comm *col_comm, Comm_t *nccl_row_comm, Comm_t *nccl_col_comm, cudaStream_t *s, int *device, int proc_rows, int proc_cols, bool conjugate)
{
  /**
    @brief Initialize MPI and NCCL communicators for the row and column groups.
    @param world_comm MPI communicator for all processes.
    @param row_comm MPI communicator for the row group.
    @param col_comm MPI communicator for the column group.
    @param nccl_row_comm NCCL communicator for the row group.
    @param nccl_col_comm NCCL communicator for the column group.
    @param s CUDA stream for the current process.
    @param device CUDA device ID for the current process.
    @param proc_rows Number of processes in the row group.
    @param proc_cols Number of processes in the column group.
    @param conjugate F/F* matvec
  */
  int world_rank, nRanks, localRank = 0;
  MPICHECK(MPI_Comm_rank(world_comm, &world_rank));
  MPICHECK(MPI_Comm_size(world_comm, &nRanks));

  int row_color = (!conjugate) ? world_rank % proc_rows : world_rank / proc_rows;

  MPI_Comm_split(world_comm, row_color, world_rank, row_comm);

  int row_group_rank, row_group_size;
  MPI_Comm_rank(*row_comm, &row_group_rank);
  MPI_Comm_size(*row_comm, &row_group_size);

  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[world_rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, world_comm));
  for (int p = 0; p < nRanks; p++)
  {
    if (p == world_rank)
      break;
    if (hostHashs[p] == hostHashs[world_rank])
      localRank++;
  }

  ncclUniqueId row_id;

  if (row_group_rank == 0)
    ncclGetUniqueId(&row_id);

  MPICHECK(MPI_Bcast((void *)&row_id, sizeof(row_id), MPI_BYTE, 0, *row_comm));

  // picking a GPU based on localRank, make stream
  *device = localRank;
  gpuErrchk(cudaSetDevice(localRank));
  gpuErrchk(cudaStreamCreate(s));

  ncclCommInitRank(nccl_row_comm, row_group_size, row_id, row_group_rank);

  int col_color = (!conjugate) ? world_rank / proc_rows : world_rank % proc_rows;

  int col_group_rank, col_group_size;
  ncclUniqueId col_id;

  MPI_Comm_split(world_comm, col_color, world_rank, col_comm);

  MPI_Comm_rank(*col_comm, &col_group_rank);
  MPI_Comm_size(*col_comm, &col_group_size);

  if (col_group_rank == 0)
    ncclGetUniqueId(&col_id);
  MPICHECK(MPI_Bcast((void *)&col_id, sizeof(col_id), MPI_BYTE, 0, *col_comm));

  ncclCommInitRank(nccl_col_comm, col_group_size, col_id, col_group_rank);
}

void compute_matvec(double *res2, double *res, const double *d_in, Complex *d_mat_freq, const unsigned int size, const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool full, const unsigned int device, double scale, Comm_t nccl_row_comm, Comm_t nccl_col_comm, cudaStream_t s, const unsigned int color, double *const d_in_pad, cufftHandle forward_plan, cufftHandle inverse_plan, cufftHandle forward_plan_conj, cufftHandle inverse_plan_conj, double *const d_out, double *const d_out_conj, Complex *const d_freq, Complex *const d_freq_conj, Complex *const d_out_freq, Complex *const d_red_freq, Complex *const d_red_freq_conj, Complex *const d_freq_t, Complex *const d_red_freq_t, Complex *const d_freq_conj_t, Complex *const d_red_freq_conj_t, cublasHandle_t cublasHandle, bool newmv, Complex *d_mat_freq_conj)
{
  /**
   * @brief Compute the matrix vector product of a matrix in frequency space with a vector in real space
   *
   * @param res2 The output vector
   * @param res The output vector (intermediate)
   * @param d_in The input vector
   * @param d_mat_freq The matrix in frequency space
   * @param size The size of the input and output vectors (2x since padded)
   * @param num_cols The number of local block columns in the matrix F
   * @param num_rows The number of local block rows in the matrix F
   * @param conjugate Whether to conjugate the matrix
   * @param full Whether to compute the full matrix vector product or just F/F*
   * @param device The device to use
   * @param scale The scale factor (inverse noise covariance)
   * @param nccl_row_comm The row communicator
   * @param nccl_col_comm The column communicator
   * @param s The stream to use
   */

#if TIME_MPI
  enum_array<ProfilerTimes, profiler_t, 9> *tl, *tl2;
  enum_array<ProfilerTimesNew, profiler_t, 10> *tl_new, *tl2_new;
  if (full)
    tl2 = (conjugate) ? &t_list_fs : &t_list_f;

  tl = (conjugate) ? &t_list_fs : &t_list_f;

  if (full)
    tl2_new = (conjugate) ? &t_list_fs_new : &t_list_f_new;

  tl_new = (conjugate) ? &t_list_fs_new : &t_list_f_new;
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (!newmv)
    (*tl)[ProfilerTimes::TOT].start();
  else
    (*tl_new)[ProfilerTimesNew::TOT].start();
#endif
  unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
  unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;

  // if (color != 0)
  //   gpuErrchk(cudaMalloc((void **)&d_in, vec_in_len * size / 2 * sizeof(double)));

#if TIME_MPI
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (!newmv)
    (*tl)[ProfilerTimes::BROADCAST].start();
  else
    (*tl_new)[ProfilerTimesNew::BROADCAST].start();
#endif
  Comm_t comm = (conjugate) ? nccl_row_comm : nccl_col_comm;
  Comm_t comm2 = (conjugate) ? nccl_col_comm : nccl_row_comm;
  NCCLCHECK(ncclBroadcast((const void *)d_in, (void *)d_in, (size_t)vec_in_len * size / 2, ncclDouble, 0, comm, s));
#if !CUDA_GRAPH
  gpuErrchk(cudaStreamSynchronize(s));
#endif
#if TIME_MPI
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (!newmv)
    (*tl)[ProfilerTimes::BROADCAST].stop();
  else
    (*tl_new)[ProfilerTimesNew::BROADCAST].stop();
#endif

    // double * h_in;
    // h_in = (double *)malloc(vec_in_len * size/2 * sizeof(double));
    // gpuErrchk(cudaMemset(d_in_pad, 0, vec_in_len * size * sizeof(double)));
    // gpuErrchk(cudaMemcpy(h_in, d_in, vec_in_len * size/2 * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < vec_in_len ; i++)
    // {
    //   for (int j = 0; j < size/2; j++)
    //   {
    //     printf("%f ", h_in[i * size/2 + j]);
    //   }
    //   printf("\n");
    // }
    // free(h_in);

    // double * d_in_pad;// = d_in;

    // gpuErrchk(cudaMalloc((void **)&d_in_pad, vec_in_len * size * sizeof(double)));

#if TIME_MPI
  if (!newmv)
    (*tl)[ProfilerTimes::PAD].start();
  else
    (*tl_new)[ProfilerTimesNew::PAD].start();
#endif
  PadVector(d_in, d_in_pad, vec_in_len, size, s);
#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
  if (!newmv)
    (*tl)[ProfilerTimes::PAD].stop();
  else
    (*tl_new)[ProfilerTimesNew::PAD].stop();
#endif

  // double * h_in_pad;
  // h_in_pad = (double *)malloc(vec_in_len * size * sizeof(double));
  // gpuErrchk(cudaMemcpy(h_in_pad, d_in_pad, vec_in_len * size * sizeof(double), cudaMemcpyDeviceToHost));
  // for (int i = 0; i < vec_in_len ; i++)
  // {
  //   for (int j = 0; j < size; j++)
  //   {
  //     printf("%f ", h_in_pad[i * size + j]);
  //   }
  //   printf("\n");
  // }
  // free(h_in_pad);

  // if (color != 0)
  //   gpuErrchk(cudaFree(d_in));

  Complex *d_mat_freq1, *d_mat_freq2;

  if (full)
  {
    if (conjugate)
    {
      d_mat_freq1 = (d_mat_freq_conj) ? d_mat_freq_conj : d_mat_freq;
      d_mat_freq2 = d_mat_freq;
    }
    else
    {
      d_mat_freq1 = d_mat_freq;
      d_mat_freq2 = (d_mat_freq_conj) ? d_mat_freq_conj : d_mat_freq;
    }
  }
  else
  {
    d_mat_freq1 = d_mat_freq;
    d_mat_freq2 = d_mat_freq;
  }

  if (!newmv)
    fft_matvec(res, d_in_pad, d_mat_freq1, size, num_cols, num_rows, conjugate, !(full), device, forward_plan, inverse_plan, d_out, d_freq, d_out_freq, d_red_freq, s);
  else
    fft_matvec_new(res, d_in_pad, d_mat_freq1, size, num_cols, num_rows, conjugate, !(full), device, forward_plan, inverse_plan, d_out, d_freq, d_red_freq, d_freq_t, d_red_freq_t, s, cublasHandle);
#if TIME_MPI
  gpuErrchk(cudaDeviceSynchronize());
#endif

  // gpuErrchk(cudaFree(d_in_pad));

  // double * h_res;
  // h_res = (double *)malloc(vec_out_len * size / 2 * sizeof(double));
  // gpuErrchk(cudaMemcpy(h_res, res, vec_out_len * size / 2 * sizeof(double), cudaMemcpyDeviceToHost));
  // for (int i = 0; i < vec_out_len ; i++)
  // {
  //   for (int j = 0; j < size/2; j++)
  //   {
  //     printf("%f ", h_res[i * size/2 + j]);
  //   }
  //   printf("\n");
  // }

  if (!full)
  {
#if TIME_MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!newmv)
      (*tl)[ProfilerTimes::NCCLC].start();
    else
      (*tl_new)[ProfilerTimesNew::NCCLC].start();
#endif
    NCCLCHECK(ncclReduce((const void *)res, (void *)res, (size_t)vec_out_len * size / 2, ncclDouble, ncclSum,
                         0, comm2, s));
#if !CUDA_GRAPH
    gpuErrchk(cudaStreamSynchronize(s));
#endif

    // if (color == 0)
    // {
    //   double * h_res;
    //   h_res = (double *)malloc(vec_out_len * size / 2 * sizeof(double));
    //   gpuErrchk(cudaMemcpy(h_res, res, vec_out_len * size / 2 * sizeof(double), cudaMemcpyDeviceToHost));
    //   for (int i = 0; i < vec_out_len ; i++)
    //   {
    //     for (int j = 0; j < size/2; j++)
    //     {
    //       printf("%f ", h_res[i * size/2 + j]);
    //     }
    //     printf("\n");
    //   }
    // }

#if TIME_MPI
    // MPI_Barrier(MPI_COMM_WORLD);
    // mpi_time += MPI_Wtime();
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!newmv)
      (*tl)[ProfilerTimes::NCCLC].stop();
    else
      (*tl_new)[ProfilerTimesNew::NCCLC].stop();

    if (!newmv)
      (*tl)[ProfilerTimes::TOT].stop();
    else
      (*tl_new)[ProfilerTimesNew::TOT].stop();
#endif
  }

  else
  {
    // if (color == 0)
    // {
    //   double *h_res;
    //   h_res = (double *)malloc(vec_out_len * size * sizeof(double));
    //   gpuErrchk(cudaMemcpy(h_res, res, vec_out_len * size * sizeof(double), cudaMemcpyDeviceToHost));
    //   for (int i = 0; i < vec_out_len; i++)
    //   {
    //     for (int j = 0; j < size; j++)
    //     {
    //       printf("%f ", h_res[i * size + j]);
    //     }
    //     printf("\n");
    //   }
    // }

#if TIME_MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!newmv)
      (*tl)[ProfilerTimes::NCCLC].start();
    else
      (*tl_new)[ProfilerTimesNew::NCCLC].start();
#endif
    NCCLCHECK(ncclAllReduce((const void *)res, (void *)res, (size_t)vec_out_len * size, ncclDouble, ncclSum,
                            comm2, s));
#if !CUDA_GRAPH
    gpuErrchk(cudaStreamSynchronize(s));
#endif
#if TIME_MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!newmv)
      (*tl)[ProfilerTimes::NCCLC].stop();
    else
      (*tl_new)[ProfilerTimesNew::NCCLC].stop();

    if (!newmv)
      (*tl)[ProfilerTimes::TOT].stop();
    else
      (*tl_new)[ProfilerTimesNew::TOT].stop();
#endif
#if TIME_MPI
    t_list[ProfilerTimesFull::SCALE].start();
#endif

    // ScaleVector(res, size, vec_out_len, scale, s);

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    t_list[ProfilerTimesFull::SCALE].stop();
#endif

#if TIME_MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!newmv)
      (*tl2)[ProfilerTimes::TOT].start();
    else
      (*tl2_new)[ProfilerTimesNew::TOT].start();
#endif

    if (!newmv)
      fft_matvec(res2, res, d_mat_freq2, size, num_cols, num_rows, !(conjugate), true, device, forward_plan_conj, inverse_plan_conj, d_out_conj, d_freq_conj, d_out_freq, d_red_freq_conj, s);
    else
      fft_matvec_new(res2, res, d_mat_freq2, size, num_cols, num_rows, !(conjugate), true, device, forward_plan_conj, inverse_plan_conj, d_out_conj, d_freq_conj, d_red_freq_conj, d_freq_conj_t, d_red_freq_conj_t, s, cublasHandle);

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!newmv)
      (*tl2)[ProfilerTimes::NCCLC].start();
    else
      (*tl2_new)[ProfilerTimesNew::NCCLC].start();
#endif
    NCCLCHECK(ncclReduce((const void *)res2, (void *)res2, (size_t)vec_in_len * size / 2, ncclDouble, ncclSum,
                         0, comm, s));
#if !CUDA_GRAPH
    gpuErrchk(cudaStreamSynchronize(s));
#endif

#if TIME_MPI
    // MPI_Barrier(MPI_COMM_WORLD);
    // mpi_time += MPI_Wtime();
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!newmv)
      (*tl2)[ProfilerTimes::NCCLC].stop();
    else
      (*tl2_new)[ProfilerTimesNew::NCCLC].stop();

    if (!newmv)
      (*tl2)[ProfilerTimes::TOT].stop();
    else
      (*tl2_new)[ProfilerTimesNew::TOT].stop();
// t_list[ProfilerTimes::FULL].stop();
#endif
  }
}

void cleanup(double *d_in, double *d_in_pad, cufftHandle forward_plan, cufftHandle inverse_plan, cufftHandle forward_plan_conj, cufftHandle inverse_plan_conj, double *d_out, double *d_out_conj, Complex *d_freq, Complex *d_freq_conj, Complex *d_out_freq, const unsigned int color, const bool full)
{
  if (color != 0)
  {
    gpuErrchk(cudaFree(d_in));
  }

  gpuErrchk(cudaFree(d_in_pad));
  gpuErrchk(cudaFree(d_out));
  gpuErrchk(cudaFree(d_freq));
  gpuErrchk(cudaFree(d_out_freq));
  cufftSafeCall(cufftDestroy(forward_plan));
  cufftSafeCall(cufftDestroy(inverse_plan));

  if (full)
  {
    gpuErrchk(cudaFree(d_out_conj));
    gpuErrchk(cudaFree(d_freq_conj));
    cufftSafeCall(cufftDestroy(forward_plan_conj));
    cufftSafeCall(cufftDestroy(inverse_plan_conj));
  }
}
