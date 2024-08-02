#ifndef __matfuncs_h__
#define __matfuncs_h__

#include "shared.cuh"

typedef struct {
  Complex *d_mat_freq, *d_mat_freq_conj;
  unsigned int size;
  unsigned int num_cols;
  unsigned int num_rows;
  bool conjugate;
  bool newmv;
  // bool full;
  int device;
  double noise_scale;
  Comm_t row_comm;
  Comm_t col_comm;
  cudaStream_t s;
  // MPI_Comm vec_comm;
  int row_color, col_color;
  double * d_in, * d_in_pad, * d_out, * d_out_conj, * res, * res2;
  Complex * d_freq, * d_out_freq, * d_freq_conj, *d_red_freq, *d_red_freq_conj, * d_freq_t, * d_red_freq_t, * d_freq_conj_t, * d_red_freq_conj_t;
  cufftHandle forward_plan, inverse_plan, forward_plan_conj, inverse_plan_conj;
  cublasHandle_t cublasHandle;
  // int test;
  // double * test2;
  Mat R, M;
} matvec_args_t;




void createMat(matvec_args_t *ctx, int in_color, int out_color, unsigned int block_size, unsigned int num_rows, unsigned int num_cols, int proc_rows, int proc_cols, bool conjugate, bool full, cudaStream_t s, bool newmv);
void destroyMat(matvec_args_t *args, int in_color, int out_color, bool conjugate, bool full, bool newmv);
void MatVec(matvec_args_t *args, double *d_in, double *res, bool conj, bool full);
void init_hmat(int num_rows, int num_cols, int size, double **hmat);
void init_vector(int len, int unpad_size, double **vec, bool init_value);















#endif