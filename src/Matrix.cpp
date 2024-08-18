

#include "Matrix.hpp"


Matrix::Matrix(Comm &comm, unsigned int num_cols, unsigned int num_rows, unsigned int block_size) : comm(comm), num_cols(num_cols), num_rows(num_rows), block_size(block_size)
{
    // Initialize the matrix data structures

    fft_int_t n[1] = {(fft_int_t)block_size};
    int rank = 1;

    fft_int_t idist = block_size;
    fft_int_t odist = (block_size / 2 + 1);

    fft_int_t inembed[] = {0};
    fft_int_t onembed[] = {0};

    fft_int_t istride = 1;
    fft_int_t ostride = 1;
#if !FFT_64
    cufftSafeCall(cufftPlanMany(&(forward_plan), rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols));
    cufftSafeCall(cufftPlanMany(&(inverse_plan), rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, num_rows));
#else
    size_t ws = 0;
    cufftSafeCall(cufftCreate(&(forward_plan)));
    cufftSafeCall(cufftMakePlanMany64(forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_cols, &ws));
    cufftSafeCall(cufftCreate(&(inverse_plan)));
    cufftSafeCall(cufftMakePlanMany64(inverse_plan, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, num_rows, &ws));
#endif

    cudaStream_t s = comm.get_stream();

    cufftSafeCall(cufftSetStream(forward_plan, s));
    cufftSafeCall(cufftSetStream(inverse_plan, s));
    gpuErrchk(cudaMalloc((void **)&(col_vec_pad), (size_t)num_cols * block_size * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&(col_vec_freq), sizeof(Complex) * (size_t)(block_size / 2 + 1) * num_cols));

    gpuErrchk(cudaMalloc((void **)&(col_vec_freq_tosi), sizeof(Complex) * (size_t)(block_size / 2 + 1) * num_cols));
    gpuErrchk(cudaMalloc((void **)&(row_vec_freq_tosi), (size_t)sizeof(Complex) * (block_size / 2 + 1) * num_rows));

    gpuErrchk(cudaMalloc((void **)&(row_vec_freq), (size_t)sizeof(Complex) * (block_size / 2 + 1) * num_rows));
    gpuErrchk(cudaMalloc((void **)&(row_vec_pad), (size_t)sizeof(double) * block_size * num_rows)); // num_cols * num_rows));

#if !FFT_64
    cufftSafeCall(cufftPlanMany(&(forward_plan_conj), rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_rows));
    cufftSafeCall(cufftPlanMany(&(inverse_plan_conj), rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, num_cols));
#else
    cufftSafeCall(cufftCreate(&(forward_plan_conj)));
    cufftSafeCall(cufftMakePlanMany64(forward_plan_conj, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, num_rows, &ws));
    cufftSafeCall(cufftCreate(&(inverse_plan_conj)));
    cufftSafeCall(cufftMakePlanMany64(inverse_plan_conj, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, num_cols, &ws));
#endif
    cufftSafeCall(cufftSetStream(forward_plan_conj, s));
    cufftSafeCall(cufftSetStream(inverse_plan_conj, s));

    int max_block_len = (num_cols > num_rows) ? num_cols : num_rows;

    gpuErrchk(cudaMalloc((void **)&(res_pad), sizeof(double) * (size_t)block_size * max_block_len));


    gpuErrchk(cudaMalloc((void **)&(col_vec_unpad), (size_t)num_cols * block_size / 2 * sizeof(double)));

    gpuErrchk(cudaMalloc((void **)&(row_vec_unpad), sizeof(double) * (size_t)block_size / 2 * num_rows));
}


Matrix::~Matrix()
{
    cufftSafeCall(cufftDestroy(forward_plan));
    cufftSafeCall(cufftDestroy(inverse_plan));
    gpuErrchk(cudaFree(col_vec_pad));
    gpuErrchk(cudaFree(col_vec_freq));
    gpuErrchk(cudaFree(col_vec_freq_tosi));
    gpuErrchk(cudaFree(row_vec_freq_tosi));
    gpuErrchk(cudaFree(row_vec_freq));
    gpuErrchk(cudaFree(row_vec_pad));

    cufftSafeCall(cufftDestroy(forward_plan_conj));
    cufftSafeCall(cufftDestroy(inverse_plan_conj));

    gpuErrchk(cudaFree(col_vec_unpad));

    gpuErrchk(cudaFree(row_vec_unpad));
    gpuErrchk(cudaFree(res_pad));

    if (initialized)
        gpuErrchk(cudaFree(mat_freq_tosi));
    if (has_mat_freq_tosi_other && initialized)
        gpuErrchk(cudaFree(mat_freq_tosi_other));
}

void Matrix::init_mat_ones()
{
    double *h_mat = new double[(size_t)block_size * num_cols * num_rows];
#pragma omp parallel for collapse(3)
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            for (int k = 0; k < block_size; k++)
            {
                // set to 1 if k < block_size / 2, 0 otherwise.
                h_mat[(size_t)i * num_cols * block_size + (size_t)j * block_size + k] = (k < block_size / 2) ? 1.0 : 0.0;
            }
        }
    }

    cublasHandle_t cublasHandle = comm.get_cublasHandle();

    Matvec::setup(&mat_freq_tosi, h_mat, block_size, num_cols, num_rows, cublasHandle);
    delete[] h_mat;
    initialized = true;
}




void Matrix::matvec(Vector &x, Vector &y, bool full)
{
    if (!initialized)
    {
        fprintf(stderr, "Matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }
    else if (!x.is_initialized())
    {
        fprintf(stderr, "Vector x not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }
    else if (!y.is_initialized())
    {
        fprintf(stderr, "Vector y not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }

    cudaStream_t s = comm.get_stream();
    cublasHandle_t cublasHandle = comm.get_cublasHandle();
    int device = comm.get_device();
    ncclComm_t row_comm = comm.get_gpu_row_comm();
    ncclComm_t col_comm = comm.get_gpu_col_comm();

    double *in_vec, *out_vec;

    int in_color = comm.get_row_color();
    int out_color = (full) ? comm.get_row_color() : comm.get_col_color();

    if (in_color == 0)
        in_vec = x.get_d_vec();
    else
        in_vec = col_vec_unpad;
    
    if (out_color == 0)
        out_vec = y.get_d_vec();
    else
        out_vec = (full) ? col_vec_unpad : row_vec_unpad;


    if (full)
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, block_size, num_cols, num_rows, false, true, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan, forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq, col_vec_freq_tosi, row_vec_freq_tosi, cublasHandle, mat_freq_tosi_other, res_pad);
    else
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, block_size, num_cols, num_rows, false, false, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan, forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq, col_vec_freq_tosi, row_vec_freq_tosi, cublasHandle, mat_freq_tosi_other, res_pad);
    gpuErrchk(cudaStreamSynchronize(s));


    if (out_color == 0)
        y.set_d_vec(out_vec);

}

void Matrix::transpose_matvec(Vector &x, Vector &y, bool full)
{
    if (!initialized)
    {
        fprintf(stderr, "Matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }
    else if (!x.is_initialized())
    {
        fprintf(stderr, "Vector x not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }
    else if (!y.is_initialized())
    {
        fprintf(stderr, "Vector y not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }

    cudaStream_t s = comm.get_stream();
    cublasHandle_t cublasHandle = comm.get_cublasHandle();
    int device = comm.get_device();
    ncclComm_t row_comm = comm.get_gpu_row_comm();
    ncclComm_t col_comm = comm.get_gpu_col_comm();

    double *in_vec, *out_vec;

    int in_color = comm.get_col_color();
    int out_color = (full) ? comm.get_col_color() : comm.get_row_color();

    if (in_color == 0)
        in_vec = x.get_d_vec();
    else
        in_vec = row_vec_unpad;

    if (out_color == 0)
        out_vec = y.get_d_vec();
    else
        out_vec = (full) ? row_vec_unpad : col_vec_unpad;

    
    if (full)
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, block_size, num_cols, num_rows, true, true, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj, inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_tosi, col_vec_freq_tosi, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_tosi_other, res_pad);
    else
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, block_size, num_cols, num_rows, true, false, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj, inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_tosi, col_vec_freq_tosi, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_tosi_other, res_pad);
    gpuErrchk(cudaStreamSynchronize(s));

    if (out_color == 0)
        y.set_d_vec(out_vec);

}
