#include "matvec.hpp"

#if TIME_MPI
enum_array<ProfilerTimesFull, profiler_t, 3> t_list;
enum_array<ProfilerTimes, profiler_t, 10> t_list_f;
enum_array<ProfilerTimes, profiler_t, 10> t_list_fs;
#endif



void Matvec::setup(Complex** mat_freq_tosi, const double* const h_mat, const unsigned int block_size,
    const unsigned int num_cols, const unsigned int num_rows, cublasHandle_t cublasHandle)
{

    double* d_mat;
    cufftHandle forward_plan_mat;
    const size_t mat_len = (size_t)block_size * num_cols * num_rows * sizeof(double);

    fft_int_t n[1] = { (fft_int_t)block_size };
    int rank = 1;

    fft_int_t idist = block_size;
    fft_int_t odist = (block_size / 2 + 1);

    fft_int_t inembed[] = { 0 };
    fft_int_t onembed[] = { 0 };

    fft_int_t istride = 1;
    fft_int_t ostride = 1;

#if !FFT_64
#if !ROW_SETUP
    cufftSafeCall(cufftPlanMany(&forward_plan_mat, rank, n, inembed, istride, idist, onembed,
        ostride, odist, CUFFT_D2Z, (size_t)num_cols * num_rows));
#else
    cufftSafeCall(cufftPlanMany(&forward_plan_mat, rank, n, inembed, istride, idist, onembed,
        ostride, odist, CUFFT_D2Z, num_cols));
#endif
#else
    size_t ws = 0;
    cufftSafeCall(cufftCreate(&forward_plan_mat));
    cufftSafeCall(cufftMakePlanMany64(forward_plan_mat, rank, n, inembed, istride, idist, onembed,
        ostride, odist, CUFFT_D2Z, num_cols * num_rows, &ws));
#endif
    gpuErrchk(cudaMalloc((void**)&d_mat, mat_len));
    gpuErrchk(cudaMemcpy(d_mat, h_mat, mat_len, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(
        (void**)mat_freq_tosi, (size_t)(block_size / 2 + 1) * num_cols * num_rows * sizeof(Complex)));

#if !ROW_SETUP
    cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat, *mat_freq_tosi));
#else
    for (int i = 0; i < num_rows; i++) {
        cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat + (size_t)i * block_size * num_cols,
            *mat_freq_tosi + (size_t)i * num_cols * (block_size / 2 + 1)));
    }
#endif

    cufftSafeCall(cufftDestroy(forward_plan_mat));
    gpuErrchk(cudaFree(d_mat));

    double scale = 1.0 / block_size;
#if !FFT_64
    cublasSafeCall(cublasZdscal(cublasHandle, (size_t)(block_size / 2 + 1) * num_cols * num_rows,
        &scale, *mat_freq_tosi, 1));
#else
    cublasSafeCall(cublasZdscal_64(cublasHandle, (size_t)(block_size / 2 + 1) * num_cols * num_rows,
        &scale, *mat_freq_tosi, 1));
#endif

    Complex* d_mat_freq_trans;
    gpuErrchk(cudaMalloc(
        (void**)&d_mat_freq_trans, sizeof(Complex) * (size_t)(block_size / 2 + 1) * num_cols * num_rows));
    if (num_cols > 1 && num_rows > 1) {
        int sz[3] = { (int)(block_size / 2 + 1), (int)num_cols, (int)num_rows };
        int perm[3] = { 2, 1, 0 };
        int elements_per_thread = 4;

        if (cut_transpose3d(d_mat_freq_trans, *mat_freq_tosi, sz, perm, elements_per_thread) < 0) {
            fprintf(stderr, "Error while performing transpose.\n");
            exit(1);
        }
    } else {
        cuDoubleComplex aa({ 1, 0 });
        cuDoubleComplex bb({ 0, 0 });
        if (num_rows == 1) {
#if !FFT_64
            cublasSafeCall(
                cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_cols, (block_size / 2 + 1), &aa,
                    *mat_freq_tosi, (block_size / 2 + 1), &bb, NULL, num_cols, d_mat_freq_trans, num_cols));
#else
            cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_cols,
                (block_size / 2 + 1), &aa, *mat_freq_tosi, (block_size / 2 + 1), &bb, NULL, num_cols,
                d_mat_freq_trans, num_cols));
#endif
        } else {
#if !FFT_64
            cublasSafeCall(
                cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_rows, (block_size / 2 + 1), &aa,
                    *mat_freq_tosi, (block_size / 2 + 1), &bb, NULL, num_rows, d_mat_freq_trans, num_rows));
#else
            cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_rows,
                (block_size / 2 + 1), &aa, *mat_freq_tosi, (block_size / 2 + 1), &bb, NULL, num_rows,
                d_mat_freq_trans, num_rows));
#endif
        }
    }

    gpuErrchk(cudaFree(*mat_freq_tosi));
    *mat_freq_tosi = d_mat_freq_trans;
}

void Matvec::local_matvec(double* const out_vec, double* const in_vec, const Complex* const d_mat_freq,
    const unsigned int block_size, const unsigned int num_cols, const unsigned int num_rows,
    const bool conjugate, const bool unpad, const unsigned int device, cufftHandle forward_plan,
    cufftHandle inverse_plan, double* const out_vec_pad, Complex* const in_vec_freq,
    Complex* const out_vec_freq_tosi, Complex* const in_vec_freq_tosi, Complex* const out_vec_freq,
    cudaStream_t s, cublasHandle_t cublasHandle)
{

    unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
    unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;

#if TIME_MPI
    enum_array<ProfilerTimes, profiler_t, 10>* tl = (conjugate) ? &t_list_fs : &t_list_f;
    (*tl)[ProfilerTimes::FFT].start();
#endif

    cufftSafeCall(cufftExecD2Z(forward_plan, in_vec, in_vec_freq));

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::FFT].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::TRANS1].start();
#endif

    cuDoubleComplex alpha({ 1, 0 });
    cuDoubleComplex beta({ 0, 0 });
#if !FFT_64
    cublasSafeCall(
        cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, vec_in_len, (block_size / 2 + 1), &alpha,
            in_vec_freq, (block_size / 2 + 1), &beta, NULL, vec_in_len, in_vec_freq_tosi, vec_in_len));

#else
    cublasSafeCall(
        cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, vec_in_len, (block_size / 2 + 1), &alpha,
            in_vec_freq, (block_size / 2 + 1), &beta, NULL, vec_in_len, in_vec_freq_tosi, vec_in_len));
#endif

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::TRANS1].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::SBGEMV].start();
#endif

    cublasOperation_t transa = (conjugate) ? CUBLAS_OP_C : CUBLAS_OP_N;

#if !FFT_64
    cublasSafeCall(cublasZgemvStridedBatched(cublasHandle, transa, num_rows, num_cols, &alpha,
        d_mat_freq, num_rows, (size_t)num_rows * num_cols, in_vec_freq_tosi, 1, vec_in_len, &beta,
        out_vec_freq_tosi, 1, vec_out_len, (block_size / 2 + 1)));

#else
    cublasSafeCall(cublasZgemvStridedBatched_64(cublasHandle, transa, num_rows, num_cols, &alpha,
        d_mat_freq, num_rows, (size_t)num_rows * num_cols, in_vec_freq_tosi, 1, vec_in_len, &beta,
        out_vec_freq_tosi, 1, vec_out_len, (block_size / 2 + 1)));
#endif

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::SBGEMV].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::TRANS2].start();
#endif

#if !FFT_64
    cublasSafeCall(cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (block_size / 2 + 1), vec_out_len,
        &alpha, out_vec_freq_tosi, vec_out_len, &beta, NULL, (block_size / 2 + 1), out_vec_freq,
        (block_size / 2 + 1)));
#else
    cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (block_size / 2 + 1),
        vec_out_len, &alpha, out_vec_freq_tosi, vec_out_len, &beta, NULL, (block_size / 2 + 1),
        out_vec_freq, (block_size / 2 + 1)));
#endif

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::TRANS2].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::IFFT].start();
#endif

    cufftSafeCall(cufftExecZ2D(inverse_plan, out_vec_freq, out_vec_pad));
#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::IFFT].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::UNPAD].start();
#endif

    Utils::UnpadRepadVector(out_vec_pad, out_vec, vec_out_len, block_size, unpad, s);

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::UNPAD].stop();
#endif
}

void Matvec::compute_matvec(double* out_vec, double* in_vec, Complex* mat_freq_tosi, const unsigned int block_size,
    const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool full,
    const unsigned int device, ncclComm_t nccl_row_comm, ncclComm_t nccl_col_comm,
    cudaStream_t s, double* const in_vec_pad, cufftHandle forward_plan, cufftHandle inverse_plan,
    cufftHandle forward_plan_conj, cufftHandle inverse_plan_conj, double* const out_vec_pad,
    Complex* const in_vec_freq, Complex* const out_vec_freq_tosi, Complex* const in_vec_freq_tosi,
    Complex* const out_vec_freq, cublasHandle_t cublasHandle, Complex* mat_freq_tosi_other,
    double* const res_pad)
{

#if TIME_MPI
    enum_array<ProfilerTimes, profiler_t, 10>*tl, *tl2;
    if (full)
        tl2 = (conjugate) ? &t_list_fs : &t_list_f;
    tl = (conjugate) ? &t_list_fs : &t_list_f;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    (*tl)[ProfilerTimes::TOT].start();
#endif
    unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
    unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;


#if TIME_MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    (*tl)[ProfilerTimes::BROADCAST].start();
#endif
    ncclComm_t comm = (conjugate) ? nccl_row_comm : nccl_col_comm;
    ncclComm_t comm2 = (conjugate) ? nccl_col_comm : nccl_row_comm;
    NCCLCHECK(ncclBroadcast(
        (const void*)in_vec, (void*)in_vec, (size_t)vec_in_len * block_size / 2, ncclDouble, 0, comm, s));
    gpuErrchk(cudaStreamSynchronize(s));
#if TIME_MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    (*tl)[ProfilerTimes::BROADCAST].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::PAD].start();
#endif
    Utils::PadVector(in_vec, in_vec_pad, vec_in_len, block_size, s);
#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::PAD].stop();
#endif

    Complex *mat_freq_tosi1, *mat_freq_tosi2;

    if (full) {
        if (conjugate) {
            mat_freq_tosi1 = (mat_freq_tosi_other) ? mat_freq_tosi_other : mat_freq_tosi;
            mat_freq_tosi2 = mat_freq_tosi;
        } else {
            mat_freq_tosi1 = mat_freq_tosi;
            mat_freq_tosi2 = (mat_freq_tosi_other) ? mat_freq_tosi_other : mat_freq_tosi;
        }
    } else {
        mat_freq_tosi1 = mat_freq_tosi;
        mat_freq_tosi2 = mat_freq_tosi;
    }

    double* res_vec = (full) ? res_pad : out_vec;

    local_matvec(res_vec, in_vec_pad, mat_freq_tosi1, block_size, num_cols, num_rows, conjugate, !(full),
        device, forward_plan, inverse_plan, out_vec_pad, in_vec_freq, out_vec_freq_tosi, in_vec_freq_tosi, out_vec_freq, s,
        cublasHandle);
#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
#endif

    if (!full) {
#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl)[ProfilerTimes::NCCLC].start();
#endif
        NCCLCHECK(ncclReduce((const void*)res_vec, (void*)res_vec, (size_t)vec_out_len * block_size / 2,
            ncclDouble, ncclSum, 0, comm2, s));
        gpuErrchk(cudaStreamSynchronize(s));

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl)[ProfilerTimes::NCCLC].stop();
        (*tl)[ProfilerTimes::TOT].stop();
#endif
    }

    else {

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl)[ProfilerTimes::NCCLC].start();
#endif
        NCCLCHECK(ncclAllReduce((const void*)res_vec, (void*)res_vec, (size_t)vec_out_len * block_size,
            ncclDouble, ncclSum, comm2, s));
        gpuErrchk(cudaStreamSynchronize(s));
#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl)[ProfilerTimes::NCCLC].stop();
        (*tl)[ProfilerTimes::TOT].stop();
#endif

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl2)[ProfilerTimes::TOT].start();
#endif

        local_matvec(out_vec, res_vec, mat_freq_tosi2, block_size, num_cols, num_rows, !(conjugate), true,
            device, forward_plan_conj, inverse_plan_conj, in_vec_pad, out_vec_freq, in_vec_freq_tosi,
            out_vec_freq_tosi, in_vec_freq, s, cublasHandle);

#if TIME_MPI
        gpuErrchk(cudaDeviceSynchronize());
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl2)[ProfilerTimes::NCCLC].start();
#endif
        NCCLCHECK(ncclReduce((const void*)out_vec, (void*)out_vec, (size_t)vec_in_len * block_size / 2,
            ncclDouble, ncclSum, 0, comm, s));
        gpuErrchk(cudaStreamSynchronize(s));

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl2)[ProfilerTimes::NCCLC].stop();
        (*tl2)[ProfilerTimes::TOT].stop();
#endif
    }
}
