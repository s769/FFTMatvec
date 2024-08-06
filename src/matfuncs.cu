#include "matfuncs.cuh"
#include "matvec.cuh"

void createMat(matvec_args_t *ctx, int in_color, int out_color, unsigned int block_size, unsigned int num_rows, unsigned int num_cols, int proc_rows, int proc_cols, bool conjugate, bool full, cudaStream_t s, bool newmv)
{
    unsigned int vec_in_len = (conjugate) ? num_rows : num_cols;
    unsigned int vec_out_len = (conjugate) ? num_cols : num_rows;

    int local_size_row, local_size_col;
    if (!full)
    {
        local_size_row = (out_color == 0) ? block_size / 2 * vec_out_len : 0;
        local_size_col = (in_color == 0) ? block_size / 2 * vec_in_len : 0;
    }
    else
    {
        local_size_row = (in_color == 0) ? block_size / 2 * vec_in_len : 0;
        local_size_col = (in_color == 0) ? block_size / 2 * vec_in_len : 0;
    }

    fft_int_t n[1] = {(fft_int_t)block_size};
    int rank = 1;

    fft_int_t idist = block_size;
    fft_int_t odist = (block_size / 2 + 1);

    fft_int_t inembed[] = {0};
    fft_int_t onembed[] = {0};

    fft_int_t istride = 1;
    fft_int_t ostride = 1;

#if !FFT_64
    cufftSafeCall(cufftPlanMany(&(ctx->forward_plan), rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_in_len));
    cufftSafeCall(cufftPlanMany(&(ctx->inverse_plan), rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_out_len));
#else
    size_t ws = 0;
    cufftSafeCall(cufftCreate(&(ctx->forward_plan)));
    cufftSafeCall(cufftMakePlanMany64(ctx->forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_in_len, &ws));
    cufftSafeCall(cufftCreate(&(ctx->inverse_plan)));
    cufftSafeCall(cufftMakePlanMany64(ctx->inverse_plan, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_out_len, &ws));
#endif

    cufftSafeCall(cufftSetStream(ctx->forward_plan, s));
    cufftSafeCall(cufftSetStream(ctx->inverse_plan, s));
    gpuErrchk(cudaMalloc((void **)&(ctx->d_in_pad), (size_t)vec_in_len * block_size * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&(ctx->d_freq), sizeof(Complex) * (size_t)(block_size / 2 + 1) * vec_in_len));
    if (!newmv)
    {
        gpuErrchk(cudaMalloc((void **)&(ctx->d_out_freq), (size_t)sizeof(Complex) * (block_size / 2 + 1) * num_cols * num_rows));
        ctx->cublasHandle = NULL;
    }
    else
    {
        cublasSafeCall(cublasCreate(&(ctx->cublasHandle)));
        cublasSafeCall(cublasSetStream(ctx->cublasHandle, s));
        gpuErrchk(cudaMalloc((void **)&(ctx->d_freq_t), sizeof(Complex) * (size_t)(block_size / 2 + 1) * vec_in_len));
        gpuErrchk(cudaMalloc((void **)&(ctx->d_red_freq_t), (size_t)sizeof(Complex) * (block_size / 2 + 1) * vec_out_len));
    }
    gpuErrchk(cudaMalloc((void **)&(ctx->d_red_freq), (size_t)sizeof(Complex) * (block_size / 2 + 1) * vec_out_len));
    gpuErrchk(cudaMalloc((void **)&(ctx->d_out), (size_t)sizeof(double) * block_size * vec_out_len)); // num_cols * num_rows));
    ctx->newmv = newmv;
    if (full)
    {
#if !FFT_64
        cufftSafeCall(cufftPlanMany(&(ctx->forward_plan_conj), rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_out_len));
        cufftSafeCall(cufftPlanMany(&(ctx->inverse_plan_conj), rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_in_len));
#else
        cufftSafeCall(cufftCreate(&(ctx->forward_plan_conj)));
        cufftSafeCall(cufftMakePlanMany64(ctx->forward_plan_conj, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, vec_out_len, &ws));
        cufftSafeCall(cufftCreate(&(ctx->inverse_plan_conj)));
        cufftSafeCall(cufftMakePlanMany64(ctx->inverse_plan_conj, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, vec_in_len, &ws));
#endif
        cufftSafeCall(cufftSetStream(ctx->forward_plan_conj, s));
        cufftSafeCall(cufftSetStream(ctx->inverse_plan_conj, s));
        gpuErrchk(cudaMalloc((void **)&(ctx->d_freq_conj), sizeof(Complex) * (size_t)(block_size / 2 + 1) * vec_out_len));
        gpuErrchk(cudaMalloc((void **)&(ctx->d_red_freq_conj), (size_t)sizeof(Complex) * (block_size / 2 + 1) * vec_in_len));
        gpuErrchk(cudaMalloc((void **)&(ctx->d_out_conj), (size_t)sizeof(double) * block_size * vec_in_len));
        if (newmv)
        {
            gpuErrchk(cudaMalloc((void **)&(ctx->d_freq_conj_t), sizeof(Complex) * (size_t)(block_size / 2 + 1) * vec_out_len));
            gpuErrchk(cudaMalloc((void **)&(ctx->d_red_freq_conj_t), (size_t)sizeof(Complex) * (block_size / 2 + 1) * vec_in_len));
        }
    }

    if (in_color != 0)
    {
        gpuErrchk(cudaMalloc((void **)&(ctx->d_in), (size_t)vec_in_len * block_size / 2 * sizeof(double)));
    }

    if (full)
    {
        gpuErrchk(cudaMalloc((void **)&(ctx->res), sizeof(double) * (size_t)block_size * vec_out_len));
        if (in_color != 0)
            gpuErrchk(cudaMalloc((void **)&(ctx->res2), sizeof(double) * (size_t)block_size / 2 * vec_in_len));
    }
    else
    {
        if (out_color != 0)
            gpuErrchk(cudaMalloc((void **)&(ctx->res), sizeof(double) * (size_t)block_size / 2 * vec_out_len));
    }

    ctx->s = s;
}

void destroyMat(matvec_args_t *args, int in_color, int out_color, bool conjugate, bool full, bool newmv)
{
    cufftSafeCall(cufftDestroy(args->forward_plan));
    cufftSafeCall(cufftDestroy(args->inverse_plan));
    gpuErrchk(cudaFree(args->d_freq));
    if (!newmv)
    {
        gpuErrchk(cudaFree(args->d_out_freq));
    }
    else
    {
        cublasSafeCall(cublasDestroy(args->cublasHandle));
        gpuErrchk(cudaFree(args->d_freq_t));
        gpuErrchk(cudaFree(args->d_red_freq_t));
    }
    gpuErrchk(cudaFree(args->d_red_freq));
    gpuErrchk(cudaFree(args->d_out));
    gpuErrchk(cudaFree(args->d_in_pad));
    if (full)
    {
        cufftSafeCall(cufftDestroy(args->forward_plan_conj));
        cufftSafeCall(cufftDestroy(args->inverse_plan_conj));
        gpuErrchk(cudaFree(args->d_freq_conj));
        gpuErrchk(cudaFree(args->d_red_freq_conj));
        gpuErrchk(cudaFree(args->d_out_conj));
        if (newmv)
        {
            gpuErrchk(cudaFree(args->d_freq_conj_t));
            gpuErrchk(cudaFree(args->d_red_freq_conj_t));
        }
    }
    if (in_color != 0)
    {
        gpuErrchk(cudaFree(args->d_in));
    }
    if (full)
    {
        gpuErrchk(cudaFree(args->res));
        if (in_color != 0)
            gpuErrchk(cudaFree(args->res2));
    }
    else
    {
        if (out_color != 0)
            gpuErrchk(cudaFree(args->res));
    }
}

void MatVec(matvec_args_t *args, double *d_in, double *res, bool conj, bool full)
{

    Complex *d_mat_freq = args->d_mat_freq;
    unsigned int size = args->size;
    unsigned int num_cols = args->num_cols;
    unsigned int num_rows = args->num_rows;

    int device = args->device;
    double noise_scale = args->noise_scale;
    Comm_t row_comm = args->row_comm;
    Comm_t col_comm = args->col_comm;
    cudaStream_t s = args->s;
    int row_color = args->row_color;
    int col_color = args->col_color;
    double *d_in_pad = args->d_in_pad;
    double *d_out = args->d_out;
    double *d_out_conj = NULL; // args->d_out_conj;
    Complex *d_freq = args->d_freq;
    Complex *d_out_freq = args->d_out_freq;
    Complex *d_freq_conj = NULL; // args->d_freq_conj;
    Complex *d_red_freq = args->d_red_freq;
    Complex *d_red_freq_conj = NULL;
    Complex *d_freq_t = args->d_freq_t;
    Complex *d_freq_conj_t = NULL;
    Complex *d_red_freq_t = args->d_red_freq_t;
    Complex *d_red_freq_conj_t = NULL;
    cufftHandle forward_plan = args->forward_plan;
    cufftHandle inverse_plan = args->inverse_plan;
    cufftHandle forward_plan_conj = NULL; // args->forward_plan_conj;
    cufftHandle inverse_plan_conj = NULL; // args->inverse_plan_conj;
    cublasHandle_t cublasHandle = args->cublasHandle;
    bool newmv = args->newmv;
    double *res2 = NULL; // args->res2;

    compute_matvec(res2, res, d_in, d_mat_freq, size, num_cols, num_rows, conj, full, device, noise_scale, row_comm, col_comm, s, 0, d_in_pad, forward_plan, inverse_plan, forward_plan_conj, inverse_plan_conj, d_out, d_out_conj, d_freq, d_freq_conj, d_out_freq, d_red_freq, d_red_freq_conj, d_freq_t, d_red_freq_t, d_freq_conj_t, d_red_freq_conj_t, cublasHandle, newmv);

    gpuErrchk(cudaStreamSynchronize(s));
}

void init_hmat(int num_rows, int num_cols, int size, double **hmat)
{
    *hmat = (double *)malloc(sizeof(double) * num_rows * num_cols * size);

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            for (int k = 0; k < size; k++)
            {
                if (k < size / 2)
                {
                    (*hmat)[i * num_cols * size + j * size + k] = 1.0;
                }
                else
                {
                    (*hmat)[i * num_cols * size + j * size + k] = 0.0;
                }
            }
        }
    }
}

void init_vector(int len, int unpad_size, double **vec, bool init_value)
{
    gpuErrchk(cudaMalloc((void **)vec, sizeof(double) * len * unpad_size));

    if (init_value)
    {
        double *h_vec = (double *)malloc(sizeof(double) * len * unpad_size);
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < unpad_size; j++)
            {
                h_vec[i * unpad_size + j] = 1.0;
            }
        }

        // Copy to device
        gpuErrchk(cudaMemcpy(*vec, h_vec, sizeof(double) * len * unpad_size, cudaMemcpyHostToDevice));
        free(h_vec);
    }
}


