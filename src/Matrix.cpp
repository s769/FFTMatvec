

#include "Matrix.hpp"

void Matrix::initialize(
    unsigned int cols, unsigned int rows, unsigned int block_size, bool global_sizes)
{
    if (global_sizes) {
        glob_num_cols = cols;
        glob_num_rows = rows;
        num_cols = Utils::global_to_local_size(
            glob_num_cols, comm.get_col_color(), comm.get_proc_cols());
        num_rows = Utils::global_to_local_size(
            glob_num_rows, comm.get_row_color(), comm.get_proc_rows());
    } else {
        num_cols = cols;
        num_rows = rows;
        glob_num_cols = Utils::local_to_global_size(num_cols, comm.get_proc_cols());
        glob_num_rows = Utils::local_to_global_size(num_rows, comm.get_proc_rows());
    }


    fft_int_t n[1] = { (fft_int_t)padded_size };
    int rank = 1;

    fft_int_t idist = padded_size;
    fft_int_t odist = (padded_size / 2 + 1);

    fft_int_t inembed[] = { 0 };
    fft_int_t onembed[] = { 0 };

    fft_int_t istride = 1;
    fft_int_t ostride = 1;
#if !FFT_64
    cufftSafeCall(cufftPlanMany(&(forward_plan), rank, n, inembed, istride, idist, onembed, ostride,
        odist, CUFFT_D2Z, num_cols));
    cufftSafeCall(cufftPlanMany(&(inverse_plan), rank, n, onembed, ostride, odist, inembed, istride,
        idist, CUFFT_Z2D, num_rows));
#else
    size_t ws = 0;
    cufftSafeCall(cufftCreate(&(forward_plan)));
    cufftSafeCall(cufftMakePlanMany64(forward_plan, rank, n, inembed, istride, idist, onembed,
        ostride, odist, CUFFT_D2Z, num_cols, &ws));
    cufftSafeCall(cufftCreate(&(inverse_plan)));
    cufftSafeCall(cufftMakePlanMany64(inverse_plan, rank, n, onembed, ostride, odist, inembed,
        istride, idist, CUFFT_Z2D, num_rows, &ws));
#endif

    cudaStream_t s = comm.get_stream();

    cufftSafeCall(cufftSetStream(forward_plan, s));
    cufftSafeCall(cufftSetStream(inverse_plan, s));
    gpuErrchk(cudaMalloc((void**)&(col_vec_pad), (size_t)num_cols * padded_size * sizeof(double)));
    gpuErrchk(cudaMalloc(
        (void**)&(col_vec_freq), sizeof(Complex) * (size_t)(padded_size / 2 + 1) * num_cols));

    gpuErrchk(cudaMalloc(
        (void**)&(col_vec_freq_tosi), sizeof(Complex) * (size_t)(padded_size / 2 + 1) * num_cols));
    gpuErrchk(cudaMalloc(
        (void**)&(row_vec_freq_tosi), (size_t)sizeof(Complex) * (padded_size / 2 + 1) * num_rows));

    gpuErrchk(cudaMalloc(
        (void**)&(row_vec_freq), (size_t)sizeof(Complex) * (padded_size / 2 + 1) * num_rows));
    gpuErrchk(cudaMalloc((void**)&(row_vec_pad),
        (size_t)sizeof(double) * padded_size * num_rows)); // num_cols * num_rows));

#if !FFT_64
    cufftSafeCall(cufftPlanMany(&(forward_plan_conj), rank, n, inembed, istride, idist, onembed,
        ostride, odist, CUFFT_D2Z, num_rows));
    cufftSafeCall(cufftPlanMany(&(inverse_plan_conj), rank, n, onembed, ostride, odist, inembed,
        istride, idist, CUFFT_Z2D, num_cols));
#else
    cufftSafeCall(cufftCreate(&(forward_plan_conj)));
    cufftSafeCall(cufftMakePlanMany64(forward_plan_conj, rank, n, inembed, istride, idist, onembed,
        ostride, odist, CUFFT_D2Z, num_rows, &ws));
    cufftSafeCall(cufftCreate(&(inverse_plan_conj)));
    cufftSafeCall(cufftMakePlanMany64(inverse_plan_conj, rank, n, onembed, ostride, odist, inembed,
        istride, idist, CUFFT_Z2D, num_cols, &ws));
#endif
    cufftSafeCall(cufftSetStream(forward_plan_conj, s));
    cufftSafeCall(cufftSetStream(inverse_plan_conj, s));

    int max_block_len = (num_cols > num_rows) ? num_cols : num_rows;

    gpuErrchk(cudaMalloc((void**)&(res_pad), sizeof(double) * (size_t)padded_size * max_block_len));

    gpuErrchk(
        cudaMalloc((void**)&(col_vec_unpad), (size_t)num_cols * padded_size / 2 * sizeof(double)));

    gpuErrchk(
        cudaMalloc((void**)&(row_vec_unpad), sizeof(double) * (size_t)padded_size / 2 * num_rows));



}

Matrix::Matrix(Comm& comm, unsigned int cols, unsigned int rows, unsigned int block_size,
    bool global_sizes)
    : comm(comm)
    , block_size(block_size)
    , padded_size(2 * block_size)
{
    // Initialize the matrix data structures
    initialize(cols, rows, block_size, global_sizes);
}

Matrix::Matrix(Comm& comm, std::string path)
    : comm(comm)
{
    std::string meta_filename = path + "/binary/meta_adj";
    std::ifstream infile(meta_filename);
    std::string line;
    int count = 0;
    std::string ext, prefix;
    int reverse_dof, reindexed, glob_num_rows, glob_num_cols, block_size;
    while (std::getline(infile, line)) {
        switch (count) {
        case 0:
            glob_num_rows = std::stoi(line);
            break;
        case 1:
            glob_num_cols = std::stoi(line);
            break;
        case 2:
            block_size = std::stoi(line);
            break;
        case 3:
            prefix = line;
            break;
        case 4:
            ext = line;
            break;
        case 5:
            reverse_dof = std::stoi(line);
            break;
        case 6:
            reindexed = std::stoi(line);
            break;
        default:
            break;
        }
        count++;
    }
    if (!reverse_dof) {
        fprintf(stderr, "reverse_dof must be true. Got reverse_dof = %d.\n", reverse_dof);
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    } else if (!reindexed) {
        fprintf(stderr, "reindex must be true. Got reindex = %d.\n", reindexed);
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    } else if (ext != ".h5") {
        fprintf(stderr, "File extension must be .h5. Got %s.\n", ext.c_str());
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    }
    infile.close();

    this->block_size = block_size;
    this->padded_size = 2 * block_size;

    initialize(glob_num_cols, glob_num_rows, block_size, true);
    init_mat_from_file(path + "/binary/" + prefix);
    if (comm.get_world_rank() == 0)
        printf("Initialized matrix from %s\n", path.c_str());
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
    double* h_mat = new double[(size_t)padded_size * num_cols * num_rows];
#pragma omp parallel for collapse(3)
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            for (int k = 0; k < padded_size; k++) {
                // set to 1 if k < padded_size / 2, 0 otherwise.
                h_mat[(size_t)i * num_cols * padded_size + (size_t)j * padded_size + k]
                    = (k < padded_size / 2) ? 1.0 : 0.0;
            }
        }
    }

    cublasHandle_t cublasHandle = comm.get_cublasHandle();

    Matvec::setup(&mat_freq_tosi, h_mat, padded_size, num_cols, num_rows, cublasHandle);
    delete[] h_mat;
    initialized = true;
}

void Matrix::init_mat_from_file(std::string dirname)
{
    using namespace HighFive;
    double* h_mat = new double[(size_t)padded_size * num_cols * num_rows];

    try {
        FileAccessProps fapl;
        fapl.add(MPIOFileAccess { comm.get_row_comm(), MPI_INFO_NULL });
        fapl.add(MPIOCollectiveMetadata {});

        auto xfer_props = DataTransferProps {};
        xfer_props.add(UseCollectiveIO {});

        size_t row_start = Utils::get_start_index(
            glob_num_rows, comm.get_row_color(), comm.get_proc_rows());

        for (int r = 0; r < num_rows; r++) {
            size_t n_zero = 6;
            std::string vec_str = std::to_string(r + row_start);
            auto zero_pad_vec_str
                = std::string(n_zero - std::min(n_zero, vec_str.length()), '0') + vec_str;
            std::string vec_filename = dirname + zero_pad_vec_str + ".h5";
            File file(vec_filename, File::ReadOnly, fapl);

            auto dataset = file.getDataSet("vec");
            std::vector<double> vec;

            int reindex, n_blocks, steps;
            dataset.getAttribute("reindex").read<int>(reindex);
            dataset.getAttribute("n_param").read<int>(n_blocks);
            dataset.getAttribute("param_steps").read<int>(steps);

            if (!reindex) {
                fprintf(stderr, "reindex must be true. Got reindex = %d.\n", reindex);
                MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
            } else if (steps != block_size) {
                fprintf(stderr,
                    "steps must be equal to block_size. Got steps = %d, block_size = %d.\n", steps,
                    block_size);
                MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
            } else if (n_blocks != glob_num_cols) {
                fprintf(stderr,
                    "n_blocks must be equal to glob_num_cols. Got n_blocks = %d, glob_num_cols = "
                    "%d.\n",
                    n_blocks, glob_num_cols);
                MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
            }

            size_t col_start
                = Utils::get_start_index(glob_num_cols, comm.get_col_color(), comm.get_proc_cols());

            dataset.select({ col_start * block_size }, { (size_t)num_cols * block_size })
                .read(vec, xfer_props);
            Utils::check_collective_io(xfer_props);
#pragma omp parallel for collapse(2)
            for (int c = 0; c < num_cols; c++) {
                for (int t = 0; t < padded_size; t++) {
                    if (t < block_size) {
                        h_mat[(size_t)r * num_cols * padded_size + (size_t)c * padded_size + t]
                            = vec[(size_t)c * block_size + t];
                    } else {
                        h_mat[(size_t)r * num_cols * padded_size + (size_t)c * padded_size + t]
                            = 0.0;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error reading matrix from file: %s\n", e.what());
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    }

    cublasHandle_t cublasHandle = comm.get_cublasHandle();

    Matvec::setup(&mat_freq_tosi, h_mat, padded_size, num_cols, num_rows, cublasHandle);

    delete[] h_mat;

    initialized = true;
}

void Matrix::matvec(Vector& x, Vector& y, bool full)
{
    if (!initialized) {
        fprintf(stderr, "Matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (!x.is_initialized()) {
        fprintf(stderr, "Vector x not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (!y.is_initialized()) {
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
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, padded_size, num_cols, num_rows,
            false, true, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan,
            forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq,
            col_vec_freq_tosi, row_vec_freq_tosi, cublasHandle, mat_freq_tosi_other, res_pad);
    else
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, padded_size, num_cols, num_rows,
            false, false, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan,
            forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq,
            col_vec_freq_tosi, row_vec_freq_tosi, cublasHandle, mat_freq_tosi_other, res_pad);
    gpuErrchk(cudaStreamSynchronize(s));

    if (out_color == 0)
        y.set_d_vec(out_vec);
}

void Matrix::transpose_matvec(Vector& x, Vector& y, bool full)
{
    if (!initialized) {
        fprintf(stderr, "Matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (!x.is_initialized()) {
        fprintf(stderr, "Vector x not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (!y.is_initialized()) {
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
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, padded_size, num_cols, num_rows,
            true, true, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj,
            inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_tosi,
            col_vec_freq_tosi, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_tosi_other,
            res_pad);
    else
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_tosi, padded_size, num_cols, num_rows,
            true, false, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj,
            inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_tosi,
            col_vec_freq_tosi, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_tosi_other,
            res_pad);
    gpuErrchk(cudaStreamSynchronize(s));

    if (out_color == 0)
        y.set_d_vec(out_vec);
}
