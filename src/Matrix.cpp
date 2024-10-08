

#include "Matrix.hpp"
#include "util_kernels.hpp"

#if TIME_MPI
enum_array<ProfilerTimesFull, profiler_t, 3> t_list;
enum_array<ProfilerTimes, profiler_t, 10> t_list_f;
enum_array<ProfilerTimes, profiler_t, 10> t_list_fs;
#endif

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
#if !INDICES_64_BIT
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
        (void**)&(col_vec_freq_TOSI), sizeof(Complex) * (size_t)(padded_size / 2 + 1) * num_cols));
    gpuErrchk(cudaMalloc(
        (void**)&(row_vec_freq_TOSI), (size_t)sizeof(Complex) * (padded_size / 2 + 1) * num_rows));

    gpuErrchk(cudaMalloc(
        (void**)&(row_vec_freq), (size_t)sizeof(Complex) * (padded_size / 2 + 1) * num_rows));
    gpuErrchk(cudaMalloc((void**)&(row_vec_pad),
        (size_t)sizeof(double) * padded_size * num_rows)); // num_cols * num_rows));

#if !INDICES_64_BIT
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
    bool global_sizes, bool QoI)
    : comm(comm)
    , block_size(block_size)
    , padded_size(2 * block_size)
    , is_QoI(QoI)
{
    // Initialize the matrix data structures
    initialize(cols, rows, block_size, global_sizes);
}

Matrix::Matrix(Comm& comm, std::string path, std::string aux_path, bool QoI)
    : comm(comm), is_QoI(QoI)
{
    if (aux_path != "" && path == "") {
        fprintf(stderr, "Primary matrix path must be provided when aux_path is provided.\n");
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    }
    std::string meta_filename = path + "/binary/meta_adj";

    std::string prefix = read_meta(meta_filename, QoI, false);
    initialize(glob_num_cols, glob_num_rows, block_size, true);
    init_mat_from_file(path + "/binary/" + prefix);
    if (comm.get_world_rank() == 0)
        printf("Initialized matrix from %s\n", path.c_str());

    if (aux_path != "") {
        std::string aux_meta_filename = aux_path + "/binary/meta_adj";
        std::string aux_prefix = read_meta(aux_meta_filename, QoI, true);
        init_mat_from_file(aux_path + "/binary/" + aux_prefix, true);
        if (comm.get_world_rank() == 0)
            printf("Initialized aux matrix from %s\n", aux_path.c_str());
    }
}

Matrix::~Matrix()
{
    cufftSafeCall(cufftDestroy(forward_plan));
    cufftSafeCall(cufftDestroy(inverse_plan));
    gpuErrchk(cudaFree(col_vec_pad));
    gpuErrchk(cudaFree(col_vec_freq));
    gpuErrchk(cudaFree(col_vec_freq_TOSI));
    gpuErrchk(cudaFree(row_vec_freq_TOSI));
    gpuErrchk(cudaFree(row_vec_freq));
    gpuErrchk(cudaFree(row_vec_pad));

    cufftSafeCall(cufftDestroy(forward_plan_conj));
    cufftSafeCall(cufftDestroy(inverse_plan_conj));

    gpuErrchk(cudaFree(col_vec_unpad));

    gpuErrchk(cudaFree(row_vec_unpad));
    gpuErrchk(cudaFree(res_pad));

    if (initialized)
        gpuErrchk(cudaFree(mat_freq_TOSI));
    if (has_mat_freq_TOSI_aux && initialized)
        gpuErrchk(cudaFree(mat_freq_TOSI_aux));
}

void Matrix::init_mat_ones(bool aux_mat)
{
    if (aux_mat && !initialized) {
        fprintf(stderr, "Primary matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }
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
    if (aux_mat) {
        Matrix::setup_matvec(&mat_freq_TOSI_aux, h_mat, padded_size, num_cols, num_rows, cublasHandle);
        has_mat_freq_TOSI_aux = true;
    } else {
        Matrix::setup_matvec(&mat_freq_TOSI, h_mat, padded_size, num_cols, num_rows, cublasHandle);
        initialized = true;
    }
    delete[] h_mat;
}

std::string Matrix::read_meta(std::string meta_filename, bool QoI, bool aux_mat)
{
    if (aux_mat && !initialized) {
        fprintf(stderr, "Primary matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }

    std::ifstream infile(meta_filename);
    std::string line;
    int count = 0;
    std::string ext, prefix;
    int reverse_dof, reindexed, g_num_rows, g_num_cols, block_sz, is_p2q;
    while (std::getline(infile, line)) {
        switch (count) {
        case 0:
            g_num_rows = std::stoi(line);
            break;
        case 1:
            g_num_cols = std::stoi(line);
            break;
        case 2:
            block_sz = std::stoi(line);
            break;
        case 3:
            prefix = line;
            break;
        case 4:
            ext = line;
            break;
        case 5:
            reindexed = std::stoi(line);
            break;
        case 6:
            is_p2q = std::stoi(line);
            break;
        case 7:
            reverse_dof = std::stoi(line);
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
    } else if (is_p2q != QoI) {
        fprintf(stderr,
            "Mat type p2o/p2q must match meta file. Got type = %d, meta file type = %d.\n", QoI,
            is_p2q);
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    }
    infile.close();

    if (aux_mat) {
        if (g_num_cols != glob_num_cols) {
            fprintf(stderr,
                "p2o-aux global number of columns must match p2o global number of columns. Got "
                "p2o-aux "
                "global cols = %d, p2o global cols = %d.\n",
                g_num_cols, glob_num_cols);
            MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
        } else if (g_num_rows != glob_num_rows) {
            fprintf(stderr,
                "p2o-aux global number of rows must match p2o global number of rows. Got p2o-aux "
                "global rows = %d, p2o global rows = %d.\n",
                g_num_rows, glob_num_rows);
            MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
        } else if (block_sz != block_size) {
            fprintf(stderr,
                "p2o-aux block size must match p2o block size. Got p2o-aux block size = %d, p2o "
                "block "
                "size = %d.\n",
                block_sz, block_size);
            MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
        }
    }

    if (QoI != is_QoI) {
        fprintf(stderr,
            "Primary matrix QoI type must match requested type. Got primary matrix QoI type = %d, "
            "requested type = %d.\n",
            is_QoI, QoI);
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    }

    if (!aux_mat) {
        glob_num_cols = g_num_cols;
        glob_num_rows = g_num_rows;
        block_size = block_sz;
        padded_size = 2 * block_size;
    }
    return prefix;
}

void Matrix::init_mat_from_file(std::string dirname, bool aux_mat)
{
    if (aux_mat && !initialized) {
        fprintf(stderr, "Primary matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }
    using namespace HighFive;
    double* h_mat = new double[(size_t)padded_size * num_cols * num_rows];

    try {
        FileAccessProps fapl;
        fapl.add(MPIOFileAccess { comm.get_row_comm(), MPI_INFO_NULL });
        fapl.add(MPIOCollectiveMetadata {});

        auto xfer_props = DataTransferProps {};
        xfer_props.add(UseCollectiveIO {});

        size_t row_start
            = Utils::get_start_index(glob_num_rows, comm.get_row_color(), comm.get_proc_rows());

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
                    "n_blocks must be equal to glob_num_cols. Got n_blocks = %d, glob_num_cols "
                    "= "
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

    if (aux_mat) {
        Matrix::setup_matvec(&mat_freq_TOSI_aux, h_mat, padded_size, num_cols, num_rows, cublasHandle);
        has_mat_freq_TOSI_aux = true;
    } else {
        Matrix::setup_matvec(&mat_freq_TOSI, h_mat, padded_size, num_cols, num_rows, cublasHandle);
        initialized = true;
    }

    delete[] h_mat;
}

void Matrix::matvec(Vector& x, Vector& y, bool use_aux_mat, bool full)
{
    check_matvec(x, y, false, full, use_aux_mat);

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
        Matrix::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
            false, true, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan,
            forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq,
            col_vec_freq_TOSI, row_vec_freq_TOSI, cublasHandle, mat_freq_TOSI_aux, res_pad,
            use_aux_mat);
    else if (use_aux_mat)
        Matrix::compute_matvec(out_vec, in_vec, mat_freq_TOSI_aux, padded_size, num_cols, num_rows,
            false, false, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan,
            forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq,
            col_vec_freq_TOSI, row_vec_freq_TOSI, cublasHandle, mat_freq_TOSI_aux, res_pad);
    else
        Matrix::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
            false, false, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan,
            forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq,
            col_vec_freq_TOSI, row_vec_freq_TOSI, cublasHandle, mat_freq_TOSI_aux, res_pad);

    gpuErrchk(cudaStreamSynchronize(s));

    if (out_color == 0)
        y.set_d_vec(out_vec);
}

void Matrix::transpose_matvec(Vector& x, Vector& y, bool use_aux_mat, bool full)
{   
    check_matvec(x, y, true, full, use_aux_mat);
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
        Matrix::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
            true, true, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj,
            inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_TOSI,
            col_vec_freq_TOSI, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_TOSI_aux, res_pad,
            use_aux_mat);
    else if (use_aux_mat)
        Matrix::compute_matvec(out_vec, in_vec, mat_freq_TOSI_aux, padded_size, num_cols, num_rows,
            true, false, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj,
            inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_TOSI,
            col_vec_freq_TOSI, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_TOSI_aux,
            res_pad);
    else
        Matrix::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
            true, false, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj,
            inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_TOSI,
            col_vec_freq_TOSI, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_TOSI_aux,
            res_pad);
    gpuErrchk(cudaStreamSynchronize(s));

    if (out_color == 0)
        y.set_d_vec(out_vec);
}

Vector Matrix::get_vec(std::string input_or_output)
{
    if (input_or_output == "input") {
        return Vector(comm, glob_num_cols, block_size, "col", true);
    } else if (input_or_output == "output") {
        return Vector(comm, glob_num_rows, block_size, "row", true);
    } else {
        fprintf(stderr, "Invalid input_or_output descriptor: %s\n", input_or_output.c_str());
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
        exit(1);
    }
}

void Matrix::check_matvec(Vector& x, Vector& y, bool transpose, bool full, bool use_aux_mat) {

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
    } else if (!x.is_SOTI_ordered()) {
        fprintf(stderr, "Vector x must be in SOTI ordering.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (!y.is_SOTI_ordered()) {
        fprintf(stderr, "Vector y must be in SOTI ordering.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (use_aux_mat && !has_mat_freq_TOSI_aux) {
        fprintf(stderr, "Auxiliary matrix not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (x.get_block_size() != block_size) {
        fprintf(stderr,
            "Block size of x must match block size of matrix. Got x block size = %d, "
            "matrix block size = %d.\n",
            x.get_block_size(), block_size);
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (y.get_block_size() != block_size) {
        fprintf(stderr,
            "Block size of y must match block size of matrix. Got y block size = %d, "
            "matrix block size = %d.\n",
            y.get_block_size(), block_size);
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } 
    int in_size, out_size;
    if (full) {
        in_size = (transpose) ? num_rows : num_cols;
        out_size = (transpose) ? num_rows : num_cols;
    }
    else {
        in_size = (transpose) ? num_rows : num_cols;
        out_size = (transpose) ? num_cols : num_rows;
    }
    
    if (x.get_num_blocks() != in_size) {
        fprintf(stderr,
            "Number of blocks in x must match input size of matrix. Got x num blocks = %d, matrix input size = %d.\n",
            x.get_num_blocks(), in_size);
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    } else if (y.get_num_blocks() != out_size) {
        fprintf(stderr,
            "Number of blocks in y must match output size of matrix. Got y num blocks = %d, matrix output size = %d.\n",
            y.get_num_blocks(), out_size);
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        exit(1);
    }

}







void Matrix::setup_matvec(Complex** mat_freq_TOSI, const double* const h_mat, const unsigned int padded_size,
    const unsigned int num_cols, const unsigned int num_rows, cublasHandle_t cublasHandle)
{

    double* d_mat;
    cufftHandle forward_plan_mat;
    const size_t mat_len = (size_t)padded_size * num_cols * num_rows * sizeof(double);

    fft_int_t n[1] = { (fft_int_t)padded_size };
    int rank = 1;

    fft_int_t idist = padded_size;
    fft_int_t odist = (padded_size / 2 + 1);

    fft_int_t inembed[] = { 0 };
    fft_int_t onembed[] = { 0 };

    fft_int_t istride = 1;
    fft_int_t ostride = 1;

#if !INDICES_64_BIT
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
        (void**)mat_freq_TOSI, (size_t)(padded_size / 2 + 1) * num_cols * num_rows * sizeof(Complex)));

#if !ROW_SETUP
    cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat, *mat_freq_TOSI));
#else
    for (int i = 0; i < num_rows; i++) {
        cufftSafeCall(cufftExecD2Z(forward_plan_mat, d_mat + (size_t)i * padded_size * num_cols,
            *mat_freq_TOSI + (size_t)i * num_cols * (padded_size / 2 + 1)));
    }
#endif

    cufftSafeCall(cufftDestroy(forward_plan_mat));
    gpuErrchk(cudaFree(d_mat));

    double scale = 1.0 / padded_size;
#if !INDICES_64_BIT
    cublasSafeCall(cublasZdscal(cublasHandle, (size_t)(padded_size / 2 + 1) * num_cols * num_rows,
        &scale, *mat_freq_TOSI, 1));
#else
    cublasSafeCall(cublasZdscal_64(cublasHandle, (size_t)(padded_size / 2 + 1) * num_cols * num_rows,
        &scale, *mat_freq_TOSI, 1));
#endif

    Complex* d_mat_freq_trans;
    gpuErrchk(cudaMalloc(
        (void**)&d_mat_freq_trans, sizeof(Complex) * (size_t)(padded_size / 2 + 1) * num_cols * num_rows));
    if (num_cols > 1 && num_rows > 1) {
        Utils::swap_axes(*mat_freq_TOSI, d_mat_freq_trans, num_cols, num_rows, (padded_size / 2 + 1));
    } else {
        cuDoubleComplex aa({ 1, 0 });
        cuDoubleComplex bb({ 0, 0 });
        if (num_rows == 1) {
#if !INDICES_64_BIT
            cublasSafeCall(
                cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_cols, (padded_size / 2 + 1), &aa,
                    *mat_freq_TOSI, (padded_size / 2 + 1), &bb, NULL, num_cols, d_mat_freq_trans, num_cols));
#else
            cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_cols,
                (padded_size / 2 + 1), &aa, *mat_freq_TOSI, (padded_size / 2 + 1), &bb, NULL, num_cols,
                d_mat_freq_trans, num_cols));
#endif
        } else {
#if !INDICES_64_BIT
            cublasSafeCall(
                cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_rows, (padded_size / 2 + 1), &aa,
                    *mat_freq_TOSI, (padded_size / 2 + 1), &bb, NULL, num_rows, d_mat_freq_trans, num_rows));
#else
            cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_rows,
                (padded_size / 2 + 1), &aa, *mat_freq_TOSI, (padded_size / 2 + 1), &bb, NULL, num_rows,
                d_mat_freq_trans, num_rows));
#endif
        }
    }

    gpuErrchk(cudaFree(*mat_freq_TOSI));
    *mat_freq_TOSI = d_mat_freq_trans;
}

void Matrix::local_matvec(double* const out_vec, double* const in_vec, const Complex* const d_mat_freq,
    const unsigned int padded_size, const unsigned int num_cols, const unsigned int num_rows,
    const bool conjugate, const bool unpad, const unsigned int device, cufftHandle forward_plan,
    cufftHandle inverse_plan, double* const out_vec_pad, Complex* const in_vec_freq,
    Complex* const out_vec_freq_TOSI, Complex* const in_vec_freq_TOSI, Complex* const out_vec_freq,
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
#if !INDICES_64_BIT
    cublasSafeCall(
        cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, vec_in_len, (padded_size / 2 + 1), &alpha,
            in_vec_freq, (padded_size / 2 + 1), &beta, NULL, vec_in_len, in_vec_freq_TOSI, vec_in_len));

#else
    cublasSafeCall(
        cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, vec_in_len, (padded_size / 2 + 1), &alpha,
            in_vec_freq, (padded_size / 2 + 1), &beta, NULL, vec_in_len, in_vec_freq_TOSI, vec_in_len));
#endif

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::TRANS1].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::SBGEMV].start();
#endif

    cublasOperation_t transa = (conjugate) ? CUBLAS_OP_C : CUBLAS_OP_N;

#if !INDICES_64_BIT
    cublasSafeCall(cublasZgemvStridedBatched(cublasHandle, transa, num_rows, num_cols, &alpha,
        d_mat_freq, num_rows, (size_t)num_rows * num_cols, in_vec_freq_TOSI, 1, vec_in_len, &beta,
        out_vec_freq_TOSI, 1, vec_out_len, (padded_size / 2 + 1)));

#else
    cublasSafeCall(cublasZgemvStridedBatched_64(cublasHandle, transa, num_rows, num_cols, &alpha,
        d_mat_freq, num_rows, (size_t)num_rows * num_cols, in_vec_freq_TOSI, 1, vec_in_len, &beta,
        out_vec_freq_TOSI, 1, vec_out_len, (padded_size / 2 + 1)));
#endif

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::SBGEMV].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::TRANS2].start();
#endif

#if !INDICES_64_BIT
    cublasSafeCall(cublasZgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (padded_size / 2 + 1), vec_out_len,
        &alpha, out_vec_freq_TOSI, vec_out_len, &beta, NULL, (padded_size / 2 + 1), out_vec_freq,
        (padded_size / 2 + 1)));
#else
    cublasSafeCall(cublasZgeam_64(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (padded_size / 2 + 1),
        vec_out_len, &alpha, out_vec_freq_TOSI, vec_out_len, &beta, NULL, (padded_size / 2 + 1),
        out_vec_freq, (padded_size / 2 + 1)));
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

    UtilKernels::unpad_repad_vector(out_vec_pad, out_vec, vec_out_len, padded_size, unpad, s);

#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::UNPAD].stop();
#endif
}

void Matrix::compute_matvec(double* out_vec, double* in_vec, Complex* mat_freq_TOSI, const unsigned int padded_size,
    const unsigned int num_cols, const unsigned int num_rows, const bool conjugate, const bool full,
    const unsigned int device, ncclComm_t nccl_row_comm, ncclComm_t nccl_col_comm,
    cudaStream_t s, double* const in_vec_pad, cufftHandle forward_plan, cufftHandle inverse_plan,
    cufftHandle forward_plan_conj, cufftHandle inverse_plan_conj, double* const out_vec_pad,
    Complex* const in_vec_freq, Complex* const out_vec_freq_TOSI, Complex* const in_vec_freq_TOSI,
    Complex* const out_vec_freq, cublasHandle_t cublasHandle, Complex* mat_freq_TOSI_aux,
    double* const res_pad, bool use_aux_mat)
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
        (const void*)in_vec, (void*)in_vec, (size_t)vec_in_len * padded_size / 2, ncclDouble, 0, comm, s));
    gpuErrchk(cudaStreamSynchronize(s));
#if TIME_MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    (*tl)[ProfilerTimes::BROADCAST].stop();
#endif

#if TIME_MPI
    (*tl)[ProfilerTimes::PAD].start();
#endif
    UtilKernels::pad_vector(in_vec, in_vec_pad, vec_in_len, padded_size, s);
#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
    (*tl)[ProfilerTimes::PAD].stop();
#endif

    Complex *mat_freq_TOSI1, *mat_freq_TOSI2;

    if (full && use_aux_mat) {
        if (conjugate) {
            mat_freq_TOSI1 = (mat_freq_TOSI_aux) ? mat_freq_TOSI_aux : mat_freq_TOSI;
            mat_freq_TOSI2 = mat_freq_TOSI;
        } else {
            mat_freq_TOSI1 = mat_freq_TOSI;
            mat_freq_TOSI2 = (mat_freq_TOSI_aux) ? mat_freq_TOSI_aux : mat_freq_TOSI;
        }
    } else {
        mat_freq_TOSI1 = mat_freq_TOSI;
        mat_freq_TOSI2 = mat_freq_TOSI;
    }

    double* res_vec = (full) ? res_pad : out_vec;

    local_matvec(res_vec, in_vec_pad, mat_freq_TOSI1, padded_size, num_cols, num_rows, conjugate, !(full),
        device, forward_plan, inverse_plan, out_vec_pad, in_vec_freq, out_vec_freq_TOSI, in_vec_freq_TOSI, out_vec_freq, s,
        cublasHandle);
#if TIME_MPI
    gpuErrchk(cudaDeviceSynchronize());
#endif

    if (!full) {
#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl)[ProfilerTimes::NCCLC].start();
#endif
        NCCLCHECK(ncclReduce((const void*)res_vec, (void*)res_vec, (size_t)vec_out_len * padded_size / 2,
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
        NCCLCHECK(ncclAllReduce((const void*)res_vec, (void*)res_vec, (size_t)vec_out_len * padded_size,
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

        local_matvec(out_vec, res_vec, mat_freq_TOSI2, padded_size, num_cols, num_rows, !(conjugate), true,
            device, forward_plan_conj, inverse_plan_conj, in_vec_pad, out_vec_freq, in_vec_freq_TOSI,
            out_vec_freq_TOSI, in_vec_freq, s, cublasHandle);

#if TIME_MPI
        gpuErrchk(cudaDeviceSynchronize());
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl2)[ProfilerTimes::NCCLC].start();
#endif
        NCCLCHECK(ncclReduce((const void*)out_vec, (void*)out_vec, (size_t)vec_in_len * padded_size / 2,
            ncclDouble, ncclSum, 0, comm, s));
        gpuErrchk(cudaStreamSynchronize(s));

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        (*tl2)[ProfilerTimes::NCCLC].stop();
        (*tl2)[ProfilerTimes::TOT].stop();
#endif
    }
}

