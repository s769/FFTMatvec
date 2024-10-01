

#include "Matrix.hpp"
#include "matvec.hpp"

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
    : comm(comm)
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
        Matvec::setup(&mat_freq_TOSI_aux, h_mat, padded_size, num_cols, num_rows, cublasHandle);
        has_mat_freq_TOSI_aux = true;
    } else {
        Matvec::setup(&mat_freq_TOSI, h_mat, padded_size, num_cols, num_rows, cublasHandle);
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
        Matvec::setup(&mat_freq_TOSI_aux, h_mat, padded_size, num_cols, num_rows, cublasHandle);
        has_mat_freq_TOSI_aux = true;
    } else {
        Matvec::setup(&mat_freq_TOSI, h_mat, padded_size, num_cols, num_rows, cublasHandle);
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
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
            false, true, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan,
            forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq,
            col_vec_freq_TOSI, row_vec_freq_TOSI, cublasHandle, mat_freq_TOSI_aux, res_pad,
            use_aux_mat);
    else if (use_aux_mat)
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_TOSI_aux, padded_size, num_cols, num_rows,
            false, false, device, row_comm, col_comm, s, col_vec_pad, forward_plan, inverse_plan,
            forward_plan_conj, inverse_plan_conj, row_vec_pad, col_vec_freq, row_vec_freq,
            col_vec_freq_TOSI, row_vec_freq_TOSI, cublasHandle, mat_freq_TOSI_aux, res_pad);
    else
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
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
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
            true, true, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj,
            inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_TOSI,
            col_vec_freq_TOSI, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_TOSI_aux, res_pad,
            use_aux_mat);
    else if (use_aux_mat)
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_TOSI_aux, padded_size, num_cols, num_rows,
            true, false, device, row_comm, col_comm, s, row_vec_pad, forward_plan_conj,
            inverse_plan_conj, forward_plan, inverse_plan, col_vec_pad, row_vec_freq_TOSI,
            col_vec_freq_TOSI, row_vec_freq, col_vec_freq, cublasHandle, mat_freq_TOSI_aux,
            res_pad);
    else
        Matvec::compute_matvec(out_vec, in_vec, mat_freq_TOSI, padded_size, num_cols, num_rows,
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