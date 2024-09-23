#include "Vector.hpp"

Vector::Vector(Comm& comm, unsigned int num_blocks, unsigned int block_size, std::string row_or_col)
    : comm(comm)
    , num_blocks(num_blocks)
    , block_size(block_size)
    , padded_size(2 * block_size)
    , row_or_col(row_or_col)
{
    // Initialize the vector data structures. If row_or_col is "row", then the vector is a row
    // vector, otherwise it is a column vector. For row vectors, initialize only on row_color == 0,
    // and for column vectors, initialize only on col_color == 0.

    if (row_or_col != "row" && row_or_col != "col") {
        fprintf(stderr, "row_or_col must be either 'row' or 'col'\n");
        MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
    }

    this->comm = comm;
    if (on_grid()) {
        gpuErrchk(cudaMalloc((void**)&d_vec, (size_t)num_blocks * block_size * sizeof(double)));
        // Use MPI to set the global number of blocks
        MPI_Comm grid_comm = (row_or_col == "col") ? comm.get_row_comm() : comm.get_col_comm();
        MPICHECK(MPI_Allreduce(&num_blocks, &glob_num_blocks, 1, MPI_INT, MPI_SUM, grid_comm));
    } else {
        d_vec = nullptr;
    }
}

Vector::Vector(Vector& vec, bool deep_copy)
    : comm(vec.comm)
    , num_blocks(vec.num_blocks)
    , padded_size(vec.padded_size)
    , block_size(vec.block_size)
    , row_or_col(vec.row_or_col)
{
    // Copy constructor for the Vector class. If deep_copy is true, then copy the data from vec,
    // otherwise just allocate memory.
    if (on_grid()) {
        gpuErrchk(cudaMalloc((void**)&d_vec, (size_t)num_blocks * block_size * sizeof(double)));
        if (deep_copy) {
            gpuErrchk(cudaMemcpy(d_vec, vec.d_vec, (size_t)num_blocks * block_size * sizeof(double),
                cudaMemcpyDeviceToDevice));
        }
    } else {
        d_vec = nullptr;
    }
}

Vector& Vector::operator=(Vector& vec)
{
    // Copy assignment operator for the Vector class. Copy the data from vec.
    if (this != &vec) {
        comm = vec.comm;
        num_blocks = vec.num_blocks;
        padded_size = vec.padded_size;
        block_size = vec.block_size;
        row_or_col = vec.row_or_col;

        if (on_grid()) {
            gpuErrchk(cudaMalloc((void**)&d_vec, (size_t)num_blocks * block_size * sizeof(double)));
            gpuErrchk(cudaMemcpy(d_vec, vec.d_vec, (size_t)num_blocks * block_size * sizeof(double),
                cudaMemcpyDeviceToDevice));
        } else {
            d_vec = nullptr;
        }
    }

    return *this;
}

Vector& Vector::operator=(Vector&& vec)
{
    // Move assignment operator for the Vector class. Move the data from vec.
    if (this != &vec) {
        comm = vec.comm;
        num_blocks = vec.num_blocks;
        padded_size = vec.padded_size;
        block_size = vec.block_size;
        row_or_col = vec.row_or_col;
        d_vec = vec.d_vec;
        initialized = vec.initialized;

        vec.d_vec = nullptr;
    }

    return *this;
}

Vector::~Vector()
{
    // Free the memory allocated for the vector
    if (on_grid()) {
        gpuErrchk(cudaFree(d_vec));
    }
}

/*
    Initialize the vector to whatever is in d_vec. Just sets initialized to true.
*/

void Vector::init_vec() { initialized = true; }

void Vector::init_vec_zeros()
{
    // Initialize the vector with zeros
    if (on_grid()) {
        gpuErrchk(cudaMemset(d_vec, 0, (size_t)num_blocks * block_size * sizeof(double)));
    }
    initialized = true;
}

void Vector::init_vec_ones()
{
    // Initialize the vector with ones
    // make double array on host

    if (on_grid()) {
        double* h_vec = new double[num_blocks * block_size];
#pragma omp parallel for
        for (int i = 0; i < num_blocks * block_size; i++) {
            h_vec[i] = 1.0;
        }
        // copy to device
        gpuErrchk(cudaMemcpy(d_vec, h_vec, (size_t)num_blocks * block_size * sizeof(double),
            cudaMemcpyHostToDevice));
        delete[] h_vec;
    }
    initialized = true;
}

void Vector::init_vec_from_file(std::string filename)
{
    // Use HighFive to read in the vector from a file
    // Initialize the vector with the data from the file
    using namespace HighFive;
    if (on_grid()) {
        // read in the vector from the file
        try {
            FileAccessProps fapl;
            MPI_Comm grid_comm = (row_or_col == "col") ? comm.get_row_comm() : comm.get_col_comm();
            fapl.add(MPIOFileAccess { grid_comm, MPI_INFO_NULL });
            fapl.add(MPIOCollectiveMetadata {});

            File file(filename, File::ReadOnly, fapl);
            std::vector<double> vec;

            auto xfer_props = DataTransferProps {};
            xfer_props.add(UseCollectiveIO {});

            auto dataset = file.getDataSet("vec");
            int reindex, n_blocks, steps;

            if (row_or_col == "col") {
                dataset.getAttribute("n_param").read<int>(n_blocks);
                dataset.getAttribute("param_steps").read<int>(steps);
            } else {
                dataset.getAttribute("n_obs").read<int>(n_blocks);
                dataset.getAttribute("obs_steps").read<int>(steps);
            }
            dataset.getAttribute("reindex").read<int>(reindex);

            if (!reindex) {
                fprintf(stderr, "reindex must be true. Got reindex = %d.\n", reindex);
                MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
            } else if (n_blocks != glob_num_blocks) {
                fprintf(stderr,
                    "n_blocks must be equal to glob_num_blocks. Got n_blocks = %d, glob_num_blocks "
                    "= %d.\n",
                    n_blocks, glob_num_blocks);
                MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
            } else if (steps != block_size) {
                fprintf(stderr,
                    "steps must be equal to block_size. Got steps = %d, block_size = %d.\n", steps,
                    block_size);
                MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
            }

            size_t start;

            if (row_or_col == "col") {
                start = (comm.get_col_color() < glob_num_blocks % comm.get_proc_cols())
                    ? (glob_num_blocks / comm.get_proc_cols() + 1) * comm.get_col_color()
                    : (glob_num_blocks / comm.get_proc_cols()) * comm.get_col_color()
                        + glob_num_blocks % comm.get_proc_cols();
            } else {
                start = (comm.get_row_color() < glob_num_blocks % comm.get_proc_rows())
                    ? (glob_num_blocks / comm.get_proc_rows() + 1) * comm.get_row_color()
                    : (glob_num_blocks / comm.get_proc_rows()) * comm.get_row_color()
                        + glob_num_blocks % comm.get_proc_rows();
            }

            dataset.select({ start * block_size }, { (size_t)num_blocks * block_size })
                .read(vec, xfer_props);
            Utils::check_collective_io(xfer_props);
            // copy to device
            gpuErrchk(cudaMemcpy(d_vec, vec.data(),
                (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyHostToDevice));
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
        }
    }
    initialized = true;
}

void Vector::print(std::string name)
{
    // Print the vector to stdout

    double* h_vec;
    if (on_grid()) {
        h_vec = new double[num_blocks * block_size];
        gpuErrchk(cudaMemcpy(h_vec, d_vec, (size_t)num_blocks * block_size * sizeof(double),
            cudaMemcpyDeviceToHost));
    }

    int rank = comm.get_world_rank();
    int group_rank = (row_or_col == "col") ? comm.get_col_color() : comm.get_row_color();
    if (group_rank == 0 && on_grid())
        printf("Vector %s: \n", name.c_str());
    int num_ranks = (row_or_col == "col") ? comm.get_proc_cols() : comm.get_proc_rows();
    for (int r = 0; r < num_ranks; r++) {
        if (group_rank == r && on_grid()) {
            printf("Group Rank %d: \n", group_rank);
            for (int i = 0; i < num_blocks; i++) {
                for (int j = 0; j < block_size; j++) {
                    printf("block: %d, t: %d, val: %f\n", i, j, h_vec[i * block_size + j]);
                }
                printf("\n");
            }
            printf("\n");
        }

        MPICHECK(MPI_Barrier(comm.get_global_comm()));
    }

    if (on_grid())
        delete[] h_vec;
}

double Vector::norm(int order)
{
    // Compute the norm of the vector
    // order is the order of the norm (e.g., "2" for the 2-norm)
    // If the vector is not initialized, print an error message and abort the program.

    if (!initialized) {
        fprintf(stderr, "Vector not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }
    double norm, global_norm = 0.0;
    if (on_grid()) {
        norm = 0.0;
        // use cuBLAS to compute the norm
        cublasHandle_t cublasHandle = comm.get_cublasHandle();
        MPI_Comm grid_comm = (row_or_col == "col") ? comm.get_row_comm() : comm.get_col_comm();

        switch (order) {
        case 1:
#if !FFT_64
            cublasSafeCall(cublasDasum(cublasHandle, num_blocks * block_size, d_vec, 1, &norm));
#else
            cublasSafeCall(
                cublasDasum_64(cublasHandle, (size_t)num_blocks * block_size, d_vec, 1, &norm));
#endif
            MPICHECK(MPI_Reduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, 0, grid_comm));

            break;
        case 2:
#if !FFT_64
            cublasSafeCall(cublasDnrm2(cublasHandle, num_blocks * block_size, d_vec, 1, &norm));
#else
            cublasSafeCall(
                cublasDnrm2_64(cublasHandle, (size_t)num_blocks * block_size, d_vec, 1, &norm));
#endif
            norm = norm * norm;
            MPICHECK(MPI_Reduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, 0, grid_comm));
            global_norm = std::sqrt(global_norm);
            break;
        case -1:
#if !FFT_64
            int max_index;
            cublasSafeCall(
                cublasIdamax(cublasHandle, num_blocks * block_size, d_vec, 1, &max_index));
            cublasSafeCall(cublasGetVector(1, sizeof(double), d_vec + max_index - 1, 1, &norm, 1));
#else
            size_t max_index;
            cublasSafeCall(cublasIdamax_64(
                cublasHandle, (size_t)num_blocks * block_size, d_vec, 1, &max_index));
            cublasSafeCall(
                cublasGetVector_64(1, sizeof(double), d_vec + max_index - 1, 1, &norm, 1));
#endif
            MPICHECK(MPI_Reduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm));

            break;
        default:
            fprintf(stderr, "Invalid vector norm order: %d\n", order);
            MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        }
    }

    return global_norm; // only rank 0 has the global norm
}

void Vector::scale(double alpha)
{
    // Scale the vector by alpha
    // If the vector is not initialized, print an error message and abort the program.

    if (!initialized) {
        fprintf(stderr, "Vector not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    if (on_grid()) {
        // use cuBLAS to scale the vector
        cublasHandle_t cublasHandle = comm.get_cublasHandle();
#if !FFT_64
        cublasSafeCall(cublasDscal(cublasHandle, num_blocks * block_size, &alpha, d_vec, 1));
#else
        cublasSafeCall(
            cublasDscal_64(cublasHandle, (size_t)num_blocks * block_size, &alpha, d_vec, 1));
#endif
    }
}

Vector Vector::wscale(double alpha)
{
    // Copy vector and scale by alpha
    // If the vector is not initialized, print an error message and abort the program.

    if (!initialized) {
        fprintf(stderr, "Vector not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    Vector w(*this, true);
    w.init_vec();
    w.scale(alpha);

    return w;
}

void Vector::axpy(double alpha, Vector& x)
{
    // Compute y = alpha * x + y
    // If the vector is not initialized, print an error message and abort the program.

    if (!initialized) {
        fprintf(stderr, "Vector not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    if (!x.is_initialized()) {
        fprintf(stderr, "Vector x (to be added) not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    if (on_grid()) {
        // use cuBLAS to compute the axpy operation
        cublasHandle_t cublasHandle = comm.get_cublasHandle();
#if !FFT_64
        cublasSafeCall(
            cublasDaxpy(cublasHandle, num_blocks * block_size, &alpha, x.get_d_vec(), 1, d_vec, 1));
#else
        cublasSafeCall(cublasDaxpy_64(
            cublasHandle, (size_t)num_blocks * block_size, &alpha, x.get_d_vec(), 1, d_vec, 1));
#endif
    }
}

Vector Vector::waxpy(double alpha, Vector& x)
{
    // Compute w = alpha * x + y
    // If the vector is not initialized, print an error message and abort the program.

    if (!initialized) {
        fprintf(stderr, "Vector not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    if (!x.is_initialized()) {
        fprintf(stderr, "Vector x (to be added) not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    Vector w(*this, true);
    w.init_vec();
    w.axpy(alpha, x);

    return w;
}

void Vector::axpby(double alpha, double beta, Vector& x)
{
    scale(beta);
    axpy(alpha, x);
}

Vector Vector::waxpby(double alpha, double beta, Vector& x)
{
    Vector w(*this, true);
    w.init_vec();
    w.axpby(alpha, beta, x);

    return w;
}

double Vector::dot(Vector& x)
{
    // Compute the dot product of the vector with x
    // If the vector is not initialized, print an error message and abort the program.

    if (!initialized) {
        fprintf(stderr, "Vector not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    if (!x.is_initialized()) {
        fprintf(stderr, "Vector x (to be dotted) not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    double dot, global_dot = 0.0;
    if (on_grid()) {
        dot = 0.0;
        // use cuBLAS to compute the dot product
        cublasHandle_t cublasHandle = comm.get_cublasHandle();
#if !FFT_64
        cublasSafeCall(
            cublasDdot(cublasHandle, num_blocks * block_size, d_vec, 1, x.get_d_vec(), 1, &dot));
#else
        cublasSafeCall(cublasDdot_64(
            cublasHandle, (size_t)num_blocks * block_size, d_vec, 1, x.get_d_vec(), 1, &dot));
#endif

        // use MPI to compute the global dot product

        MPI_Comm grid_comm = (row_or_col == "col") ? comm.get_row_comm() : comm.get_col_comm();
        MPICHECK(MPI_Reduce(&dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, grid_comm));
    }

    return global_dot; // only rank 0 has the global dot product
}

void Vector::save(std::string filename)
{
    // Use HighFive to save the vector to a file
    // If the vector is not initialized, print an error message and abort the program.

    if (!initialized) {
        fprintf(stderr, "Vector not initialized.\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    using namespace HighFive;
    // use collective I/O to write the vector to a file
    if (on_grid()) {
        try {
            FileAccessProps fapl;
            MPI_Comm grid_comm = (row_or_col == "col") ? comm.get_row_comm() : comm.get_col_comm();
            fapl.add(MPIOFileAccess { grid_comm, MPI_INFO_NULL });
            fapl.add(MPIOCollectiveMetadata {});

            File file(filename, File::Overwrite, fapl);
            std::vector<double> vec(num_blocks * block_size);

            // copy to host
            gpuErrchk(cudaMemcpy(vec.data(), d_vec,
                (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyDeviceToHost));

            auto xfer_props = DataTransferProps {};
            xfer_props.add(UseCollectiveIO {});

            size_t start;

            if (row_or_col == "col") {
                start = (comm.get_col_color() < glob_num_blocks % comm.get_proc_cols())
                    ? (glob_num_blocks / comm.get_proc_cols() + 1) * comm.get_col_color()
                    : (glob_num_blocks / comm.get_proc_cols()) * comm.get_col_color()
                        + glob_num_blocks % comm.get_proc_cols();
            } else {
                start = (comm.get_row_color() < glob_num_blocks % comm.get_proc_rows())
                    ? (glob_num_blocks / comm.get_proc_rows() + 1) * comm.get_row_color()
                    : (glob_num_blocks / comm.get_proc_rows()) * comm.get_row_color()
                        + glob_num_blocks % comm.get_proc_rows();
            }

            std::vector<size_t> dims = { (size_t)glob_num_blocks * block_size };

            DataSet dataset = file.createDataSet<double>("vec", DataSpace(dims));
            dataset.select({ start * block_size }, { (size_t)num_blocks * block_size })
                .write(vec, xfer_props);
            Utils::check_collective_io(xfer_props);

            if (row_or_col == "col") {
                dataset.createAttribute<int>("n_param", glob_num_blocks);
                dataset.createAttribute<int>("param_steps", block_size);
            } else {
                dataset.createAttribute<int>("n_obs", glob_num_blocks);
                dataset.createAttribute<int>("obs_steps", block_size);
            }
            dataset.createAttribute<int>("reindex", 1);

            file.flush();

            if (comm.get_world_rank() == 0)
                printf("Saved vector to %s\n", filename.c_str());
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
        }
    }
}