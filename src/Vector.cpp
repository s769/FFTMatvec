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

void Vector::axpby(double alpha, double beta, Vector& x)
{
    scale(beta);
    axpy(alpha, x);
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