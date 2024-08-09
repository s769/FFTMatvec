#include "Vector.hpp"

/*
    Constructor for Vector class
    @param comm Comm object for the communicator
    @param num_blocks Number of blocks in the vector
    @param block_size Size of each block in the vector

    Initializes the vector object with the given number of blocks and block size.
*/
Vector::Vector(Comm& comm, unsigned int num_blocks, unsigned int block_size, std::string row_or_col) : comm(comm), num_blocks(num_blocks), block_size(block_size), row_or_col(row_or_col)
{
    // Initialize the vector data structures. If row_or_col is "row", then the vector is a row vector, otherwise it is a column vector.
    // For row vectors, initialize only on row_color == 0, and for column vectors, initialize only on col_color == 0.
    this->comm = comm;
    if (on_grid())
    {
        gpuErrchk(cudaMalloc((void **)&d_vec, (size_t)num_blocks * block_size * sizeof(double)));
    }
    else
    {
        d_vec = nullptr;
    }


}

/*
    Copy constructor for Vector class
    @param vec Vector object to copy
    @param deep_copy Whether to perform a deep copy of the vector data or just allocate memory.
*/

Vector::Vector(Vector& vec, bool deep_copy) : comm(vec.comm), num_blocks(vec.num_blocks), block_size(vec.block_size), row_or_col(vec.row_or_col)
{
    // Copy constructor for the Vector class. If deep_copy is true, then copy the data from vec, otherwise just allocate memory.
    if (on_grid())
    {
        gpuErrchk(cudaMalloc((void **)&d_vec, (size_t)num_blocks * block_size * sizeof(double)));
        if (deep_copy)
        {
            gpuErrchk(cudaMemcpy(d_vec, vec.d_vec, (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyDeviceToDevice));
        }
    }
    else
    {
        d_vec = nullptr;
    }
}




/*
    Destructor for Vector class

    Frees the memory allocated for the vector.
*/

Vector::~Vector()
{
    // Free the memory allocated for the vector
    if (on_grid())
    {
        gpuErrchk(cudaFree(d_vec));
    }
}

/*
    Initialize the vector to whatever is in d_vec. Just sets initialized to true.
*/

void Vector::init_vec()
{
    initialized = true;
}


/*
    Initializes the vector with zeros.

    Sets all elements of the vector to zero.
*/

void Vector::init_vec_zeros()
{
    // Initialize the vector with zeros
    if (on_grid())
    {
        gpuErrchk(cudaMemset(d_vec, 0, (size_t)num_blocks * block_size * sizeof(double)));
    }
    initialized = true;
}

/*
    Initializes the vector with ones.

    Sets all elements of the vector to one.
*/

void Vector::init_vec_ones()
{
    // Initialize the vector with ones
    // make double array on host

    if (on_grid())
    {
        double *h_vec = new double[num_blocks * block_size];
#pragma omp parallel for
        for (int i = 0; i < num_blocks * block_size; i++)
        {
            h_vec[i] = 1.0;
        }
        // copy to device
        gpuErrchk(cudaMemcpy(d_vec, h_vec, (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyHostToDevice));
        delete[] h_vec;
    }
    initialized = true;
}


/*
    Prints the vector to stdout.
*/

void Vector::print(std::string name)
{
    // Print the vector to stdout
    if (on_grid())
    {   
        int unpad_size = block_size / 2;
        double *h_vec = new double[num_blocks * unpad_size];
        gpuErrchk(cudaMemcpy(h_vec, d_vec, (size_t)num_blocks * unpad_size * sizeof(double), cudaMemcpyDeviceToHost));


        int rank = comm.get_world_rank();
        int group_rank = (row_or_col == "row") ? comm.get_col_color() : comm.get_row_color();
        if (group_rank == 0)
            printf("Vector %s: \n", name.c_str());
        int num_ranks = (row_or_col == "row") ? comm.get_proc_cols() : comm.get_proc_rows();

        double * h_vec_full;
        if (group_rank == 0)
            h_vec_full = new double[(size_t)num_blocks * unpad_size * num_ranks];

        MPI_Comm group_comm = (row_or_col == "row") ? comm.get_col_comm() : comm.get_row_comm();

        MPICHECK(MPI_Gather(h_vec, (size_t)num_blocks * unpad_size, MPI_DOUBLE, h_vec_full, (size_t)num_blocks * unpad_size, MPI_DOUBLE, 0, group_comm));

        if (group_rank == 0)
        {
            for (int r = 0; r < num_ranks; r++)
            {
                printf("Group Rank %d: \n", group_rank);
                for (int i = 0; i < num_blocks; i++)
                {
                    for (int j = 0; j < unpad_size; j++)
                    {
                    printf("block: %d, t: %d, val: %f\n", i, j, h_vec_full[(size_t)r * num_blocks * unpad_size + (size_t)i * unpad_size + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }

            delete[] h_vec_full;
        }
        
        
        delete[] h_vec;
    }
}


