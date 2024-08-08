#include "Comm.hpp"

/*
    Constructor for Comm class
    @param comm MPI_Comm object for the global communicator
    @param proc_rows Number of rows in the processor grid
    @param proc_cols Number of columns in the processor grid

    Initializes the global communicator and creates row and column communicators
    for the processor grid. Also creates CUDA streams for each processor and GPU communicators (NCCL).

*/
Comm::Comm(MPI_Comm comm, int proc_rows, int proc_cols) : global_comm(comm), proc_rows(proc_rows), proc_cols(proc_cols)
{
    int local_rank = 0;
    global_comm = comm;
    MPI_Comm_rank(global_comm, &world_rank);
    MPI_Comm_size(global_comm, &world_size);

    row_color = world_rank % proc_rows;

    MPICHECK(MPI_Comm_split(global_comm, row_color, world_rank, &row_comm));
    int row_group_rank, row_group_size;
    MPI_Comm_rank(row_comm, &row_group_rank);
    MPI_Comm_size(row_comm, &row_group_size);

    uint64_t hostHashs[world_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[world_rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, global_comm));
    for (int p = 0; p < world_size; p++)
    {
        if (p == world_rank)
            break;
        if (hostHashs[p] == hostHashs[world_rank])
            local_rank++;
    }

    ncclUniqueId row_id;

    if (row_group_rank == 0)
        ncclGetUniqueId(&row_id);

    MPICHECK(MPI_Bcast((void *)&row_id, sizeof(row_id), MPI_BYTE, 0, row_comm));

    // picking a GPU based on local_rank, make stream
    device = local_rank;
    gpuErrchk(cudaSetDevice(local_rank));
    gpuErrchk(cudaStreamCreate(&s));

    NCCLCHECK(ncclCommInitRank(&gpu_row_comm, row_group_size, row_id, row_group_rank));

    col_color = world_rank / proc_rows;
    int col_group_rank, col_group_size;
    ncclUniqueId col_id;

    MPICHECK(MPI_Comm_split(global_comm, col_color, world_rank, &col_comm));

    MPI_Comm_rank(col_comm, &col_group_rank);
    MPI_Comm_size(col_comm, &col_group_size);

    if (col_group_rank == 0)
        ncclGetUniqueId(&col_id);
    MPICHECK(MPI_Bcast((void *)&col_id, sizeof(col_id), MPI_BYTE, 0, col_comm));

    NCCLCHECK(ncclCommInitRank(&gpu_col_comm, col_group_size, col_id, col_group_rank));

    cublasSafeCall(cublasCreate(&(cublasHandle)));
    cublasSafeCall(cublasSetStream(cublasHandle, s));
}


/*
    Destructor for Comm class

    Frees the row and column communicators and destroys the NCCL communicators.
*/
Comm::~Comm()
{
    MPICHECK(MPI_Comm_free(&row_comm));
    MPICHECK(MPI_Comm_free(&col_comm));
    NCCLCHECK(ncclCommDestroy(gpu_row_comm));
    NCCLCHECK(ncclCommDestroy(gpu_col_comm));
    cublasSafeCall(cublasDestroy(cublasHandle));
}

/*

*/