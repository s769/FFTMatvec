#include "Comm.hpp"
#include "gtest-mpi-listener.hpp"
#include "comm_error_checkers.h"
#include "error_checkers.h"
#include "shared.hpp"
#include <gtest/gtest.h>

int proc_rows, proc_cols;


TEST(CommTest, Constructor)
{
    Comm comm(MPI_COMM_WORLD, proc_rows, proc_cols);
    ASSERT_EQ(comm.get_global_comm(), MPI_COMM_WORLD);
    ASSERT_EQ(comm.get_proc_rows(), proc_rows);
    ASSERT_EQ(comm.get_proc_cols(), proc_cols);

    int rank, size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    ASSERT_EQ(comm.get_world_rank(), rank);
    ASSERT_EQ(comm.get_world_size(), size);
    ASSERT_EQ(comm.get_row_color(), rank % proc_rows);
    ASSERT_EQ(comm.get_col_color(), rank / proc_rows);

    MPI_Comm row_comm = comm.get_row_comm();
    MPI_Comm col_comm = comm.get_col_comm();
    int row_sz, col_sz;
    MPICHECK(MPI_Comm_size(row_comm, &row_sz));
    MPICHECK(MPI_Comm_size(col_comm, &col_sz));
    ASSERT_EQ(row_sz, proc_cols);
    ASSERT_EQ(col_sz, proc_rows);

    ncclComm_t gpu_row_comm = comm.get_gpu_row_comm();
    ncclComm_t gpu_col_comm = comm.get_gpu_col_comm();

    int gpu_row_size, gpu_col_size;
    NCCLCHECK(ncclCommCount(gpu_row_comm, &gpu_row_size));
    NCCLCHECK(ncclCommCount(gpu_col_comm, &gpu_col_size));
    ASSERT_EQ(gpu_row_size, proc_cols);
    ASSERT_EQ(gpu_col_size, proc_rows);

    int nccl_device;
    NCCLCHECK(ncclCommCuDevice(gpu_row_comm, &nccl_device));
    ASSERT_EQ(nccl_device, comm.get_device());

    cudaStream_t cublas_stream;
    cublasHandle_t cublas_handle = comm.get_cublasHandle();
    cublasSafeCall(cublasGetStream(cublas_handle, &cublas_stream));
    ASSERT_EQ(cublas_stream, comm.get_stream());

    ASSERT_EQ(false, comm.has_external_stream());




}

TEST(CommTest, ConstructorExternalStream)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Comm comm(MPI_COMM_WORLD, proc_rows, proc_cols, stream);
    ASSERT_EQ(comm.get_global_comm(), MPI_COMM_WORLD);
    ASSERT_EQ(comm.get_proc_rows(), proc_rows);
    ASSERT_EQ(comm.get_proc_cols(), proc_cols);

    int rank, size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    ASSERT_EQ(comm.get_world_rank(), rank);
    ASSERT_EQ(comm.get_world_size(), size);
    ASSERT_EQ(comm.get_row_color(), rank % proc_rows);
    ASSERT_EQ(comm.get_col_color(), rank / proc_rows);

    MPI_Comm row_comm = comm.get_row_comm();
    MPI_Comm col_comm = comm.get_col_comm();
    int row_sz, col_sz;
    MPICHECK(MPI_Comm_size(row_comm, &row_sz));
    MPICHECK(MPI_Comm_size(col_comm, &col_sz));
    ASSERT_EQ(row_sz, proc_cols);
    ASSERT_EQ(col_sz, proc_rows);

    ncclComm_t gpu_row_comm = comm.get_gpu_row_comm();
    ncclComm_t gpu_col_comm = comm.get_gpu_col_comm();

    int gpu_row_size, gpu_col_size;
    NCCLCHECK(ncclCommCount(gpu_row_comm, &gpu_row_size));
    NCCLCHECK(ncclCommCount(gpu_col_comm, &gpu_col_size));
    ASSERT_EQ(gpu_row_size, proc_cols);
    ASSERT_EQ(gpu_col_size, proc_rows);

    int nccl_device;
    NCCLCHECK(ncclCommCuDevice(gpu_row_comm, &nccl_device));
    ASSERT_EQ(nccl_device, comm.get_device());

    cudaStream_t cublas_stream;
    cublasHandle_t cublas_handle = comm.get_cublasHandle();
    cublasSafeCall(cublasGetStream(cublas_handle, &cublas_stream));
    ASSERT_EQ(cublas_stream, comm.get_stream());
    ASSERT_EQ(stream, comm.get_stream());
    ASSERT_EQ(true, comm.has_external_stream());




}


int main(int argc, char** argv)
{
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));
    // Run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
    // pass and 1 if some test fails.

    int world_size;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    proc_cols = sqrt(world_size);
    proc_rows = world_size / proc_cols;
    if (proc_rows > proc_cols) {
        int temp = proc_cols;
        proc_cols = proc_rows;
        proc_rows = temp;
    }
    int result = RUN_ALL_TESTS();

    return result; // Run tests, then clean up and exit
}