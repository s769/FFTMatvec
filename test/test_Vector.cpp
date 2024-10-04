#include "Comm.hpp"
#include "gtest-mpi-listener.hpp"
#include "shared.hpp"
#include "Vector.hpp"
#include <gtest/gtest.h>

int proc_rows, proc_cols;

class VectorTest : public ::testing::Test {
protected:
    static Comm* comm;
    static Vector *x, *y, *x2, *y2;
    static void SetUpTestSuite()
    {
        int num_rows = 2;
        int num_cols = 3;
        int block_size = 4;
        if (comm == nullptr) {
            comm = new Comm(MPI_COMM_WORLD, proc_rows, proc_cols);
        }
        if (x == nullptr) {
            x = new Vector(*comm, num_cols, block_size, "col");
        }
        if (y == nullptr) {
            y = new Vector(*comm, num_rows, block_size, "row");
        }
        if (x2 == nullptr) {
            x2 = new Vector(*comm, num_cols, block_size, "col");
        }
        if (y2 == nullptr) {
            y2 = new Vector(*comm, num_rows, block_size, "row");
        }
    }
    static void TearDownTestSuite()
    {
        if (comm != nullptr) {
            delete comm;
            comm = nullptr;
        }
        if (x != nullptr) {
            delete x;
            x = nullptr;
        }
        if (y != nullptr) {
            delete y;
            y = nullptr;
        }
        if (x2 != nullptr) {
            delete x2;
            x2 = nullptr;
        }
    }
};

Comm* VectorTest::comm = nullptr;
Vector* VectorTest::x = nullptr;
Vector* VectorTest::y = nullptr;
Vector* VectorTest::x2 = nullptr;
Vector* VectorTest::y2 = nullptr;



TEST_F(VectorTest, Example)
{
    printf("proc_rows = %d, proc_cols = %d\n", proc_rows, proc_cols);
    printf(
        "comm proc_rows = %d, comm proc_cols = %d\n", comm->get_proc_rows(), comm->get_proc_cols());
    ASSERT_EQ(comm->get_proc_rows(), proc_rows);
    ASSERT_EQ(comm->get_proc_cols(), proc_cols);
}

TEST_F(VectorTest, Example2)
{
    printf("proc_rows = %d, proc_cols = %d\n", proc_rows, proc_cols);
    printf(
        "comm proc_rows = %d, comm proc_cols = %d\n", comm->get_proc_rows(), comm->get_proc_cols());
    ASSERT_EQ(comm->get_proc_rows(), proc_rows);
    ASSERT_EQ(comm->get_proc_cols(), proc_cols);
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
    printf("proc_rows = %d, proc_cols = %d\n", proc_rows, proc_cols);
    int result = RUN_ALL_TESTS();

    return result; // Run tests, then clean up and exit
}