
#include "Comm.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "error_checkers.h"
#include "gtest-mpi-listener.hpp"
#include "shared.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

int proc_rows, proc_cols;

class MatrixTest : public ::testing::Test {
protected:
    static Comm* comm;
    static Matrix* F;
    static Vector *x, *y, *x2, *y2;

    static void SetUpTestSuite()
    {
        int num_rows = 2;
        int num_cols = 3;
        int block_size = 4;
        if (comm == nullptr) {
            comm = new Comm(MPI_COMM_WORLD, proc_rows, proc_cols);
        }
        if (F == nullptr) {
            F = new Matrix(*comm, num_cols, num_rows, block_size);
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
        if (F != nullptr) {
            delete F;
            F = nullptr;
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
        if (y2 != nullptr) {
            delete y2;
            y2 = nullptr;
        }
    }
};

Comm* MatrixTest::comm = nullptr;
Matrix* MatrixTest::F = nullptr;
Vector* MatrixTest::x = nullptr;
Vector* MatrixTest::y = nullptr;
Vector* MatrixTest::x2 = nullptr;
Vector* MatrixTest::y2 = nullptr;

TEST_F(MatrixTest, MatrixConstructorLocal)
{
    int num_rows = 2;
    int num_cols = 3;
    int block_size = 4;
    Matrix A = Matrix(*comm, num_cols, num_rows, block_size);
    ASSERT_EQ(A.get_num_cols(), num_cols);
    ASSERT_EQ(A.get_num_rows(), num_rows);
    ASSERT_EQ(A.get_block_size(), block_size);
    ASSERT_EQ(A.get_padded_size(), 2 * block_size);
    ASSERT_EQ(A.get_glob_num_cols(), num_cols * proc_cols);
    ASSERT_EQ(A.get_glob_num_rows(), num_rows * proc_rows);
    ASSERT_NE(A.get_col_vec_freq(), nullptr);
    ASSERT_NE(A.get_row_vec_freq(), nullptr);
    ASSERT_NE(A.get_col_vec_freq_TOSI(), nullptr);
    ASSERT_NE(A.get_row_vec_freq_TOSI(), nullptr);
    ASSERT_NE(A.get_col_vec_pad(), nullptr);
    ASSERT_NE(A.get_row_vec_pad(), nullptr);
    ASSERT_NE(A.get_col_vec_unpad(), nullptr);
    ASSERT_NE(A.get_row_vec_unpad(), nullptr);
    size_t sz1, sz2, sz3, sz4;
    cufftSafeCall(cufftGetSize(A.get_forward_plan(), &sz1));
    cufftSafeCall(cufftGetSize(A.get_forward_plan_conj(), &sz2));
    cufftSafeCall(cufftGetSize(A.get_inverse_plan(), &sz3));
    cufftSafeCall(cufftGetSize(A.get_inverse_plan_conj(), &sz4));
    ASSERT_NE(sz1, 0);
    ASSERT_NE(sz2, 0);
    ASSERT_NE(sz3, 0);
    ASSERT_NE(sz4, 0);
    ASSERT_FALSE(A.is_p2q_mat());
    ASSERT_FALSE(A.is_initialized());
}

TEST_F(MatrixTest, MatrixConstructorGlobal)
{
    size_t glob_num_rows = 2;
    size_t glob_num_cols = 3;
    int block_size = 4;
    Matrix A = Matrix(*comm, glob_num_cols, glob_num_rows, block_size, true);
    ASSERT_EQ(A.get_block_size(), block_size);
    ASSERT_EQ(A.get_padded_size(), 2 * block_size);
    ASSERT_EQ(A.get_glob_num_cols(), glob_num_cols);
    ASSERT_EQ(A.get_glob_num_rows(), glob_num_rows);
    ASSERT_NE(A.get_col_vec_freq(), nullptr);
    ASSERT_NE(A.get_row_vec_freq(), nullptr);
    ASSERT_NE(A.get_col_vec_freq_TOSI(), nullptr);
    ASSERT_NE(A.get_row_vec_freq_TOSI(), nullptr);
    ASSERT_NE(A.get_col_vec_pad(), nullptr);
    ASSERT_NE(A.get_row_vec_pad(), nullptr);
    ASSERT_NE(A.get_col_vec_unpad(), nullptr);
    ASSERT_NE(A.get_row_vec_unpad(), nullptr);
    size_t sz1, sz2, sz3, sz4;
    cufftSafeCall(cufftGetSize(A.get_forward_plan(), &sz1));
    cufftSafeCall(cufftGetSize(A.get_forward_plan_conj(), &sz2));
    cufftSafeCall(cufftGetSize(A.get_inverse_plan(), &sz3));
    cufftSafeCall(cufftGetSize(A.get_inverse_plan_conj(), &sz4));
    ASSERT_NE(sz1, 0);
    ASSERT_NE(sz2, 0);
    ASSERT_NE(sz3, 0);
    ASSERT_NE(sz4, 0);
    ASSERT_FALSE(A.is_p2q_mat());
    ASSERT_FALSE(A.is_initialized());
}

TEST_F(MatrixTest, MatrixConstructorP2Q)
{
    int num_rows = 2;
    int num_cols = 3;
    int block_size = 4;
    Matrix A = Matrix(*comm, num_cols, num_rows, block_size, false, true);
    ASSERT_EQ(A.get_block_size(), block_size);
    ASSERT_EQ(A.get_padded_size(), 2 * block_size);
    ASSERT_EQ(A.get_num_cols(), num_cols);
    ASSERT_EQ(A.get_num_rows(), num_rows);
    ASSERT_NE(A.get_col_vec_freq(), nullptr);
    ASSERT_NE(A.get_row_vec_freq(), nullptr);
    ASSERT_NE(A.get_col_vec_freq_TOSI(), nullptr);
    ASSERT_NE(A.get_row_vec_freq_TOSI(), nullptr);
    ASSERT_NE(A.get_col_vec_pad(), nullptr);
    ASSERT_NE(A.get_row_vec_pad(), nullptr);
    ASSERT_NE(A.get_col_vec_unpad(), nullptr);
    ASSERT_NE(A.get_row_vec_unpad(), nullptr);
    size_t sz1, sz2, sz3, sz4;
    cufftSafeCall(cufftGetSize(A.get_forward_plan(), &sz1));
    cufftSafeCall(cufftGetSize(A.get_forward_plan_conj(), &sz2));
    cufftSafeCall(cufftGetSize(A.get_inverse_plan(), &sz3));
    cufftSafeCall(cufftGetSize(A.get_inverse_plan_conj(), &sz4));
    ASSERT_NE(sz1, 0);
    ASSERT_NE(sz2, 0);
    ASSERT_NE(sz3, 0);
    ASSERT_NE(sz4, 0);
    ASSERT_TRUE(A.is_p2q_mat());
    ASSERT_FALSE(A.is_initialized());
}

TEST_F(MatrixTest, InitMatOnes)
{
    F->init_mat_ones();
    ASSERT_TRUE(F->is_initialized());
    // Complex* h_F = new Complex[F->get_num_cols() * F->get_num_rows() * F->get_padded_size()];

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