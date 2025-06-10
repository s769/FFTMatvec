
#include "Comm.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "error_checkers.h"
#include "gtest-mpi-listener.hpp"
#include "shared.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

int proc_rows, proc_cols;
int NUM_ROWS = 2;
int NUM_COLS = 3;
int BLOCK_SIZE = 4;

class MatrixTest : public ::testing::Test
{
protected:
    static Comm *comm;
    static Matrix *F;
    static Vector *x, *y, *x2, *y2;

    static void SetUpTestSuite()
    {
        if (comm == nullptr)
        {
            comm = new Comm(MPI_COMM_WORLD, proc_rows, proc_cols);
        }
        if (F == nullptr)
        {
            F = new Matrix(*comm, NUM_COLS, NUM_ROWS, BLOCK_SIZE);
        }
        if (x == nullptr)
        {
            x = new Vector(*comm, NUM_COLS, BLOCK_SIZE, "col");
        }
        if (y == nullptr)
        {
            y = new Vector(*comm, NUM_ROWS, BLOCK_SIZE, "row");
        }
        if (x2 == nullptr)
        {
            x2 = new Vector(*comm, NUM_COLS, BLOCK_SIZE, "col");
        }
        if (y2 == nullptr)
        {
            y2 = new Vector(*comm, NUM_ROWS, BLOCK_SIZE, "row");
        }
    }
    static void TearDownTestSuite()
    {
        if (comm != nullptr)
        {
            delete comm;
            comm = nullptr;
        }
        if (F != nullptr)
        {
            delete F;
            F = nullptr;
        }
        if (x != nullptr)
        {
            delete x;
            x = nullptr;
        }
        if (y != nullptr)
        {
            delete y;
            y = nullptr;
        }
        if (x2 != nullptr)
        {
            delete x2;
            x2 = nullptr;
        }
        if (y2 != nullptr)
        {
            delete y2;
            y2 = nullptr;
        }
    }
    static void check_element(
        double elem, size_t b, size_t j, size_t Nt, size_t Nm, size_t Nd, bool conj, bool full)
    {
        double correct_elem;
        if (conj)
        {
            if (full)
            {
                correct_elem = (Nm * Nd * ((j + 1) * (2 * Nt - j))) / 2.0;
            }
            else
            {
                correct_elem = (Nt - j) * Nd;
            }
        }
        else
        {
            if (full)
            {
                correct_elem = (Nm * Nd * ((Nt - j) * (2 * (j + 1) + (Nt - j - 1)))) / 2.0;
            }
            else
            {
                correct_elem = (j + 1) * Nm;
            }
        }
        ASSERT_NEAR(elem, correct_elem, 1e-10);
    }

    static void check_ones_matvec(Matrix &mat, Vector &vec, bool conj, bool full)
    {

        int proc_rows = comm->get_proc_rows();
        int proc_cols = comm->get_proc_cols();

        int Nt = mat.get_block_size();

        int Nm = mat.get_glob_num_cols();
        int Nd = mat.get_glob_num_rows();

        if (vec.on_grid())
        {
            double *d_vec = vec.get_d_vec();
            int num_blocks = vec.get_num_blocks();
            double *h_vec = new double[num_blocks * Nt];

            gpuErrchk(
                cudaMemcpy(h_vec, d_vec, num_blocks * Nt * sizeof(double), cudaMemcpyDeviceToHost));

            for (size_t i = 0; i < num_blocks; i++)
            {
                for (size_t j = 0; j < Nt; j++)
                {
                    check_element(h_vec[i * Nt + j], i, j, Nt, Nm, Nd, conj, full);
                }
            }

            delete[] h_vec;
        }
    }

    static std::string dirname(const std::string &fname)
    {
        size_t pos = fname.find_last_of("\\/");
        return (std::string::npos == pos) ? "" : fname.substr(0, pos);
    }
};

Comm *MatrixTest::comm = nullptr;
Matrix *MatrixTest::F = nullptr;
Vector *MatrixTest::x = nullptr;
Vector *MatrixTest::y = nullptr;
Vector *MatrixTest::x2 = nullptr;
Vector *MatrixTest::y2 = nullptr;

TEST_F(MatrixTest, MatrixConstructorLocal)
{
    Matrix A = Matrix(*comm, NUM_COLS, NUM_ROWS, BLOCK_SIZE);
    ASSERT_EQ(A.get_num_cols(), NUM_COLS);
    ASSERT_EQ(A.get_num_rows(), NUM_ROWS);
    ASSERT_EQ(A.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(A.get_padded_size(), 2 * BLOCK_SIZE);
    ASSERT_EQ(A.get_glob_num_cols(), NUM_COLS * proc_cols);
    ASSERT_EQ(A.get_glob_num_rows(), NUM_ROWS * proc_rows);
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
    ASSERT_FALSE(A.is_p2q_mat());
    ASSERT_FALSE(A.is_initialized());
}

TEST_F(MatrixTest, MatrixConstructorGlobal)
{
    size_t glob_num_cols = NUM_COLS * proc_cols;
    size_t glob_num_rows = NUM_ROWS * proc_rows;
    Matrix A = Matrix(*comm, glob_num_cols, glob_num_rows, BLOCK_SIZE, true);
    ASSERT_EQ(A.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(A.get_padded_size(), 2 * BLOCK_SIZE);
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
    ASSERT_FALSE(A.is_p2q_mat());
    ASSERT_FALSE(A.is_initialized());
}

TEST_F(MatrixTest, MatrixConstructorP2Q)
{
    Matrix A = Matrix(*comm, NUM_COLS, NUM_ROWS, BLOCK_SIZE, false, true);
    ASSERT_EQ(A.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(A.get_padded_size(), 2 * BLOCK_SIZE);
    ASSERT_EQ(A.get_num_cols(), NUM_COLS);
    ASSERT_EQ(A.get_num_rows(), NUM_ROWS);
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
    ASSERT_TRUE(A.is_p2q_mat());
    ASSERT_FALSE(A.is_initialized());
}

TEST_F(MatrixTest, InitMatOnes)
{
    F->init_mat_ones();
    ASSERT_TRUE(F->is_initialized());
    ComplexD *h_mat = new ComplexD[NUM_COLS * NUM_ROWS * (BLOCK_SIZE + 1)];
    ComplexD *d_mat = F->get_mat_freq_TOSI();
    gpuErrchk(cudaMemcpy(h_mat, d_mat, NUM_COLS * NUM_ROWS * (BLOCK_SIZE + 1) * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));
    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, 2 * BLOCK_SIZE, CUFFT_D2Z, 1));
    double *h_vec = new double[2 * BLOCK_SIZE];
    for (int i = 0; i < 2 * BLOCK_SIZE; i++)
    {
        h_vec[i] = (i < BLOCK_SIZE) ? 1.0 : 0.0;
    }
    double *d_vec;
    gpuErrchk(cudaMalloc(&d_vec, 2 * BLOCK_SIZE * sizeof(double)));
    ComplexD *d_fft;
    gpuErrchk(cudaMalloc(&d_fft, (BLOCK_SIZE + 1) * sizeof(ComplexD)));
    gpuErrchk(cudaMemcpy(d_vec, h_vec, 2 * BLOCK_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    cufftSafeCall(cufftExecD2Z(plan, d_vec, (cufftDoubleComplex *)d_fft));
    cufftSafeCall(cufftDestroy(plan));
    ComplexD *h_fft = new ComplexD[BLOCK_SIZE + 1];
    gpuErrchk(cudaMemcpy(h_fft, d_fft, (BLOCK_SIZE + 1) * sizeof(ComplexD), cudaMemcpyDeviceToHost));

    for (int t = 0; t < BLOCK_SIZE + 1; t++)
    {
        for (int r = 0; r < NUM_ROWS; r++)
        {
            for (int c = 0; c < NUM_COLS; c++)
            {
                size_t ind = t * NUM_ROWS * NUM_COLS + r * NUM_COLS + c;
                ASSERT_NEAR(h_mat[ind].x, h_fft[t].x / (2 * BLOCK_SIZE), 1e-12);
                ASSERT_NEAR(h_mat[ind].y, h_fft[t].y / (2 * BLOCK_SIZE), 1e-12);
            }
        }
    }
    delete[] h_mat;
    delete[] h_vec;
    delete[] h_fft;
    gpuErrchk(cudaFree(d_vec));
    gpuErrchk(cudaFree(d_fft));
}

TEST_F(MatrixTest, InitMatOnesAux)
{
    F->init_mat_ones();
    F->init_mat_ones(true);
    ASSERT_TRUE(F->is_initialized());
    ComplexD *h_mat = new ComplexD[NUM_COLS * NUM_ROWS * (BLOCK_SIZE + 1)];
    ComplexD *d_mat = F->get_mat_freq_TOSI_aux();
    gpuErrchk(cudaMemcpy(h_mat, d_mat, NUM_COLS * NUM_ROWS * (BLOCK_SIZE + 1) * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));
    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, 2 * BLOCK_SIZE, CUFFT_D2Z, 1));
    double *h_vec = new double[2 * BLOCK_SIZE];
    for (int i = 0; i < 2 * BLOCK_SIZE; i++)
    {
        h_vec[i] = (i < BLOCK_SIZE) ? 1.0 : 0.0;
    }
    double *d_vec;
    gpuErrchk(cudaMalloc(&d_vec, 2 * BLOCK_SIZE * sizeof(double)));
    ComplexD *d_fft;
    gpuErrchk(cudaMalloc(&d_fft, (BLOCK_SIZE + 1) * sizeof(ComplexD)));
    gpuErrchk(cudaMemcpy(d_vec, h_vec, 2 * BLOCK_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    cufftSafeCall(cufftExecD2Z(plan, d_vec, (cufftDoubleComplex *)d_fft));
    cufftSafeCall(cufftDestroy(plan));
    ComplexD *h_fft = new ComplexD[BLOCK_SIZE + 1];
    gpuErrchk(cudaMemcpy(h_fft, d_fft, (BLOCK_SIZE + 1) * sizeof(ComplexD), cudaMemcpyDeviceToHost));

    for (int t = 0; t < BLOCK_SIZE + 1; t++)
    {
        for (int r = 0; r < NUM_ROWS; r++)
        {
            for (int c = 0; c < NUM_COLS; c++)
            {
                size_t ind = t * NUM_ROWS * NUM_COLS + r * NUM_COLS + c;
                ASSERT_NEAR(h_mat[ind].x, h_fft[t].x / (2 * BLOCK_SIZE), 1e-12);
                ASSERT_NEAR(h_mat[ind].y, h_fft[t].y / (2 * BLOCK_SIZE), 1e-12);
            }
        }
    }
    delete[] h_mat;
    delete[] h_vec;
    delete[] h_fft;
    gpuErrchk(cudaFree(d_vec));
    gpuErrchk(cudaFree(d_fft));
}

TEST_F(MatrixTest, GetVector)
{
    F->init_mat_ones();
    Vector m = F->get_vec("input");
    ASSERT_EQ(m.get_glob_num_blocks(), NUM_COLS * proc_cols);
    ASSERT_EQ(m.get_num_blocks(), NUM_COLS);
    ASSERT_EQ(m.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(m.get_padded_size(), 2 * BLOCK_SIZE);
    ASSERT_EQ(m.is_initialized(), false);
    ASSERT_EQ(m.get_row_or_col(), "col");

    Vector d = F->get_vec("output");
    ASSERT_EQ(d.get_glob_num_blocks(), NUM_ROWS * proc_rows);
    ASSERT_EQ(d.get_num_blocks(), NUM_ROWS);
    ASSERT_EQ(d.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(d.get_padded_size(), 2 * BLOCK_SIZE);
    ASSERT_EQ(d.is_initialized(), false);
    ASSERT_EQ(d.get_row_or_col(), "row");
}

TEST_F(MatrixTest, OnesMatvec)
{
    F->init_mat_ones();
    x->init_vec_ones();
    y->init_vec_ones();
    F->matvec(*x, *y);
    check_ones_matvec(*F, *y, false, false);
}

TEST_F(MatrixTest, OnesMatvecConj)
{
    F->init_mat_ones();
    x->init_vec_ones();
    y->init_vec_ones();
    F->transpose_matvec(*y, *x);
    check_ones_matvec(*F, *x, true, false);
}

TEST_F(MatrixTest, OnesMatvecFull)
{
    F->init_mat_ones();
    x->init_vec_ones();
    x2->init_vec_ones();
    F->matvec(*x, *x2, false, true);
    check_ones_matvec(*F, *x2, false, true);
}

TEST_F(MatrixTest, OnesMatvecFullConj)
{
    F->init_mat_ones();
    y->init_vec_ones();
    y2->init_vec_ones();
    F->transpose_matvec(*y, *y2, false, true);
    check_ones_matvec(*F, *y2, true, true);
}

TEST_F(MatrixTest, OnesMatvecAux)
{
    F->init_mat_ones();
    F->init_mat_ones(true);
    x->init_vec_ones();
    y->init_vec_ones();
    F->matvec(*x, *y, true);
    check_ones_matvec(*F, *y, false, false);
}

TEST_F(MatrixTest, OnesMatvecConjAux)
{
    F->init_mat_ones();
    F->init_mat_ones(true);
    x->init_vec_ones();
    y->init_vec_ones();
    F->transpose_matvec(*y, *x, true);
    check_ones_matvec(*F, *x, true, false);
}

TEST_F(MatrixTest, OnesMatvecFullAux)
{
    F->init_mat_ones();
    F->init_mat_ones(true);
    x->init_vec_ones();
    x2->init_vec_ones();
    F->matvec(*x, *x2, true, true);
    check_ones_matvec(*F, *x2, false, true);
}

TEST_F(MatrixTest, OnesMatvecFullConjAux)
{
    F->init_mat_ones();
    F->init_mat_ones(true);
    y->init_vec_ones();
    y2->init_vec_ones();
    F->transpose_matvec(*y, *y2, true, true);
    check_ones_matvec(*F, *y2, true, true);
}

TEST_F(MatrixTest, ReadFromFile)
{
    std::string path = dirname(__FILE__) + "/data/test_mat/";
    Matrix F2 = Matrix(*comm, path);
    size_t glob_num_cols = 289;
    size_t glob_num_rows = 15;
    int block_size = 10;
    ASSERT_EQ(F2.get_glob_num_cols(), glob_num_cols);
    ASSERT_EQ(F2.get_glob_num_rows(), glob_num_rows);
    ASSERT_EQ(F2.get_block_size(), block_size);

    ComplexD *h_mat = new ComplexD[F2.get_num_cols() * F2.get_num_rows() * (block_size + 1)];
    ComplexD *d_mat = F2.get_mat_freq_TOSI();
    gpuErrchk(cudaMemcpy(h_mat, d_mat,
                         F2.get_num_cols() * F2.get_num_rows() * (block_size + 1) * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));

    ComplexD result = {0.0, 0.0};
    for (int t = 0; t < block_size + 1; t++)
    {
        for (int r = 0; r < F2.get_num_rows(); r++)
        {
            for (int c = 0; c < F2.get_num_cols(); c++)
            {
                size_t ind = t * F2.get_num_rows() * F2.get_num_cols() + r * F2.get_num_cols() + c;
                result.x += h_mat[ind].x;
                result.y += h_mat[ind].y;
            }
        }
    }
    double glob_result_x, glob_result_y;
    MPICHECK(MPI_Allreduce(&result.x, &glob_result_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPICHECK(MPI_Allreduce(&result.y, &glob_result_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    ASSERT_NEAR(glob_result_x, 5.1536338732773492, 1e-6);
    ASSERT_NEAR(glob_result_y, -3.2036157296392416, 1e-6);

    delete[] h_mat;

    std::string aux_path = dirname(__FILE__) + "/data/test_mat_2/binary/adj/vec_";
    F2.init_mat_from_file(aux_path, true);

    ComplexD *h_mat_aux = new ComplexD[F2.get_num_cols() * F2.get_num_rows() * (block_size + 1)];
    ComplexD *d_mat_aux = F2.get_mat_freq_TOSI_aux();
    gpuErrchk(cudaMemcpy(h_mat_aux, d_mat_aux,
                         F2.get_num_cols() * F2.get_num_rows() * (block_size + 1) * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));

    ComplexD result_aux = {0.0, 0.0};
    for (int t = 0; t < block_size + 1; t++)
    {
        for (int r = 0; r < F2.get_num_rows(); r++)
        {
            for (int c = 0; c < F2.get_num_cols(); c++)
            {
                size_t ind = t * F2.get_num_rows() * F2.get_num_cols() + r * F2.get_num_cols() + c;
                result_aux.x += h_mat_aux[ind].x;
                result_aux.y += h_mat_aux[ind].y;
            }
        }
    }
    double glob_result_x_aux, glob_result_y_aux;
    MPICHECK(
        MPI_Allreduce(&result_aux.x, &glob_result_x_aux, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPICHECK(
        MPI_Allreduce(&result_aux.y, &glob_result_y_aux, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    ASSERT_NEAR(glob_result_x_aux, 393796.81238437677, 1e-6);
    ASSERT_NEAR(glob_result_y_aux, -244782.78303411166, 1e-6);

    delete[] h_mat_aux;
}

TEST_F(MatrixTest, ReadFromFileQoI)
{
    std::string path = dirname(__FILE__) + "/data/test_mat_qoi/";
    std::string aux_path = dirname(__FILE__) + "/data/test_mat_qoi_2/";
    Matrix F2 = Matrix(*comm, path, aux_path, true);
    size_t glob_num_cols = 289;
    size_t glob_num_rows = 15;
    int block_size = 10;
    ASSERT_EQ(F2.get_glob_num_cols(), glob_num_cols);
    ASSERT_EQ(F2.get_glob_num_rows(), glob_num_rows);
    ASSERT_EQ(F2.get_block_size(), block_size);

    ComplexD *h_mat = new ComplexD[F2.get_num_cols() * F2.get_num_rows() * (block_size + 1)];
    ComplexD *d_mat = F2.get_mat_freq_TOSI();
    gpuErrchk(cudaMemcpy(h_mat, d_mat,
                         F2.get_num_cols() * F2.get_num_rows() * (block_size + 1) * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));

    ComplexD result = {0.0, 0.0};
    for (int t = 0; t < block_size + 1; t++)
    {
        for (int r = 0; r < F2.get_num_rows(); r++)
        {
            for (int c = 0; c < F2.get_num_cols(); c++)
            {
                size_t ind = t * F2.get_num_rows() * F2.get_num_cols() + r * F2.get_num_cols() + c;
                result.x += h_mat[ind].x;
                result.y += h_mat[ind].y;
            }
        }
    }
    double glob_result_x, glob_result_y;
    MPICHECK(MPI_Allreduce(&result.x, &glob_result_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPICHECK(MPI_Allreduce(&result.y, &glob_result_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    ASSERT_NEAR(glob_result_x, 5.1536338732773492, 1e-6);
    ASSERT_NEAR(glob_result_y, -3.2036157296392416, 1e-6);

    delete[] h_mat;

    ComplexD *h_mat_aux = new ComplexD[F2.get_num_cols() * F2.get_num_rows() * (block_size + 1)];
    ComplexD *d_mat_aux = F2.get_mat_freq_TOSI_aux();
    gpuErrchk(cudaMemcpy(h_mat_aux, d_mat_aux,
                         F2.get_num_cols() * F2.get_num_rows() * (block_size + 1) * sizeof(ComplexD),
                         cudaMemcpyDeviceToHost));

    ComplexD result_aux = {0.0, 0.0};
    for (int t = 0; t < block_size + 1; t++)
    {
        for (int r = 0; r < F2.get_num_rows(); r++)
        {
            for (int c = 0; c < F2.get_num_cols(); c++)
            {
                size_t ind = t * F2.get_num_rows() * F2.get_num_cols() + r * F2.get_num_cols() + c;
                result_aux.x += h_mat_aux[ind].x;
                result_aux.y += h_mat_aux[ind].y;
            }
        }
    }
    double glob_result_x_aux, glob_result_y_aux;
    MPICHECK(
        MPI_Allreduce(&result_aux.x, &glob_result_x_aux, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    MPICHECK(
        MPI_Allreduce(&result_aux.y, &glob_result_y_aux, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    ASSERT_NEAR(glob_result_x_aux, 393796.81238437677, 1e-6);
    ASSERT_NEAR(glob_result_y_aux, -244782.78303411166, 1e-6);

    delete[] h_mat_aux;

    ASSERT_EQ(F2.is_p2q_mat(), true);
}

int main(int argc, char **argv)
{
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener *l = listeners.Release(listeners.default_result_printer());

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));
    // Run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
    // pass and 1 if some test fails.

    int world_size;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    proc_cols = sqrt(world_size);
    proc_rows = world_size / proc_cols;
    if (proc_rows > proc_cols)
    {
        int temp = proc_cols;
        proc_cols = proc_rows;
        proc_rows = temp;
    }
    int result = RUN_ALL_TESTS();

    return result; // Run tests, then clean up and exit
}