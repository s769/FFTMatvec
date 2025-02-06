#include "Comm.hpp"
#include "Vector.hpp"
#include "gtest-mpi-listener.hpp"
#include "shared.hpp"
#include <gtest/gtest.h>


int proc_rows, proc_cols;
int NUM_BLOCKS = 3, BLOCK_SIZE = 4;

class VectorTest : public ::testing::Test {
protected:
    static Comm* comm;
    static void SetUpTestSuite()
    {
        if (comm == nullptr) {
            comm = new Comm(MPI_COMM_WORLD, proc_rows, proc_cols);
        }
    }
    static void TearDownTestSuite()
    {
        if (comm != nullptr) {
            delete comm;
            comm = nullptr;
        }
    }

    static std::string dirname(const std::string& fname)
    {
        size_t pos = fname.find_last_of("\\/");
        return (std::string::npos == pos) ? "" : fname.substr(0, pos);
    }
};

Comm* VectorTest::comm = nullptr;

TEST_F(VectorTest, ConstructorTestCol)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    ASSERT_EQ(x.get_num_blocks(), NUM_BLOCKS);
    ASSERT_EQ(x.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(x.get_padded_size(), 2 * BLOCK_SIZE);
    ASSERT_EQ(x.get_glob_num_blocks(), NUM_BLOCKS * proc_cols);
    ASSERT_EQ(x.get_row_or_col(), "col");
    ASSERT_EQ(x.is_SOTI_ordered(), true);
    ASSERT_EQ(x.is_initialized(), false);
    ASSERT_EQ(x.on_grid(),
        (x.get_row_or_col() == "col") ? (comm->get_row_color() == 0)
                                      : (comm->get_col_color() == 0));
    if (x.on_grid()) {
        ASSERT_NE(x.get_d_vec(), nullptr);
    } else {
        ASSERT_EQ(x.get_d_vec(), nullptr);
    }
}

TEST_F(VectorTest, ConstructorTestRow)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "row");
    ASSERT_EQ(x.get_num_blocks(), NUM_BLOCKS);
    ASSERT_EQ(x.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(x.get_padded_size(), 2 * BLOCK_SIZE);
    ASSERT_EQ(x.get_glob_num_blocks(), NUM_BLOCKS * proc_rows);
    ASSERT_EQ(x.get_row_or_col(), "row");
    ASSERT_EQ(x.is_SOTI_ordered(), true);
    ASSERT_EQ(x.is_initialized(), false);
    ASSERT_EQ(x.on_grid(),
        (x.get_row_or_col() == "col") ? (comm->get_row_color() == 0)
                                      : (comm->get_col_color() == 0));
    if (x.on_grid()) {
        ASSERT_NE(x.get_d_vec(), nullptr);
    } else {
        ASSERT_EQ(x.get_d_vec(), nullptr);
    }
}

TEST_F(VectorTest, ConstructorTestGlobal)
{
    size_t glob_num_blocks = NUM_BLOCKS * proc_cols;
    Vector x = Vector(*comm, glob_num_blocks, BLOCK_SIZE, "col", true);
    ASSERT_EQ(x.get_block_size(), BLOCK_SIZE);
    ASSERT_EQ(x.get_padded_size(), 2 * BLOCK_SIZE);
    ASSERT_EQ(x.get_glob_num_blocks(), glob_num_blocks);
    ASSERT_EQ(x.get_row_or_col(), "col");
    ASSERT_EQ(x.is_SOTI_ordered(), true);
    ASSERT_EQ(x.is_initialized(), false);
    ASSERT_EQ(x.on_grid(),
        (x.get_row_or_col() == "col") ? (comm->get_row_color() == 0)
                                      : (comm->get_col_color() == 0));
    if (x.on_grid()) {
        ASSERT_NE(x.get_d_vec(), nullptr);
    } else {
        ASSERT_EQ(x.get_d_vec(), nullptr);
    }
}

TEST_F(VectorTest, InitializeZeros)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_zeros();
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 0);
        }
        delete[] h_vec;
    }
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "row");
    y.init_vec_zeros();
    if (y.on_grid()) {
        double* h_vec = new double[(size_t)y.get_num_blocks() * y.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, y.get_d_vec(),
            (size_t)y.get_num_blocks() * y.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < y.get_num_blocks() * y.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, Initialize)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec();
    ASSERT_EQ(x.is_initialized(), true);
}

TEST_F(VectorTest, InitializeOnes)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 1.0);
        }
        delete[] h_vec;
    }
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "row");
    y.init_vec_ones();
    if (y.on_grid()) {
        double* h_vec = new double[(size_t)y.get_num_blocks() * y.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, y.get_d_vec(),
            (size_t)y.get_num_blocks() * y.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < y.get_num_blocks() * y.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 1.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, InitializeConsecutive)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_consecutive();
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        size_t start
            = (x.get_row_or_col() == "col") ? comm->get_col_color() : comm->get_row_color();
        start *= x.get_num_blocks() * x.get_block_size();
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], start + i);
        }
        delete[] h_vec;
    }
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "row");
    y.init_vec_consecutive();
    if (y.on_grid()) {
        double* h_vec = new double[(size_t)y.get_num_blocks() * y.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, y.get_d_vec(),
            (size_t)y.get_num_blocks() * y.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        size_t start
            = (y.get_row_or_col() == "col") ? comm->get_col_color() : comm->get_row_color();
        start *= y.get_num_blocks() * y.get_block_size();
        for (size_t i = 0; i < y.get_num_blocks() * y.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], start + i);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, Copy)
{

    Vector xx = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector x = Vector(xx, false);
    ASSERT_EQ(x.get_num_blocks(), x.get_num_blocks());
    ASSERT_EQ(x.get_block_size(), x.get_block_size());
    ASSERT_EQ(x.get_padded_size(), x.get_padded_size());
    ASSERT_EQ(x.get_glob_num_blocks(), x.get_glob_num_blocks());
    ASSERT_EQ(x.get_row_or_col(), x.get_row_or_col());
    ASSERT_EQ(x.is_SOTI_ordered(), x.is_SOTI_ordered());
    ASSERT_EQ(x.is_initialized(), false);
    if (x.on_grid()) {
        ASSERT_NE(x.get_d_vec(), nullptr);
    } else {
        ASSERT_EQ(x.get_d_vec(), nullptr);
    }
}

TEST_F(VectorTest, DeepCopy)
{

    Vector xx = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    xx.init_vec_ones();
    Vector x = Vector(xx);
    ASSERT_EQ(x.get_num_blocks(), x.get_num_blocks());
    ASSERT_EQ(x.get_block_size(), x.get_block_size());
    ASSERT_EQ(x.get_padded_size(), x.get_padded_size());
    ASSERT_EQ(x.get_glob_num_blocks(), x.get_glob_num_blocks());
    ASSERT_EQ(x.get_row_or_col(), x.get_row_or_col());
    ASSERT_EQ(x.is_SOTI_ordered(), x.is_SOTI_ordered());
    ASSERT_EQ(x.is_initialized(), true);
    if (x.on_grid()) {
        double* h_vec_1 = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec_1, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        double* h_vec_2 = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec_2, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec_1[i], h_vec_2[i]);
        }
        delete[] h_vec_1;
        delete[] h_vec_2;
    } else {
        ASSERT_EQ(x.get_d_vec(), nullptr);
    }
}

TEST_F(VectorTest, Move)
{

    Vector xx = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    size_t glob_num_blocks = xx.get_glob_num_blocks();
    int block_size = xx.get_block_size();
    int padded_size = xx.get_padded_size();
    int num_blocks = xx.get_num_blocks();
    std::string row_or_col = xx.get_row_or_col();
    bool SOTI_ordered = xx.is_SOTI_ordered();
    bool initialized = xx.is_initialized();
    double* h_vec_1;
    if (xx.on_grid()) {
        h_vec_1 = new double[(size_t)xx.get_num_blocks() * xx.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec_1, xx.get_d_vec(),
            (size_t)xx.get_num_blocks() * xx.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    Vector x = std::move(xx);
    ASSERT_EQ(x.get_num_blocks(), num_blocks);
    ASSERT_EQ(x.get_block_size(), block_size);
    ASSERT_EQ(x.get_padded_size(), padded_size);
    ASSERT_EQ(x.get_glob_num_blocks(), glob_num_blocks);
    ASSERT_EQ(x.get_row_or_col(), row_or_col);
    ASSERT_EQ(x.is_SOTI_ordered(), SOTI_ordered);
    ASSERT_EQ(x.is_initialized(), initialized);
    if (x.on_grid()) {
        double* h_vec_2 = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec_2, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec_1[i], h_vec_2[i]);
        }
        delete[] h_vec_2;
        delete[] h_vec_1;
    } else {
        ASSERT_EQ(x.get_d_vec(), nullptr);
    }
}

TEST_F(VectorTest, Assignment)
{

    Vector xx = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    size_t glob_num_blocks = xx.get_glob_num_blocks();
    int block_size = xx.get_block_size();
    int padded_size = xx.get_padded_size();
    int num_blocks = xx.get_num_blocks();
    std::string row_or_col = xx.get_row_or_col();
    bool SOTI_ordered = xx.is_SOTI_ordered();
    bool initialized = xx.is_initialized();
    double* h_vec_1;
    if (xx.on_grid()) {
        h_vec_1 = new double[(size_t)xx.get_num_blocks() * xx.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec_1, xx.get_d_vec(),
            (size_t)xx.get_num_blocks() * xx.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    Vector x = xx;
    ASSERT_EQ(x.get_num_blocks(), num_blocks);
    ASSERT_EQ(x.get_block_size(), block_size);
    ASSERT_EQ(x.get_padded_size(), padded_size);
    ASSERT_EQ(x.get_glob_num_blocks(), glob_num_blocks);
    ASSERT_EQ(x.get_row_or_col(), row_or_col);
    ASSERT_EQ(x.is_SOTI_ordered(), SOTI_ordered);
    ASSERT_EQ(x.is_initialized(), initialized);
    if (x.on_grid()) {
        double* h_vec_2 = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec_2, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec_1[i], h_vec_2[i]);
        }
        delete[] h_vec_2;
        delete[] h_vec_1;
    } else {
        ASSERT_EQ(x.get_d_vec(), nullptr);
    }
}

TEST_F(VectorTest, AXPY)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    x.axpy(2.0, y);
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 3.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, WAXPY)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    Vector z = x.waxpy(2.0, y);
    if (z.on_grid()) {
        double* h_vec = new double[(size_t)z.get_num_blocks() * z.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z.get_d_vec(),
            (size_t)z.get_num_blocks() * z.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z.get_num_blocks() * z.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 3.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, AXPBY)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    x.axpby(2.0, 3.0, y);
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 5.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, WAXPBY)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    Vector z = x.waxpby(2.0, 3.0, y);
    if (z.on_grid()) {
        double* h_vec = new double[(size_t)z.get_num_blocks() * z.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z.get_d_vec(),
            (size_t)z.get_num_blocks() * z.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z.get_num_blocks() * z.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 5.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, Scale)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    x.scale(2.0);
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 2.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, WScale)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    Vector z = x.wscale(2.0);
    if (z.on_grid()) {
        double* h_vec = new double[(size_t)z.get_num_blocks() * z.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z.get_d_vec(),
            (size_t)z.get_num_blocks() * z.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z.get_num_blocks() * z.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 2.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, Dot)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    double dot = x.dot(y);
    if (comm->get_world_rank() == 0) {
        ASSERT_EQ(dot, x.get_glob_num_blocks() * x.get_block_size());
    }
}

TEST_F(VectorTest, Norm2)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    double norm = x.norm();
    if (comm->get_world_rank() == 0) {
        ASSERT_EQ(norm, sqrt(x.get_glob_num_blocks() * x.get_block_size()));
    }
}

TEST_F(VectorTest, Norm1)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    double norm = x.norm(1);
    if (comm->get_world_rank() == 0) {
        ASSERT_EQ(norm, x.get_glob_num_blocks() * x.get_block_size());
    }
}

TEST_F(VectorTest, NormInf)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    double norm = x.norm(-1);
    if (comm->get_world_rank() == 0) {
        ASSERT_EQ(norm, 1.0);
    }
}

TEST_F(VectorTest, PlusOperator)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    Vector z = x + y;
    if (z.on_grid()) {
        double* h_vec = new double[(size_t)z.get_num_blocks() * z.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z.get_d_vec(),
            (size_t)z.get_num_blocks() * z.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z.get_num_blocks() * z.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 2.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, MinusOperator)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    Vector z = x - y;
    if (z.on_grid()) {
        double* h_vec = new double[(size_t)z.get_num_blocks() * z.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z.get_d_vec(),
            (size_t)z.get_num_blocks() * z.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z.get_num_blocks() * z.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 0.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, ScalarMult)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    Vector z = 2.0 * x;
    if (z.on_grid()) {
        double* h_vec = new double[(size_t)z.get_num_blocks() * z.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z.get_d_vec(),
            (size_t)z.get_num_blocks() * z.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z.get_num_blocks() * z.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 2.0);
        }
        delete[] h_vec;
    }

    x.init_vec_ones();
    Vector z2 = x * 2.0;
    if (z2.on_grid()) {
        double* h_vec = new double[(size_t)z2.get_num_blocks() * z2.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z2.get_d_vec(),
            (size_t)z2.get_num_blocks() * z2.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z2.get_num_blocks() * z2.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 2.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, ScalarDiv)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    Vector z = x / 2.0;
    if (z.on_grid()) {
        double* h_vec = new double[(size_t)z.get_num_blocks() * z.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, z.get_d_vec(),
            (size_t)z.get_num_blocks() * z.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < z.get_num_blocks() * z.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 0.5);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, PlusEqual)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    x += y;
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 2.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, MinusEqual)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    x -= y;
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 0.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, ScalarMultEqual)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    x *= 2.0;
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 2.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, ScalarDivEqual)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    x /= 2.0;
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 0.5);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, SetDVec)
{
    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
    for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
        h_vec[i] = i;
    }
    double* d_vec;
    gpuErrchk(cudaMalloc(&d_vec, (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_vec, h_vec,
        (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double), cudaMemcpyHostToDevice));
    x.set_d_vec(d_vec);

    if (x.on_grid()) {
        double* h_vec_2 = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec_2, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], h_vec_2[i]);
        }
        delete[] h_vec_2;
    }
}

TEST_F(VectorTest, DotOperator)
{

    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    y.init_vec_ones();
    double dot = x * y;
    if (comm->get_world_rank() == 0) {
        ASSERT_EQ(dot, x.get_glob_num_blocks() * x.get_block_size());
    }
}

TEST_F(VectorTest, CopyToVector)
{
    
    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_ones();
    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    y.init_vec();
    x.copy(y);
    if (y.on_grid()) {
        double* h_vec = new double[(size_t)y.get_num_blocks() * y.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, y.get_d_vec(),
            (size_t)y.get_num_blocks() * y.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < y.get_num_blocks() * y.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], 1.0);
        }
        delete[] h_vec;
    }
}

TEST_F(VectorTest, ReadFromFile)
{
    std::string filename_param = dirname(__FILE__) + "/data/test_param_vec_SOTI.h5";
    Vector x = Vector(*comm, 289, 10, "col",true);
    x.init_vec_from_file(filename_param);
    double norm = x.norm();
    if (comm->get_world_rank() == 0) {
        ASSERT_NEAR(norm, 0.0009506339491893347, 1e-10);
    }
    std::string filename_obs = dirname(__FILE__) + "/data/test_obs_vec_SOTI.h5";
    Vector y = Vector(*comm, 15, 10, "row",true);
    y.init_vec_from_file(filename_obs);
    double norm2 = y.norm();
    if (comm->get_world_rank() == 0) {
        ASSERT_NEAR(norm2, 0.0004194688794230295, 1e-10);
    }
    std::string filename_qoi = dirname(__FILE__) + "/data/test_qoi_vec_SOTI.h5";
    Vector z = Vector(*comm, 15, 10, "row",true);
    z.init_vec_from_file(filename_qoi,true);
    double norm3 = z.norm();
    if (comm->get_world_rank() == 0) {
        ASSERT_NEAR(norm3, 0.0004194688794230295, 1e-10);
    }
}

TEST_F(VectorTest, WriteToFile)
{
    std::string filename = dirname(__FILE__) + "/data/test_vec.h5";
    Vector x = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x.init_vec_consecutive();
    x.save(filename);
    Vector x2 = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "col");
    x2.init_vec_from_file(filename);
    if (x.on_grid()) {
        double* h_vec = new double[(size_t)x.get_num_blocks() * x.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, x.get_d_vec(),
            (size_t)x.get_num_blocks() * x.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        double* h_vec2 = new double[(size_t)x2.get_num_blocks() * x2.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec2, x2.get_d_vec(),
            (size_t)x2.get_num_blocks() * x2.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.get_num_blocks() * x.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], h_vec2[i]);
        }
        delete[] h_vec;
        delete[] h_vec2;
    }

    Vector y = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "row");
    y.init_vec_consecutive();
    y.save(filename);
    Vector y2 = Vector(*comm, NUM_BLOCKS, BLOCK_SIZE, "row");
    y2.init_vec_from_file(filename);
    if (y.on_grid()) {
        double* h_vec = new double[(size_t)y.get_num_blocks() * y.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec, y.get_d_vec(),
            (size_t)y.get_num_blocks() * y.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        double* h_vec2 = new double[(size_t)y2.get_num_blocks() * y2.get_block_size()];
        gpuErrchk(cudaMemcpy(h_vec2, y2.get_d_vec(),
            (size_t)y2.get_num_blocks() * y2.get_block_size() * sizeof(double),
            cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < y.get_num_blocks() * y.get_block_size(); i++) {
            ASSERT_EQ(h_vec[i], h_vec2[i]);
        }
        delete[] h_vec;
        delete[] h_vec2;
    }
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

