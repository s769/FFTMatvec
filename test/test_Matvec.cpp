#include "Matrix.hpp"
#include "Vector.hpp"
#include "gtest-mpi-listener.hpp"
#include "shared.hpp"
#include <gtest/gtest.h>

int proc_rows, proc_cols;


class MatvecTest : public ::testing::Test {
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

Comm* MatvecTest::comm = nullptr;

TEST_F(MatvecTest, FmDTest)
{
    std::string path = dirname(__FILE__) + "/data/";
    Matrix F = Matrix(*comm, path + "test_mat/");
    Vector m = F.get_vec("input");
    m.init_vec_from_file(path + "test_param_vec_SOTI.h5");
    Vector d = F.get_vec("output");
    d.init_vec_from_file(path + "test_obs_vec_SOTI.h5");
    Vector d2 = Vector(d, false);
    d2.init_vec();
    F.matvec(m, d2);
    d2 -= d;
    double norm = d2.norm();
    double norm_true = d.norm();
    double rel_err;
    if (comm->get_world_rank() == 0)
    {
        rel_err = norm / norm_true;
        ASSERT_LT(rel_err, 1e-6);
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