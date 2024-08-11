#include "Comm.hpp"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace testing;

class CommTest : public Test {
protected:
    MPI_Comm comm;
    int proc_rows, proc_cols, world_size, world_rank, row_color, col_color;


    void SetUp() override {
        // Initialize MPI
        MPI_Init(nullptr, nullptr);
        comm = MPI_COMM_WORLD;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

        // Set up the processor grid
        proc_rows = 1;
        proc_cols = 1;
    }

    void TearDown() override {
        // Finalize MPI
        MPI_Finalize();
    }
};

TEST_F(CommTest, Constructor) {
    // Create the Comm object
    Comm commObj(comm, proc_rows, proc_cols);

    // Verify the properties of the Comm object
    EXPECT_EQ(commObj.get_world_size(), world_size);
    EXPECT_EQ(commObj.get_world_rank(), world_rank);
    EXPECT_EQ(commObj.get_proc_rows(), proc_rows);
    EXPECT_EQ(commObj.get_proc_cols(), proc_cols);
    EXPECT_EQ(commObj.get_row_color(), world_rank % proc_rows);
    EXPECT_EQ(commObj.get_col_color(), world_rank / proc_rows);
    // Add more assertions as needed
}

TEST_F(CommTest, ConstructorWithInvalidComm) {
    // Create an invalid MPI_Comm object
    MPI_Comm invalidComm = MPI_COMM_NULL;

    // Verify that the constructor throws an exception when an invalid comm is passed
    EXPECT_THROW(Comm commObj(invalidComm, proc_rows, proc_cols), std::exception);
    // Add more assertions as needed
}

// Add more test cases as needed
