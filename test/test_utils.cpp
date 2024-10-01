#include <gtest/gtest.h>
#include "shared.hpp"
#include "Comm.hpp"

TEST(UtilsTest, Example)
{
    EXPECT_EQ(1, 1);
}

// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     MPICHECK(MPI_Init(&argc, &argv));
//     int result;
//     {
//         Comm comm(MPI_COMM_WORLD, 2, 2);
//         result = RUN_ALL_TESTS();
//     }
//     MPICHECK(MPI_Finalize());
//     return result;
// }