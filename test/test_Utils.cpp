#include "shared.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

TEST(UtilsTest, GetHostHash)
{
    uint64_t hash = Utils::get_host_hash("localhost");
    ASSERT_EQ(hash, 249786565182708392);
}

TEST(UtilsTest, GetHostName)
{
    char hostname[256];
    Utils::get_host_name(hostname, 256);
    ASSERT_TRUE(strlen(hostname) > 0);
}

TEST(UtilsTest, SwapAxes)
{
    int num_cols = 3;
    int num_rows = 2;
    int padded_size = 4;
    Complex* d_in;
    Complex* d_out;
    gpuErrchk(cudaMalloc(&d_in, (size_t) num_cols * num_rows * padded_size * sizeof(Complex)));
    Complex* h_in = new Complex[(size_t) num_cols * num_rows * padded_size * 2];
    for (size_t i = 0; i < (size_t) num_cols * num_rows * padded_size * 2; i++) {
        h_in[i] = { i*1.0, i*1.0 + 1 };
    }
    gpuErrchk(cudaMemcpy(
        d_in, h_in, (size_t) num_cols * num_rows * padded_size * sizeof(Complex), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_out, (size_t) num_cols * num_rows * padded_size * sizeof(Complex)));
    Utils::swap_axes(d_in, d_out, num_cols, num_rows, padded_size);
    Complex* h_out = new Complex[(size_t) num_cols * num_rows * padded_size];
    gpuErrchk(cudaMemcpy(
        h_out, d_out, num_cols * num_rows * padded_size * sizeof(Complex), cudaMemcpyDeviceToHost));
    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_cols; c++) {
            for (int t = 0; t < padded_size; t++) {
                size_t idx = r * num_cols * padded_size + c * padded_size + t;
                size_t idx2 = t * num_rows * num_cols + c * num_rows + r;
                ASSERT_EQ(h_in[idx].x, h_out[idx2].x);
                ASSERT_EQ(h_in[idx].y, h_out[idx2].y);
            }
        }
    }

    delete[] h_in;
    delete[] h_out;
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
}

TEST(UtilsTest, GetStartIndex)
{
    size_t glob_num_blocks = 10;
    int comm_size = 4;
    int correct_start_indices[4] = {0, 3, 6, 8};
    for (int color = 0; color < comm_size; color ++){
        size_t start_index = Utils::get_start_index(glob_num_blocks, color, comm_size);
        ASSERT_EQ(start_index, correct_start_indices[color]);
    }
}

TEST(UtilsTest, GlobalToLocalSize)
{
    int global_size = 10;
    int comm_size = 4;
    int correct_local_sizes[4] = {3, 3, 2, 2};
    for (int color = 0; color < comm_size; color ++){
        int local_size = Utils::global_to_local_size(global_size, color, comm_size);
        ASSERT_EQ(local_size, correct_local_sizes[color]);
    }
}

TEST(UtilsTest, LocalToGlobalSize)
{
    int local_size = 3;
    int comm_size = 4;
    int global_size = Utils::local_to_global_size(local_size, comm_size);
    ASSERT_EQ(global_size, 12);
}
