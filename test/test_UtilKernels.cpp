#include "shared.hpp"
#include "util_kernels.hpp"
#include <gtest/gtest.h>

TEST(UtilsKernelsTest, PadVectorEven)
{
    int num_blocks = 3;
    int block_size = 4;
    double* d_in;
    double* d_pad;
    gpuErrchk(cudaMalloc(&d_in, (size_t)num_blocks * block_size * sizeof(double)));
    double* h_in = new double[(size_t)num_blocks * block_size];
    for (size_t i = 0; i < (size_t)num_blocks * block_size; i++) {
        h_in[i] = i;
    }
    gpuErrchk(cudaMemcpy(
        d_in, h_in, (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_pad, (size_t)num_blocks * 2 * block_size * sizeof(double)));
    UtilKernels::pad_vector(d_in, d_pad, num_blocks, 2 * block_size, nullptr);
    double* h_pad = new double[(size_t)num_blocks * 2 * block_size];
    gpuErrchk(cudaMemcpy(h_pad, d_pad, (size_t)num_blocks * 2 * block_size * sizeof(double),
        cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; b++) {
        for (int t = 0; t < 2 * block_size; t++) {
            size_t idx = b * 2 * block_size + t;
            if (t < block_size) {
                ASSERT_EQ(h_pad[idx], h_in[b * block_size + t]);
            } else {
                ASSERT_EQ(h_pad[idx], 0);
            }
        }
    }

    delete[] h_in;
    delete[] h_pad;
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_pad));
}

TEST(UtilsKernelsTest, PadVectorOdd)
{
    int num_blocks = 3;
    int block_size = 5;
    double* d_in;
    double* d_pad;
    gpuErrchk(cudaMalloc(&d_in, (size_t)num_blocks * block_size * sizeof(double)));
    double* h_in = new double[(size_t)num_blocks * block_size];
    for (size_t i = 0; i < (size_t)num_blocks * block_size; i++) {
        h_in[i] = i;
    }
    gpuErrchk(cudaMemcpy(
        d_in, h_in, (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_pad, (size_t)num_blocks * 2 * block_size * sizeof(double)));
    UtilKernels::pad_vector(d_in, d_pad, num_blocks, 2 * block_size, nullptr);
    double* h_pad = new double[(size_t)num_blocks * 2 * block_size];
    gpuErrchk(cudaMemcpy(h_pad, d_pad, (size_t)num_blocks * 2 * block_size * sizeof(double),
        cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; b++) {
        for (int t = 0; t < 2 * block_size; t++) {
            size_t idx = b * 2 * block_size + t;
            if (t < block_size) {
                ASSERT_EQ(h_pad[idx], h_in[b * block_size + t]);
            } else {
                ASSERT_EQ(h_pad[idx], 0);
            }
        }
    }

    delete[] h_in;
    delete[] h_pad;
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_pad));
}

TEST(UtilsKernelsTest, UnpadVectorEven)
{
    int num_blocks = 3;
    int block_size = 4;
    double* d_in;
    double* d_out;
    gpuErrchk(cudaMalloc(&d_in, (size_t)num_blocks * 2 * block_size * sizeof(double)));
    double* h_in = new double[(size_t)num_blocks * 2 * block_size];
    for (size_t i = 0; i < (size_t)num_blocks * 2 * block_size; i++) {
        h_in[i] = (i < block_size) ? i : 0;
    }
    gpuErrchk(cudaMemcpy(
        d_in, h_in, (size_t)num_blocks * 2 * block_size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_out, (size_t)num_blocks * block_size * sizeof(double)));
    UtilKernels::unpad_repad_vector(d_in, d_out, num_blocks, 2 * block_size, true, nullptr);
    double* h_out = new double[(size_t)num_blocks * block_size];
    gpuErrchk(cudaMemcpy(
        h_out, d_out, (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; b++) {
        for (int t = 0; t < block_size; t++) {
            size_t idx = b * block_size + t;
            ASSERT_EQ(h_out[idx], h_in[b * 2 * block_size + t]);
        }
    }

    delete[] h_in;
    delete[] h_out;
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
}

TEST(UtilsKernelsTest, UnpadVectorOdd)
{
    int num_blocks = 3;
    int block_size = 5;
    double* d_in;
    double* d_out;
    gpuErrchk(cudaMalloc(&d_in, (size_t)num_blocks * 2 * block_size * sizeof(double)));
    double* h_in = new double[(size_t)num_blocks * 2 * block_size];
    for (size_t i = 0; i < (size_t)num_blocks * 2 * block_size; i++) {
        h_in[i] = (i < block_size) ? i : 0;
    }
    gpuErrchk(cudaMemcpy(
        d_in, h_in, (size_t)num_blocks * 2 * block_size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_out, (size_t)num_blocks * block_size * sizeof(double)));
    UtilKernels::unpad_repad_vector(d_in, d_out, num_blocks, 2 * block_size, true, nullptr);
    double* h_out = new double[(size_t)num_blocks * block_size];
    gpuErrchk(cudaMemcpy(
        h_out, d_out, (size_t)num_blocks * block_size * sizeof(double), cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; b++) {
        for (int t = 0; t < block_size; t++) {
            size_t idx = b * block_size + t;
            ASSERT_EQ(h_out[idx], h_in[b * 2 * block_size + t]);
        }
    }

    delete[] h_in;
    delete[] h_out;
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
}

TEST(UtilKernelsTest, RepadVector)
{
    int num_blocks = 3;
    int block_size = 4;
    double* d_in;
    double* d_out;
    gpuErrchk(cudaMalloc(&d_in, (size_t)num_blocks * 2 * block_size * sizeof(double)));
    double* h_in = new double[(size_t)num_blocks * 2 * block_size];
    for (size_t i = 0; i < (size_t)num_blocks * 2 * block_size; i++) {
        h_in[i] = i;
    }
    gpuErrchk(cudaMemcpy(
        d_in, h_in, (size_t)num_blocks * 2 * block_size * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_out, (size_t)num_blocks * 2 * block_size * sizeof(double)));
    UtilKernels::unpad_repad_vector(d_in, d_out, num_blocks, 2 * block_size, false, nullptr);
    double* h_out = new double[(size_t)num_blocks * 2 * block_size];
    gpuErrchk(cudaMemcpy(h_out, d_out, (size_t)num_blocks * 2 * block_size * sizeof(double),
        cudaMemcpyDeviceToHost));
    for (int b = 0; b < num_blocks; b++) {
        for (int t = 0; t < 2 * block_size; t++) {
            size_t idx = b * 2 * block_size + t;
            if (t < block_size) {
                ASSERT_EQ(h_out[idx], h_in[b * 2 * block_size + t]);
            } else {
                ASSERT_EQ(h_out[idx], 0);
            }
        }
    }

    delete[] h_in;
    delete[] h_out;
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
}