#include "shared.hpp"       // Your header with ComplexF/D, gpuErrchk, etc.
#include "util_kernels.hpp" // Your header with the UtilKernels namespace
#include <gtest/gtest.h>
#include <numeric>
#include <type_traits> // For std::is_same
#include <vector>

//============================================================================//
//            CPU HELPER FOR REPAD VERIFICATION (from before)                 //
//============================================================================//
template <typename T_in, typename T_out>
void repad_cpu(const std::vector<T_in> &in, std::vector<T_out> &out,
               const int num_blocks, const int padded_size) {
  const int unpadded_size = padded_size / 2;
  out.resize(num_blocks * padded_size);

  for (int b = 0; b < num_blocks; ++b) {
    for (int t = 0; t < padded_size; ++t) {
      const size_t idx = (size_t)b * padded_size + t;
      if (t < unpadded_size) {
        // Perform a cast during the copy
        out[idx] = static_cast<T_out>(in[idx]);
      } else {
        out[idx] = T_out(0.0);
      }
    }
    if (padded_size % 2 == 1) {
      const size_t nyquist_idx = (size_t)b * padded_size + (padded_size / 2);
      if (nyquist_idx + 1 < out.size()) {
        out[nyquist_idx + 1] = T_out(0.0);
      }
    }
  }
}

//============================================================================//
//                TEST FIXTURE FOR REAL-VALUED TESTS                          //
//============================================================================//

template <typename T> class UtilKernelsRealTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UtilKernelsRealTest, RealTypes);

TYPED_TEST(UtilKernelsRealTest, PadVector) {
  using T_in = TypeParam;
  using T_out = typename std::conditional<std::is_same<T_in, float>::value,
                                          double, float>::type;

  const int num_blocks = 3;
  const int unpadded_size = 5;
  const int padded_size = 2 * unpadded_size;

  // --- Test 1: Same-precision padding (e.g., float -> float) ---
  {
    T_in *d_in = nullptr;
    T_in *d_out = nullptr;
    const size_t in_count = num_blocks * unpadded_size;
    const size_t out_count = num_blocks * padded_size;

    gpuErrchk(cudaMalloc(&d_in, in_count * sizeof(T_in)));
    gpuErrchk(cudaMalloc(&d_out, out_count * sizeof(T_in)));

    std::vector<T_in> h_in(in_count);
    std::iota(h_in.begin(), h_in.end(), 1);
    gpuErrchk(cudaMemcpy(d_in, h_in.data(), in_count * sizeof(T_in),
                         cudaMemcpyHostToDevice));

    UtilKernels::pad_vector<T_in, T_in>(d_in, d_out, num_blocks, padded_size,
                                        nullptr);

    std::vector<T_in> h_out(out_count);
    gpuErrchk(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(T_in),
                         cudaMemcpyDeviceToHost));

    for (int b = 0; b < num_blocks; ++b) {
      for (int t = 0; t < padded_size; ++t) {
        T_in expected = (t < unpadded_size)
                            ? h_in[(size_t)b * unpadded_size + t]
                            : T_in(0.0);
        if (std::is_same<T_in, float>::value)
          ASSERT_FLOAT_EQ(h_out[(size_t)b * padded_size + t], expected);
        else
          ASSERT_DOUBLE_EQ(h_out[(size_t)b * padded_size + t], expected);
      }
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }

  // --- Test 2: Mixed-precision padding (e.g., float -> double) ---
  {
    T_in *d_in = nullptr;
    T_out *d_out = nullptr;
    const size_t in_count = num_blocks * unpadded_size;
    const size_t out_count = num_blocks * padded_size;

    gpuErrchk(cudaMalloc(&d_in, in_count * sizeof(T_in)));
    gpuErrchk(cudaMalloc(&d_out, out_count * sizeof(T_out)));

    std::vector<T_in> h_in(in_count);
    std::iota(h_in.begin(), h_in.end(), 1);
    gpuErrchk(cudaMemcpy(d_in, h_in.data(), in_count * sizeof(T_in),
                         cudaMemcpyHostToDevice));

    UtilKernels::pad_vector<T_in, T_out>(d_in, d_out, num_blocks, padded_size,
                                         nullptr);

    std::vector<T_out> h_out(out_count);
    gpuErrchk(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(T_out),
                         cudaMemcpyDeviceToHost));

    for (int b = 0; b < num_blocks; ++b) {
      for (int t = 0; t < padded_size; ++t) {
        T_out expected =
            (t < unpadded_size)
                ? static_cast<T_out>(h_in[(size_t)b * unpadded_size + t])
                : T_out(0.0);
        if (std::is_same<T_out, float>::value)
          ASSERT_FLOAT_EQ(h_out[(size_t)b * padded_size + t], expected);
        else
          ASSERT_DOUBLE_EQ(h_out[(size_t)b * padded_size + t], expected);
      }
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }
}

TYPED_TEST(UtilKernelsRealTest, UnpadVector) {
  using T_in = TypeParam;
  using T_out = typename std::conditional<std::is_same<T_in, float>::value,
                                          double, float>::type;

  const int num_blocks = 3;
  const int unpadded_size = 5;
  const int padded_size = 2 * unpadded_size;

  // --- Test 1: Same-precision unpadding ---
  {
    T_in *d_in = nullptr;
    T_in *d_out = nullptr;
    const size_t padded_count = num_blocks * padded_size;
    const size_t unpadded_count = num_blocks * unpadded_size;

    gpuErrchk(cudaMalloc(&d_in, padded_count * sizeof(T_in)));
    gpuErrchk(cudaMalloc(&d_out, unpadded_count * sizeof(T_in)));

    std::vector<T_in> h_padded_in(padded_count);
    for (size_t i = 0; i < padded_count; ++i)
      h_padded_in[i] = T_in(i + 1);
    gpuErrchk(cudaMemcpy(d_in, h_padded_in.data(), padded_count * sizeof(T_in),
                         cudaMemcpyHostToDevice));

    UtilKernels::unpad_repad_vector<T_in, T_in>(d_in, d_out, num_blocks,
                                                padded_size, true, nullptr);

    std::vector<T_in> h_unpadded_out(unpadded_count);
    gpuErrchk(cudaMemcpy(h_unpadded_out.data(), d_out,
                         unpadded_count * sizeof(T_in),
                         cudaMemcpyDeviceToHost));

    for (int b = 0; b < num_blocks; ++b) {
      for (int t = 0; t < unpadded_size; ++t) {
        T_in expected = h_padded_in[(size_t)b * padded_size + t];
        if (std::is_same<T_in, float>::value)
          ASSERT_FLOAT_EQ(h_unpadded_out[(size_t)b * unpadded_size + t],
                          expected);
        else
          ASSERT_DOUBLE_EQ(h_unpadded_out[(size_t)b * unpadded_size + t],
                           expected);
      }
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }

  // --- Test 2: Mixed-precision unpadding ---
  {
    T_in *d_in = nullptr;
    T_out *d_out = nullptr;
    const size_t padded_count = num_blocks * padded_size;
    const size_t unpadded_count = num_blocks * unpadded_size;

    gpuErrchk(cudaMalloc(&d_in, padded_count * sizeof(T_in)));
    gpuErrchk(cudaMalloc(&d_out, unpadded_count * sizeof(T_out)));

    std::vector<T_in> h_padded_in(padded_count);
    for (size_t i = 0; i < padded_count; ++i)
      h_padded_in[i] = T_in(i + 1);
    gpuErrchk(cudaMemcpy(d_in, h_padded_in.data(), padded_count * sizeof(T_in),
                         cudaMemcpyHostToDevice));

    UtilKernels::unpad_repad_vector<T_in, T_out>(d_in, d_out, num_blocks,
                                                 padded_size, true, nullptr);

    std::vector<T_out> h_unpadded_out(unpadded_count);
    gpuErrchk(cudaMemcpy(h_unpadded_out.data(), d_out,
                         unpadded_count * sizeof(T_out),
                         cudaMemcpyDeviceToHost));

    for (int b = 0; b < num_blocks; ++b) {
      for (int t = 0; t < unpadded_size; ++t) {
        T_out expected =
            static_cast<T_out>(h_padded_in[(size_t)b * padded_size + t]);
        if (std::is_same<T_out, float>::value)
          ASSERT_FLOAT_EQ(h_unpadded_out[(size_t)b * unpadded_size + t],
                          expected);
        else
          ASSERT_DOUBLE_EQ(h_unpadded_out[(size_t)b * unpadded_size + t],
                           expected);
      }
    }
    cudaFree(d_in);
    cudaFree(d_out);
  }
}

TYPED_TEST(UtilKernelsRealTest, RepadVector) {
  // Repad is only instantiated for same-type conversion, so we only test that
  // case.
  using T = TypeParam;
  const int num_blocks = 3;
  const int unpadded_size = 5;
  const int padded_size = 2 * unpadded_size + 1; // Padded size 11, unpadded 5
  const size_t count = num_blocks * padded_size;

  T *d_in = nullptr, *d_out = nullptr;
  gpuErrchk(cudaMalloc(&d_in, count * sizeof(T)));
  gpuErrchk(cudaMalloc(&d_out, count * sizeof(T)));

  std::vector<T> h_dirty_padded_in(count);
  for (size_t i = 0; i < count; ++i)
    h_dirty_padded_in[i] = T(i + 1);
  gpuErrchk(cudaMemcpy(d_in, h_dirty_padded_in.data(), count * sizeof(T),
                       cudaMemcpyHostToDevice));

  // Execute the repad kernel (via the dispatcher)
  UtilKernels::unpad_repad_vector<T, T>(d_in, d_out, num_blocks, padded_size,
                                        false, nullptr);

  std::vector<T> h_expected_repad;
  repad_cpu(h_dirty_padded_in, h_expected_repad, num_blocks, padded_size);

  std::vector<T> h_gpu_out(count);
  gpuErrchk(cudaMemcpy(h_gpu_out.data(), d_out, count * sizeof(T),
                       cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < count; ++i) {
    if (std::is_same<T, float>::value)
      ASSERT_FLOAT_EQ(h_gpu_out[i], h_expected_repad[i]);
    else
      ASSERT_DOUBLE_EQ(h_gpu_out[i], h_expected_repad[i]);
  }
  cudaFree(d_in);
  cudaFree(d_out);
}

//============================================================================//
//                      CASTING AND SWAPPING TESTS                            //
//============================================================================//

TEST(UtilKernelsNonTypedTest, CastVector) {
  const size_t size = 1024;

  // --- Test float -> double ---
  {
    float *d_in_f;
    double *d_out_d;
    gpuErrchk(cudaMalloc(&d_in_f, size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_out_d, size * sizeof(double)));
    std::vector<float> h_in_f(size);
    std::iota(h_in_f.begin(), h_in_f.end(), 0.5f);
    gpuErrchk(cudaMemcpy(d_in_f, h_in_f.data(), size * sizeof(float),
                         cudaMemcpyHostToDevice));
    UtilKernels::cast_vector<float, double>(d_in_f, d_out_d, size, nullptr);
    std::vector<double> h_out_d(size);
    gpuErrchk(cudaMemcpy(h_out_d.data(), d_out_d, size * sizeof(double),
                         cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < size; ++i) {
      ASSERT_DOUBLE_EQ(h_out_d[i], static_cast<double>(h_in_f[i]));
    }
    cudaFree(d_in_f);
    cudaFree(d_out_d);
  }

  // --- Test double -> float ---
  {
    double *d_in_d;
    float *d_out_f;
    gpuErrchk(cudaMalloc(&d_in_d, size * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_out_f, size * sizeof(float)));
    std::vector<double> h_in_d(size);
    std::iota(h_in_d.begin(), h_in_d.end(), 0.5);
    gpuErrchk(cudaMemcpy(d_in_d, h_in_d.data(), size * sizeof(double),
                         cudaMemcpyHostToDevice));
    UtilKernels::cast_vector<double, float>(d_in_d, d_out_f, size, nullptr);
    std::vector<float> h_out_f(size);
    gpuErrchk(cudaMemcpy(h_out_f.data(), d_out_f, size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < size; ++i) {
      ASSERT_FLOAT_EQ(h_out_f[i], static_cast<float>(h_in_d[i]));
    }
    cudaFree(d_in_d);
    cudaFree(d_out_f);
  }
}

template <typename T> class UtilKernelsComplexTest : public ::testing::Test {
protected:
  void SetUp() override {
    d_in = nullptr;
    d_out = nullptr;
  }
  void TearDown() override {
    if (d_in)
      cudaFree(d_in);
    if (d_out)
      cudaFree(d_out);
  }
  T *d_in;
  T *d_out;
};

using ComplexTypes = ::testing::Types<ComplexF, ComplexD>;
TYPED_TEST_SUITE(UtilKernelsComplexTest, ComplexTypes);

template <typename T_complex> T_complex make_complex_from_int(size_t i);
template <> ComplexF make_complex_from_int<ComplexF>(size_t i) {
  return make_cuComplex(static_cast<float>(i), -static_cast<float>(i));
}
template <> ComplexD make_complex_from_int<ComplexD>(size_t i) {
  return make_cuDoubleComplex(static_cast<double>(i), -static_cast<double>(i));
}

TYPED_TEST(UtilKernelsComplexTest, SwapAxes) {
  using T_complex = TypeParam;

  const unsigned int num_cols = 48;
  const unsigned int num_rows = 64;
  const unsigned int block_size = 32;
  const size_t count = num_cols * num_rows * block_size;

  gpuErrchk(cudaMalloc(&this->d_in, count * sizeof(T_complex)));
  gpuErrchk(cudaMalloc(&this->d_out, count * sizeof(T_complex)));

  std::vector<T_complex> h_in(count);
  for (size_t i = 0; i < count; ++i) {
    h_in[i] = make_complex_from_int<T_complex>(i);
  }
  gpuErrchk(cudaMemcpy(this->d_in, h_in.data(), count * sizeof(T_complex),
                       cudaMemcpyHostToDevice));

  UtilKernels::swap_axes_cutranspose<T_complex>(
      this->d_in, this->d_out, num_cols, num_rows, block_size, nullptr);

  std::vector<T_complex> h_out(count);
  gpuErrchk(cudaMemcpy(h_out.data(), this->d_out, count * sizeof(T_complex),
                       cudaMemcpyDeviceToHost));

  for (size_t z = 0; z < num_rows; ++z) {
    for (size_t y = 0; y < num_cols; ++y) {
      for (size_t x = 0; x < block_size; ++x) {
        size_t in_idx = x + (y + z * num_cols) * block_size;
        size_t out_idx = z + (y + x * num_cols) * num_rows;
        if (std::is_same<T_complex, ComplexF>::value) {
          ASSERT_FLOAT_EQ(h_out[out_idx].x, h_in[in_idx].x);
          ASSERT_FLOAT_EQ(h_out[out_idx].y, h_in[in_idx].y);
        } else {
          ASSERT_DOUBLE_EQ(h_out[out_idx].x, h_in[in_idx].x);
          ASSERT_DOUBLE_EQ(h_out[out_idx].y, h_in[in_idx].y);
        }
      }
    }
  }
}

TEST(UtilKernelsResizeTest, ExtendVector) {
  // Configuration
  const unsigned int num_blocks = 10;
  const unsigned int current_block_size = 5;
  const unsigned int new_block_size = 12; // Must be > current_block_size

  const size_t in_count = num_blocks * current_block_size;
  const size_t out_count = num_blocks * new_block_size;

  // Allocate Device Memory
  double *d_in = nullptr;
  double *d_out = nullptr;
  gpuErrchk(cudaMalloc(&d_in, in_count * sizeof(double)));
  gpuErrchk(cudaMalloc(&d_out, out_count * sizeof(double)));

  // Prepare Input Data
  std::vector<double> h_in(in_count);
  // Fill with 1.0, 2.0, 3.0...
  std::iota(h_in.begin(), h_in.end(), 1.0);

  gpuErrchk(cudaMemcpy(d_in, h_in.data(), in_count * sizeof(double),
                       cudaMemcpyHostToDevice));

  // Run Kernel
  UtilKernels::extend_vector(d_in, d_out, num_blocks, current_block_size,
                             new_block_size, nullptr);

  // Retrieve Output
  std::vector<double> h_out(out_count);
  gpuErrchk(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(double),
                       cudaMemcpyDeviceToHost));

  // Verification
  for (unsigned int b = 0; b < num_blocks; ++b) {
    for (unsigned int i = 0; i < new_block_size; ++i) {
      size_t out_idx = (size_t)b * new_block_size + i;

      if (i < current_block_size) {
        // Should match the input data
        size_t in_idx = (size_t)b * current_block_size + i;
        ASSERT_DOUBLE_EQ(h_out[out_idx], h_in[in_idx])
            << "Mismatch at block " << b << ", index " << i;
      } else {
        // Should be zero padded
        ASSERT_DOUBLE_EQ(h_out[out_idx], 0.0)
            << "Padding mismatch at block " << b << ", index " << i;
      }
    }
  }

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out);
}

TEST(UtilKernelsResizeTest, ShrinkVector) {
  // Configuration
  const unsigned int num_blocks = 10;
  const unsigned int current_block_size = 12;
  const unsigned int new_block_size = 5; // Must be < current_block_size

  const size_t in_count = num_blocks * current_block_size;
  const size_t out_count = num_blocks * new_block_size;

  // Allocate Device Memory
  double *d_in = nullptr;
  double *d_out = nullptr;
  gpuErrchk(cudaMalloc(&d_in, in_count * sizeof(double)));
  gpuErrchk(cudaMalloc(&d_out, out_count * sizeof(double)));

  // Prepare Input Data
  std::vector<double> h_in(in_count);
  std::iota(h_in.begin(), h_in.end(), 1.0);

  gpuErrchk(cudaMemcpy(d_in, h_in.data(), in_count * sizeof(double),
                       cudaMemcpyHostToDevice));

  // Run Kernel
  UtilKernels::shrink_vector(d_in, d_out, num_blocks, current_block_size,
                             new_block_size, nullptr);

  // Retrieve Output
  std::vector<double> h_out(out_count);
  gpuErrchk(cudaMemcpy(h_out.data(), d_out, out_count * sizeof(double),
                       cudaMemcpyDeviceToHost));

  // Verification
  for (unsigned int b = 0; b < num_blocks; ++b) {
    for (unsigned int i = 0; i < new_block_size; ++i) {
      // We expect the output to match the first 'new_block_size' elements
      // of the corresponding input block.
      size_t out_idx = (size_t)b * new_block_size + i;
      size_t in_idx = (size_t)b * current_block_size + i;

      ASSERT_DOUBLE_EQ(h_out[out_idx], h_in[in_idx])
          << "Mismatch at block " << b << ", index " << i;
    }
  }

  // Cleanup
  cudaFree(d_in);
  cudaFree(d_out);
}
