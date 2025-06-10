#include "util_kernels.hpp" // Your header with the UtilKernels namespace
#include "shared.hpp"       // Your header with ComplexF/D, gpuErrchk, etc.
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <type_traits> // For std::is_same

//============================================================================//
//                      TEST FIXTURE SETUP                                    //
//============================================================================//

template <typename T>
class UtilKernelsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        d_in = nullptr;
        d_out1 = nullptr;
        d_out2 = nullptr;
    }

    void TearDown() override
    {
        if (d_in)
            cudaFree(d_in);
        if (d_out1)
            cudaFree(d_out1);
        if (d_out2)
            cudaFree(d_out2);
    }

    T *d_in;
    T *d_out1;
    T *d_out2;
};

// Define the list of types for standard tests
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UtilKernelsTest, RealTypes);

//============================================================================//
//                      PADDING / UNPADDING TESTS                             //
//============================================================================//

TYPED_TEST(UtilKernelsTest, PadVector)
{
    using T = TypeParam;

    const int num_blocks = 3;
    const int unpadded_size[] = {4, 5};

    for (int size : unpadded_size)
    {
        const int padded_size = 2 * size;
        const size_t in_count = num_blocks * size;
        const size_t out_count = num_blocks * padded_size;

        gpuErrchk(cudaMalloc(&this->d_in, in_count * sizeof(T)));
        gpuErrchk(cudaMalloc(&this->d_out1, out_count * sizeof(T)));

        std::vector<T> h_in(in_count);
        std::iota(h_in.begin(), h_in.end(), 0);
        gpuErrchk(cudaMemcpy(this->d_in, h_in.data(), in_count * sizeof(T), cudaMemcpyHostToDevice));

        UtilKernels::pad_vector<T>(this->d_in, this->d_out1, num_blocks, padded_size, nullptr);

        std::vector<T> h_out(out_count);
        gpuErrchk(cudaMemcpy(h_out.data(), this->d_out1, out_count * sizeof(T), cudaMemcpyDeviceToHost));

        for (int b = 0; b < num_blocks; b++)
        {
            for (int t = 0; t < padded_size; t++)
            {
                size_t out_idx = (size_t)b * padded_size + t;
                if (t < size)
                {
                    size_t in_idx = (size_t)b * size + t;
                    if (std::is_same<T, float>::value)
                        ASSERT_FLOAT_EQ(h_out[out_idx], h_in[in_idx]);
                    else
                        ASSERT_DOUBLE_EQ(h_out[out_idx], h_in[in_idx]);
                }
                else
                {
                    if (std::is_same<T, float>::value)
                        ASSERT_FLOAT_EQ(h_out[out_idx], T(0.0));
                    else
                        ASSERT_DOUBLE_EQ(h_out[out_idx], T(0.0));
                }
            }
        }
        gpuErrchk(cudaFree(this->d_in));
        this->d_in = nullptr;
        gpuErrchk(cudaFree(this->d_out1));
        this->d_out1 = nullptr;
    }
}

TYPED_TEST(UtilKernelsTest, UnpadVector)
{
    using T = TypeParam;
    const int num_blocks = 3;
    const int unpadded_size = 5;
    const int padded_size = 2 * unpadded_size;
    const size_t padded_count = num_blocks * padded_size;
    const size_t unpadded_count = num_blocks * unpadded_size;

    gpuErrchk(cudaMalloc(&this->d_in, padded_count * sizeof(T)));
    gpuErrchk(cudaMalloc(&this->d_out1, unpadded_count * sizeof(T)));

    std::vector<T> h_padded_in(padded_count);
    for (int b = 0; b < num_blocks; ++b)
    {
        for (int t = 0; t < padded_size; ++t)
        {
            h_padded_in[(size_t)b * padded_size + t] = (t < unpadded_size) ? T((size_t)b * unpadded_size + t + 1) : T(99.0); // Non-zero padding
        }
    }
    gpuErrchk(cudaMemcpy(this->d_in, h_padded_in.data(), padded_count * sizeof(T), cudaMemcpyHostToDevice));

    UtilKernels::unpad_repad_vector<T>(this->d_in, this->d_out1, num_blocks, padded_size, true, nullptr);

    std::vector<T> h_unpadded_out(unpadded_count);
    gpuErrchk(cudaMemcpy(h_unpadded_out.data(), this->d_out1, unpadded_count * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < unpadded_count; ++i)
    {
        if (std::is_same<T, float>::value)
            ASSERT_FLOAT_EQ(h_unpadded_out[i], T(i + 1));
        else
            ASSERT_DOUBLE_EQ(h_unpadded_out[i], T(i + 1));
    }
}

TYPED_TEST(UtilKernelsTest, RepadVector)
{
    using T = TypeParam;
    const int num_blocks = 3;
    const int unpadded_size = 5;                   // Odd size to test Nyquist
    const int padded_size = 2 * unpadded_size + 1; // Padded size is 11, unpadded is 5
    const size_t count = num_blocks * padded_size;

    gpuErrchk(cudaMalloc(&this->d_in, count * sizeof(T)));
    gpuErrchk(cudaMalloc(&this->d_out1, count * sizeof(T)));

    // Create a "dirty" padded vector: data in the right place, but garbage in the padding.
    std::vector<T> h_dirty_padded_in(count);
    for (int b = 0; b < num_blocks; ++b)
    {
        for (int t = 0; t < padded_size; ++t)
        {
            T val = (t < unpadded_size) ? T((size_t)b * unpadded_size + t + 1) : T(99.0); // Garbage value
            h_dirty_padded_in[(size_t)b * padded_size + t] = val;
        }
    }
    gpuErrchk(cudaMemcpy(this->d_in, h_dirty_padded_in.data(), count * sizeof(T), cudaMemcpyHostToDevice));

    // Execute the repad kernel to clean up the dirty vector
    UtilKernels::unpad_repad_vector<T>(this->d_in, this->d_out1, num_blocks, padded_size, false, nullptr);

    // Create the "clean" ground truth on the CPU
    std::vector<T> h_expected_repad(count);

    // Copy result back and verify
    std::vector<T> h_gpu_out(count);
    gpuErrchk(cudaMemcpy(h_gpu_out.data(), this->d_out1, count * sizeof(T), cudaMemcpyDeviceToHost));

    for (int b = 0; b < num_blocks; ++b)
    {
        for (int t = 0; t < padded_size; ++t)
        {
            size_t idx = (size_t)b * padded_size + t;
            if (t < unpadded_size)
            {
                h_expected_repad[idx] = T((size_t)b * unpadded_size + t + 1);
            }
            else
            {
                h_expected_repad[idx] = T(0.0);
            }
            if (std::is_same<T, float>::value)
                ASSERT_FLOAT_EQ(h_gpu_out[idx], h_expected_repad[idx]);
            else
                ASSERT_DOUBLE_EQ(h_gpu_out[idx], h_expected_repad[idx]);
        }
    }
}

//============================================================================//
//                      CASTING AND SWAPPING TESTS                            //
//============================================================================//

TEST(UtilKernelsNonTypedTest, CastVector)
{
    const size_t size = 1024;

    // --- Test float -> double ---
    {
        float *d_in_f;
        double *d_out_d;
        gpuErrchk(cudaMalloc(&d_in_f, size * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_out_d, size * sizeof(double)));
        std::vector<float> h_in_f(size);
        std::iota(h_in_f.begin(), h_in_f.end(), 0.5f);
        gpuErrchk(cudaMemcpy(d_in_f, h_in_f.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        UtilKernels::cast_vector<float, double>(d_in_f, d_out_d, size, nullptr);

        std::vector<double> h_out_d(size);
        gpuErrchk(cudaMemcpy(h_out_d.data(), d_out_d, size * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < size; ++i)
        {
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
        gpuErrchk(cudaMemcpy(d_in_d, h_in_d.data(), size * sizeof(double), cudaMemcpyHostToDevice));

        UtilKernels::cast_vector<double, float>(d_in_d, d_out_f, size, nullptr);

        std::vector<float> h_out_f(size);
        gpuErrchk(cudaMemcpy(h_out_f.data(), d_out_f, size * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < size; ++i)
        {
            ASSERT_FLOAT_EQ(h_out_f[i], static_cast<float>(h_in_d[i]));
        }
        cudaFree(d_in_d);
        cudaFree(d_out_f);
    }
}

//============================================================================//
//                      CASTING AND SWAPPING TESTS                            //
//============================================================================//

/* ... other tests like CastVector remain the same ... */

// A new fixture is needed for the complex-typed swap test
template <typename T>
class UtilKernelsComplexTest : public UtilKernelsTest<T>
{
};

using ComplexTypes = ::testing::Types<ComplexF, ComplexD>;
TYPED_TEST_SUITE(UtilKernelsComplexTest, ComplexTypes);

// --- Type-safe helper functions to create complex numbers for tests ---
// This is the key to solving the compilation errors.

// Generic template declaration
template <typename T_complex>
T_complex make_complex_from_int(size_t i);

// Specialization for float complex
template <>
ComplexF make_complex_from_int<ComplexF>(size_t i)
{
    return make_cuComplex(static_cast<float>(i), -static_cast<float>(i));
}

// Specialization for double complex
template <>
ComplexD make_complex_from_int<ComplexD>(size_t i)
{
    return make_cuDoubleComplex(static_cast<double>(i), -static_cast<double>(i));
}

TYPED_TEST(UtilKernelsComplexTest, SwapAxes)
{
    using T_complex = TypeParam;

    const unsigned int num_cols = 48;
    const unsigned int num_rows = 64;
    const unsigned int block_size = 32;
    const size_t count = num_cols * num_rows * block_size;

    gpuErrchk(cudaMalloc(&this->d_in, count * sizeof(T_complex)));
    gpuErrchk(cudaMalloc(&this->d_out1, count * sizeof(T_complex)));

    std::vector<T_complex> h_in(count);
    for (size_t i = 0; i < count; ++i)
    {
        // The call is now simple and generic. The compiler chooses the
        // correct specialized helper function at compile time.
        h_in[i] = make_complex_from_int<T_complex>(i);
    }
    gpuErrchk(cudaMemcpy(this->d_in, h_in.data(), count * sizeof(T_complex), cudaMemcpyHostToDevice));

    // Execute kernel
    UtilKernels::swap_axes_cutranspose<T_complex>(
        this->d_in, this->d_out1, num_cols, num_rows, block_size, nullptr);

    // Verify
    std::vector<T_complex> h_out(count);
    gpuErrchk(cudaMemcpy(h_out.data(), this->d_out1, count * sizeof(T_complex), cudaMemcpyDeviceToHost));

    // CPU-side swap for verification
    for (size_t z = 0; z < num_rows; ++z)
    {
        for (size_t y = 0; y < num_cols; ++y)
        {
            for (size_t x = 0; x < block_size; ++x)
            {
                size_t in_idx = x + (y + z * num_cols) * block_size;
                size_t out_idx = z + (y + x * num_cols) * num_rows;

                // CORRECTED: The verification logic is also now simpler and fully type-safe.
                if (std::is_same<T_complex, ComplexF>::value)
                {
                    ASSERT_FLOAT_EQ(h_out[out_idx].x, h_in[in_idx].x);
                    ASSERT_FLOAT_EQ(h_out[out_idx].y, h_in[in_idx].y);
                }
                else
                {
                    ASSERT_DOUBLE_EQ(h_out[out_idx].x, h_in[in_idx].x);
                    ASSERT_DOUBLE_EQ(h_out[out_idx].y, h_in[in_idx].y);
                }
            }
        }
    }
}