#include "shared.hpp"
#include "utils.hpp"
#include <gtest/gtest.h>

//============================================================================//
//                      CPU HELPER FUNCTIONS FOR VERIFICATION                 //
//============================================================================//

/**
 * @brief CPU equivalent of the transpose operation for verification.
 */
template<typename T_complex>
void transpose_cpu(const std::vector<T_complex>& in, std::vector<T_complex>& out,
                   int num_rows, int num_cols)
{
    out.resize(in.size());
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            // Input is (r, c), output is (c, r)
            out[c * num_rows + r] = in[r * num_cols + c];
        }
    }
}

/**
 * @brief CORRECTED CPU equivalent of the strided batched GEMV, assuming COLUMN-MAJOR matrix storage.
 */
template<typename T_complex>
void sbgemv_cpu(const std::vector<T_complex>& mat, const std::vector<T_complex>& vec_in,
                std::vector<T_complex>& vec_out, int num_rows, int num_cols,
                int block_size, bool conjugate)
{
    int vec_in_len = conjugate ? num_rows : num_cols;
    int vec_out_len = conjugate ? num_cols : num_rows;
    vec_out.assign(block_size * vec_out_len, T_complex{0.0f, 0.0f});

    for (int k = 0; k < block_size; ++k) { // Batch loop
        const T_complex* A = &mat[k * num_rows * num_cols];
        const T_complex* x = &vec_in[k * vec_in_len];
        T_complex* y = &vec_out[k * vec_out_len];

        if (conjugate) { // y = A^H * x
            for (int i = 0; i < num_cols; ++i) { // Output vector elements
                for (int j = 0; j < num_rows; ++j) { // Input vector elements
                    // Access A in column-major order: A(j, i)
                    T_complex A_val = A[i * num_rows + j];
                    T_complex conj_A = {A_val.x, -A_val.y};
                    y[i].x += conj_A.x * x[j].x - conj_A.y * x[j].y;
                    y[i].y += conj_A.x * x[j].y + conj_A.y * x[j].x;
                }
            }
        } else { // y = A * x
            for (int i = 0; i < num_rows; ++i) { // Output vector elements
                for (int j = 0; j < num_cols; ++j) { // Input vector elements
                    // Access A in column-major order: A(i, j)
                    T_complex A_val = A[j * num_rows + i];
                    T_complex x_val = x[j];
                    y[i].x += A_val.x * x_val.x - A_val.y * x_val.y;
                    y[i].y += A_val.x * x_val.y + A_val.y * x_val.x;
                }
            }
        }
    }
}
//============================================================================//
//               TEST FIXTURE FOR COMPLEX-VALUED UTILS                        //
//============================================================================//

template <typename T>
class UtilComplexKernelsTest : public ::testing::Test {
protected:
    // Using individual pointers for clarity in tests
    void SetUp() override {
        d_mat = nullptr;
        d_vec_in = nullptr;
        d_vec_out = nullptr;
    }
    void TearDown() override {
        if (d_mat) cudaFree(d_mat);
        if (d_vec_in) cudaFree(d_vec_in);
        if (d_vec_out) cudaFree(d_vec_out);
    }
    T* d_mat;
    T* d_vec_in;
    T* d_vec_out;
    cublasHandle_t handle;
};

using ComplexTypes = ::testing::Types<ComplexF, ComplexD>;
TYPED_TEST_SUITE(UtilComplexKernelsTest, ComplexTypes);

// Helper to initialize test data
template <typename T_complex> T_complex make_complex_from_int(size_t i);
template <> ComplexF make_complex_from_int<ComplexF>(size_t i) { return make_cuComplex(static_cast<float>(i), -static_cast<float>(i)); }
template <> ComplexD make_complex_from_int<ComplexD>(size_t i) { return make_cuDoubleComplex(static_cast<double>(i), -static_cast<double>(i)); }

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
    int block_size = 4;
    ComplexD *d_in;
    ComplexD *d_out;
    gpuErrchk(cudaMalloc(&d_in, (size_t)num_cols * num_rows * block_size * sizeof(ComplexD)));
    ComplexD *h_in = new ComplexD[(size_t)num_cols * num_rows * block_size];
    for (size_t i = 0; i < (size_t)num_cols * num_rows * block_size; i++)
    {
        h_in[i] = {i * 1.0, i * 1.0 + 1};
    }
    gpuErrchk(cudaMemcpy(
        d_in, h_in, (size_t)num_cols * num_rows * block_size * sizeof(ComplexD), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_out, (size_t)num_cols * num_rows * block_size * sizeof(ComplexD)));
    Utils::swap_axes(Precision::DOUBLE, d_in, d_out, num_cols, num_rows, block_size);
    ComplexD *h_out = new ComplexD[(size_t)num_cols * num_rows * block_size];
    gpuErrchk(cudaMemcpy(
        h_out, d_out, num_cols * num_rows * block_size * sizeof(ComplexD), cudaMemcpyDeviceToHost));
    for (int r = 0; r < num_rows; r++)
    {
        for (int c = 0; c < num_cols; c++)
        {
            for (int t = 0; t < block_size; t++)
            {
                size_t idx = r * num_cols * block_size + c * block_size + t;
                size_t idx2 = t * num_rows * num_cols + c * num_rows + r;
                ASSERT_NEAR(h_in[idx].x, h_out[idx2].x, 1e-10);
                ASSERT_NEAR(h_in[idx].y, h_out[idx2].y, 1e-10);
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
    for (int color = 0; color < comm_size; color++)
    {
        size_t start_index = Utils::get_start_index(glob_num_blocks, color, comm_size);
        ASSERT_EQ(start_index, correct_start_indices[color]);
    }
}

TEST(UtilsTest, GlobalToLocalSize)
{
    int global_size = 10;
    int comm_size = 4;
    int correct_local_sizes[4] = {3, 3, 2, 2};
    for (int color = 0; color < comm_size; color++)
    {
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


TEST(UtilsTest, GenerateDouble)
{
    size_t index = 123456789;
    double value = Utils::generate_double(index);
    // check that the number is in [1, 2)
    ASSERT_TRUE(value >= 1.0);
    ASSERT_TRUE(value < 2.0);
}

//============================================================================//
//                            THE ACTUAL TESTS                                //
//============================================================================//

TYPED_TEST(UtilComplexKernelsTest, Transpose2D) {
    using T_complex = TypeParam;
    // Determine the precision enum for the dispatcher function
    constexpr Precision P = (std::is_same_v<T_complex, ComplexF>) ? Precision::SINGLE : Precision::DOUBLE;

    const int num_rows = 64;
    const int num_cols = 48;
    const size_t count = num_rows * num_cols;

    gpuErrchk(cudaMalloc(&this->d_vec_in, count * sizeof(T_complex)));
    gpuErrchk(cudaMalloc(&this->d_vec_out, count * sizeof(T_complex)));
    cublasCreate(&this->handle);

    std::vector<T_complex> h_in(count);
    for (size_t i = 0; i < count; ++i) {
        h_in[i] = make_complex_from_int<T_complex>(i);
    }
    gpuErrchk(cudaMemcpy(this->d_vec_in, h_in.data(), count * sizeof(T_complex), cudaMemcpyHostToDevice));

    // Call the public-facing dispatcher function
    Utils::transpose_2d(P, this->d_vec_in, this->d_vec_out, num_cols, num_rows, this->handle, nullptr);

    std::vector<T_complex> h_gpu_out(count);
    gpuErrchk(cudaMemcpy(h_gpu_out.data(), this->d_vec_out, count * sizeof(T_complex), cudaMemcpyDeviceToHost));
    
    std::vector<T_complex> h_expected;
    transpose_cpu(h_in, h_expected, num_rows, num_cols);

    for (size_t i = 0; i < count; ++i) {
        if (std::is_same_v<T_complex, ComplexF>) {
            ASSERT_FLOAT_EQ(h_gpu_out[i].x, h_expected[i].x);
            ASSERT_FLOAT_EQ(h_gpu_out[i].y, h_expected[i].y);
        } else {
            ASSERT_DOUBLE_EQ(h_gpu_out[i].x, h_expected[i].x);
            ASSERT_DOUBLE_EQ(h_gpu_out[i].y, h_expected[i].y);
        }
    }
    cublasDestroy(this->handle);
}

TYPED_TEST(UtilComplexKernelsTest, Sbgemv) {
    using T_complex = TypeParam;
    constexpr Precision P = (std::is_same_v<T_complex, ComplexF>) ? Precision::SINGLE : Precision::DOUBLE;

    const int num_rows = 8;
    const int num_cols = 6;
    const int block_size = 4; // batch size

    // --- Test Case 1: Regular (No Conjugate) ---
    {
        const size_t mat_count = block_size * num_rows * num_cols;
        const size_t vec_in_count = block_size * num_cols;
        const size_t vec_out_count = block_size * num_rows;
        
        gpuErrchk(cudaMalloc(&this->d_mat, mat_count * sizeof(T_complex)));
        gpuErrchk(cudaMalloc(&this->d_vec_in, vec_in_count * sizeof(T_complex)));
        gpuErrchk(cudaMalloc(&this->d_vec_out, vec_out_count * sizeof(T_complex)));
        cublasCreate(&this->handle);

        std::vector<T_complex> h_mat(mat_count);
        std::vector<T_complex> h_vec_in(vec_in_count);
        for(size_t i=0; i<mat_count; ++i) h_mat[i] = make_complex_from_int<T_complex>(i);
        for(size_t i=0; i<vec_in_count; ++i) h_vec_in[i] = make_complex_from_int<T_complex>(i);

        gpuErrchk(cudaMemcpy(this->d_mat, h_mat.data(), mat_count * sizeof(T_complex), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(this->d_vec_in, h_vec_in.data(), vec_in_count * sizeof(T_complex), cudaMemcpyHostToDevice));

        Utils::sbgemv(P, this->d_mat, this->d_vec_in, this->d_vec_out, num_rows, num_cols, block_size, false, this->handle, nullptr);

        std::vector<T_complex> h_gpu_out(vec_out_count);
        gpuErrchk(cudaMemcpy(h_gpu_out.data(), this->d_vec_out, vec_out_count * sizeof(T_complex), cudaMemcpyDeviceToHost));
        
        std::vector<T_complex> h_expected;
        sbgemv_cpu(h_mat, h_vec_in, h_expected, num_rows, num_cols, block_size, false);

        for (size_t i = 0; i < vec_out_count; ++i) {
            if (std::is_same_v<T_complex, ComplexF>) {
                ASSERT_NEAR(h_gpu_out[i].x, h_expected[i].x, 1e-3);
                ASSERT_NEAR(h_gpu_out[i].y, h_expected[i].y, 1e-3);
            } else {
                ASSERT_NEAR(h_gpu_out[i].x, h_expected[i].x, 1e-9);
                ASSERT_NEAR(h_gpu_out[i].y, h_expected[i].y, 1e-9);
            }
        }
        cublasDestroy(this->handle);
        cudaFree(this->d_mat); this->d_mat = nullptr;
        cudaFree(this->d_vec_in); this->d_vec_in = nullptr;
        cudaFree(this->d_vec_out); this->d_vec_out = nullptr;
    }

    // --- Test Case 2: Conjugate Transpose ---
    {
        const size_t mat_count = block_size * num_rows * num_cols;
        const size_t vec_in_count = block_size * num_rows; // Input vec has num_rows length
        const size_t vec_out_count = block_size * num_cols; // Output vec has num_cols length
        
        gpuErrchk(cudaMalloc(&this->d_mat, mat_count * sizeof(T_complex)));
        gpuErrchk(cudaMalloc(&this->d_vec_in, vec_in_count * sizeof(T_complex)));
        gpuErrchk(cudaMalloc(&this->d_vec_out, vec_out_count * sizeof(T_complex)));
        cublasCreate(&this->handle);

        std::vector<T_complex> h_mat(mat_count);
        std::vector<T_complex> h_vec_in(vec_in_count);
        for(size_t i=0; i<mat_count; ++i) h_mat[i] = make_complex_from_int<T_complex>(i);
        for(size_t i=0; i<vec_in_count; ++i) h_vec_in[i] = make_complex_from_int<T_complex>(i);

        gpuErrchk(cudaMemcpy(this->d_mat, h_mat.data(), mat_count * sizeof(T_complex), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(this->d_vec_in, h_vec_in.data(), vec_in_count * sizeof(T_complex), cudaMemcpyHostToDevice));

        Utils::sbgemv(P, this->d_mat, this->d_vec_in, this->d_vec_out, num_rows, num_cols, block_size, true, this->handle, nullptr);

        std::vector<T_complex> h_gpu_out(vec_out_count);
        gpuErrchk(cudaMemcpy(h_gpu_out.data(), this->d_vec_out, vec_out_count * sizeof(T_complex), cudaMemcpyDeviceToHost));
        
        std::vector<T_complex> h_expected;
        sbgemv_cpu(h_mat, h_vec_in, h_expected, num_rows, num_cols, block_size, true);

        for (size_t i = 0; i < vec_out_count; ++i) {
            if (std::is_same_v<T_complex, ComplexF>) {
                ASSERT_NEAR(h_gpu_out[i].x, h_expected[i].x, 1e-3);
                ASSERT_NEAR(h_gpu_out[i].y, h_expected[i].y, 1e-3);
            } else {
                ASSERT_NEAR(h_gpu_out[i].x, h_expected[i].x, 1e-9);
                ASSERT_NEAR(h_gpu_out[i].y, h_expected[i].y, 1e-9);
            }
        }
        cublasDestroy(this->handle);
        cudaFree(this->d_mat); this->d_mat = nullptr;
        cudaFree(this->d_vec_in); this->d_vec_in = nullptr;
        cudaFree(this->d_vec_out); this->d_vec_out = nullptr;
    }
}
