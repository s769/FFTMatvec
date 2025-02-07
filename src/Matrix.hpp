/**
 * @file Matrix.hpp
 * @brief This file contains the declaration of the Matrix class.
 */

#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include "Comm.hpp"
#include "Vector.hpp"
#include "shared.hpp"

class Vector; // forward declaration

/**
 * @class Matrix
 * @brief Represents a matrix and provides matrix operations.
 */
class Matrix {

private:
    Complex* mat_freq_TOSI; /**< Pointer to the matrix frequency in TOSI format. */
    Complex* mat_freq_TOSI_aux
        = nullptr; /**< Pointer to the other matrix frequency in TOSI format. */
    unsigned int padded_size; /**< The padded block size of the matrix. */
    unsigned int block_size; /**< The unpadded block size of the matrix. */
    unsigned int num_cols; /**< The number of columns in the matrix. */
    unsigned int num_rows; /**< The number of rows in the matrix. */
    size_t glob_num_cols; /**< The global number of columns in the matrix. */
    size_t glob_num_rows; /**< The global number of rows in the matrix. */
    double* col_vec_unpad; /**< Pointer to the unpadded column vector. */
    double* col_vec_pad; /**< Pointer to the padded column vector. */
    double* row_vec_pad; /**< Pointer to the padded row vector. */
    double* row_vec_unpad; /**< Pointer to the unpadded row vector. */
    double* res_pad; /**< Pointer to the padded result vector. */
    Complex* col_vec_freq; /**< Pointer to the column vector frequency. */
    Complex* row_vec_freq; /**< Pointer to the row vector frequency. */
    Complex* col_vec_freq_TOSI; /**< Pointer to the column vector frequency in TOSI format. */
    Complex* row_vec_freq_TOSI; /**< Pointer to the row vector frequency in TOSI format. */
    cufftHandle forward_plan; /**< The forward plan for FFT. */
    cufftHandle inverse_plan; /**< The inverse plan for FFT. */
    cufftHandle forward_plan_conj; /**< The forward plan for conjugate FFT. */
    cufftHandle inverse_plan_conj; /**< The inverse plan for conjugate FFT. */
    bool initialized = false; /**< Flag indicating if the matrix is initialized. */
    bool has_mat_freq_TOSI_aux
        = false; /**< Flag indicating if the other matrix frequency in TOSI format exists. */
    bool is_QoI = false; /**< Flag indicating if the matrix is the p2q map. */
    int checksum = 0; /**< Checksum for the matrix. */
    /**
     * @brief Reads the meta file to get the matrix dimensions.
     * @param meta_filename The name of the meta file.
     * @param QoI Flag indicating if the matrix is the p2q map instead of the p2o map.
     * @param aux_mat Flag indicating if the matrix to be initialized is the auxiliary matrix. The
     * primary matrix must be initialized first.
     * @return The prefix path to the adjoint vectors.
     */
    std::string read_meta(std::string meta_filename, bool QoI = false, bool aux_mat = false);

    /**
     * @brief Checks for errors before performing matrix-vector multiplication.
     * @param transpose Flag indicating if the matvec is a transpose matvec.
     * @param full Flag indicating if the matvec is with a full matrix.
     * @param use_aux_mat Flag indicating if the auxiliary matrix G is used for the matvec.
     */
    void check_matvec(Vector& x, Vector& y, bool transpose, bool full, bool use_aux_mat);

    /*
     * @brief Initializes the matrix (usued by the constructors).
     * @param cols The number of columns in the matrix (local or global based on value of
     * global_sizes).
     * @param rows The number of rows in the matrix (local or global based on value of
     * global_sizes).
     * @param block_size The block size of the matrix without padding.
     * @param global_sizes Flag indicating whether the sizes are global.
     */
    void initialize(
        unsigned int cols, unsigned int rows, unsigned int block_size, bool global_sizes);

    /**
     * @brief Set up the matrix and perform necessary initialization for matrix-vector operations.
     *
     * This function sets up the matrix and performs necessary initialization for matrix-vector
     * operations.
     *
     * @param d_mat_freq Pointer to the matrix in device memory.
     * @param h_mat Pointer to the matrix in host memory.
     * @param padded_size The block size for matrix operations.
     * @param num_cols The number of columns in the matrix.
     * @param num_rows The number of rows in the matrix.
     * @param cublasHandle The handle for the cuBLAS library.
     */
    void setup_matvec(Complex** d_mat_freq, const double* const h_mat,
        const unsigned int padded_size, const unsigned int num_cols, const unsigned int num_rows,
        cublasHandle_t cublasHandle);

    /**
     * @brief Perform local matrix-vector multiplication.
     *
     * This function performs local matrix-vector multiplication using the provided matrix and
     * vectors.
     *
     * @param out_vec Pointer to the output vector.
     * @param in_vec Pointer to the input vector.
     * @param d_mat_freq Pointer to the matrix in device memory.
     * @param size The size of the vectors.
     * @param num_cols The number of columns in the matrix.
     * @param num_rows The number of rows in the matrix.
     * @param conjugate Flag indicating whether to perform conjugate multiplication.
     * @param unpad Flag indicating whether to unpad the output vector.
     * @param device The device ID.
     * @param forward_plan The forward FFT plan.
     * @param inverse_plan The inverse FFT plan.
     * @param out_vec_pad Pointer to the padded output vector.
     * @param in_vec_freq Pointer to the input vector in frequency domain.
     * @param out_vec_freq_TOSI Pointer to the output vector in frequency domain (transpose of
     * input, scaled).
     * @param in_vec_freq_TOSI Pointer to the input vector in frequency domain (transpose of input,
     * scaled).
     * @param out_vec_freq Pointer to the output vector in frequency domain.
     * @param s The CUDA stream.
     * @param cublasHandle The handle for the cuBLAS library.
     */
    void local_matvec(double* const out_vec, double* const in_vec, const Complex* const d_mat_freq,
        const unsigned int size, const unsigned int num_cols, const unsigned int num_rows,
        const bool conjugate, const bool unpad, const unsigned int device, cufftHandle forward_plan,
        cufftHandle inverse_plan, double* const out_vec_pad, Complex* const in_vec_freq,
        Complex* const out_vec_freq_TOSI, Complex* const in_vec_freq_TOSI,
        Complex* const out_vec_freq, cudaStream_t s, cublasHandle_t cublasHandle);

    /**
     * @brief Perform matrix-vector multiplication.
     *
     * This function performs matrix-vector multiplication using the provided matrix and vectors.
     *
     * @param out_vec Pointer to the output vector.
     * @param in_vec Pointer to the input vector.
     * @param mat_freq_TOSI Pointer to the matrix in frequency domain (transpose of input, scaled).
     * @param padded_size The block size for matrix operations.
     * @param num_cols The number of columns in the matrix.
     * @param num_rows The number of rows in the matrix.
     * @param conjugate Flag indicating whether to perform conjugate multiplication.
     * @param full Flag indicating whether to perform full matrix-vector multiplication.
     * @param device The device ID.
     * @param scale The scaling factor.
     * @param nccl_row_comm The NCCL communicator for row-wise communication.
     * @param nccl_col_comm The NCCL communicator for column-wise communication.
     * @param s The CUDA stream.
     * @param in_vec_pad Pointer to the padded input vector.
     * @param forward_plan The forward FFT plan.
     * @param inverse_plan The inverse FFT plan.
     * @param forward_plan_conj The forward FFT plan for conjugate multiplication.
     * @param inverse_plan_conj The inverse FFT plan for conjugate multiplication.
     * @param out_vec_pad Pointer to the padded output vector.
     * @param in_vec_freq Pointer to the input vector in frequency domain.
     * @param out_vec_freq_TOSI Pointer to the output vector in frequency domain (transpose of
     * input, scaled).
     * @param in_vec_freq_TOSI Pointer to the input vector in frequency domain (transpose of input,
     * scaled).
     * @param out_vec_freq Pointer to the output vector in frequency domain.
     * @param cublasHandle The handle for the cuBLAS library.
     * @param mat_freq_TOSI_aux Pointer to the matrix in frequency domain (transpose of input,
     * scaled) on other devices.
     * @param res_pad Pointer to the padded result vector.
     * @param use_aux_mat Flag indicating whether to use the auxiliary matrix for the full
     * multiplication (i.e. FG^* or G^*F)
     */
    void compute_matvec(double* out_vec, double* in_vec, Complex* mat_freq_TOSI,
        const unsigned int padded_size, const unsigned int num_cols, const unsigned int num_rows,
        const bool conjugate, const bool full, const unsigned int device, ncclComm_t nccl_row_comm,
        ncclComm_t nccl_col_comm, cudaStream_t s, double* const in_vec_pad,
        cufftHandle forward_plan, cufftHandle inverse_plan, cufftHandle forward_plan_conj,
        cufftHandle inverse_plan_conj, double* const out_vec_pad, Complex* const in_vec_freq,
        Complex* const out_vec_freq_TOSI, Complex* const in_vec_freq_TOSI,
        Complex* const out_vec_freq, cublasHandle_t cublasHandle, Complex* mat_freq_TOSI_aux,
        double* const res_pad, bool use_aux_mat = false);

public:
    Comm& comm; /**< Reference to the communication object. */
    /**
     * @brief Constructs a Matrix object.
     * @param comm The communication object (passed as reference).
     * @param cols The number of columns in the matrix (local or global based on value of
     * global_sizes).
     * @param rows The number of rows in the matrix (local or global based on value of
     * global_sizes).
     * @param block_size The block size of the matrix without padding.
     * @param global_sizes Flag indicating whether the sizes are global.
     * @param QoI Flag indicating if the matrix is the p2q map instead of the p2o map.
     */
    Matrix(Comm& comm, unsigned int cols, unsigned int rows, unsigned int block_size,
        bool global_sizes = false, bool QoI = false);

    /**
     * @brief Constructs a Matrix object from a meta file.
     * @param comm The communication object (passed as reference).
     * @param path Path to the directory containing the matrix data.
     * @param aux_path Path to the directory containing the auxiliary matrix data. Cannot be
     * nonempty if path is empty.
     * @param QoI Flag indicating if the matrix is the p2q map instead of the p2o map.
     *
     */
    Matrix(Comm& comm, std::string path, std::string aux_path = "", bool QoI = false);

    /**
     * @brief Destroys the Matrix object. Frees the memory allocated for the matrix data.
     */
    ~Matrix();

    /**
     * @brief Initializes the matrix from a file.
     * @param dirname The path to the directory containing the adjoint vectors.
     * @param aux_mat Flag indicating if the matrix to be initialized is the auxiliary matrix. The
     * primary matrix must be initialized first.
     */
    void init_mat_from_file(std::string dirname, bool aux_mat = false);

    /**
     * @brief Initializes the matrix with ones.
     * @param aux_mat Flag indicating if the matrix to be initialized is the auxiliary matrix. The
     * primary matrix must be initialized first.
     */
    void init_mat_ones(bool aux_mat = false);

    /**
     * @brief Performs matrix-vector multiplication.
     * @param x The input vector.
     * @param y The output vector.
     * @param use_aux_mat Flag indicating if the auxiliary matrix G is used for the matvec.
     * @param full Flag indicating if the matvec is with the full matrix F*F or just F.
     */
    void matvec(Vector& x, Vector& y, bool use_aux_mat = false, bool full = false);

    /**
     * @brief Performs conjugate transpose matrix-vector multiplication.
     * @param x The input vector.
     * @param y The output vector.
     * @param use_aux_mat Flag indicating if the auxiliary matrix G is used for the matvec.
     * @param full Flag indicating if the matvec is with the full matrix FF* or just F*.
     */
    void transpose_matvec(Vector& x, Vector& y, bool use_aux_mat = false, bool full = false);

    /**
     * @brief Get an input or output vector compatible with the matrix.
     * @param input_or_output The string "input" or "output". In the expression y = Fx, y is the
     * output, and x is the input. x will have size glob_num_cols * block_size, and y will have size
     * glob_num_rows * block_size.
     * @return The input or output vector.
     */
    Vector get_vec(std::string input_or_output);

    // Getters
    double* get_col_vec_unpad()
    {
        return col_vec_unpad;
    } /**< Returns the unpadded column vector. */
    double* get_col_vec_pad() { return col_vec_pad; } /**< Returns the padded column vector. */
    double* get_row_vec_pad() { return row_vec_pad; } /**< Returns the padded row vector. */
    double* get_row_vec_unpad() { return row_vec_unpad; } /**< Returns the unpadded row vector. */
    double* get_res_pad() { return res_pad; } /**< Returns the padded result vector. */
    Complex* get_col_vec_freq()
    {
        return col_vec_freq;
    } /**< Returns the column vector frequency. */
    Complex* get_row_vec_freq() { return row_vec_freq; } /**< Returns the row vector frequency. */
    Complex* get_col_vec_freq_TOSI()
    {
        return col_vec_freq_TOSI;
    } /**< Returns the column vector frequency in TOSI format. */
    Complex* get_row_vec_freq_TOSI()
    {
        return row_vec_freq_TOSI;
    } /**< Returns the row vector frequency in TOSI format. */
    Complex* get_mat_freq_TOSI()
    {
        return mat_freq_TOSI;
    } /**< Returns the matrix frequency in TOSI format. */
    Complex* get_mat_freq_TOSI_aux()
    {
        return mat_freq_TOSI_aux;
    } /**< Returns the other matrix frequency in TOSI format. */
    cufftHandle get_forward_plan()
    {
        return forward_plan;
    } /**< Returns the forward plan for FFT. */
    cufftHandle get_inverse_plan()
    {
        return inverse_plan;
    } /**< Returns the inverse plan for FFT. */
    cufftHandle get_forward_plan_conj()
    {
        return forward_plan_conj;
    } /**< Returns the forward plan for conjugate FFT. */
    cufftHandle get_inverse_plan_conj()
    {
        return inverse_plan_conj;
    } /**< Returns the inverse plan for conjugate FFT. */
    unsigned int get_num_cols()
    {
        return num_cols;
    } /**< Returns the number of columns in the matrix. */
    unsigned int get_num_rows()
    {
        return num_rows;
    } /**< Returns the number of rows in the matrix. */
    unsigned int get_glob_num_cols()
    {
        return glob_num_cols;
    } /**< Returns the global number of columns in the matrix. */
    unsigned int get_glob_num_rows()
    {
        return glob_num_rows;
    } /**< Returns the global number of rows in the matrix. */
    unsigned int get_padded_size()
    {
        return padded_size;
    } /**< Returns the padded block size of the matrix. */
    unsigned int get_block_size()
    {
        return block_size;
    } /**< Returns the unpadded block size of the matrix. */
    bool is_initialized()
    {
        return initialized;
    } /**< Returns true if the matrix is initialized, false otherwise. */
    bool has_aux_mat()
    {
        return has_mat_freq_TOSI_aux;
    } /**< Returns true if the other matrix frequency in TOSI format exists (and is initialized),
         false otherwise. */
    bool is_p2q_mat()
    {
        return is_QoI;
    } /**< Returns true if the matrix is the p2q map, false otherwise. */
    int get_checksum() { return checksum; } /**< Returns the checksum for the matrix. */
};

#endif // __MATRIX_HPP__