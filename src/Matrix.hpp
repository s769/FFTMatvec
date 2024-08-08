/**
 * @file Matrix.hpp
 * @brief This file contains the declaration of the Matrix class.
 */

#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include "shared.hpp"
#include "Comm.hpp"
#include "Vector.hpp"
#include "matvec.hpp"

/**
 * @class Matrix
 * @brief Represents a matrix and provides matrix operations.
 */
class Matrix {

private:
    Comm& comm; /**< Reference to the communication object. */
    bool conjugate; /**< Flag indicating if the matrix is conjugate. */
    bool full; /**< Flag indicating if the matrix is full. */
    double noise_scale; /**< The noise scale of the matrix. */
    Complex *mat_freq_tosi; /**< Pointer to the matrix frequency in TOSI format. */
    Complex *mat_freq_tosi_other=nullptr; /**< Pointer to the other matrix frequency in TOSI format. */
    unsigned int block_size; /**< The block size of the matrix. */
    unsigned int num_cols; /**< The number of columns in the matrix. */
    unsigned int num_rows; /**< The number of rows in the matrix. */
    double * col_vec_unpad; /**< Pointer to the unpadded column vector. */
    double * col_vec_pad; /**< Pointer to the padded column vector. */
    double * row_vec_pad; /**< Pointer to the padded row vector. */
    double * row_vec_unpad; /**< Pointer to the unpadded row vector. */
    double * res_pad; /**< Pointer to the padded result vector. */
    Complex * col_vec_freq; /**< Pointer to the column vector frequency. */
    Complex *row_vec_freq; /**< Pointer to the row vector frequency. */
    Complex * col_vec_freq_tosi; /**< Pointer to the column vector frequency in TOSI format. */
    Complex * row_vec_freq_tosi; /**< Pointer to the row vector frequency in TOSI format. */
    cufftHandle forward_plan; /**< The forward plan for FFT. */
    cufftHandle inverse_plan; /**< The inverse plan for FFT. */
    cufftHandle forward_plan_conj; /**< The forward plan for conjugate FFT. */
    cufftHandle inverse_plan_conj; /**< The inverse plan for conjugate FFT. */
    bool initialized = false; /**< Flag indicating if the matrix is initialized. */
    bool has_mat_freq_tosi_other = false; /**< Flag indicating if the other matrix frequency in TOSI format exists. */

public:
    /**
     * @brief Constructs a Matrix object.
     * @param comm The communication object.
     * @param num_cols The number of columns in the matrix.
     * @param num_rows The number of rows in the matrix.
     * @param block_size The block size of the matrix.
     * @param conjugate Flag indicating if the matrix is conjugate.
     * @param full Flag indicating if the matrix is full.
     * @param noise_scale The noise scale of the matrix.
     */
    Matrix(Comm& comm, unsigned int num_cols, unsigned int num_rows, unsigned int block_size, bool conjugate, bool full, double noise_scale = 1.0);

    /**
     * @brief Destroys the Matrix object.
     */
    ~Matrix();

    /**
     * @brief Initializes the matrix from a file.
     * @param filename The name of the file.
     */
    void init_mat_from_file(std::string filename);

    /**
     * @brief Initializes the matrix with ones.
     */
    void init_mat_ones();

    /**
     * @brief Performs matrix-vector multiplication.
     * @param x The input vector.
     * @param y The output vector.
     * @param full Flag indicating if the matrix is full.
     */
    void matvec(Vector &x, Vector &y, bool full = false);

    /**
     * @brief Performs transpose matrix-vector multiplication.
     * @param x The input vector.
     * @param y The output vector.
     * @param full Flag indicating if the matrix is full.
     */
    void transpose_matvec(Vector &x, Vector &y, bool full = false);

    // Getters
    double * get_col_vec_unpad() { return col_vec_unpad; } /**< Returns the unpadded column vector. */
    double * get_col_vec_pad() { return col_vec_pad; } /**< Returns the padded column vector. */
    double * get_row_vec_pad() { return row_vec_pad; } /**< Returns the padded row vector. */
    double * get_row_vec_unpad() { return row_vec_unpad; } /**< Returns the unpadded row vector. */
    double * get_res_pad() { return res_pad; } /**< Returns the padded result vector. */
    Complex * get_col_vec_freq() { return col_vec_freq; } /**< Returns the column vector frequency. */
    Complex * get_row_vec_freq() { return row_vec_freq; } /**< Returns the row vector frequency. */
    Complex * get_col_vec_freq_tosi() { return col_vec_freq_tosi; } /**< Returns the column vector frequency in TOSI format. */
    Complex * get_row_vec_freq_tosi() { return row_vec_freq_tosi; } /**< Returns the row vector frequency in TOSI format. */
    Complex * get_mat_freq_tosi() { return mat_freq_tosi; } /**< Returns the matrix frequency in TOSI format. */
    Complex * get_mat_freq_tosi_other() { return mat_freq_tosi_other; } /**< Returns the other matrix frequency in TOSI format. */
    cufftHandle get_forward_plan() { return forward_plan; } /**< Returns the forward plan for FFT. */
    cufftHandle get_inverse_plan() { return inverse_plan; } /**< Returns the inverse plan for FFT. */
    cufftHandle get_forward_plan_conj() { return forward_plan_conj; } /**< Returns the forward plan for conjugate FFT. */
    cufftHandle get_inverse_plan_conj() { return inverse_plan_conj; } /**< Returns the inverse plan for conjugate FFT. */
    Comm get_comm() { return comm; } /**< Returns the communication object. */
    unsigned int get_num_cols() { return num_cols; } /**< Returns the number of columns in the matrix. */
    unsigned int get_num_rows() { return num_rows; } /**< Returns the number of rows in the matrix. */
    unsigned int get_block_size() { return block_size; } /**< Returns the block size of the matrix. */
    bool is_conjugate() { return conjugate; } /**< Returns true if the matrix is conjugate, false otherwise. */
    bool is_full() { return full; } /**< Returns true if the matrix is full, false otherwise. */
    double get_noise_scale() { return noise_scale; } /**< Returns the noise scale of the matrix. */
    bool is_initialized() { return initialized; } /**< Returns true if the matrix is initialized, false otherwise. */
    bool get_has_mat_freq_tosi_other() { return has_mat_freq_tosi_other; } /**< Returns true if the other matrix frequency in TOSI format exists, false otherwise. */

};

#endif