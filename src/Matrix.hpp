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
    unsigned int glob_num_cols; /**< The global number of columns in the matrix. */
    unsigned int glob_num_rows; /**< The global number of rows in the matrix. */
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
     * output, and x is the input.
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
};

#endif // __MATRIX_HPP__