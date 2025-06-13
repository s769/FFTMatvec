/**
 * @file Matrix.hpp
 * @brief This file contains the declaration of the Matrix class.
 */

#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include "Comm.hpp"
#include "Vector.hpp"
#include "shared.hpp"
#include "precision.hpp"

class Vector; // forward declaration

/**
 * @struct MatvecConfig
 * @brief Represents the configuration for matrix-vector operations.
 * @var MatvecConfig::transpose
 * Flag indicating if the matvec is a transpose matvec.
 * @var MatvecConfig::full
 * Flag indicating if the matvec is with a full matrix (F*F) or just F.
 * @var MatvecConfig::use_aux_mat
 * Flag indicating if the auxiliary matrix G is used for the matvec.
 */
struct MatvecConfig {
    bool transpose = false; /**< Flag indicating if the matvec is a transpose matvec. */
    bool full = false;       /**< Flag indicating if the matvec is with a full matrix. */
    bool use_aux_mat = false; /**< Flag indicating if the auxiliary matrix G is used for the matvec. */
};

/**
 * @class Matrix
 * @brief Represents a matrix and provides matrix operations.
 */
class Matrix
{

private:
    MatvecPrecisionConfig p_config;          /**< The precision configuration for matrix-vector operations. */
    ComplexD *mat_freq_TOSI = nullptr;       /**< Pointer to the matrix frequency in TOSI format. */
    ComplexD *mat_freq_TOSI_aux = nullptr;   /**< Pointer to the other matrix frequency in TOSI format. */
    ComplexF *mat_freq_TOSI_F = nullptr;     /**< Pointer to the matrix frequency in TOSI format (float). */
    ComplexF *mat_freq_TOSI_aux_F = nullptr; /**< Pointer to the other matrix frequency in TOSI format (float). */
    unsigned int padded_size;                /**< The padded block size of the matrix. */
    unsigned int block_size;                 /**< The unpadded block size of the matrix. */
    unsigned int num_cols;                   /**< The number of columns in the matrix. */
    unsigned int num_rows;                   /**< The number of rows in the matrix. */
    size_t glob_num_cols;                    /**< The global number of columns in the matrix. */
    size_t glob_num_rows;                    /**< The global number of rows in the matrix. */
    double *col_vec_unpad = nullptr;         /**< Pointer to the unpadded column vector. */
    double *col_vec_pad = nullptr;           /**< Pointer to the padded column vector. */
    double *row_vec_pad = nullptr;           /**< Pointer to the padded row vector. */
    double *row_vec_unpad = nullptr;         /**< Pointer to the unpadded row vector. */
    double *res_pad = nullptr;               /**< Pointer to the padded result vector. */
    double *res_unpad = nullptr;             /**< Pointer to the unpadded result vector. */
    float *res_pad_F = nullptr;              /**< Pointer to the padded result vector (float). */
    float *res_unpad_F = nullptr;            /**< Pointer to the unpadded result vector (float). */
    float *col_vec_unpad_F = nullptr;        /**< Pointer to the unpadded column vector (float). */
    float *col_vec_pad_F = nullptr;          /**< Pointer to the padded column vector (float). */
    float *row_vec_pad_F = nullptr;          /**< Pointer to the padded row vector (float). */
    float *row_vec_unpad_F = nullptr;        /**< Pointer to the unpadded row vector (float). */
    ComplexD *col_vec_freq = nullptr;        /**< Pointer to the column vector frequency. */
    ComplexD *row_vec_freq = nullptr;        /**< Pointer to the row vector frequency. */
    ComplexD *col_vec_freq_TOSI = nullptr;   /**< Pointer to the column vector frequency in TOSI format. */
    ComplexD *row_vec_freq_TOSI = nullptr;   /**< Pointer to the row vector frequency in TOSI format. */
    ComplexF *col_vec_freq_F = nullptr;      /**< Pointer to the column vector frequency (float). */
    ComplexF *row_vec_freq_F = nullptr;      /**< Pointer to the row vector frequency (float). */
    ComplexF *col_vec_freq_TOSI_F = nullptr; /**< Pointer to the column vector frequency in TOSI format (float). */
    ComplexF *row_vec_freq_TOSI_F = nullptr; /**< Pointer to the row vector frequency in TOSI format (float). */
    cufftHandle forward_plan;                /**< The forward plan for FFT. */
    cufftHandle inverse_plan;                /**< The inverse plan for FFT. */
    cufftHandle forward_plan_conj;           /**< The forward plan for conjugate FFT. */
    cufftHandle inverse_plan_conj;           /**< The inverse plan for conjugate FFT. */
    bool initialized = false;                /**< Flag indicating if the matrix is initialized. */
    bool has_mat_freq_TOSI_aux = false;      /**< Flag indicating if the other matrix frequency in TOSI format exists. */
    bool is_QoI = false;                     /**< Flag indicating if the matrix is the p2q map. */
    int checksum = 0;                        /**< Checksum for the matrix. */
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
    void check_matvec(Vector &x, Vector &y, bool transpose, bool full, bool use_aux_mat);

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
     * @param mat_freq_TOSI Pointer to pointer to the matrix in device memory.
     * @param h_mat Pointer to the matrix in host memory.
     */
    void setup_matvec(ComplexD **mat_freq_TOSI, const double *const h_mat);
    
    /**
     * @brief Casts mat_freq_TOSI to float. Allocates memory for mat_freq_TOSI_F.
     * 
     * @param mat_freq_TOSI_F Pointer to the matrix in device memory (float).
     * @param mat_freq_TOSI Pointer to the matrix in device memory.
     */
    void setup_mat_freq_TOSI_F(ComplexF **mat_freq_TOSI_F, const ComplexD *const mat_freq_TOSI);



    /**
     * @brief Main function for matrix-vector multiplication.
     * 
     * @param out_vec The output vector.
     * @param in_vec The input vector.
     * @param config The configuration for the matvec.
     * 
     * */   
    void compute_matvec(double* out_vec, double* in_vec, const MatvecConfig& config);


    

public:
    Comm &comm; /**< Reference to the communication object. */
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
     * @param p_config The precision configuration for matrix-vector operations.
     */
    Matrix(Comm &comm, unsigned int cols, unsigned int rows, unsigned int block_size,
           bool global_sizes = false, bool QoI = false, const MatvecPrecisionConfig &p_config = MatvecPrecisionConfig());

    /**
     * @brief Constructs a Matrix object from a meta file.
     * @param comm The communication object (passed as reference).
     * @param path Path to the directory containing the matrix data.
     * @param aux_path Path to the directory containing the auxiliary matrix data. Cannot be
     * nonempty if path is empty.
     * @param QoI Flag indicating if the matrix is the p2q map instead of the p2o map.
     * @param p_config The precision configuration for matrix-vector operations.
     *
     */
    Matrix(Comm &comm, std::string path, std::string aux_path = "", bool QoI = false,
           const MatvecPrecisionConfig &p_config = MatvecPrecisionConfig());

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
    void matvec(Vector &x, Vector &y, bool use_aux_mat = false, bool full = false);

    /**
     * @brief Performs conjugate transpose matrix-vector multiplication.
     * @param x The input vector.
     * @param y The output vector.
     * @param use_aux_mat Flag indicating if the auxiliary matrix G is used for the matvec.
     * @param full Flag indicating if the matvec is with the full matrix FF* or just F*.
     */
    void transpose_matvec(Vector &x, Vector &y, bool use_aux_mat = false, bool full = false);

    /**
     * @brief Get an input or output vector compatible with the matrix.
     * @param input_or_output The string "input" or "output". In the expression y = Fx, y is the
     * output, and x is the input. x will have size glob_num_cols * block_size, and y will have size
     * glob_num_rows * block_size.
     * @return The input or output vector.
     */
    Vector get_vec(std::string input_or_output);

    // Getters
    double *get_col_vec_unpad()
    {
        return col_vec_unpad;
    } /**< Returns the unpadded column vector. */
    double *get_col_vec_pad() { return col_vec_pad; }     /**< Returns the padded column vector. */
    double *get_row_vec_pad() { return row_vec_pad; }     /**< Returns the padded row vector. */
    double *get_row_vec_unpad() { return row_vec_unpad; } /**< Returns the unpadded row vector. */
    double *get_res_pad() { return res_pad; }             /**< Returns the padded result vector. */
    ComplexD *get_col_vec_freq()
    {
        return col_vec_freq;
    } /**< Returns the column vector frequency. */
    ComplexD *get_row_vec_freq() { return row_vec_freq; } /**< Returns the row vector frequency. */
    ComplexD *get_col_vec_freq_TOSI()
    {
        return col_vec_freq_TOSI;
    } /**< Returns the column vector frequency in TOSI format. */
    ComplexD *get_row_vec_freq_TOSI()
    {
        return row_vec_freq_TOSI;
    } /**< Returns the row vector frequency in TOSI format. */
    ComplexD *get_mat_freq_TOSI()
    {
        return mat_freq_TOSI;
    } /**< Returns the matrix frequency in TOSI format. */
    ComplexD *get_mat_freq_TOSI_aux()
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

    float *get_col_vec_unpad_F()
    {
        return col_vec_unpad_F;
    } /**< Returns the unpadded column vector (float). */
    float *get_col_vec_pad_F() { return col_vec_pad_F; }     /**< Returns the padded column vector (float). */
    float *get_row_vec_pad_F() { return row_vec_pad_F; }     /**< Returns the padded row vector (float). */
    float *get_row_vec_unpad_F() { return row_vec_unpad_F; } /**< Returns the unpadded row vector (float). */   

    ComplexF *get_col_vec_freq_F()
    {
        return col_vec_freq_F;
    } /**< Returns the column vector frequency (float). */
    ComplexF *get_row_vec_freq_F() { return row_vec_freq_F; } /**< Returns the row vector frequency (float). */
    ComplexF *get_col_vec_freq_TOSI_F()
    {
        return col_vec_freq_TOSI_F;
    } /**< Returns the column vector frequency in TOSI format (float). */       
    ComplexF *get_row_vec_freq_TOSI_F()
    {
        return row_vec_freq_TOSI_F;
    } /**< Returns the row vector frequency in TOSI format (float). */
    ComplexF *get_mat_freq_TOSI_F()
    {
        return mat_freq_TOSI_F;
    } /**< Returns the matrix frequency in TOSI format (float). */
    ComplexF *get_mat_freq_TOSI_aux_F()
    {
        return mat_freq_TOSI_aux_F;
    } /**< Returns the other matrix frequency in TOSI format (float). */
    MatvecPrecisionConfig get_precision_config()
    {
        return p_config;
    } /**< Returns the precision configuration for matrix-vector operations. */
};

#endif // __MATRIX_HPP__