/**
 * @file Vector.hpp
 * @brief Header file for the Vector class.
 */

#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

#include "Comm.hpp"
#include "shared.hpp"

/**
 * @class Vector
 * @brief Represents a vector.
 */
class Vector {
private:
    Comm& comm; /**< Reference to the Comm object. */
    unsigned int num_blocks; /**< Number of blocks. */
    unsigned int padded_size; /**< Size of each block with padding. */
    unsigned int block_size; /**< Size of each block without padding. */
    double* d_vec; /**< Pointer to the vector data. */
    std::string row_or_col; /**< Indicates whether the vector is row or column. */
    bool initialized = false; /**< Flag indicating if the vector is initialized. */

public:
    /**
     * @brief Constructor for the Vector class.
     * @param comm The Comm object (passed as reference).
     * @param num_blocks The number of blocks.
     * @param block_size The size of each block without padding.
     * @param row_or_col Indicates whether the vector is row or column.
     */
    Vector(Comm& comm, unsigned int num_blocks, unsigned int block_size, std::string row_or_col);

    /**
     * @brief Copy constructor for the Vector class.
     * @param vec The Vector object to be copied.
     * @param deep_copy Flag indicating whether to perform a deep copy.
     */
    Vector(Vector& vec, bool deep_copy = false);

    /**
     * @brief Destructor for the Vector class. Frees the memory allocated for the vector data.
     */
    ~Vector();

    /**
     * @brief Initializes the vector.
     */
    void init_vec();


    /**
     * @brief Initializes the vector with all ones.
     */
    void init_vec_ones();

    /**
     * @brief Initializes the vector with all zeros.
     */
    void init_vec_zeros();

    /**
     * @brief Checks if the calling process has the vector data.
     * @return True if the calling process has the vector data, false otherwise.
     */
    bool on_grid()
    {
        if (row_or_col == "col") {
            return comm.get_row_color() == 0;
        } else if (row_or_col == "row") {
            return comm.get_col_color() == 0;
        } else {
            fprintf(stderr, "Invalid grid descriptor: %s\n", row_or_col.c_str());
            MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
            exit(1);
        }
    }

    /**
     * @brief Prints the vector.
     * @param name The name of the vector.
     */
    void print(std::string name = "");

    /**
     * @brief Computes the norm of the vector.
     * @param order The order of the norm (e.g., "2" for the 2-norm, "1" for the 1-norm, "-1" for the infinity norm).
     * @return The norm of the vector. Only rank 0 has the global norm.
     */
    double norm(int order = 2);

    /**
     * @brief Scales the vector by a constant.
     * @param alpha The constant by which to scale the vector.
     */
    void scale(double alpha);

    /**
     * @brief Computes the operation y = alpha * x + y.
     * @param alpha The constant by which to scale the vector x.
     * @param x The vector to be added.
     */

    void axpy(double alpha, Vector& x);

    /**
     * @brief Computes the operation y = alpha * x + beta * y.
     * @param alpha The constant by which to scale the vector x.
     * @param beta The constant by which to scale the vector y.
     * @param x The vector to be added.
     */
    void axpby(double alpha, double beta, Vector& x);

    /**
     * @brief Computes the dot product of the vector with another vector.
     * @param x The vector with which to compute the dot product.
     * @return The dot product of the two vectors. Only rank 0 has the global dot product.
     */
    double dot(Vector& x);
    

    // Getters

    /**
     * @brief Gets the pointer to the vector data.
     * @return The pointer to the vector data.
     */
    double* get_d_vec() { return d_vec; }

    /**
     * @brief Gets the number of blocks.
     * @return The number of blocks.
     */
    unsigned int get_num_blocks() { return num_blocks; }

    /**
     * @brief Gets the size of each block with padding.
     * @return The size of each block with padding.
     */
    unsigned int get_padded_size() { return padded_size; }

    /**
     * @brief Gets the size of each block without padding.
     * @return The size of each block without padding.
     */
    unsigned int get_block_size() { return block_size; }

    /**
     * @brief Gets the communication object.
     * @return The communication object.
     */
    Comm& get_comm() { return comm; }

    /**
     * @brief Gets the row or column descriptor.
     * @return The row or column descriptor.
     */
    std::string get_row_or_col() { return row_or_col; }

    /**
     * @brief Checks if the vector data is initialized.
     * @return True if the vector data is initialized, false otherwise.
     */
    bool is_initialized() { return initialized; }

    // Setters

    /**
     * @brief Sets the pointer to the vector data.
     * @param d_vec The pointer to the vector data.
     */
    void set_d_vec(double* d_vec)
    {
        if (on_grid())
            this->d_vec = d_vec;
    }
};

#endif // __VECTOR_HPP__