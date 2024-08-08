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
    unsigned int block_size; /**< Size of each block. */
    double* d_vec; /**< Pointer to the vector data. */
    std::string row_or_col; /**< Indicates whether the vector is row or column. */
    bool initialized = false; /**< Flag indicating if the vector is initialized. */

public:
    /**
     * @brief Constructor for the Vector class.
     * @param comm The Comm object.
     * @param num_blocks The number of blocks.
     * @param block_size The size of each block.
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
     * @brief Destructor for the Vector class.
     */
    ~Vector();

    /**
     * @brief Initializes the vector.
     */
    void init_vec();

    /**
     * @brief Initializes the vector from a file.
     * @param filename The name of the file.
     */
    void init_vec_from_file(std::string filename);

    /**
     * @brief Initializes the vector with all ones.
     */
    void init_vec_ones();

    /**
     * @brief Initializes the vector with all zeros.
     */
    void init_vec_zeros();

    /**
     * @brief Performs the operation y = alpha * x + y.
     * @param alpha The scalar value.
     * @param x The Vector object x.
     * @param y The Vector object y.
     */
    void vec_axpy(double alpha, Vector& x, Vector& y);

    /**
     * @brief Performs the operation w = alpha * x + y.
     * @param alpha The scalar value.
     * @param x The Vector object x.
     * @param y The Vector object y.
     * @param w The Vector object w.
     */
    void vec_waxpy(double alpha, Vector& x, Vector& y, Vector& w);

    /**
     * @brief Scales the vector by a scalar value.
     * @param alpha The scalar value.
     */
    void vec_scale(double alpha);

    /**
     * @brief Checks if the vector is on the grid.
     * @return True if the vector is on the grid, false otherwise.
     */
    bool on_grid()
    {
        if (row_or_col == "row") {
            return comm.get_row_color() == 0;
        } else if (row_or_col == "col") {
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
     * @brief Gets the size of each block.
     * @return The size of each block.
     */
    unsigned int get_block_size() { return block_size; }

    /**
     * @brief Gets the row or column descriptor.
     * @return The row or column descriptor.
     */
    std::string get_row_or_col() { return row_or_col; }

    /**
     * @brief Checks if the vector is initialized.
     * @return True if the vector is initialized, false otherwise.
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

#endif