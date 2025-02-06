/**
 * @file Vector.hpp
 * @brief Header file for the Vector class.
 */

#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

#include "Comm.hpp"
#include "shared.hpp"

class Matrix; // forward declaration

/**
 * @class Vector
 * @brief Represents a vector.
 */
class Vector {
private:
    unsigned int num_blocks; /**< Number of blocks. */
    size_t glob_num_blocks; /**< Global number of blocks. */
    unsigned int padded_size; /**< Size of each block with padding. */
    unsigned int block_size; /**< Size of each block without padding. */
    bool SOTI_ordering; /**< Flag indicating whether to use SOTI ordering. */
    double* d_vec; /**< Pointer to the vector data. */
    std::string row_or_col; /**< Indicates whether the vector is row or column. */
    bool initialized = false; /**< Flag indicating if the vector is initialized. */

    // void to_TOSI_local(); /**< Reorder the vector to TOSI ordering (locally). */
    // void to_SOTI_local(); /**< Reorder the vector to SOTI ordering (locally). */

    // void switch_ordering(); /**< Switch the ordering of the vector. */

public:
    Comm& comm; /**< Reference to the communication object. */
    /**
     * @brief Constructor for the Vector class.
     * @param comm The Comm object (passed as reference).
     * @param blocks The number of blocks (local or global based on value of global_sizes).
     * @param block_size The size of each block without padding.
     * @param row_or_col Indicates whether the vector is row or column.
     * @param global_sizes Flag indicating whether the sizes are global.
     * @param SOTI_ordering Flag indicating whether to use SOTI ordering.
     */
    Vector(Comm& comm, unsigned int blocks, unsigned int block_size, std::string row_or_col,
        bool global_sizes = false, bool SOTI_ordering = true);

    /**
     * @brief Copy constructor for the Vector class.
     * @param vec The Vector object to be copied.
     * @param deep_copy Flag indicating whether to perform a deep copy.
     */
    Vector(Vector& vec, bool deep_copy);

    /**
     * @brief Copy constructor for the Vector class.
     * @param vec The Vector object to be copied.
     *
     */
    Vector(Vector& vec)
        : Vector(vec, true)
    {
    } // default to deep copy

    /**
     * @brief Move constructor for the Vector class.
     * @param vec The Vector object to be moved.
     *
     */
    Vector(Vector&& vec) noexcept
        : comm(vec.comm)
        , num_blocks(vec.num_blocks)
        , glob_num_blocks(vec.glob_num_blocks)
        , padded_size(vec.padded_size)
        , block_size(vec.block_size)
        , d_vec(vec.d_vec)
        , row_or_col(vec.row_or_col)
        , initialized(vec.initialized)
        , SOTI_ordering(vec.SOTI_ordering)
    {
        vec.d_vec = nullptr;
    }

    /**
     * @brief Copy assignment operator for the Vector class.
     * @param vec The Vector object to be copied.
     * @return The copied Vector object.
     */
    Vector& operator=(Vector& vec);

    /**
     * @brief Move assignment operator for the Vector class.
     * @param vec The Vector object to be moved.
     * @return The moved Vector object.
     */
    Vector& operator=(Vector&& vec);

    /**
     * @brief Addition operator for the Vector class.
     * @param x The Vector object to be added.
     * @return The sum of the two vectors.
     */
    Vector operator+(Vector& x) { return waxpy(1.0, x); }

    /**
     * @brief Subtraction operator for the Vector class.
     * @param x The Vector object to be subtracted.
     * @return The difference of the two vectors.
     */
    Vector operator-(Vector& x) { return waxpy(-1.0, x); }

    /**
     * @brief Scalar multiplication operator for the Vector class.
     * @param alpha The constant by which to scale the vector.
     * @return The scaled vector.
     *
     */
    Vector operator*(double alpha) { return wscale(alpha); }
    friend Vector operator*(double alpha, Vector& x) { return x.wscale(alpha); }

    /**
     * @brief Dot product operator for the Vector class.
     * @param x The Vector object with which to compute the dot product.
     * @return The dot product of the two vectors.
     */
    double operator*(Vector& x) { return dot(x); }

    /**
     * @brief Scalar division operator for the Vector class.
     * @param alpha The constant by which to divide the vector.
     * @return The scaled vector.
     *
     */
    Vector operator/(double alpha) { return wscale(1.0 / alpha); }

    /**
     * @brief Additive assignment operator for the Vector class.
     * @param x The Vector object to be added.
     * @return The sum of the two vectors.
     */
    Vector& operator+=(Vector& x)
    {
        this->axpy(1.0, x);
        return *this;
    }

    /**
     * @brief Subtractive assignment operator for the Vector class.
     * @param x The Vector object to be subtracted.
     * @return The difference of the two vectors.
     */
    Vector& operator-=(Vector& x)
    {
        this->axpy(-1.0, x);
        return *this;
    }

    /**
     * @brief Scalar Multiplicative assignment operator for the Vector class.
     * @param alpha The constant by which to scale the vector.
     * @return The scaled vector.
     */
    Vector& operator*=(double alpha)
    {
        this->scale(alpha);
        return *this;
    }

    /**
     * @brief Scalar Division assignment operator for the Vector class.
     * @param alpha The constant by which to divide the vector.
     * @return The scaled vector.
     */
    Vector& operator/=(double alpha)
    {
        this->scale(1.0 / alpha);
        return *this;
    }

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
     * @brief Initializes the vector with consecutive integers.
     */
    void init_vec_consecutive();

    /**
     * @brief Initializes the vector from a file.
     * @param filename The name of the file.
     * @param QoI Flag indicating whether the vector is a quantity of interest or regular
     * observation (if applicable).
     */
    void init_vec_from_file(std::string filename, bool QoI = false);

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
            if (comm.get_world_rank() == 0)
                fprintf(stderr, "Invalid grid descriptor: %s\n", row_or_col.c_str());
            MPICHECK(MPI_Abort(comm.get_global_comm(), 1));
        }
    }

    /**
     * @brief Prints the vector.
     * @param name The name of the vector.
     */
    void print(std::string name = "");

    /**
     * @brief Computes the norm of the vector.
     * @param order The order of the norm (e.g., "2" for the 2-norm, "1" for the 1-norm, "-1" for
     * the infinity norm).
     * @param name The name of the vector. Prints the norm if name is not empty.
     * @return The norm of the vector. Only rank 0 has the global norm.
     */
    double norm(int order = 2, std::string name = "");

    /**
     * @brief Scales the vector by a constant (in-place).
     * @param alpha The constant by which to scale the vector.
     */
    void scale(double alpha);

    /**
     * @brief Scales the vector by a constant (out-of-place).
     * @param alpha The constant by which to scale the vector.
     * @return The scaled vector.
     */
    Vector wscale(double alpha);

    /**
     * @brief Computes the operation y = alpha * x + y.
     * @param alpha The constant by which to scale the vector x.
     * @param x The vector to be added.
     */

    void axpy(double alpha, Vector& x);

    /**
     * @brief Computes the operation w = alpha * x + y.
     * @param alpha The constant by which to scale the vector x.
     * @param x The vector to be added.
     */
    Vector waxpy(double alpha, Vector& x);

    /**
     * @brief Computes the operation y = alpha * x + beta * y.
     * @param alpha The constant by which to scale the vector x.
     * @param beta The constant by which to scale the vector y.
     * @param x The vector to be added.
     */
    void axpby(double alpha, double beta, Vector& x);

    /**
     * @brief Computes the operation w = alpha * x + beta * y.
     * @param alpha The constant by which to scale the vector x.
     * @param beta The constant by which to scale the vector y.
     * @param x The vector to be added.
     */
    Vector waxpby(double alpha, double beta, Vector& x);

    /**
     * @brief Computes the dot product of the vector with another vector.
     * @param x The vector with which to compute the dot product.
     * @return The dot product of the two vectors. Only rank 0 has the global dot product.
     */
    double dot(Vector& x);

    /**
     * @brief Saves the vector to a file (HDF5 format).
     * @param filename The name of the file.
     */
    void save(std::string filename);

    // /**
    //  * @brief Converts from SOTI to TOSI ordering.
    //  */
    // void to_TOSI();

    // /**
    //  * @brief Converts from TOSI to SOTI ordering.
    //  */
    // void to_SOTI();

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
     * @brief Gets the global number of blocks.
     * @return The global number of blocks.
     */
    unsigned int get_glob_num_blocks() { return glob_num_blocks; }

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
     * @brief Gets the row or column descriptor.
     * @return The row or column descriptor.
     */
    std::string get_row_or_col() { return row_or_col; }

    /**
     * @brief Checks if the vector data is initialized.
     * @return True if the vector data is initialized, false otherwise.
     */
    bool is_initialized() { return initialized; }

    /**
     * @brief Checks if the vector is using SOTI ordering.
     * @return True if the vector is using SOTI ordering, false otherwise.
     */
    bool is_SOTI_ordered() { return SOTI_ordering; }

    // Setters

    /**
     * @brief Sets the pointer to the vector data.
     * @param vec The pointer to the vector data. Must be allocated on the GPU.
     */
    void set_d_vec(double* vec);
};

#endif // __VECTOR_HPP__