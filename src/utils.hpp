/**
 * @file utils.hpp
 * @brief Contains utility functions for the project.
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include "shared.hpp"
#include "table.hpp"

/**
 * @namespace Utils
 * @brief Namespace containing utility functions.
 */
namespace Utils {
/**
 * @brief Get the host hash.
 *
 * This function returns the hash of the host name.
 *
 * @param string The host name.
 * @return The hash of the host name.
 */
uint64_t get_host_hash(const char* string);

/**
 * @brief Get the host name.
 *
 * This function gets the host name and stores it in the provided buffer.
 *
 * @param hostname The buffer to store the host name.
 * @param maxlen The maximum length of the buffer.
 */
void get_host_name(char* hostname, int maxlen);

/**
 * @brief Prints the elements of a vector.
 *
 * This function prints the elements of a vector to the console.
 *
 * @param vec Pointer to the vector.
 * @param len The length of the vector.
 * @param block_size The size of the unpadded vector.
 * @param name (Optional) The name of the vector. Defaults to "Vector".
 */
void print_vec(double* vec, int len, int block_size, std::string name = "Vector");

/**
 * @brief Prints a complex vector.
 *
 * This function prints the elements of a complex vector to the console.
 *
 * @param vec The complex vector to be printed.
 * @param len The length of the vector.
 * @param block_size The size of the unpadded vector.
 * @param name The name of the vector (optional).
 */
void print_vec_complex(Complex* vec, int len, int block_size, std::string name = "Vector");

/**
 * @brief Prints a vector using MPI.
 *
 * This function prints the elements of a vector using MPI. It takes the following parameters:
 * - `vec`: A pointer to the vector to be printed.
 * - `len`: The length of the vector.
 * - `block_size`: The size of the unpadded vector.
 * - `rank`: The rank of the current process.
 * - `world_size`: The total number of processes.
 * - `name`: (Optional) The name of the vector (default is "Vector").
 *
 * @param vec A pointer to the vector to be printed.
 * @param len The length of the vector.
 * @param block_size The size of the unpadded vector.
 * @param rank The rank of the current process.
 * @param world_size The total number of processes.
 * @param name (Optional) The name of the vector (default is "Vector").
 */
void print_vec_mpi(
    double* vec, int len, int block_size, int rank, int world_size, std::string name = "Vector");

/**
 * @brief Print the times for the different parts of the code.
 * @param reps The number of repetitions of the code.
 * @param table Flag indicating whether to print the times in a table or print raw values.
 */
void print_times(int reps = 1, bool table = true);

/**
 * @brief Make a table of timing data
 * @param col_names The names of the columns (first entry is the title).
 * @param mean The mean times.
 * @param min The minimum times.
 * @param max The maximum times.
 */
void make_table(std::vector<std::string> col_names, std::vector<long double> mean,
    std::vector<long double> min, std::vector<long double> max);

/**
 * @brief Print the raw timing data.
 * @param mean_times The mean times.
 * @param min_times The minimum times.
 * @param max_times The maximum times.
 * @param mean_times_f The mean times for the forward FFT.
 * @param min_times_f The minimum times for the forward FFT.
 * @param max_times_f The maximum times for the forward FFT.
 * @param mean_times_fs The mean times for the forward FFT in TOSI format.
 * @param min_times_fs The minimum times for the forward FFT in TOSI format.
 * @param max_times_fs The maximum times for the forward FFT in TOSI format.
 * @param times_len The number of timing segments.
 */
void print_raw(long double* mean_times, long double* min_times, long double* max_times,
    long double* mean_times_f, long double* min_times_f, long double* max_times_f,
    long double* mean_times_fs, long double* min_times_fs, long double* max_times_fs,
    int times_len);

/**
 * @brief Computes the 3D swapaxes operation on the matrix blocks.
 * @param d_in Pointer to the input matrix.
 * @param d_out Pointer to the output matrix.
 * @param num_cols The number of block columns in the matrix.
 * @param num_rows The number of block rows in the matrix.
 * @param padded_size The size of each block. Note: this is used inside the matvec setup function,
 * so it is called with Nt + 1 and not 2 * Nt.
 * @param s The CUDA stream to use for the operation (optional).
 */
void swap_axes(Complex* d_in, Complex* d_out, int num_cols, int num_rows, int padded_size,
    cudaStream_t s = nullptr);

/**
 * @brief Check if the collective I/O is working.
 * @param xfer_props The data transfer properties.
 */
void check_collective_io(const HighFive::DataTransferProps& xfer_props);

/**
 * @brief Get the starting index for a given color (row or col).
 * @param glob_num_blocks The global number of blocks (distributed across the communicator).
 * @param color The color of the process in the communicator.
 * @param comm_size The number of processes in the communicator.
 *
 */
size_t get_start_index(int glob_num_blocks, int color, int comm_size);

/**
 * @brief Get the global size of a vector.
 * @param local_size The local size of the vector.
 * @param color The color of the process in the communicator.
 * @param comm_size The number of processes in the communicator.
 * @return The global size of the vector.
 */
int global_to_local_size(int global_size, int color, int comm_size);

/**
 * @brief Get the global size of a vector.
 * @param local_size The local size of the vector.
 * @param color The color of the process in the communicator.
 * @param comm_size The number of processes in the communicator.
 * @return The global size of the vector (just local_size * comm_size).
 */
int local_to_global_size(int local_size, int comm_size);

} // namespace Utils

#endif // __UTILS_H__
