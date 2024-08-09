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
uint64_t getHostHash(const char* string);

/**
 * @brief Get the host name.
 *
 * This function gets the host name and stores it in the provided buffer.
 *
 * @param hostname The buffer to store the host name.
 * @param maxlen The maximum length of the buffer.
 */
void getHostName(char* hostname, int maxlen);

/**
 * @brief Pads each block of a vector to twice the length with zeros.
 *
 * This function takes an input vector `d_in` and pads each block of the vector to twice the length
 * with zeros. The padded vector is stored in the output vector `d_pad`. The number of columns in
 * each block is specified by `num_cols`. The total size of the vector is specified by `size`.
 * The padding operation is performed asynchronously on the CUDA stream `s`.
 *
 * @param d_in      Pointer to the input vector.
 * @param d_pad     Pointer to the output padded vector.
 * @param num_cols  Number of columns in each block.
 * @param size      Total size of the vector.
 * @param s         CUDA stream for asynchronous execution.
 */
void PadVector(const double* const d_in, double* const d_pad, const unsigned int num_cols,
    const unsigned int size, cudaStream_t s);

/**
 * @brief Unpads or repads a vector.
 *
 * This function either unpads each block of the vector back to the original length or resets the
 * second half of each block to zeros.
 *
 * @param d_in Pointer to the input vector.
 * @param d_out Pointer to the output vector.
 * @param num_cols The number of columns in the vector.
 * @param size The size of the vector.
 * @param unpad Flag indicating whether to unpad or repad the vector. If true, the vector will be
 * unpadded. If false, the second half of each block will be reset to zeros.
 * @param s The CUDA stream to use for the operation.
 */
void UnpadRepadVector(const double* const d_in, double* const d_out, const unsigned int num_cols,
    const unsigned int size, const bool unpad, cudaStream_t s);

/**
 * @brief Prints the elements of a vector.
 *
 * This function prints the elements of a vector to the console.
 *
 * @param vec Pointer to the vector.
 * @param len The length of the vector.
 * @param unpad_size The size of the unpadded vector.
 * @param name (Optional) The name of the vector. Defaults to "Vector".
 */
void printVec(double* vec, int len, int unpad_size, std::string name = "Vector");

/**
 * @brief Prints a complex vector.
 *
 * This function prints the elements of a complex vector to the console.
 *
 * @param vec The complex vector to be printed.
 * @param len The length of the vector.
 * @param unpad_size The size of the unpadded vector.
 * @param name The name of the vector (optional).
 */
void printVecComplex(Complex* vec, int len, int unpad_size, std::string name = "Vector");

/**
 * @brief Prints a vector using MPI.
 *
 * This function prints the elements of a vector using MPI. It takes the following parameters:
 * - `vec`: A pointer to the vector to be printed.
 * - `len`: The length of the vector.
 * - `unpad_size`: The size of the unpadded vector.
 * - `rank`: The rank of the current process.
 * - `world_size`: The total number of processes.
 * - `name`: (Optional) The name of the vector (default is "Vector").
 *
 * @param vec A pointer to the vector to be printed.
 * @param len The length of the vector.
 * @param unpad_size The size of the unpadded vector.
 * @param rank The rank of the current process.
 * @param world_size The total number of processes.
 * @param name (Optional) The name of the vector (default is "Vector").
 */
void printVecMPI(
    double* vec, int len, int unpad_size, int rank, int world_size, std::string name = "Vector");

/**
 * @brief Print the times for the different parts of the code.
 * @param reps The number of repetitions of the code.
 * @param table Flag indicating whether to print the times in a table or print raw values.
 */
void printTimes(int reps = 1, bool table = true);

/**
 * @brief Make a table of timing data
 * @param col_names The names of the columns (first entry is the title).
 * @param mean The mean times.
 * @param min The minimum times.
 * @param max The maximum times.
 */
void makeTable(std::vector<std::string> col_names, std::vector<long double> mean,
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
 * @param world_size The total number of processes.
 * @param times_len The number of timing segments.
 */
void printRaw(long double* mean_times, long double* min_times, long double* max_times,
    long double* mean_times_f, long double* min_times_f, long double* max_times_f,
    long double* mean_times_fs, long double* min_times_fs, long double* max_times_fs,
    int world_size, int times_len);

}

#endif