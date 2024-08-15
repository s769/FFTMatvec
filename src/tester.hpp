/**
 * @file tester.hpp
 * @brief Header file for the Tester namespace.
 * 
 * This file contains the declaration of the Tester namespace, which provides functions for testing matrix-vector multiplication.
 */
#ifndef __TESTER_HPP__
#define __TESTER_HPP__

#include "Comm.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"

/**
 * @namespace Tester
 * @brief Namespace containing functions for testing matrix-vector multiplication.
 */
namespace Tester {

/**
 * @brief Checks the results of a matrix-vector multiplication with a ones matrix and ones vectors.
 *
 * This function takes a communication object, a matrix, a vector, and two boolean flags as input.
 * It performs a matrix-vector multiplication with a ones matrix and ones vectors and checks the results.
 * The boolean flag `conj` indicates whether the matrix should be conjugated during the multiplication.
 * The boolean flag `full` indicates whether the full matrix should be used for the multiplication.
 *
 * @param comm The communication object.
 * @param mat The matrix for the multiplication.
 * @param out The output vector.
 * @param conj Flag indicating whether the matrix should be conjugated.
 * @param full Flag indicating whether the full matrix should be used (F*F/FF* vs F/F*).
 */
void check_ones_matvec(Comm& comm, Matrix& mat, Vector& out, bool conj, bool full);

} // namespace Tester


#endif // __TESTER_HPP__