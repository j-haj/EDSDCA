#ifndef __EDSDCA_MEMORY_HPP
#define __EDSDCA_MEMORY_HPP

#include <Eigen/Dense>

namespace edsdca {
namespace memory {

/**
 * The @p MemoryConversion class is used to convert Eigen data structures as
 * well as std::vector to and from raw C arrays via pointers. This is used
 * primarily when loading data to and off the GPU.
 */

/**
 * Returns a pointer to a C array of doubles. Note that the C array is in column
 * major order (as in Fortran)
 *
 * @param m Eigen::MatrixXd matrix
 *
 * @return C pointer to array of doubles in column-major order
 */
double *ConvertEigenMatrixToPtr(Eigen::MatrixXd &m);

/**
 * Returns a pointer to a C array of doubles
 *
 * @param Eigen::VectorXd of double to be converted to C array
 *
 * @return C array of doubles containing the data of @p v
 */
double *ConvertEigenVectorToPtr(Eigen::VectorXd &v);

/**
 * Returns an Eigen::MatrixXd object with the given data
 *
 * @param c C array of the data
 * @param row number of rows
 * @param col number of columns
 *
 * @return @p Eigen::MatrixXd array
 */
Eigen::MatrixXd ConvertPtrToEigenMatrix(double *c, long row, long col);

/**
 * Returns an @p Eigen::VectorXd for the given data @p c
 *
 * @param c data to be converted to @p Eigen::VectorXd
 * @param row number of rows in the vector
 *
 * @return @p Eigen::VectorXd of the given data @p c
 */
Eigen::VectorXd ConvertPtrToEigenVector(double *c, long row);

} // namespace memory
} // namespace edsdca

#endif // __EDSDCA_MEMORY_HPP
