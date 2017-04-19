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

double* ConvertEigenMatrixToPtr(Eigen::MatrixXd& m);
double* ConvertEigenVectorToPtr(Eigen::VectorXd& v);

Eigen::MatrixXd ConvertPtrToEigenMatrix(double* c);
Eigen::VectorXd ConvertPtrToEigenVector(double* c);

} // namespace memory
} // namespace edsdca

#endif // __EDSDCA_MEMORY_HPP
