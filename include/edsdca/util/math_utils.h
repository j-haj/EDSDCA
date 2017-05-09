#ifndef __MATH_UTILS_HPP
#define __MATH_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "edsdca/memory/memsync.h"

// Returns a vector that is the result of a matrix-vector multiplication
Eigen::VectorXd MatrixVectorMultiply(const std::vector<Eigen::VectorXd> &X, const Eigen::VectorXd &y);
Eigen::VectorXd MatrixVectorMultiply_cpu(const std::vector<Eigen::VectorXd> &X,
                                         const Eigen::VectorXd &y);
Eigen::VectorXd MatrixVectorMultiply_gpu(const std::vector<Eigen::VectorXd> &X,
                                         const Eigen::VectorXd &y);


// Returns a normalized vector for the given input
Eigen::VectorXd NormalizeVector(const Eigen::VectorXd &x);

// Norm for Eigen vector
double NormSquared_cpu(const Eigen::VectorXd &x);
double NormSquared(const Eigen::VectorXd &x);

// Vector dot product for Eigen vectors
double VectorDotProd(const Eigen::VectorXd &x, const Eigen::VectorXd &y);
double VectorDotProd_cpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

void VectorInPlaceSum(std::vector<double> &x, const std::vector<double> &y);
void VectorInPlaceSum_cpu(std::vector<double> &x, const std::vector<double> &y);

Eigen::VectorXd VectorReduce(const std::vector<Eigen::VectorXd> &v);
Eigen::VectorXd VectorReduce_cpu(const std::vector<Eigen::VectorXd> &v);

// GPU functions

#ifdef GPU
double NormSquared_gpu(const Eigen::VectorXd &x);
double VectorDotProd_gpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y);
Eigen::VectorXd VectorReduce_gpu(const std::vector<Eigen::VectorXd> &v);
#endif

#endif // __MATH_UTILS_HPP
