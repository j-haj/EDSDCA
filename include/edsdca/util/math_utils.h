#ifndef __MATH_UTILS_HPP
#define __MATH_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "edsdca/memory/memsync.h"

// Norm for std::vector
double NormSquared_cpu(const std::vector<double> &x);
double NormSquared_gpu(const std::vector<double> &x);
double NormSquared(const std::vector<double> &x);

// Norm for Eigen vector
double NormSquared_cpu(const Eigen::VectorXd &x);
double NormSquared_gpu(const Eigen::VectorXd &x);
double NormSquared(const Eigen::VectorXd &x);

// Vector dot product for std::vector vectors
double VectorDotProd(const std::vector<double> &x,
                     const std::vector<double> &y);
double VectorDotProd_cpu(const std::vector<double> &x,
                         const std::vector<double> &y);
double VectorDotProd_gpu(const std::vector<double> &x,
                         const std::vector<double> &y);

// Vector dot product for Eigen vectors
double VectorDotProd(const Eigen::VectorXd &x, const Eigen::VectorXd &y);
double VectorDotProd_cpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y);
double VectorDotProd_gpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

void VectorInPlaceSum(std::vector<double> &x, const std::vector<double> &y);
void VectorInPlaceSum_cpu(std::vector<double> &x, const std::vector<double> &y);
void VectorInPlaceSum_gpu(std::vector<double> &x, const std::vector<double> &y);

std::vector<double> VectorReduce(const std::vector<std::vector<double>> &v);
std::vector<double> VectorReduce_cpu(const std::vector<std::vector<double>> &v);
std::vector<double> VectorReduce_gpu(const std::vector<std::vector<double>> &v);

Eigen::VectorXd VectorReduce(const std::vector<Eigen::VectorXd> &v);
Eigen::VectorXd VectorReduce_cpu(const std::vector<Eigen::VectorXd> &v);
Eigen::VectorXd VecotrReduce_gpu(const std::vector<Eigen::VectorXd> &v);

#endif // __MATH_UTILS_HPP
