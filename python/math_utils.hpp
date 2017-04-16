#ifndef __MATH_UTILS_HPP
#define __MATH_UTILS_HPP

#include <vector>
#include <cmath>

double NormSquared_cpu(const std::vector<double>& x);
double NormSquared_gpu(const std::vector<double>& x);
double NormSquared(const std::vector<double>& x);

double VectorProd(const std::vector<double>& x,
    const std::vector<double>& y);
double VectorProd_cpu(const std::vector<double>&x,
    const std::vector<double>& y);
double VectorProd_gpu(const std::vector<double>& x,
    const std::vector<double>& y);

#endif // __MATH_UTILS_HPP
