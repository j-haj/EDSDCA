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

void VectorInPlaceSum(std::vector<double>& x,
    std::vector<double>& y);
void VectorInPlaceSum_cpu(std::vector<double>& x,
    std::vector<double>& y);
void VectorInPlaceSum_gpu(std::vector<double>& x,
    std::vector<double>& y);

std::vector<double> VectorReduce(const std::vector<std::vector<double>>& v);
std::vector<double> VectorReduce_cpu(const std::vector<std::vector<double>>& v);
std::vector<double> VectorReduce_gpu(const std::vector<std::vector<double>>& v);

#endif // __MATH_UTILS_HPP
