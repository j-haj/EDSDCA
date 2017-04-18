#include "math_utils.hpp"

double NormSquared_cpu(const std::vector<double>& x) {
  const double res = VectorProd_cpu(x, x);
  return res;
}

double NormSquared_gpu(const std::vector<double>& x) {
  // Call CUDA kernel for NormSquared here
  double res = 0;
  return res;  
}

double NormSquared(const std::vector<double>& x) {
#ifndef GPU
  const double res = NormSquared_cpu(x);
#else
  const double res = NormSquared_gpu(x);
#endif
  return res;
}

double VectorProd(const std::vector<double>& x,
  const std::vector<double>& y) {
#ifndef GPU
  const double res = VectorProd_cpu(x, y);
#else
  const double res = VectorProd_gpu(x, y);
#endif
    return res;
}

double VectorProd_cpu(const std::vector<double>& x,
  const std::vector<double>& y) {
    long n = x.size();
    double res = 0;
    for (long i = 0; i < n; ++i) {
      res += x[i] * y[i];
    }
    return res;
}

double VectorProd_cpu(const std::vector<double>& x,
  const std::vector<double>& y) {
    // Call gpu prod

    // Reduce result
    double res = 0;
    return res;
}

