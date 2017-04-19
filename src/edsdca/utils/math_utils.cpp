#include "math_utils.hpp"

double NormSquared_cpu(const std::vector<double>& x) {
  const double res = VectorProd_cpu(x, x);
  return res;
}

double NormSquared_cpu(const Eigen::VectorXd& x) {
  return x.squaredNorm();
}

// TODO: add cuda call
double NormSquared_gpu(const std::vector<double>& x) {
  // Call CUDA kernel for NormSquared here
  double res = 0;
  return res;  
}

// TODO: add CUDA call
double NormSquared_gpu(const Eigen::VectorXd& x) {
  // Call CUDA kernel
}

double NormSquared(const std::vector<double>& x) {
#ifndef GPU
  const double res = NormSquared_cpu(x);
#else
  const double res = NormSquared_gpu(x);
#endif
  return res;
}

double NormSquared(const Eigen::VectorXd<double>& x) {
#ifndef GPU
  const double res = NormSquared_cpu(x);
#else
  const double res = NormSquared_gpu(x);
#endif
  return res;
}

double VectorDotProd(const std::vector<double>& x,
  const std::vector<double>& y) {
#ifndef GPU
  const double res = VectorProd_cpu(x, y);
#else
  const double res = VectorProd_gpu(x, y);
#endif
    return res;
}

double VectorDotProd(const Eigen::VectorXd& x,
  const Eigen::VectorXd& y) {
#ifndef GPU
  const double res = VectorProd_cpu(x, y);
#else
  const double res = VectorProd_gpu(x, y);
#endif
    return res;
}

double VectorDotProd_cpu(const std::vector<double>& x,
  const std::vector<double>& y) {
    long n = x.size();
    double res = 0;
    for (long i = 0; i < n; ++i) {
      res += x[i] * y[i];
    }
    return res;
}

double VectorDotProd_cpu(const Eigen::VectorXd& x,
    const Eigen::VectorXd& y) {
  return x.dot(y);
}

// TODO: add CUDA call
double VectorDotProd_gpu(const std::vector<double>& x,
  const std::vector<double>& y) {
    // Call gpu prod

    // Reduce result
    double res = 0;
    return res;
}

// TODO: add CUDA call
double VectorDotProd(const Eigen::VectorXd& x,
    const Eigen::VectorXd& y) {
  // Call GPU kernel here
}

void VectorInPlaceSum(std::vector<double>& x, std::vector<double>& y) {
#ifndef GPU
  VectorInPlaceSum_cpu(x, y);
#else
  VectorInPlaceSum_gpu(x, y);
#endif
}

void VectorInPlaceSum_cpu(std::vector<double>& x, std::vector<double>& y) {
  long n = x.size();
  for (long i = 0; i < n; ++i) {
    x[i] += y[i];
  }
}

// TODO: add CUDA call
void VectorInPlaceSum_gpu(std::vector<double>& x, std::vector<double>& y) {
  // GPU implementation
}

std::vector<double> VectorReduce(const std::vector<std::vector<double>>& v) {
#ifndef GPU
  return VectorReduce_cpu(const std::vector<std::vector<double>>& v);
#else
  return VectorReduce_gpu(const std::vector<std::vector<double>>& v);
#endif
}

std::vector<double> VectorReduce_cpu(const std::vector<std::vector<double>>& v) {
  std::vector<double> accumulator(v.first().size());
  for (const auto& x : v) {
    VectorInPlaceSum_cpu(accumulator, x);
  }
  return accumulator;
}

// TODO: add CUDA call
std::vector<double> VectorReduce_gpu(const std::vector<std::vector<double>>& v) {
  // implement GPU version
  std::vector<double> accumulator(v.first().size());
  return accumulator;
}

Eigen::VectorXd VectorReduce(const Eigen::MatrixXd& m) {

}

Eigen::VectorXd VectorReduce_cpu(const Eigen::MatrixXd& m) {

}

Eigen::VectorXd VectorReduce_gpu(const Eigen::MatrixXd& m) {

}



