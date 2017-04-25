#include "edsdca/util/math_utils.h"

double NormSquared_cpu(const std::vector<double> &x) {
  const double res = VectorDotProd_cpu(x, x);
  return res;
}

double NormSquared_cpu(const Eigen::VectorXd &x) { return x.squaredNorm(); }

double NormSquared(const std::vector<double> &x) {
#ifndef GPU
  const double res = NormSquared_cpu(x);
#else
  const double res = NormSquared_gpu(x);
#endif
  return res;
}

double NormSquared(const Eigen::VectorXd &x) {
#ifndef GPU
  const double res = NormSquared_cpu(x);
#else
  const double res = NormSquared_gpu(x);
#endif
  return res;
}

double VectorDotProd(const std::vector<double> &x,
                     const std::vector<double> &y) {
#ifndef GPU
  const double res = VectorDotProd_cpu(x, y);
#else
  const double res = VectorDotProd_gpu(x, y);
#endif
  return res;
}

double VectorDotProd(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
#ifndef GPU
  const double res = VectorDotProd_cpu(x, y);
#else
  const double res = VectorDotProd_gpu(x, y);
#endif
  return res;
}

double VectorDotProd_cpu(const std::vector<double> &x,
                         const std::vector<double> &y) {
  long n = x.size();
  double res = 0;
  for (long i = 0; i < n; ++i) {
    res += x[i] * y[i];
  }
  return res;
}

double VectorDotProd_cpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  return x.dot(y);
}

void VectorInPlaceSum(std::vector<double> &x, const std::vector<double> &y) {
#ifndef GPU
  VectorInPlaceSum_cpu(x, y);
#else
  VectorInPlaceSum_gpu(x, y);
#endif
}

void VectorInPlaceSum_cpu(std::vector<double> &x,
                          const std::vector<double> &y) {
  long n = x.size();
  for (long i = 0; i < n; ++i) {
    x[i] += y[i];
  }
}

std::vector<double> VectorReduce(const std::vector<std::vector<double>> &v) {
#ifndef GPU
  return VectorReduce_cpu(v);
#else
  return VectorReduce_gpu(v);
#endif
}

std::vector<double>
VectorReduce_cpu(const std::vector<std::vector<double>> &v) {
  std::vector<double> accumulator(v.front().size());
  for (const auto &x : v) {
    VectorInPlaceSum_cpu(accumulator, x);
  }
  return accumulator;
}

// TODO: add CUDA call
std::vector<double>
VectorReduce_gpu(const std::vector<std::vector<double>> &v) {
  // implement GPU version
  std::vector<double> accumulator(v.front().size());
  return accumulator;
}

Eigen::VectorXd VectorReduce(const std::vector<Eigen::VectorXd> &v) {
#ifndef GPU
  return VectorReduce_cpu(v);
#else
  return VectorReduce_gpu(v);
#endif
}

Eigen::VectorXd VectorReduce_cpu(const std::vector<Eigen::VectorXd> &v) {
  Eigen::VectorXd accumulator = Eigen::VectorXd::Zero(v.front().size());
  for (const Eigen::VectorXd &x : v) {
    accumulator += x;
  }
  return accumulator;
}
