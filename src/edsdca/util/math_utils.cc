#include "edsdca/util/math_utils.h"

double NormSquared(const Eigen::VectorXd &x) {
#ifndef GPU
  const double res = NormSquared_cpu(x);
#else
  const double res = NormSquared_gpu(x);
#endif
  return res;
}

double NormSquared_cpu(const Eigen::VectorXd& x) {
  return VectorDotProd_cpu(x, x);
}

double VectorDotProd(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
#ifndef GPU
  const double res = VectorDotProd_cpu(x, y);
#else
  const double res = VectorDotProd_gpu(x, y);
#endif
  return res;
}


double VectorDotProd_cpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  return x.dot(y);
}

void VectorInPlaceSum(std::vector<double> &x, const std::vector<double> &y) {
  VectorInPlaceSum_cpu(x, y);
}

void VectorInPlaceSum_cpu(std::vector<double> &x,
                          const std::vector<double> &y) {
  long n = x.size();
  for (long i = 0; i < n; ++i) {
    x[i] += y[i];
  }
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

