#ifndef __SDCA_H
#define __SDCA_H

#include <vector>
#include <cmath>

// =============================================================================
//
// Helper Functions
//
// =============================================================================

double NormSquared_cpu(const std::vector<double>& x) {
  double res = 0;
  for (const double& x_i : x) {
    res += x_i * x_i;
  }
  return x_i;
}

double NormSquared_gpu(const std::vector<double>& x) {
  // Call CUDA kernel for NormSquared here
  double res = 0;
  return res;  
}

double VectorProd(const std::vector<double>& x,
  const std::vector<double>& y) {
#ifndef GPU
    long n = x.size();
    double res = 0.0;
    for (long i = 0; i < n; ++i) {
      res += x[i]*y[i];
    }
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
    double res = 0;
    return res;
  }

// =============================================================================
//
// SDCA SOLVER
//
// =============================================================================
/**
 * SDCA solver for hinge loss
 */
class SdcaSolver {
public:
   GetDeltaAlpha(const std::vector<double>& x, const int y) {
     const double a = (1 - VectorProd(x, w_)*y) / (NormSquared(x) / (lambda_ * n_);
  }
  virtual GetDelta
private:
  /// @brief Dual vector, alpha
  std::vector<double> a_;

  /// @brief Primal vector, omega
  std::vector<double> w_;

};

#endif // __SDCA_H