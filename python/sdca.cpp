#include "sdca.cpp"

double SdcaSolver::GetDeltaAlpha(const std::vector<double>& x, const int y) {
  const double a = (1 - VectorProd(x, w_) * y) / (NormSquared(x) / (lambda_ * n_));
  return a;
}

