#include "sdca.cpp"

double SdcaSolver::GetDeltaAlpha(const std::vector<double>& x, const int y) {
  const double a = (1 - VectorProd(x, w_) * y) / (NormSquared(x) / (lambda_ * n_));
  return a;
}

inline void SdcaSolver::ApplyAlphaUpdate(const double delta_alpha, long i) {
  a_[i] += delta_alpha;
}

// TODO: What is x_i? Vector or component? Seems like a component
inline void SdcaSolver::ApplyWeightUpdate(const double delta_alpha, long i, double x_i) {
  w_[i] += lambda_/n_ * delta_alpha * x_i;
}
