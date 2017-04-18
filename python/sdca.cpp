#include "sdca.cpp"

double SdcaSolver::GetDeltaAlpha(const std::vector<double>& x, const int y) {
  const double a = (1 - VectorProd(x, w_) * y) / (NormSquared(x) / (lambda_ * n_));
  return a;
}

inline void SdcaSolver::ApplyAlphaUpdate(const double delta_alpha, long i) {
  a_[i] += delta_alpha;
  accumulated_a_.push_back(a_);
}

// TODO: What is x_i? Vector or component? Seems like a component
inline void SdcaSolver::ApplyWeightUpdate(const double delta_alpha, long i, double x_i) {
  w_[i] += lambda_/n_ * delta_alpha * x_i;
  accumulated_w_.push_back(w_);
}

void SdcaSolver::ComputeAlphaBar(SdcaUpdateType update_type = SdcaUpdateType::Average) {
  double accumulated_scale_factor = accumulated_a_.size();
  std::vector<double> recuced_accumulated_a = VectorReduce(accumulated_a_);
  VectorInPlaceSum(a_, reduced_accumulated_a);
  for (auto& x : a_) {
   x /= accumulated_scale_factor;
  } 
}
