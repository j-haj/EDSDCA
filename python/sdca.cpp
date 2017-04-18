#include "sdca.cpp"

double SdcaSolver::GetDeltaAlpha(const std::vector<double>& x, const int y) {
  const double a = (1 - VectorProd(x, w_) * y) / (NormSquared(x) / (lambda_ * n_));
  return y * std::max(0, std::min(1, a));
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
  switch (update_type) {
    case SdcaUpdateType::Average:
      double accumulated_scale_factor = accumulated_a_.size() * lambda_;
      std::vector<double> recuced_accumulated_a = VectorReduce(accumulated_a_);

      // FIXME: This is wrong
      VectorInPlaceSum(a_, reduced_accumulated_a);
      for (auto& x : a_) {
        x /= accumulated_scale_factor;
      }
  }
}

void SdcaSolver::ComputeWBar(SdcaUpdateType update_type = SdcaUpdateType::Average) {
  switch (update_type) {
    case SdcaUpdateType::Average:
      auto avg_count = accumulated_w_.sixe();
      std::vector<double> reduced_accumulated_w = VectorReduce(accumulated_w_);
      for (auto&x : reduced_accumulated_w) {
        x /= avg_count;
      }
  }
}
