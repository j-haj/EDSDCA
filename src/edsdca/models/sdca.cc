#include "edsdca/solvers/sdca.h"

namespace edsdca {
namespace model {

double Sdca::DeltaAlpha(const Eigen::VectorXd& x, const y, long i) {
  const double a = (1 - VectorProd(x, w_) * y) / (NormSquared(x) / (lambda_ * n_));
  return y * std::max(0, std::min(1, a));
}


void Sdca::ComputeAlphaBar(SdcaUpdateType update_type = SdcaUpdateType::Average) {
  switch (update_type) {
    case SdcaUpdateType::Average:
      double accumulated_scale_factor = accumulated_a_.size() * lambda_;
      Eigen::VectorXd reduced_accumulated_a = VectorReduce(accumulated_a_);
      // TODO: Not complete
      accumulated_a_.clear();
  }
}

void Sdca::ComputeWBar(SdcaUpdateType update_type = SdcaUpdateType::Average) {
  switch (update_type) {
    case SdcaUpdateType::Average:
      Eigen::VectorXd reduced_accumulated_w = VectorReduce(accumulated_w_);
      reduced_accumulated_w /= accumulated_w_.size();

      accumulated_w_.clear();
      // TODO: Not complete
  }
}

void Fit(const Eigen::MatriXd& X, const Eigen::VectorXd& y) {
  // Get mini_batch
}

} // namespace model
} // namespace edsdca
