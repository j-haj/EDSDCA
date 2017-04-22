#include "edsdca/models/sdca.h"

namespace edsdca {
namespace models {

double Sdca::DeltaAlpha(const Eigen::VectorXd& x, const double y, long i) {
  const double a = (1.0 - VectorDotProd(x, w_) * y) / (NormSquared(x) / (lambda_ * n_));
  return y * std::max(0.0, std::min(1.0, a));
}


void Sdca::ComputeAlphaBar(SdcaUpdateType update_type) {
  switch (update_type) {
    case SdcaUpdateType::Average:
      double accumulated_scale_factor = accumulated_a_.size() * lambda_;
      Eigen::VectorXd reduced_accumulated_a = VectorReduce(accumulated_a_);
      // TODO: Not complete
      accumulated_a_.clear();
  }
}

void Sdca::ComputeWBar(SdcaUpdateType update_type) {
  switch (update_type) {
    case SdcaUpdateType::Average:
      Eigen::VectorXd reduced_accumulated_w = VectorReduce(accumulated_w_);
      reduced_accumulated_w /= accumulated_w_.size();

      accumulated_w_.clear();
      // TODO: Not complete
  }
}

void Sdca::Fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
  // Set model probertiem
  n_ = X.rows();
  d_ = X.cols();

  // Get mini_batch

  for (long cur_epoch = 0; cur_epoch < max_epochs_; ++cur_epoch) {
    auto remaining_batch_indices = std::vector<long>(n_)

    // Calculate number of mini-batches
    long num_batches = (long) n_ / batch_size_;
    std::cout << "num_batches: " << num_batches
              << "\nbatch_size_: " << batch_size_ << "\nn_: " << n_ << std::endl;
  }
}

void Sdca::RunUpdateOnMiniBatch(std::vector<Eigen::VectorXd>& X,
    std::vector<double>& y) {
#ifndef GPU
  Sdca::RunUpdateOnMiniBatch_cpu(X, y);
#else
  Sdca::RunUpdateOnMiniBatch_gpu(X, y);
#endif
}

void Sdca::RunUpdateOnMiniBatch_cpu(std::vector<Eigen::VectorXd>& X,
    std::vector<double>& y) {
  // 
}

// TODO: complete the gpu mini-batch update
void Sdca::RunUpdateOnMiniBatch_gpu(std::vector<Eigen::VectorXd>& X,
    std::vector<double>& y) {
  std::cout << "NEED TO IMPLEMENT\n";
}

} // namespace model
} // namespace edsdca
