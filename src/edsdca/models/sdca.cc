#include "edsdca/models/sdca.h"

namespace edsdca {
namespace models {

double Sdca::DeltaAlpha(const Eigen::VectorXd& x, const double y, long i) {
  const double a = (1.0 - VectorDotProd(x, w_) * y) / (NormSquared(x) / (lambda_ * n_)) +
                      a_(i) * y;
  return y * std::max(0.0, std::min(1.0, a)) - a_(i);
}


void Sdca::ComputeAlphaBar(SdcaUpdateType update_type) {
  switch (update_type) {
    case SdcaUpdateType::Average:
      Eigen::VectorXd reduced_accumulated_a = VectorReduce(accumulated_a_);
      reduced_accumulated_a *= 1.0/accumulated_a_.size();
      accumulated_a_.clear();
  }
}

void Sdca::ComputeWBar(SdcaUpdateType update_type) {
  switch (update_type) {
    case SdcaUpdateType::Average:
      Eigen::VectorXd reduced_accumulated_w = VectorReduce(accumulated_w_);
      reduced_accumulated_w *= 1.0/accumulated_w_.size();
      accumulated_w_.clear();
  }
}

void Sdca::Fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
  // Set model proberties
  n_ = X.rows();
  d_ = X.cols();

  // Initialize w_ and a_
  InitializeAlpha();
  InitializeWeights(); 

  for (long cur_epoch = 0; cur_epoch < max_epochs_; ++cur_epoch) {
    auto remaining_batch_indices = std::vector<long>(n_);

    // Calculate number of mini-batches
    const long num_batches = (long) n_ / batch_size_;
    for (long batch_num = 0; batch_num < num_batches; ++batch_num) {
      // Get mini-batch
      const std::vector<long> mb_indices = GenerateMiniBatchIndexVector(batch_size_, 0, d_);
      std::vector<Eigen::VectorXd> mb_X(batch_size_);
      std::vector<double> mb_y(batch_size_);
      for (long i = 0; i < batch_size_; ++i) {
        mb_X[i] = X.row(i);
        mb_y[i] = y(i);
      }
      RunUpdateOnMiniBatch(mb_X, mb_y);
      ComputeAlphaBar();
      ComputeWBar();
    }
  }
}

double Sdca::Predict(const Eigen::VectorXd& x) {
  double res = VectorDotProd(w_, x);
  if (res > 0) {
    return 1.0;
  } else {
    return 0.0;
  }
}

void Sdca::RunUpdateOnMiniBatch(const std::vector<Eigen::VectorXd>& X,
    const std::vector<double>& y) {
#ifndef GPU
  Sdca::RunUpdateOnMiniBatch_cpu(X, y);
#else
  Sdca::RunUpdateOnMiniBatch_gpu(X, y);
#endif
}

void Sdca::RunUpdateOnMiniBatch_cpu(const std::vector<Eigen::VectorXd>& X,
    const std::vector<double>& y) {
  const long batch_size = y.size();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, d_ - 1);

  for (long i = 0; i < batch_size; ++i) {
    const long current_dim = dist(gen);
    const Eigen::VectorXd x_i = X[i];
    const double y_i = y[i];
    const double delta_a = DeltaAlpha(x_i, y_i, current_dim);
    ApplyAlphaUpdate(delta_a, current_dim);
    ApplyWeightUpdates(delta_a, x_i);
  } 
}

// TODO: complete the gpu mini-batch update
void Sdca::RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd>& X,
    const std::vector<double>& y) {
  std::cout << "NEED TO IMPLEMENT\n";
}

std::vector<long> Sdca::GenerateMiniBatchIndexVector(const long size, const long low, 
    const long high) const {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(low, high - 1);
  std::vector<long> res(size);
  for (long i = 0; i < size; ++i) {
    res[i] = dist(gen);
  }
  return res;
}

void Sdca::InitializeAlpha() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> norm_dist(0, 1);
  a_ = Eigen::VectorXd(d_);
  for (long i = 0; i < d_; ++i) {
    a_(i) = norm_dist(gen);
  }
}

void Sdca::InitializeWeights() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> norm_dist(0, 1);
  w_ = Eigen::VectorXd(d_);
  for (long i = 0; i < d_; ++i) {
    w_(i) = norm_dist(gen);
  }
}

} // namespace model
} // namespace edsdca
