#include "edsdca/models/sdca.h"

namespace edsdca {
namespace models {

double Sdca::DeltaAlpha(const Eigen::VectorXd &x, const double y, long i) {
  Eigen::VectorXd tmp_w = accumulated_w_.size() == 0 ? w_ : accumulated_w_.back();
  const double a =
      (1.0 - VectorDotProd(x, tmp_w) * y) / (NormSquared(x) / (lambda_ * n_)) +
      a_(i) * y;
  return y * std::max(0.0, std::min(1.0, a)) - a_(i);
}

void Sdca::ComputeAlphaBar(SdcaUpdateType update_type) {
  switch (update_type) {
  case SdcaUpdateType::Average:
    Eigen::VectorXd reduced_accumulated_a = VectorReduce(accumulated_a_);
    reduced_accumulated_a *= 1.0 / accumulated_a_.size();
    accumulated_a_.clear();
  }
}

void Sdca::ComputeWBar(SdcaUpdateType update_type) {
  switch (update_type) {
  case SdcaUpdateType::Average:
    Eigen::VectorXd reduced_accumulated_w = VectorReduce(accumulated_w_);
    reduced_accumulated_w *= 1.0 / accumulated_w_.size();
    accumulated_w_.clear();
  }
}

void Sdca::ComputeW(const Eigen::MatrixXd &X) {
  w_ = X.transpose() * a_;
  w_ *= (1.0 / (lambda_ * n_));
}

void Sdca::Fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
  // Set model proberties
  n_ = X.rows();
  d_ = X.cols();

  // Initialize w_ and a_
  InitializeAlpha();
  InitializeWeights();

  // Calculate number of mini-batches
  const long num_batches = long(n_ / batch_size_);
  std::cout << "Running " << num_batches << " number of batches\n";
  // training_hist_ tracks the training error and training time (s)
  training_hist_ =
      std::vector<std::pair<double, double>>(num_batches * max_epochs_);
  long log_index = 0;

  // Computes $\omega$ based on current $\alpha$
  ComputeW(X);
  for (long cur_epoch = 0; cur_epoch < max_epochs_; ++cur_epoch) {

    // Currently not used
    auto remaining_batch_indices = std::vector<long>(n_);
    for (long batch_num = 0; batch_num < num_batches; ++batch_num) {

      // Get mini-batch by generating random indices
      const std::vector<long> mb_indices =
          GenerateMiniBatchIndexVector(batch_size_, 0, n_);
      std::vector<Eigen::VectorXd> mb_X(batch_size_);
      ComputeW(X);
      std::vector<double> mb_y(batch_size_);
      for (long i = 0; i < batch_size_; ++i) {
        mb_X[i] = X.row(mb_indices[i]);
        mb_y[i] = y(mb_indices[i]);
      }

      // Start timer
      timer_.Start();

      // Call the mini-batch update algorithm
      RunUpdateOnMiniBatch(mb_X, mb_y, mb_indices);
      if (batch_num * batch_size_ == update_interval_) {
        ComputeAlphaBar();
        ComputeWBar();
      }

      // Stop timer and get cumulative time
      timer_.Stop();
      double cumulative_time = timer_.cumulative_time();

      // Compute training loss with current weights
      double loss = ComputeLoss(X, y);

      // Store loss and runtime
      auto tmp_pair = std::make_pair(cumulative_time, loss);
      training_hist_[log_index] = tmp_pair;
      ++log_index;
    }
  }
  SaveHistory("results_test.csv");
  for (int i = 0; i < d_; ++i) {
    std::cout << w_[i] << " ";
  }
  std::cout << "\n";
}

double Sdca::Predict(const Eigen::VectorXd &x) {
  double res = VectorDotProd(w_, x);
  if (res > 0) {
    return 1.0;
  } else {
    return -1.0;
  }
}

double Sdca::ComputeLoss(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
  double aggregate_loss(0.0);
  
  Eigen::VectorXd Xw = X * w_;
  for (long i = 0; i < n_; ++i) {
    aggregate_loss +=
        loss_.Evaluate(Xw(i), y(i));
  }
  return aggregate_loss / n_ + (lambda_ / 2.0) * NormSquared(w_);
}

void Sdca::SaveHistory(const std::string &filename) {
  // Open file buffer
  std::ofstream results_file;
  results_file.open(filename);

  // Write results to file
  for (const auto &x : training_hist_) {
    results_file << x.first << "," << x.second << "\n";
  }

  results_file.close();
}

void Sdca::RunUpdateOnMiniBatch(const std::vector<Eigen::VectorXd> &X,
                                const std::vector<double> &y,
                                const std::vector<long> &indices) {
#ifndef GPU
  Sdca::RunUpdateOnMiniBatch_cpu(X, y, indices);
#else
  Sdca::RunUpdateOnMiniBatch_gpu(X, y, indices);
#endif
}

void Sdca::RunUpdateOnMiniBatch_cpu(const std::vector<Eigen::VectorXd> &X,
                                    const std::vector<double> &y,
                                    const std::vector<long> &indices) {
  const long batch_size = y.size();

  for (long i = 0; i < batch_size; ++i) {
    const long current_dim = indices[i];
    const Eigen::VectorXd x_i = X[i];
    const double y_i = y[i];
    const double delta_a = DeltaAlpha(x_i, y_i, current_dim);
    ApplyAlphaUpdate(delta_a, current_dim);
    ApplyWeightUpdates(delta_a, x_i);
  }
  ComputeAlphaBar();
  ComputeWBar();
}

// TODO: complete the gpu mini-batch update
void Sdca::RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd> &X,
                                    const std::vector<double> &y,
                                    const std::vector<long> &indices) {
  // std::cout << "NEED TO IMPLEMENT DISTRIBUTED VERSION -- FALLING BACK TO SEQUENTIAL WITH GPU ACCELERATION\n";
  Sdca::RunUpdateOnMiniBatch_cpu(X, y, indices);
}

std::vector<long> Sdca::GenerateMiniBatchIndexVector(const long size,
                                                     const long low,
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

  a_ = Eigen::VectorXd(n_);
  for (long i = 0; i < n_; ++i) {
    a_(i) = 0;
    //a_(i) = norm_dist(gen);
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

} // namespace models
} // namespace edsdca
