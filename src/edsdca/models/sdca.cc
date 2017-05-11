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
    a_bar_ = reduced_accumulated_a;
  }
}

void Sdca::ComputeWBar() {
  switch (sdca_type_) {
  case SdcaModelType::Sequential:
    {
    Eigen::VectorXd reduced_accumulated_w = VectorReduce(accumulated_w_);
    reduced_accumulated_w *= 1.0 / accumulated_w_.size();
    accumulated_w_.clear();
    w_bar_ = reduced_accumulated_w;
    break;
    }
  case SdcaModelType::Distributed:
    {
    Eigen::VectorXd reduced_accumulated_v = VectorReduce(accumulated_v_);
    reduced_accumulated_v *= 1.0 / (lambda_ * n_);
    w_bar_ = reduced_accumulated_v;
    break;
    }
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
  DLOG("a_ and w_ initialized");
  
  // Calculate number of mini-batches
  const long num_batches = long(n_ / batch_size_);
  // training_hist_ tracks the training error and training time (s)
  training_hist_ =
      std::vector<std::pair<double, double>>(num_batches * max_epochs_);
  long log_index = 0;

  // Computes $\omega$ based on current $\alpha$
  ComputeW(X);
  DLOG("w_ properly initialized");
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
      timer_.Stop();

      if ((batch_num + num_batches * cur_epoch) % update_interval_  == 0) {
        ComputeAlphaBar();
        ComputeWBar();
      }

      // Stop timer and get cumulative time
      double cumulative_time = timer_.cumulative_time();

      // Compute training loss with current weights
      if ((batch_num + num_batches * cur_epoch) % loss_interval_ == 0) {
        double loss = ComputeLoss(X, y);

        // Store loss and runtime
        auto tmp_pair = std::make_pair(cumulative_time, loss);
        training_hist_[log_index] = tmp_pair;
        ++log_index;
      }
      DLOG("Epoch complete");
    }
  }

  w_ = w_bar_;
  ComputeW(X);

  SaveHistory("results_test.csv");
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
  
  Eigen::VectorXd Xw = X * w_bar_;
  for (long i = 0; i < n_; ++i) {
    aggregate_loss +=
        loss_.Evaluate(Xw(i), y(i));
  }
  return aggregate_loss / n_ + (lambda_ / 2.0) * NormSquared(w_bar_);
}

void Sdca::SaveHistory(const std::string &filename) {
  // Open file buffer
  std::ofstream results_file;
  results_file.open(filename);

  // Write results to file
  for (const auto &x : training_hist_) {
    if (x.first == 0 && x.second == 0) {
      continue;
    }
    results_file << x.first << "," << x.second << "\n";
  }

  results_file.close();
}

void Sdca::RunUpdateOnMiniBatch(const std::vector<Eigen::VectorXd> &X,
                                const std::vector<double> &y,
                                const std::vector<long> &indices) {
  switch (sdca_type_) {
    case SdcaModelType::Sequential:
      Sdca::RunUpdateOnMiniBatch_cpu(X, y, indices);
      break;
    case SdcaModelType::Distributed:
#ifndef GPU
      DLOG("WARNING! Attempting to run GPU model without a GPU! Running CPU model instead.");
      Sdca::RunUpdateOnMiniBatch_cpu(X, y, indices);
#else
      DLOG("Running GPU mini-batch update");
      Sdca::RunUpdateOnMiniBatch_gpu(X, y, indices);
#endif
      break;
  }
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
}

void Sdca::RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd> &X,
				    const std::vector<double> &y,
				    const std::vector<long> &indices) {
  // Our 'batch size' is only one data poiunt over X.rows() number of workers
  // thus ``scl`` is set to X.rows
  double scl = X.size();

  Eigen::VectorXd wx = MatrixVectorMultiply(X, w_);

  DLOG("Computing delta_alphas");

  for (long i = 0; i < indices.size(); ++i) {
    double nmrtr = 1.0 - wx(i) * y[i];
    auto tmpX = X[i];
    double dnmntr = scl * tmpX.squaredNorm() / (lambda_ * n_);
    double d_alpha = y[i] * std::max(0.0, std::min(1.0,
						   nmrtr / dnmntr + a_(indices[i]) * y[i])) - a_(indices[i]);
    DLOG("Applying alpha update");
    ApplyAlphaUpdate(d_alpha, indices[i]);
    accumulated_v_.push_back(d_alpha * X[i]);
  }
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
  a_bar_ = Eigen::VectorXd(n_);

  for (long i = 0; i < n_; ++i) {
    a_(i) = 0;
    a_bar_(i) = 0;
  }
}

void Sdca::InitializeWeights() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> norm_dist(0, 1);

  w_ = Eigen::VectorXd(d_);
  w_bar_ = Eigen::VectorXd(d_);
  for (long i = 0; i < d_; ++i) {
    w_(i) = norm_dist(gen);
    w_bar_(i) = w_(i);
  }
}

} // namespace models
} // namespace edsdca
