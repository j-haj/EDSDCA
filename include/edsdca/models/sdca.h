#ifndef __EDSDCA_SDCA_H
#define __EDSDCA_SDCA_H

#include <algorithm>
#include <fstream>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "edsdca/edsdca.h"
#include "edsdca/loss/hingeloss.h"
#include "edsdca/loss/loss.h"
#include "edsdca/tools/timer.h"
#include "edsdca/util/math_utils.h"

namespace edsdca {
namespace models {

enum SdcaUpdateType { Average };
enum SdcaModelType { Sequential, Distributed };

class Sdca {

public:
  Sdca(double l, long interval=100)
      : lambda_(l), timer_(edsdca::tools::Timer()),
        loss_(edsdca::loss::HingeLoss()),
            update_interval_(interval),
            sdca_type_(SdcaModelType::Sequential) {}

  /**
   * Computes delta alpha for index i
   *
   * @param x current data point
   * @param y current feature label
   * @param i index of the direction being optimized
   *
   * @return delta alpha for the ith index
   */
  double DeltaAlpha(const Eigen::VectorXd &x, const double y, long i);

  /**
   * Handles the weight updates
   *
   * @param delta_alpha the delta alpha to apply
   * @param i index of the direction being optimized
   * @param x the current data point
   */
  inline void ApplyWeightUpdates(const double delta_alpha,
                                 const Eigen::VectorXd &x) {
    const double scale_factor = (lambda_ / double(n_)) * delta_alpha;
    w_ = w_ +  x * scale_factor;
    accumulated_w_.push_back(w_);
  }

  inline void ApplyAlphaUpdate(const double delta_alpha, const long i) {
    a_(i) = a_(i) + delta_alpha;
    accumulated_a_.push_back(a_);
  }

  /**
   * Computes $\bar \alpha$ from the accumulated updates
   *
   * @param update_type The type of update (default is average)
   */
  void ComputeAlphaBar(SdcaUpdateType update_type = SdcaUpdateType::Average);

  /**
   * Computes $\bar \omega$ from the accumulated updates
   *
   * @param update_type The update method (default is average)
   */
  void ComputeWBar(SdcaUpdateType update_type = SdcaUpdateType::Average);

  /**
   * Computes $omega$ given $alpha$. This is called at the beginning of each
   * epoch of SDCA.
   *
   * The formula for this update is equation (3) from SDCA for Reg. Loss
   * Minimization (Shalev-Schwartz, et al)
   *
   *    \omega(\alpha) = \frac{1}{\lambda n}\sum_{i=1}^n \alpha_i x_i
   *
   * @param X matrix of all the data
   */
  void ComputeW(const Eigen::MatrixXd &X);

  /**
   * Fits an SVM problem via SDCA for the given data
   */
  void Fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

  /**
   * Given a feature vector, computes the predicted class and return the result
   *
   * @param feature vector
   *
   * @return @p double representing the estimate of the predicted class
   */
  double Predict(const Eigen::VectorXd &x);

  double ComputeLoss(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

  /**
   * Saves the training history to a CSV file with filename @p filename
   *
   * @param filename name of the file used to save the training history
   */
  void SaveHistory(const std::string& filename);

  // ------------------------------------------------------------------------
  // Getters
  // ------------------------------------------------------------------------
  inline double lambda() { return lambda_; }

  inline long n() { return n_; }

  inline long dimension() { return d_; }

  inline Eigen::VectorXd w() { return w_; }

  inline Eigen::VectorXd a() { return a_; }

  inline std::vector<Eigen::VectorXd> accumulated_a() { return accumulated_a_; }

  inline std::vector<Eigen::VectorXd> accumulated_w() { return accumulated_w_; }

  // ---------------------------------------------------------------------------
  // Setters
  // ---------------------------------------------------------------------------

  void set_max_epochs(long n) { max_epochs_ = n; }

  void set_batch_size(long n) { batch_size_ = n; }

private:
  
  /// @brief Interval at which alpha_bar and w_bar are upated
  long update_interval_;

  /// @brief Interval at which training loss is computed
  long loss_interval_ = 100;

  /// @brief Number of data points
  long n_;

  /// @brief Dimension of the data
  long d_;

  /// @brief Handle for timer
  edsdca::tools::Timer timer_;

  /// @brief Loss function for model
  edsdca::loss::HingeLoss loss_;

  /// @brief stores pairs of elapsed time and error values
  std::vector<std::pair<double, double>> training_hist_;

  /// @brief Dual variable, $\alpha$
  Eigen::VectorXd a_;
  
  /// @brief Dual variable of averaged updates
  Eigen::VectorXd a_bar_;

  /// @brief Accumulates @p a_ updates
  std::vector<Eigen::VectorXd> accumulated_a_;

  /// @brief Primal variable, $\omega$
  Eigen::VectorXd w_;

  /// @brief Averaged weights $\bar \omega$
  Eigen::VectorXd w_bar_;

  /// @brief Accumulates the @p w_ updates
  std::vector<Eigen::VectorXd> accumulated_w_;

  /// @brief Regularization term
  double lambda_;

  // Default number of epochs is 20
  long max_epochs_ = 50;

  /// @brief default mini-batch size of 1
  long batch_size_ = 1;

  /// @brief Type of optimizer used - Sequential or Distributed
  SdcaModelType sdca_type_;

  /**
   * Runs the updates for $\alpha$ and $\omega$
   *
   * @param X mini-batch feature data
   * @param y mini-batch label data
   * @param indices vector of random integers used in selecting the mini-bath
   * data
   */
  void RunUpdateOnMiniBatch(const std::vector<Eigen::VectorXd> &X,
                            const std::vector<double> &y,
                            const std::vector<long> &indices);

  /**
   * Runs the updates for $\alpha$ and $\omega$ on the CPU
   *
   * @param X mini-batch feature data
   * @param y mini-batch label data
   * @param indices vector of random integers used in selecting the mini-batch
   */
  void RunUpdateOnMiniBatch_cpu(const std::vector<Eigen::VectorXd> &X,
                                const std::vector<double> &y,
                                const std::vector<long> &indices);

  /**
   * Runs the updates for $\alpha$ and $\omega$ on the GPU
   *
   * @param X mini-batch feature data
   * @param y mini-batch label data
   * @param indices vector of random integers used in selecting the mini-batch
   */
  void RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd> &X,
                                const std::vector<double> &y,
                                const std::vector<long> &indices);

  /**
   * Creates a @p std::vector of indices from 0 to max - 1 to be used in
   * creating a mini-batch sample. The indices are randomly selected from a
   * uniform distribution
   *
   * @param size the number of indices to generate
   * @param low the lower bound on the value of indicies
   * @param high the upper bound on the value of indices
   *
   * @return a @p std::vector of the indices
   */
  std::vector<long> GenerateMiniBatchIndexVector(const long size,
                                                 const long low,
                                                 const long high) const;

  /**
   * Initializes @p a_ to random values generated by a normal distribution
   * with mean 0 and variance 1
   */
  void InitializeAlpha();

  /**
   * Initializes @p w_ to random values generated by a normal distributions
   * with mean 0 and variance 1
   */
  void InitializeWeights();

}; // class Sdca

} // namespace models
} // namespace edsdca
#endif // __EDSDCA_SDCA_H
