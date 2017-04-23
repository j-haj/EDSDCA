#ifndef __EDSDCA_SDCA_H
#define __EDSDCA_SDCA_H

#include <algorithm>
#include <vector>
#include <random>

#include <Eigen/Dense>

#include "edsdca/edsdca.h"
#include "edsdca/util/math_utils.h"

namespace edsdca {
namespace models {

enum SdcaUpdateType {
  Average
};

class Sdca {

  public:

    Sdca(double l) : lambda_(l) {}

    /**
     * Computes delta alpha for index i
     *
     * @param x current data point
     * @param y current feature label
     * @param i index of the direction being optimized
     *
     * @return delta alpha for the ith index
     */
    double DeltaAlpha(const Eigen::VectorXd& x, const double y, long i);

    /**
     * Handles the weight updates
     *
     * @param delta_alpha the delta alpha to apply
     * @param i index of the direction being optimized
     * @param x the current data point
     */
     inline void ApplyWeightUpdates(const double delta_alpha, const Eigen::VectorXd& x) {
       w_ +=  lambda_ / (double)n_ * delta_alpha * x; 
       accumulated_w_.push_back(w_);
     }

     inline void ApplyAlphaUpdate(const double delta_alpha, const long i) {
       a_(i) += delta_alpha;
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
      * Fits an SVM problem via SDCA for the given data
      */
     void Fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);


     /**
      * Given a feature vector, computes the predicted class and return the result
      *
      * @param feature vector
      *
      * @return @p double representing the estimate of the predicted class
      */
     double Predict(const Eigen::VectorXd* x);

     // ------------------------------------------------------------------------
     // Getters
     // ------------------------------------------------------------------------
     inline double lambda() {
       return lambda_;
     }

     inline long n() {
       return n_;
     }

     inline long dimension() {
       return d_;
     }

     inline Eigen::VectorXd w() {
       return w_;
     }

     inline Eigen::VectorXd a() {
       return a_;
     }

     inline std::vector<Eigen::VectorXd> accumulated_a() {
       return accumulated_a_;
     }

     inline std::vector<Eigen::VectorXd> accumulated_w() {
       return accumulated_w_;
     }

  private:
    /// @brief Number of data points
    long n_;

    /// @brief Dimension of the data
    long d_;

    /// @brief Dual variable, $\alpha$
    Eigen::VectorXd a_;

    /// @brief Accumulates @p a_ updates
    std::vector<Eigen::VectorXd> accumulated_a_;

    /// @brief Primal variable, $\omega$
    Eigen::VectorXd w_;

    /// @brief Accumulates the @p w_ updates
    std::vector<Eigen::VectorXd> accumulated_w_;

    /// @brief Regularization term
    double lambda_;

    // Default number of epochs is 20
    long max_epochs_ = 20;

    /// @brief default mini-batch size of 1
    long batch_size_ = 1;

    /**
     * Runs the updates for $\alpha$ and $\omega$
     *
     * @param X mini-batch feature data
     * @param y mini-batch label data
     */
    void RunUpdateOnMiniBatch(const std::vector<Eigen::VectorXd>& X,
            const std::vector<double>& y);

    /**
     * Runs the updates for $\alpha$ and $\omega$ on the CPU
     *
     * @param X mini-batch feature data
     * @param y mini-batch label data
     */
    void RunUpdateOnMiniBatch_cpu(const std::vector<Eigen::VectorXd>& X,
            const std::vector<double>& y);
    
    /**
     * Runs the updates for $\alpha$ and $\omega$ on the GPU
     *
     * @param X mini-batch feature data
     * @param y mini-batch label data
     */
    void RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd>& X,
            const std::vector<double>& y);

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
            const long low, const long high) const;
}; // class Sdca

} // namespace models
} // namespace edsdca
#endif // __EDSDCA_SDCA_H
