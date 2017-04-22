#ifndef __EDSDCA_SDCA_H
#define __EDSDCA_SDCA_H

#include <algorithm>
#include <vector>

#include <Eigen/Dense>

#include "edsdca/edsdca.h"

namespace edsdca {
namespace models {

class Sdca {

  public:

    Sdca(double lamba) : lambda_(lambda) {}

    /**
     * Computes delta alpha for index i
     *
     * @param x current data point
     * @param y current feature label
     * @param i index of the direction being optimized
     *
     * @return delta alpha for the ith index
     */
    double DeltaAlpha(const Eigen::VectorXd& x, const int y, long i);

    /**
     * Handles the weight updates
     *
     * @param delta_alpha the delta alpha to apply
     * @param i index of the direction being optimized
     * @param x the current data point
     */
     inline void ApplyWeightUpdates(const double delta_alpha, long i, Eigen::VectorXd* x) {
       w_(i) += lambda_ / n_ * delta_alpha * x(i); // TODO: Verify this!
       accumulated_w_.push_back(w_);
     }

     inline void ApplyAlphaUpdate(const double delta_alpha, long i) {
       a_[i] += delta_alpha;
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

     // ------------------------------------------------------------------------
     // Getters
     // ------------------------------------------------------------------------
     inline long n() {
       return n_;
     }

     inline long dimension() {
       return d_;
     }

     inline Eigen::VectorXd w() {
       return w_;
     }

     inline VectorXd a() {
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
    long num_epochs_ = 20;

    /// @brief default mini-batch size of 1
    long batch_size_ = 1;
     
}; // class Sdca

} // namespace model
} // namespace edsdca
#endif // __EDSDCA_SDCA_H
