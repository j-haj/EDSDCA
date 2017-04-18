#ifndef __SDCA_H
#define __SDCA_H

#include "math_utils.hpp"

#include <random>
#include <vector>
#include <cmath>

enum SdcaUpdateType {
  Average,
  Random
};

// =============================================================================
//
// SDCA SOLVER
//
// =============================================================================
/**
 * SDCA solver for hinge loss.
 *
 * See SDCA for Regularized Loss Minimization Shalev-Schwartz & T. Zhong for
 * implementation details.
 */
class SdcaSolver {

public:
  double GetDeltaAlpha(const std::vector<double>& x, const int y);

  /**
   * Handle alpha vector value updates
   *
   * @param delta_alpha $\delta\alpha$ value computed by @p GetDeltaAlpha method
   * @param i component being updated
   */
  inline void ApplyAlphaUpdate(const double delta_alpha, long i);

  /**
   * Handle weight updates
   *
   * @param delta_alpha $\Delta\alpha$ computed by @p GetDeltaAlpha method
   * @param i component of the weight vector being updated
   * @param x_i data point being considered // FIXME: need to checkthis!
   */
  inline void ApplyWeightUpdate(const double delta_alpha, long i, double x_i);
  
  std::vector<double>& ComputeAlphaBar(SdcaUpdateType update_type = SdcaUpdateType::Average);
  std::vector<double>& ComputeWBar(SdcaUpdateType update_type = SdcaUpdateType::Average);

private:
  /// @brief number of data points 
  // TODO: Verify this!
  long n_;

  /// @brief Dual vector, alpha
  std::vector<double> a_;

  /// @brief Primal vector, omega
  std::vector<double> w_;

  /// @brief regularization term
  double lambda_;
};

#endif // __SDCA_H
