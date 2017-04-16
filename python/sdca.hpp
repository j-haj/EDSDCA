#ifndef __SDCA_H
#define __SDCA_H

#include "math_utils.hpp"

#include <random>
#include <vector>
#include <cmath>

// =============================================================================
//
// SDCA SOLVER
//
// =============================================================================
/**
 * SDCA solver for hinge loss
 */
class SdcaSolver {

public:
   double GetDeltaAlpha(const std::vector<double>& x, const int y);

private:
  /// @brief Dual vector, alpha
  std::vector<double> a_;

  /// @brief Primal vector, omega
  std::vector<double> w_;

};

#endif // __SDCA_H
