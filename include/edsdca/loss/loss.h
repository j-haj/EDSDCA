#ifndef __EDSDCA_LOSS_H
#define __EDSDCA_LOSS_H

#include <Eigen/Dense>

namespace edsdca {
namespace loss {
class AbstractLoss {
public:
  virtual double Evaluate(const double y_pred, const double y_actual) = 0;

}; // class AbstractLoss
} // namespace loss

} // namespace edsdca

#endif // __EDSDCA_LOSS_H