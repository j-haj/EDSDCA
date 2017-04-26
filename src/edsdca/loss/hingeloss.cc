#include "edsdca/loss/hingeloss.h"

namespace edsdca {
namespace loss {
double HingeLoss::Evaluate(const double y_pred, const double y_actual) {
  return std::max(0.0, 1.0 - y_pred * y_actual);
}
} // namespace loss
} // namespace edsdca