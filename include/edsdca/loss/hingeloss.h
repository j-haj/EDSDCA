#ifndef __EDSDCA_HINGELOSS_H
#define __EDSDCA_HINGELOSS_H

#include <algorithm>

#include "edsdca/loss/loss.h"
#include "edsdca/util/math_utils.h"

namespace edsdca {
namespace loss {
class HingeLoss : public AbstractLoss {
public:
  double Evaluate(const double y_pred, const double y_actual);
};
} // namespace loss
} // namespace edsdca

#endif // __EDSDCA_HINGELOSS_H