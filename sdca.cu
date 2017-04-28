#include "edsdca/models/sdca.h"

#ifdef GPU
#include "edsdca/util/math utils.h"
#include "edsdca/memory/memsync.h"
#include "edsdca/util/cuda_util.h"

namespace edsdca {
namespace models {

void Sdca::RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd> &X,
                                    const std::vector<double> &y) {
  // Use threads and blocks
}

} // namespace models
} // namespace edsdca
