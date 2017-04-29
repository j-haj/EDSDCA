#include "edsdca/models/sdca.h"

#ifdef GPU
#include "edsdca/util/math utils.h"
#include "edsdca/memory/memsync.h"
#include "edsdca/util/cuda_util.h"

__global__
void distributed_sdca(double* x, double* y, double* a, double* w) {
  // IMPLEMENT
}

__device__
double compute_delta_alpha() {

}

namespace edsdca {
namespace models {

void Sdca::RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd> &X,
                                    const std::vector<double> &y) {
  // Use threads and blocks
}

} // namespace models
} // namespace edsdca
