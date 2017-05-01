#include "edsdca/models/sdca.h"

#ifdef GPU
#include "edsdca/util/math utils.h"
#include "edsdca/memory/memsync.h"
#include "edsdca/util/cuda_util.h"

__global__
void distributed_sdca(double* X, double* y, double* a, double* w, long* indices,
    long m, long n) {
  // IMPLEMENT
}

__device__
double compute_delta_alpha() {

}

namespace edsdca {
namespace models {

void Sdca::RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd> &X,
                                    const std::vector<double> &y,
                                    const std::vector<long> &indices) {
  // Perform matrix-vector multiplication

  // Compute delta-alpha

  // Possibly call ComputeW?
}

} // namespace models
} // namespace edsdca
