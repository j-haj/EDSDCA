#include "edsdca/models/sdca.h"

#include "edsdca/util/math utils.h"
#include "edsdca/memory/memsync.h"
n#include "edsdca/util/cuda_util.h"

namespace edsdca {
namespace models {

void Sdca::RunUpdateOnMiniBatch_gpu(const std::vector<Eigen::VectorXd> &X,
                                    const std::vector<double> &y,
                                    const std::vector<long> &indices) {

  // Our "batch size" is only one data point over ``X.rows()`` number of workers, thus
  // ``scl`` is set to 1/X.rows()
  double scl = 1.0 / X.rows();
  DLOG("Calling MatrixVector multiply");
  // Perform matrix-vector multiplication
  Eigen::VectorXd wx = MatrixVectorMultiply(X, w_);
  DLOG("Computing delta_alphas");
  
  // Computes delta_alphas
  Eigen::VectorXd d_alphas(X.rows());
  for (long i = 0; i < indices.size(); ++i) {
    double numerator = 1 - wx(i) * y[i];
    double denominator = scl * X[i].normSquared() / (lambda_ * n_);
    double d_alpha = y[i] * std::max(0.0,std::min(1.0,
	          numerator / denominator + a_(indices[i]) * y[i])) - a_(indices[i]);
    DLOG("Applying alpha update");
    ApplyAlphaUpdate(d_alpha, indices[i]);
    accumulated_v_.push_back(d_alpha * X[i]);
  }
  DLOG("Exiting update");
  // Possibly call ComputeW?
}

} // namespace models
} // namespace edsdca
