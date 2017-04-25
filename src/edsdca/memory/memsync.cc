#include "edsdca/memory/memsync.h"

namespace edsdca {
namespace memory {

double *MemSync::PushToGpu(const Eigen::VectorXd &x) {
  long size = x.size();
  double *carray_x;
  Eigen::Map<Eigen::VectorXd>(carray_x, size) = x;
  double *d_x;
  CUDA_CHECK(cudaMalloc(&d_x, size * sizeof(double)));
  CUDA_CHECK(cudaMemcypy(d_x, carray_x, size * sizeof(double),
                         cudaMemcpyHostToDevice));
  return carray_x;
}

bool MemSync::PullToGpu(std::vector<double> &x) {
  CUDA_CHECK(cudaMemcpy(&x[0], d_x, cudaMemcpyDeviceToHost));
  return true;
}

Eigen::VectorXd MemSync::PullFromGpu(double *d_x, long size) {
  Eigen::VectorXd eigen_x;
  double *x = (double *)malloc(sizeof(double) * size);
  CUDA_CHECK(cudaMemcpy(&x[0], d_x, cudamemcpyDeviceToHost));
  Eigen::Map<Eigen::VectorXd> eigen_x(x, size);
  return eigen_x;
}

double *MemSync::AllocateMemOnGpu(const long size) {
  double *d_v = (double*)malloc(sizeof(double) * size);
  CUDA_CHECK(cudaMalloc((void**)&d_v, size * (sizeof(double))));
  return d_v;
}

} // namespace memory
} // namespace edsdca