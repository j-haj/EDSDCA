#include "memsync.hpp"

bool MemSync::PushToGpu(const std::vector<double>& x) {
  int n = x.size();
  double* d_x;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_x, &x[0], cudaMemcpyHostToDevice));
  return true;
}

bool MemSync::PullToGpu(std::vector<double>& x) {
  CUDA_CHECK(cudaMemcpy(&x[0], d_x, cudaMemcpyDeviceToHost));
  return true;
}
