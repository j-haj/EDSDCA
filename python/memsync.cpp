#include "memsync.hpp"

bool MemSync::PushToGpu(const std::vector<double>& x) {
  int n = x.size();
  double* d_x;
  cudaMalloc(&d_x, n * sizeof(double));
  cudaMemcpy(d_x, &x[0], cudaMemcpyHostToDevice);
  // TODO: Check CUDA error status
  
  return true;
}

bool MemSync::PullToGpu(std::vector<double>& x) {

}
