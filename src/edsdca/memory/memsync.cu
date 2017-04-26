#ifndef __SYNCMEM_HPP
#define __SYNCMEM_HPP

#ifdef GPU

#include "edsdca/memory/memsync.h"

namespace edsdca {
namespace memory {

  /**
   * Puts data stored by @p v onto GPU memory
   *
   * @param v data to be transfered to GPU
   * @param size size of the required data buffer (in bytes)
   *
   * @return a pointer to the memory location on the GPU where the pushed data
   *        is stored
   */
  double* MemSync::PushToGpu(const double* v, int size) {
    double* d_v = AllocateOnGpu(sizeof(double) * size);
    CUDA_CHECK(cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice));
    return d_v;
  }

  /**
   * Pulls data from the GPU to host
   *
   * @param d_v pointer to the data on GPU
   * @param size size of the data buffer required to store the data (in bytes)
   *
   * @return pointer to the location of the data on host memory
   */
  Eigen::VectorXd PullFromGpu(double* d_v, int size) {
    double* v = (double*)malloc(sizeof(double) *size);
    CUDA_CHECK(cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost));
    Eigen::VectorXd eig_v(size);
    for (long i = 0; i < size; ++i) {
        eig_v(i) = v[i];
    }
    return eig_v;
  }

  /**
   * Allocates memory on the GPU
   *
   * @param size size of requested memory in bytes
   *
   * @return returns a pointer to the allocated memory on GPU
   */
  double* AllocateOnGpu(int size) {
    double* d_v = (double*)malloc(sizeof(double) * size);
    CUDA_CHECK(cudaMalloc((void**)&d_v, size));
    return d_v;
  }

  double MemSync::PullValFromGpu(double* d_x) {
    double res;
    double* tmp = (double*)malloc(sizeof(double));
    CUDA_CHECK(cudaMemcpy(res, d_x, sizeof(double), cudaMemcpyDeviceToHost));
    res = *tmp;
    free(tmp);
    return res;
  
} // memory
} // edsdca

#endif // GPU
#endif // __SYNCMEM_HPP
