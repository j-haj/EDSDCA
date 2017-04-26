#ifndef __SYNCMEM_HPP
#define __SYNCMEM_HPP

#ifdef GPU

#include <cuda.h>
#include "cuda_util.h"

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
  template <typename T>
  T* PushToGpu(const T* v, int size) {
    T* d_v = AllocateOnGpu(size);
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
  template <typename T>
  T* PullFromGpu(T* d_v, int size) {
    T* v = (T*)malloc(size);
    CUDA_CHECK(cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost));
  }

  /**
   * Allocates memory on the GPU
   *
   * @param size size of requested memory in bytes
   *
   * @return returns a pointer to the allocated memory on GPU
   */
  template <typename T>
  T* AllocateOnGpu(int size) {
    T* d_v = (T*)malloc(sizeof(T*));
    CUDA_CHECK(cudaMalloc((void**)&d_v, size));
    return d_v;
  }

} // memory
} // edsdca

#endif // GPU
#endif // __SYNCMEM_HPP
