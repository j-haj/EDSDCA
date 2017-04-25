#ifndef __EDSDCA_MEMSYNC_H
#define __EDSDCA_MEMSYNC_H

#include <vector>

#include <Eigen/Dense>

#include "edsdca/util/cuda_util.h"

namespace edsdca {
namespace memory {

class MemSync {

  /**
   * Moves the @p Eigen::VectorXd from RAM to GPU and returns a pointer to the
   * memory in GPU
   *
   * @param x reference to the @p Eigen::VectorXd in memory
   *
   * @return pointer to the data on the GPU
   */
  static double *PushToGpu(const Eigen::VectorXd &x);

  /**
   * Copies the data from GPU to RAM and creates and Eigen::VectorXd which is
   * returned
   *
   * @param d_x pointer to data on the gpu
   * @param size size of the vector on GPU (and to be created)
   *
   * @return @p Eigen::VectorXd ector containing the data that was on the
   *        GPU
   */
  static Eigen::VectorXd PullFromGpu(double *d_x, long size);

  /**
   * Pulls a single value from memory on the GPU
   *
   * @param d_x pointer to the value on the GPU
   *
   * @return the value pulled off the GPU
   */
#ifdef GPU
  template <typename T>
  static T PullValFromGpu(T* d_x) {
    T res;
    T* tmp_res = (T*)malloc(sizeof(T));
    CUDA_CHECK(cudaMemcpy(res, d_x, sizeof(T), cudaMemcpyDeviceToJost));
    res = *tmp_res;
    free(tmp_res);
    return res;
  }
#endif
  /**
   * Allocates memory on the GPU and returns a pointer to the allocated
   * memory
   *
   * @param size size of the memory to be allocated (in bytes)
   *
   * @return a @p double* pointer to the memory on gpu
   */
  double *AllocateMemOnGpu(const long size);
}; // class MemSync

} // namespace memory
} // namespace edsdca

#endif // __EDSDCA_MEMSYNC_H