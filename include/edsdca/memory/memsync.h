#ifndef __EDSDCA_MEMSYNC_H
#define __EDSDCA_MEMSYNC_H

#ifdef GPU

#include <vector>

#include <cuda.h>
#include <Eigen/Dense>

#include "edsdca/util/cuda_util.h"

namespace edsdca {
namespace memory {

class MemSync {
  public:
  
  /**
   * Gets a reference do dx_
   */
  static double* GetDx();

  /**
   * Gets a reference to dy_
   */
  static double* GetDy();

  /**
   *
   */
  static double *PushToGpuMatrix(const std::vector<Eigen::VectorXd> &X);

  /**
   * Moves the @p Eigen::VectorXd from RAM to GPU and returns a pointer to the
   * memory in GPU
   *
   * @param x reference to the @p Eigen::VectorXd in memory
   *
   * @return pointer to the data on the GPU
   */
  static double *PushToGpuX(const Eigen::VectorXd &x);
  
  /**
   * Moves the @p Eigen::VectorXd from RAM to GPU and returns a pointer to the
   * memory in GPU
   *
   * @param x reference to the @p Eigen::VectorXd in memory
   *
   * @return pointer to the data on the GPU
   */
  static double *PushToGpuY(const Eigen::VectorXd &x);

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
   * Copes the data at @p dX_ from GPU to RAM
   */
  static Eigen::MatrixXd PullMatrixFromGpu();

  /**
   * Copies the data from @p res_ from GPU to RAM
   */
  static Eigen::VectorXd PullResFromGpu();

  /**
   * Pulls a single value from memory on the GPU
   *
   * @param d_x pointer to the value on the GPU
   *
   * @return the value pulled off the GPU
   */
  static double PullValFromGpu(double* d_x);
  
  /**
   * Allocates memory on the GPU and returns a pointer to the allocated
   * memory
   *
   * @param size size of the memory to be allocated (in bytes)
   *
   * @return a @p double* pointer to the memory on gpu
   */
  static double *AllocateMemOnGpu(const long size);

  /**
   * Returns a pointer to a pre-allocated result vecotr of size @p d_
   */
  static double *GetResultPointer();

  /**
   * Static to pointer to dx - this is used to prevent excessive allocations
   */
  static double* dx_;

  /**
   * Static pointer to DX - this is a reference to a matrix on the GPU
   */
  static double* dX_;

  /**
   * Static pointer to dy - this is used to prevent excessive allocations
   */
  static double* dy_;

  /**
   * Static pointer to res_ - this is used for pulling results off the GPU
   */
  static double* res_;

  /**
   * This is an internal flag used to determine if dx_ and dy_ memory
   * allocations need to be made
   */
  static bool memory_is_allocated_;

  /**
   * Flag that gets set to true once there is a heap allocation for @p dx_, @p
   * dy_, and @p res_
   */
  static bool is_heap_allocated_;

  /**
   * Size of the allocated memory. This is equal to total memory allocated on
   * GPU divided by 2 divided by sizeof(double).
   */
  static long d_;

  /**
   * Size of the allocated memory for dX_, a matrix.
   */
  static long matrix_size_;

  /**
   * This is is called to set the shared memroy allocation size. This is
   * typically the size of the feature vectors.
   *
   * @param n the size (e.g. count) of the data (NOT size in bytes)
   */
  static void SetMemoryAllocationSize(long n);

  /**
   * This is called to set the shared memory allocation size of the matrix dX_
   *
   * @param n the size of the matrix (e.g., number of elements)
   */
  static void SetMatrixMemoryAllocationSize(long n);

  /**
   * Allocates memory if memory has not yet been allocated or needs to be
   * allocated (due to new size requirements)
   */
  static void AllocateGlobalSharedMem();
}; // class MemSync



} // namespace memory
} // namespace edsdca

#endif // GPU
#endif // __EDSDCA_MEMSYNC_H
