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
  double* MemSync::PushToGpu(const Eigen::VectorXd& x) {
    int size = x.size();
    double *cv = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; ++i) {
        cv[i] = x(i);
    }
    double* d_v = MemSync::AllocateMemOnGpu(size);
    cudaMemcpy(d_v, cv, sizeof(double) * size, cudaMemcpyHostToDevice);
    free(cv);
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
  Eigen::VectorXd MemSync::PullFromGpu(double* d_v, long size) {
    double* v = (double*)malloc(sizeof(double) * size);
    cudaMemcpy(v, d_v, sizeof(double) * size, cudaMemcpyDeviceToHost);
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
  double* MemSync::AllocateMemOnGpu(const long size) {
    double* d_v = (double*)malloc(sizeof(double) * size);
    cudaMalloc((void**)&d_v, sizeof(double) * size);
    return d_v;
  }

  double MemSync::PullValFromGpu(double* d_x) {
    double res;
    double* tmp = (double*)malloc(sizeof(double));
    cudaMemcpy(tmp, d_x, sizeof(double), cudaMemcpyDeviceToHost);
    res = *tmp;
    free(tmp);
    return res;
  }  
} // memory
} // edsdca

#endif // GPU
