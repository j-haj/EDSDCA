#ifdef GPU

#include "edsdca/memory/memsync.h"

namespace edsdca {
namespace memory {
    
  /**
   * Puts data from matrix @p X onto GPU memory
   *
   * @param X data to be transfered to GPU memory
   */
  double *MemSync::PushToGpuMatrix(const std::vector<Eigen::VectorXd> &X) {
    int rows = X.size();
    int cols = X.front().size();
    double *cx = (double*)malloc(sizeof(double) * rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        cx[i * rows + j] = X[i](j);
      }
    }

    if (!memory_is_allocated_ || rows * cols != MemSync::matrix_size_) {
      SetMatrixMemoryAllocationSize(rows * cols);
      AllocateGlobalSharedMem();
    }
    cudaMemcpy(MemSync::dX_, cx, sizeof(double) * rows * cols,
            cudaMemcpyHostToDevice);
    free(cx);
    return dX_;
  }


  /**
   * Puts data stored by @p v onto GPU memory
   *
   * @param v data to be transfered to GPU
   * @param size size of the required data buffer (in bytes)
   *
   * @return a pointer to the memory location on the GPU where the pushed data
   *        is stored
   */
  double* MemSync::PushToGpuX(const Eigen::VectorXd& x) {
    int size = x.size();
    double *cv = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; ++i) {
        cv[i] = x(i);
    }

    // Make sure memory is allocated before proceeding
    if (!memory_is_allocated_ || size != MemSync::d_) {
        SetMemoryAllocationSize(size);
        AllocateGlobalSharedMem();
    }
    cudaMemcpy(MemSync::dx_, cv, sizeof(double) * size, cudaMemcpyHostToDevice);
    free(cv);
    return dx_;
  }
    
  /**
   * Puts data stored by @p v onto GPU memory
   *
   * @param v data to be transfered to GPU
   * @param size size of the required data buffer (in bytes)
   *
   * @return a pointer to the memory location on the GPU where the pushed data
   *        is stored
   */
  double* MemSync::PushToGpuY(const Eigen::VectorXd& x) {
    int size = x.size();
    double *cv = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; ++i) {
        cv[i] = x(i);
    }

    // Make sure memory is allocated before proceeding
    if (!memory_is_allocated_ || size != MemSync::d_) {
        SetMemoryAllocationSize(size);
        AllocateGlobalSharedMem();
    }
    cudaMemcpy(MemSync::dy_, cv, sizeof(double) * size, cudaMemcpyHostToDevice);
    free(cv);
    return dy_;
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

  Eigen::MatrixXd MemSync::PullMatrixFromGpu() {
    int cols = d_;
    int rows = int(matrix_size_ / d_);
    double *tmpX = (double*)malloc(sizeof(double) * matrix_size_);
    cudaMemcpy(tmpX, dX_, sizeof(double) * matrix_size_,
            cudaMemcpyDeviceToHost);
    Eigen::MatrixXd result(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        result(i, j) = tmpX[i * rows + i];
      }
    }
    free(tmpX);
    return result;
  }

  Eigen::VectorXd MemSync::PullResFromGpu() {
    double *tmpR = (double*)malloc(sizeof(double) * d_);
    cudaMemcpy(tmpR, res_, sizeof(double) * d_, cudaMemcpyDeviceToHost);
    Eigen::VectorXd result(d_);
    for (long i = 0; i < d_; ++i) {
      result(i) = tmpR[i];
    }
    free(tmpR);
    return result;
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

  void MemSync::AllocateGlobalSharedMem() {
    if (memory_is_allocated_) {
        cudaFree(dx_);
        cudaFree(dy_);
        cudaFree(res_);
        cudaFree(dX_);
    }
    cudaMalloc((double**)&dx_, d_ * sizeof(double));
    cudaMalloc((double**)&dy_, d_ * sizeof(double));
    cudaMalloc((double**)&res_, d_ * sizeof(double));
    cudaMalloc((double**)&dX_, matrix_size_ * sizeof(double));
    memory_is_allocated_ = true;
  }
} // memory
} // edsdca

#endif // GPU
