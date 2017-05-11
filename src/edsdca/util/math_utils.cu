#include "edsdca/util/math_utils.h"
#include <iostream>
#ifdef GPU

__global__
void vector_prod_gpu(double* x, double* y, double* res, long n) {
  //
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    res[i] = x[i] * y[i];
  }
}

__global__
void matrix_vector_prod(double *X, double *y, double *res, long m, long n) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  if (idx < m) {
    for (long i = 0; i < n; ++i) {
      sum += X[(i * m) + idx] * y[i];
    }
    res[idx] = sum;
  }
}

#define MAX_SIZE 2048
__global__
void vector_dot_gpu(double* x, double* y, double* res, long n) {
  __shared__ double temp[MAX_SIZE];
  temp[threadIdx.x] = x[threadIdx.x] * y[threadIdx.x];

  __syncthreads();

  if (0 == threadIdx.x) {
    double sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += temp[i];
    }
    *res = sum;
  }
}


double NormSquared_gpu(const Eigen::VectorXd &x) {
  return VectorDotProd_gpu(x, x);
}


double VectorDotProd_gpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  int block_size, grid_size, min_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                     vector_dot_gpu, 0, 0);
  grid_size = (x.size() + block_size - 1) / block_size;
  grid_size = std::max(grid_size, min_grid_size);

  double *d_x = edsdca::memory::MemSync::PushToGpuX(x);
  double *d_y = edsdca::memory::MemSync::PushToGpuY(y);

  // Call the kernel
  vector_prod_gpu<<<grid_size, block_size>>>(edsdca::memory::MemSync::dx_,
         edsdca::memory::MemSync::dy_,
         edsdca::memory::MemSync::res_, x.size());

  // Copy back from gpu
  Eigen::VectorXd result =
      edsdca::memory::MemSync::PullFromGpu(edsdca::memory::MemSync::res_, x.size());
  return result.sum();
}


Eigen::VectorXd VectorReduce_gpu(const std::vector<Eigen::VectorXd> &v) {
  Eigen::VectorXd accumulator = Eigen::VectorXd::Zero(v.front().size());
  for (const Eigen::VectorXd& x : v) {
      accumulator += x;
  }
  return accumulator;
}

Eigen::VectorXd MatrixVectorMultiply_gpu(const std::vector<Eigen::VectorXd> &X, const
    Eigen::VectorXd &y) {
  int block_size, grid_size, min_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
      matrix_vector_prod, 0, 0);
  grid_size = int((X.size() * X.front().size() + block_size - 1) / block_size);
  grid_size = std::max(grid_size, min_grid_size);
    
  double *d_X = edsdca::memory::MemSync::PushToGpuMatrix(X);
  double *d_y = edsdca::memory::MemSync::PushToGpuY(y);

  // Call the kernel
  matrix_vector_prod<<<grid_size, block_size>>>(edsdca::memory::MemSync::dX_,
      edsdca::memory::MemSync::dy_,
      edsdca::memory::MemSync::res_, X.size(), X.front().size());
  
  // Copy back from GPU
  Eigen::VectorXd result =
    edsdca::memory::MemSync::PullResFromGpu();
  return result;
}

#endif // GPU
