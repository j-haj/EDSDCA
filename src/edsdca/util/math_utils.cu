#include "edsdca/util/math_util.h"

__global__
void vector_prod_gpu(double* x, double* y, double* res, long n) {
  //
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    res[i] = x[i] * y[i];
  }
}

__global__
void vector_dot_gpu(double* x, double* y, double* res, long n) {
  __shared__ double temp[n];
  double temp = x[threadIdx.x] * y[threadIdx.x];

  __syncthreads();

  if (0 == threadIdx.x) {
    double sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += temp[i];
    }
    *res = sum;
  }
}

// TODO: add cuda call
double NormSquared_gpu(const std::vector<double> &x) {
  // Call CUDA kernel for NormSquared here
  double res = 0;
  return res;
}

double NormSquared_gpu(const Eigen::VectorXd &x) {
  int block_size, grid_size, min_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                     vector_dot_gpu, 0, 0);
  grid_size = (x.size() + block_size - 1) / block_size;
  grid_size = std::max(grid_size, min_grid_size);

  // Call CUDA kernel
  // Copy data to gpu
  double *d_x = edsdca::memory::MemSync::PushToGpu(x);

  // Call the kernel
  double *res = edsdca::memory::MemSync::AllocateMemOnGpu(1);
  vector_dot_gpu<<<grid_size, block_size>>>(d_x, d_x, res, x.size());

  // Copy back from gpu
  double result = edsdca::memory::MemSync::PullValFromGpu(res);

  cudaFree(d_x); cudaFree(res);

  return result;
}

double VectorDotProd_gpu(const std::vector<double> &x,
                         const std::vector<double> &y) {
  int block_size, grid_size, min_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                     vector_dot_gpu, 0, 0);
  grid_size = (x.size() + block_size - 1) / block_size;
  grid_size = std::max(grid_size, min_grid_size);

  double *d_x = edsdca::memory::MemSync::PushToGpu(x);
  double *d_y = edsdca::memory::MemSync::PushToGpu(y);

  // Call the kernel
  double *res = edsdca::memory::MemSync::AllocateMemOnGpu(1);
  vector_dot_gpu<<<grid_size, block_size>>>(d_x, d_y, res, x.size());

  // Copy back from gpu
  double result = edsdca::memory::MemSync::PullValFromGpu(res);

  cudaFree(d_x); cudaFree(d_y); cudaFree(res);

  return result
}

double VectorDotProd_gpu(const Eigen::VectorXd &x, const Eigen::VectorXd &y) {
  int block_size, grid_size, min_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                     vector_dot_gpu, 0, 0);
  grid_size = (x.size() + block_size - 1) / block_size;
  grid_size = std::max(grid_size, min_grid_size);

  double *d_x = edsdca::memory::MemSync::PushToGpu(x);
  double *d_y = edsdca::memory::MemSync::PushToGpu(y);

  // Call the kernel
  double *res = edsdca::memory::MemSync::AllocateMemOnGpu(1);
  vector_dot_gpu<<<grid_size, block_size>>>(d_x, d_y, res, x.size());

  // Copy back from gpu
  double result = edsdca::memory::MemSync::PullValFromGpu(res);

  cudaFree(d_x); cudaFree(d_y); cudaFree(res)
  return result;
}

// TODO: add CUDA call
void VectorInPlaceSum_gpu(std::vector<double> &x,
                          const std::vector<double> &y) {
  // GPU implementation
}

Eigen::VectorXd VectorReduce_gpu(const std::vector<Eigen::VectorXd> &v) {
  Eigen::VectorXd accumulator = Eigen::VectorXd::Zero(v.front().size());
  return accumulator;
}