
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
