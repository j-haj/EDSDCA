
__global__
void vector_prod_cuda(int n, double* res, double* x, double* y) {
  //
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    res[i] = x[i] * y[i];
  }
} 
