#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

#define CUDA_CHECK(res) { cuda_tools::gpuAssert((ans), __FILE__, __LINE__); }

namespace cuda_tools {

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cdoe),
        file, line);
    if (abort) exit(code);
  }
}

bool GetDeviceInfo();

};
#endif // __CUDA_UTILS_H
