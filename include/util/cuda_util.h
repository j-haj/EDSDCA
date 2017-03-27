#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include <cuda.h>
#include <string>
#include <sstream>

#ifdef GPU
std::string get_device_info() {
  // Get number of devices
  int n_devices;
  cudaGetDeviceCount(&n_devices);

  std::stringstream ss;
  ss << "Found " << n_devices << " CUDA device(s)\n\n";

  for (int i = 0; i < n_devices; i++) {
    ss << "Device number: " << i << "\n"
       << "\tDevice name: " << prop.name << "\n"
       << "\tMemory clock rate (KHz): " << prop.memoryClockRate << "\n"
       << "\tMemory bus width (bits): " << prop.memoryBusWidth << "\n"
       << "\tPeak memory bandwidth (GB/s): " 
       << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << "\n\n";
  }

  return ss.str();
}

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#endif // GPU
#endif // CUDUA_UTIL_H_

