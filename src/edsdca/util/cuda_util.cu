
#include "edsdca/util/cuda_util.h"

#ifdef GPU

std::string get_device_info() {
  // Get number of devices
  int n_devices;
  cudaGetDeviceCount(&n_devices);

  std::stringstream ss;
  ss << "Found " << n_devices << " CUDA device(s)\n\n";

  for (int i = 0; i < n_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    ss << "Device number: " << i << "\n"
       << "\tDevice name: " << prop.name << "\n"
       << "\tMemory clock rate (KHz): " << prop.memoryClockRate << "\n"
       << "\tMemory bus width (bits): " << prop.memoryBusWidth << "\n"
       << "\tPeak memory bandwidth (GB/s): "
       << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
       << "\n\n";
  }

  return ss.str();
}


#endif // GPU
