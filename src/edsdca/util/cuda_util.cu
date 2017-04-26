
#include "edsdca/uitl/cuda_util.h"

#ifdef GPU

std::pair<int, int> get_grid_and_block_size() {
   int block_size;
   int grid_size, min_grid_size;
   cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, )
}


std::string edsdca::cuda::get_device_info() {
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
       << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
       << "\n\n";
  }

  return ss.str();
}

static bool edsdca::cuda::HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
  return true;
}

#endif // GPU