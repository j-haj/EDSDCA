#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#ifdef GPU

#include <cuda.h>
#include <sstream>
#include <string>
#include <utility>

#define CUDA_CHECK(err) (edsdca::cuda::HandleError(err, __FILE__, __LINE__))

namespace edsdca {
namespace cuda {

std::pair<int, int> get_grid_and_block_size();

std::string get_device_info();

bool HandleError(cudaError_t err, const char *file, int line);

} // namespace cuda
} // namespace edsdca
#endif
#endif // CUDUA_UTIL_H_
