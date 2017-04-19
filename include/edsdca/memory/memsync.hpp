#ifndef __SYNCMEM_HPP
#define __SYNCMEM_HPP

#include <cuda.h>
#include "cuda_util.h"

namespace edsdca {
namespace memory {

class MemSync {

public:

  static bool PushToGpu(const std::vector<double>& x);
  static bool PullFromGpu(const std::vector<double>& x);

}; // class MemSync


} // memory
} // edsdca
#endif // __SYNCMEM_HPP
