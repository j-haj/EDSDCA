#ifndef __SYNCMEM_HPP
#define __SYNCMEM_HPP

#include "cuda.h"
#include "cuda_utils.h"

class MemSync {

public:

  static bool PushToGpu(const std::vector<double>& x);
  static bool PullFromGpu(const std::vector<double>& x);

};

#endif // __SYNCMEM_HPP
