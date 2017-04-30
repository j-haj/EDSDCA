#include "edsdca/memory/memsync.h"

namespace edsdca {
namespace memory {


double *MemSync::GetDx() { return MemSync::dx_; }
double *MemSync::GetDy() { return MemSync::dy_; }

double *MemSync::GetResultPointer() { return MemSync::res_; }

void MemSync::SetMemoryAllocationSize(long n) {
  MemSync::d_ = n;
  AllocateGlobalSharedMem();
  MemSync::memory_is_allocated_ = true;
}



long MemSync::d_ = 0;
double* MemSync::dx_ = nullptr;
double* MemSync::dy_ = nullptr;
double* MemSync::res_ = nullptr;
bool MemSync::memory_is_allocated_ = false;

} // namespace memory
} // namespace edsdca
