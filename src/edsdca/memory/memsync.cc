#include "edsdca/memory/memsync.h"

namespace edsdca {
namespace memory {
#ifdef GPU

double *MemSync::GetDx() { return MemSync::dx_; }
double *MemSync::GetDy() { return MemSync::dy_; }

double *MemSync::GetResultPointer() { return MemSync::res_; }

void MemSync::SetMemoryAllocationSize(long n) {
  MemSync::d_ = n;
}

long MemSync::d_ = 0;
double* MemSync::dx_ = nullptr;
double* MemSync::dy_ = nullptr;
double* MemSync::res_ = nullptr;
bool MemSync::memory_is_allocated_ = false;
bool MemSync::is_heap_allocated_ = false;
#endif // GPU
} // namespace memory
} // namespace edsdca
