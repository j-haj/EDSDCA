#include "edsdca/tools/timer.h"

namespace edsdca {
namespace tools {

void Timer::Start() {
  is_running_ = true;
  start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::Stop() {
  if (is_running_) {
    stop_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = false;
  }
}

double Timer::elapsed() {
  elapsed_time_ = stop_time_ - start_time_;
  double t = elapsed_time_.count();
  cumulative_runtime_ += t;
  return t;
}

double Timer::cumulative_time() { return cumulative_runtime_; }

} // namespace tools
} // namespace edsdca