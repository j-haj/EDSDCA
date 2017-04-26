#ifndef __EDSDCA_TIMER_H
#define __EDSDCA_TIMER_H

#include <chrono>
#include <vector>

#include "edsdca/edsdca.h"

namespace edsdca {
namespace tools {
class Timer {

public:
  /**
   * Starts the timer
   */
  void Start();

  /**
   * Stops the timer
   */
  void Stop();

  /**
   * Returns elapsed time in seconds between calls to @p Start() and @p Stop()
   *
   * @return elapsed time in seconds
   */
  double elapsed();

  /**
   * Returns the total time the @p Timer instance has been running
   *
   * @return cumulative runtime in seconds
   */
  double cumulative_time();

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> stop_time_;
  bool is_running_ = false;
  std::chrono::duration<double> elapsed_time_;
  double cumulative_runtime_ = 0.0;
};
} // namespace tools
} // namespace edsdca

#endif // __EDSDCA_TIMER_H