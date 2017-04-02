#ifndef __UTIL_DATETIME_H
#define __UTIL_DATETIME_H

#include <sstream>
#include <string>
#include <chrono>

namespace util {
namespace date_time {

class FormattedTime {
  public:
  static std::string now(std::string fmt="%d-%m-%Y__%H_%M_%S") {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, fmt.c_str());
    return oss.str();
  }
}; // class FormattedTime
} // namespace date_time
} // namespace util
#endif // __UTIL_DATETIME_H
