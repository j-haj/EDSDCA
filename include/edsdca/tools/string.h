#ifndef __EDSDCA_STRING_H
#define __EDSDCA_STRING_H

#include <string>
#include <sstream>
#include <vector>

#include "edsdca/edsdca.h"

namespace edsdca {
namespace tools {
class String {

  public:
  
    static std::vector<std::string> split(const std::string& s, const char delim);

}; // class String
} // namespace tools
} // namespace edsdca

#endif // __EDSDCA_STRING_H
