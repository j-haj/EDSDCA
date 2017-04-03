
#include "edsdca/tools/string.h"

std::vector<std::string> edsdca::tools::String::split(const std::string& s,
                                                      const char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  while (!ss.eof()) {
    std::string field;
    std::getline(ss, field, delim);
    result.push_back(field);
  }
  return result;                                                  
}

