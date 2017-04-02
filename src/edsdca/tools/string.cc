
#include <string>
#include <sstream>
#include <vector>

#include "edsdca.h"

namespace edsdca {
namespace tools {
class String {

  public:

    static std::vector<std::string> split(const std::string& s, const char delim) {
      std::vector<std::string> result;
      std::stringstream ss(s);
      while (!ss.eof()) {
        std::string field;
        std::getline(ss, field, delimiter);
        result.push_back(field);
      }
      return result;
    }
  }; // class String
} // namespace tools
} // namespace edsdca
