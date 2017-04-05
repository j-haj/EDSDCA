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
    /**
     * Splits the string @p s into parts based on @p delim and returns them
     * in a @p std::vector
     *
     * @param s @p std::string to be split into componenets based on @p delim
     * @param delim @p const char delimiter used for splitting @p s
     *
     * @return Returns a @p std::vector containing the component @p
     * std::strings of @p s
     */  
    static std::vector<std::string> split(const std::string& s, const char delim);

    /**
     * Converts the given string @p s to type @p T, which is expected to be a
     * numeric type.
     *
     * @param s @p std::string to be converted
     *
     * @return value of type @p T, where @p T should be a numeric type
     */
    template <typename T>
    static T cast_to_type(std::string& s) {
      std::stringstream ss(s);

      T val;
      ss >> val;
      return val;
    }

}; // class String
} // namespace tools
} // namespace edsdca

#endif // __EDSDCA_STRING_H
