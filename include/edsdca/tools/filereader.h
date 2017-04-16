#ifndef __EDSDCA_FILEREADER_H
#define __EDSDCA_FILEREADER_H

#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "edsdca/edsdca.h"

namespace edsdca {
namespace tools {
class FileReader {

  public:
    /**
     * Reads the data from @p in into a specified @p Container (in memory).
     * This should not be used for large files (<1GB), since the files are read into
     * memory.
     */
    template <typename Container = std::string,
           typename CharT = char,
           typename Traits = std::char_traits<char>>
    static auto ReadStreamIntoContainer(std::basic_istream<CharT, Traits>& in,
        typename Container::allocator_type alloc = {}) -> Container {

      // Compile time checks on Container type
      // ----------------------------------------------------------------------
      static_assert(
          std::is_same<Container, std::basic_string<CharT, Traits,
          typename Container::allocator_type>>::value ||

          std::is_same<Container, std::vector<CharT,
          typename Container::allocator_type>>::value ||

          std::is_same<Container, std::vector<std::make_unsigned<CharT>,
          typename Container::allocator_type>>::value ||

          std::is_same<Container, std::vector<std::make_signed<CharT>,
          typename Container::allocator_type>>::value, "Static assertion failed -- bad allocator type");

      // Get size of file
      // ----------------------------------------------------------------------
      auto const start_pos = in.tellg();
      if (std::streamsize(-1) == start_pos) {
        DLOG("Issue with start position");
        throw std::ios_base::failure{"start position error"};
      }

      if (!in.ignore(std::numeric_limits<std::streamsize>::max())) {
        DLOG("error encountered when reading to end of file");
        throw std::ios_base::failure{"error encountered while reading to EOF"};
      }

      auto const char_count = in.gcount();

      // Go back to the beginning of the file
      if (!in.seekg(start_pos)) {
        DLOG("error encountered when reading to beginning of file");
        throw std::ios_base::failure{"error encountered when seeking to BOF"};
      }

      // Allocate memory for the data and read in the file
      // ----------------------------------------------------------------------
      auto container = Container(std::move(alloc));
      container.resize(char_count);

      if (container.size() != 0) {
        if (!in.read(reinterpret_cast<CharT*>(&container[0]), container.size())) {
          throw std::ios_base::failure{"error encountered when attempting to read file into container"};
        }
      }

      DLOG("File buffer successfully read into file");

      return container;
    }
}; // class FileReader
} // namespace tools
} // namespace edsdca
#endif // __EDSDCA_FILEREADER_H
