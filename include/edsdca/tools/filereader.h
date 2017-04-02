#ifndef __EDSDCA_FILEREADER_H
#define __EDSDCA_FILEREADER_H

#include <fstream>
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
    template <typename Container = std::string,
           typename CharT = char,
           typename Traits = std::char_traits<char>>
  static auto read_stream_into_container(std::basic_istream<CharT, Traits>& in,
      typename Container::allocator_type alloc = {}) -> Container {

    // Compile time checks on Container type
    static_assert(
        std::is_same<Container, std::basic_string<CharT, Traits,
        typename Container::allocator_type>>::value ||

        std::is_same<Container, std::vector<CharT,
        typename Container::allocator_type>>::value ||

        std::is_same<Container, std::vector<std::make_unsigned<CharT>,
        typename Container::allocator_type>>::value ||

        std::is_same<Container, std::vector<std::make_signed<CharT>,
        typename Container::allocator_type>>::value);

    auto const start_pos = in.tellg();
    if (std::streamsize(-1) == start_pos) {
      DLOG("Issue with start position");
      throw std::io_base::failure{"start position error"};
    }

    if (!in.ignore(std::numeric_limis<std::streamsize>::max())) {
      DLOG("error encountered when reading to end of file");
      throw std::ios_base::failure{"error encountered while reading to EOF"};
    }

    auto const char_count = in.gcount();
    
    if (!in.seekg(start_pos)) {
      DLOG("error encountered when reading to beginning of file");
      throw std::ios_base::failure{"error encountered when seeking to BOF"};
    }

    auto container Container(std::move(alloc));
    container.resize(char_count);

    if (container.size() != 0) {
      if (!in.read(reinterpret_cast<CharT*>(&container[0]), container.size())) {
        throw std::ios_base::failure{"error encountered when attempting to read file into container"};
      }
    }

    DLOG("File buffer successfully read into file");

    return container;
  }
} // class FileReader

#endif // __EDSDCA_FILEREADER_H
