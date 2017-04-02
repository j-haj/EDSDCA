#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

template <typename Container = std::string,
          typename CharT = char,
          typename Traits = std::char_traits<char>>
auto read_stream_into_container (
    std::basic_istream<CharT, Traits>& in,
    typename Container::allocator_type alloc = {}) -> Container {

  // Compile time checks
  static_assert(
      std::is_same<Container, std::basic_string<CharT, Traits,
      typename Container::allocator_type>>::value ||
      
      std::is_same<Container, std::vector<CharT,
      typename Container::allocator_type>>::value ||

      std::is_same<Container, std::vector<std::make_unsigned<CharT>,
      typename Container::allocator_type>>::value ||

      std::is_same<Container, std::vector<std::make_signed<CharT>,
      typename Container::allocator_type>>::value);

  // Read in file
  auto const start_pos = in.tellg();
  if (std::streamsize(-1) == start_pos) {
    throw std::ios_base::failure{"error"};
  }

  if (!in.ignore(
        std::numeric_limits<std::streamsize>::max())) {
    throw std::ios_base::failure{"error"};
  }

  auto const char_count = in.gcount();

  if (!in.seekg(start_pos)) {
    throw std::ios_base::failure{"error"};
  }

  auto container = Container(std::move(alloc));
  container.resize(char_count);

  if (0 != container.size()) {
    if (!in.read(reinterpret_cast<CharT*>(&container[0]),
          container.size())) {
        throw std::ios_base::failure{"error"};
    }
  }

  return container;
}

int main() {
  std::cout << "Opening file...\n";
  std::ifstream f;
  f.open("kddb-raw");

  std::cout << "File open. Reading file into container...\n";
  auto c = read_stream_into_container(f);
  std::cout << "File loaded. Printing file...\n";

  //for (auto& x : c) {
  //  std::cout << x;
  //}

  return 0;
}
