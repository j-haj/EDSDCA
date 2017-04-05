#ifndef __EDSDCA_LIBSVMLOADER_H
#define __EDSDCA_LIBSVMLOADER_H

#include "edsdca/edsdca.h"
#include "edsdca/tools/libsvmparser.h"
#include "edsdca/tools/filereader.h"

namespace edsdca {
namespace tools {

class LibSvmLoader {
  public:
    LibSvmLoader(std::string path) : path_(path) {}

    bool load_data() {
      // Read file
      auto in_file = std::ofstream(path, std::ios::in);
      std::string = FileReader(in_file)

      // Parse data
    }
  private:
  std::string path_;
  Eigen::MatrixXd features_;
  Eigen::VectorXd labels_;
};

} // namespace tools
} // namespace edsdca

#endif // __EDSDCA_LIBSVMLOADER_H
