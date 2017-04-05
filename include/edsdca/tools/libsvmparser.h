#ifndef __EDSDCA_LIBSVMPARSER_H
#define __EDSDCA_LIBSVMPARSER_H

#include <pair>
#include <string>
#include <vector>

#include "edsdca/tools/string.h"

namespace edsdca {
namespace tools {

enum LibSvmDataSet {
  CovTypeBinary,
  CovTypeBinaryScale
}

/**
 * @p LibSvmParser class handles the parsing of libsvm files, which have the
 * format
 *  <label> <index>:<feature val> ... <index>:<veature val>
 *
 * To correctly parse a libsvm file it is necessary to know the correct number
 * of features for the given dataset. If this isn't known, we have no guarantee
 * of the size of the feature space, as this isn't encoded in the libsvm file
 * format.
 *
 * NOTE: indices start at 1
 *
 * The @p Container type is the type of the container holding the data that is
 * passed to the constructor (via reference). @p Dtype is the type of the 
 * feature data, and @p Ltype is the type of the labels. It is assumed that
 * feature data is of type @p float and labels are of type @p int
 */
template <typename Container = std::string,
          // typename Dtype = float,
          typename Ltype = int>
class LibSvmParser {

  public:
  LibSvmParser(LibSvmDataSet data_set, Container& data) :
    data_set_(data_set), data_(data) {
    num_features_ = LibSvmParser::GetNumberFeatures(data_set);
  }

  std::pair<std::vector<Ltype>, std::vector<Dtype>> parse() {
    // First grab new lines
    auto lines = edsdca::tools::String::split(this->data_, "\n");
    long num_data = lines.size();

    // For each line, split line on white space
    for (auto& line : lines) {

    }
  }

  private:
  /**
   * Name of the data set. The names are of type @p LibSvmDataSet enum, which 
   * contains the available data sets.
   */
  LibSvmDataSet data_set_;

  /**
   * Number of features for the given @p data_set_
   */
  long num_features_;

  /**
   * Unparsed data
   */
  Container unparsed_data_;

  /**
   * Label data
   */
  std::vector<Ltype> labels_;

  /**
   * Feature data
   */
  std::vector<Dtype> features_;

  /**
   * Gets the number of features for the selected data set
   *
   * @return feature size for specified data set
   */
  static long GetNumberFeatures(LibSvmDataSet ds) {
    switch (ds) {
      case CovTypeBinary:
        DLOG("Loading CovTypeBinary\n");
        return 54;
      case CovTypeBinaryScale:
        DLOG("Loading CovTypeBinaryScale\n");
        return 54;
      default:
        DLOG("Failed to find data set\n")
        break;
    }
  }

};

} // namespace tools
} // namespace edsdca
#endif // __EDSDCA_LIBSVMPARSER_H
