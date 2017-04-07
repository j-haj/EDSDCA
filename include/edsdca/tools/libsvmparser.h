#ifndef __EDSDCA_LIBSVMPARSER_H
#define __EDSDCA_LIBSVMPARSER_H

#include <pair>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "edsdca/edsdca.h"
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
// TODO: Class is not yet complete!
  public:
  LibSvmParser(LibSvmDataSet data_set, Container& data) :
    data_set_(data_set), unparsed_data_(data) {
    num_features_ = LibSvmParser::GetNumberFeatures(data_set);
  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> Parse() {
    // First grab new lines and initialize @p labels_ and @p features_
    auto lines = String::split(this->data_, "\n");
    long num_data = lines.size();
    labels= Eigen::VectorXf(num_data);
    features = Eigen::MatrixXd::Zero(this->num_features_, num_data);

    // For each line, split line on white space
    long row_num = 0;
    for (auto& line : lines) {
      std::vector<std::string> split_row = String::split(line, " ");
      
      bool is_label = true;
      // Iterate over each element of the line, where the first element is the
      // label
      for (auto& element : split_row) {
        if (is_label) {
          // Cast and add the label
          labels << String::cast_to_type<float>(element);
          is_label = false;
        } else {
          // Parse the feature data -- first element is index+1, second element
          // is feature value
          std::vector<std::string> split_element = 
            String::splitline(element, ":");
          long f_idx = String::cast_to_type<long>(split_element[0]) - 1;
          auto feature = String::cast_to_type<double>(split_element[1]);
          features(row_num, f_idx) = feature;
        } 
      }

      // Increment row index as we move to the next row
      ++row_num;
    }
    return std::pair<Eigen::VectorXd, Eigen::MatrixXd>(labels, features);
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
