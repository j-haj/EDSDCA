#ifndef __EDSDCA_CSVPARSER_H
#define __EDSDCA_CSVPARSER_H

#include <array>
#include <vector>
#include <string>
#include <tuple>
#include <utility>

#include <Eigen/Dense>

#include "edsdca/edsdca.h"
#include "edsdca/tools/string.h"

namespace edsdca {
namespace tools {

class CsvParser {
  public:

    /**
     * Parses the given data, returning a @p std::pair that contains the labels
     * and features @p (labels, features)
     *
     * @param data data to be parsed
     * @param label_pos @p long representing the index of the label within the
     *                  data
     * @param num_features @p long representing the number of expected features
     *
     * @return @p std::pair of label data and feature data, where the label data
     *          is the first element of the pair and the feature data is the
     *          second
     */
    static std::pair<Eigen::VectorXd, Eigen::MatrixXd> Parse(const std::string& data,
        const long label_pos, const long num_features);

    /**
     * Parses the given data, returning a @p std::pair that contains the labels
     * and features
     *
     * @param data data to be parsed
     * @param label_pos @p long representing the index of the label within the data
     * @param num_features @p long representing the expected number of features
     *
     * @return @p std::pair of label data and feature data, where the label data
     *        is the first element of the pair and the feature data is the
     *        second element of the pair.
     */
     /*
    static std::pair<std::array<double>, std::array<std::array<double>>> ParseToStdArr(
      const std::string& data, const long label_pos, const long num_features);
  */
  private:
    std::string unparsed_data_;

}; // class CsvParser
} // namespace tools
} // namespace edsdca


#endif // __EDSDCA_CSVPARSER_H
