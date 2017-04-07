#ifndef __EDSDCA_CSVPARSER_H
#define __EDSDCA_CSVPARSER_H

#include <pair>
#include <vector>
#include <string>

#include <Eigen/Dense>

#include "edsdca/edsdca.h"
#include "edsdca/tools/string.h"

namespace edsdca {
namespace tools {

class CsvParser {
  public:

    /**
     * Parses the given data
     *
     */
    static std::pair<Eigen::VectorXd, Eigen::MatrixXd> Parse(const std::string& data,
        const long label_pos, const long num_features);


  private:
    std::string unparsed_data_;

}; // class CsvParser
} // namespace tools
} // namespace edsdca


#endif // __EDSDCA_CSVPARSER_H
