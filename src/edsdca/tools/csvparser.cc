
#include "edsdca/tools/csvparser.h"


std::pair<Eigen::VectorXd, Eigen::MatrixXd> edsdca::tools::CsvParser::Parse(
    const std::string& data,
    const long label_pos,
    const long num_features) {
  // Split data based on newlines
  auto lines = tools::String::split(data, '\n');
  const long num_data = lines.size() - 1;
  Eigen::VectorXd labels = Eigen::VectorXd(num_data);
  Eigen::MatrixXd features = Eigen::MatrixXd::Zero(num_data, num_features);

  // For each line, split on commas
  long row_num = 0;
  for (const auto& line : lines) {
    if (row_num == num_data) {
      break;
    }
    std::vector<std::string> split_row = tools::String::split(line, ',');

    for (long i = 0; i < split_row.size() - 1; ++i) {
      //std::cout << "row: " << row_num << " col: " << i << std::endl;
      double val = tools::String::cast_to_type<double>(split_row[i]);
      if (i == label_pos) {
        labels(row_num) = val;
      } else {
        features(row_num, i) = val;
      }
    }
    ++row_num;
  }
  return std::pair<Eigen::VectorXd, Eigen::MatrixXd>(labels, features);
}
