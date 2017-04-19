#include "edsdca/tools/csvloader.h"

#include <Eigen/Dense>

int main() {
  std::cout << "Opening file...\n";
  auto loader = edsdca::tools::CsvLoader("../data/australian_scale");
  loader.LoadData(14l, 0l);
  std::cout << "File open. Reading file into container...\n";
  std::cout << "File loaded. Printing file...\n";

  auto features = loader.features();
  if (std::is_same<decltype(features), Eigen::MatrixXd>::value) {
    std::cout << "features is of type Eigen::MatrixXd\n";
  }
  std::cout << "Dimensions of data: " << features.rows() << "x" << features.cols() << std::endl;
  return 0;
}
