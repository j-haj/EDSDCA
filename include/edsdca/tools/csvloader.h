#ifndef __EDSDCA_CSVLOADER_H
#define __EDSDCA_CSVLOADER_H

#include <string>

#include <Eigen/Dense>

#include "edsdca/edsdca.h"
#include "edsdca/tools/filereader.h"

namespace edsdca {
namespace tools {

class CsvLoader {
  public:
    CsvLoader(std::string path) : path_(path) {}

    bool load_data(const long num_features, const long label_pos);
    
    Eigen::MatrixXd features() {
      return this->features_;
    }

    Eigen::VectorXd labels() {
      return this->labels_;
    }

    void set_data_path(const std::string& path) {
      path_ = path;
    }
  private:
    std::string path_;
    Eigen::MatrixXd features_;
    Eigen::VectorXd labels_;

} // namespace tools
} // namespace edsdca
