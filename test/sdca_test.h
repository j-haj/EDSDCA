
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "edsdca/models/sdca.h"

namespace edsdca_tests {

class SdcaTest : public ::testing::Test {
  
  protected:
    virtual void SetUp() {
      // Load data
      auto loader = edsdca::tools::CsvLoader("data/test_data_3d.csv");

      loader.LoadData(3, 0);
      Eigen::MatrxXd features = loader.features();
      Eigen::VectorXd labels = loader.labels();

      // Create Sdca instance
      sdca = edsdca::models::Sdcas(0);
    }

    edsdca::model::Sdca sdca;

};

TEST_F(SdcaTest, LoadData) {

}

TEST(EDSDCASolverTest, 1DTest) {
  // Create a toy problem where the feature space is one-dimensional such that
  // all data to the left or -1 is negatively labeled and all data to the
  // right of -1 is positively labeled

  Eigen::MatrixXd X = 
  
}

} // namespace edsdca_tests