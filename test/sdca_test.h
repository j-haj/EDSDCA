#ifndef __TEST_SDCA_TEST_H
#define __TEST_SDCA_TEST_H

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "edsdca/tools/csvloader.h"
#include "edsdca/models/sdca.h"

class SdcaTest : public ::testing::Test {
  
  protected:
    SdcaTest() : sdca_(edsdca::models::Sdca(0)) {}

    void SetUp() {
      // Load data
      auto loader = edsdca::tools::CsvLoader("../test/data/test_data_3d.csv");

      loader.LoadData(3, 0);
      Eigen::MatrixXd features = loader.features();
      Eigen::VectorXd labels = loader.labels();

      // Fit the model
      sdca_.Fit(features, labels);      

    }

    edsdca::models::Sdca sdca_;

};

TEST_F(SdcaTest, CheckLambda) {
  ASSERT_EQ(sdca_.lambda(), 0);
}


#endif