#ifndef __CSV_LOAD_TEST_H
#define __CSV_LOAD_TEST_H

#include <gtest/gtest.h>

#include "edsdca/tools/csvloader.h"

#define EDSDCA_LOAD_TEST_LE_THRESHOLD 0.00000001

class LoaderTest : public ::testing::Test {

  protected:
    LoaderTest() : loader_(edsdca::tools::CsvLoader("data/test_data_3d.csv")) {}

    void SetUp() {
      loader_.LoadData(3, 0);
    }

    edsdca::tools::CsvLoader loader_;

};

TEST_F(LoaderTest, TestFeatures) {
  Eigen::MatrixXd expected_X(6,3);
  expected_X << 2, 3, 0, -3, 8, 2, 12, -5, 0, -2, -6, 0, 0, 7, 2, 10, 2, 2;

  Eigen::MatrixXd actual_X = loader_.features();

  // Check for equality
  ASSERT_LE((actual_X - expected_X).norm(), EDSDCA_LOAD_TEST_LE_THRESHOLD);
}

TEST_F(LoaderTest, TestLabels) {
  Eigen::VectorXd expected_y(6);
  expected_y << 1, -1, 1, 1, -1, -1;
  
  Eigen::VectorXd actual_y = loader_.labels();

  // Check for equality
  
  ASSERT_LE((actual_y - expected_y).norm(), EDSDCA_LOAD_TEST_LE_THRESHOLD);
}

#endif // __CSV_LOAD_TEST_H