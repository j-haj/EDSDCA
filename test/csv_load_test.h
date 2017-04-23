#ifndef __CSV_LOAD_TEST_H
#define __CSV_LOAD_TEST_H

#include <gtest/gtest.h>

#include "edsdca/tools/csvloader.h"

#define EDSDCA_LOAD_TEST_LE_THRESHOLD 0.00000001

class LoaderTest : public ::testing::Test {

  protected:
    LoaderTest() : loader_(edsdca::tools::CsvLoader("../test/data/test_data_3d.csv")) {}

    void SetUp() {
      loader_.LoadData(3, 0);
    }

    edsdca::tools::CsvLoader loader_;

};

TEST_F(LoaderTest, TestFeatures) {
  Eigen::MatrixXd expected_X(6,3);
  expected_X << 2, 3, 0,
                -3, 8, 2,
                12, -5, 0,
                -2, -6, 0, 
                0, 7, 2,
                10, 2, 2;

  Eigen::MatrixXd actual_X = loader_.features();

  // Check for equality
  ASSERT_EQ(expected_X.rows(), 6);
  ASSERT_EQ(expected_X.cols(), 3);
  ASSERT_EQ(actual_X.rows(), 6);
  ASSERT_EQ(actual_X.cols(), 3);
  ASSERT_DOUBLE_EQ((actual_X - expected_X).norm(), 0);
}

TEST_F(LoaderTest, TestLabels) {
  Eigen::VectorXd expected_y(6);
  expected_y << 1, 0, 1, 1, 0, 0;
   
  Eigen::VectorXd actual_y = loader_.labels();

  // Check for equality
  ASSERT_EQ(expected_y.size(), 6);
  ASSERT_EQ(actual_y.size(), 6);
  ASSERT_DOUBLE_EQ((actual_y - expected_y).norm(), 0);
}

#endif // __CSV_LOAD_TEST_H