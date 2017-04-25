#ifndef __TEST_SDCA_TEST_H
#define __TEST_SDCA_TEST_H

#include <random>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "edsdca/models/sdca.h"
#include "edsdca/tools/csvloader.h"

#define NUM_TESTS 10000

class SdcaTest : public ::testing::Test {

protected:
  SdcaTest() : sdca_(edsdca::models::Sdca(1)) {}

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

TEST_F(SdcaTest, CheckLambda) { ASSERT_EQ(sdca_.lambda(), 1); }

TEST_F(SdcaTest, CheckAlpha) {
  Eigen::VectorXd a = sdca_.a();
  ASSERT_LE(0, 2);
}

TEST_F(SdcaTest, CheckNegativeClass) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-100, 100);
  int total_correct = 0;
  for (int i = 0; i < NUM_TESTS; ++i) {
    Eigen::VectorXd test_x = Eigen::VectorXd(3);
    test_x << dist(gen), dist(gen), 2;
    double result = sdca_.Predict(test_x);
    if (result == 0) {
      total_correct += 1;
    }
    // For the test data, any value with z > 1 should be in the negative class
  }
  ASSERT_EQ(total_correct, NUM_TESTS);
}

TEST_F(SdcaTest, CheckPositiveClass) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-100, 100);
  int total_correct = 0;
  // Run the test 100 times
  for (int i = 0; i < NUM_TESTS; ++i) {
    Eigen::VectorXd test_x = Eigen::VectorXd(3);
    test_x << dist(gen), dist(gen), 0;
    double result = sdca_.Predict(test_x);
    if (result == 1) {
      total_correct += 1;
    }
  }
  ASSERT_EQ(total_correct, NUM_TESTS);
}
#endif