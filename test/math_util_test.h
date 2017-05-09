#ifndef __TEST_MATH_UTIL_TEST_H
#define __TEST_MATH_UTIL_TEST_H

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "edsdca/util/math_utils.h"

TEST(NormSquared, Cpu) {
  auto eigen_x = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  // Norm squared should be 55
  EXPECT_DOUBLE_EQ(NormSquared_cpu(eigen_x), 55);
}

TEST(NormSquared, CpuOrGpu) {
  auto eigen_x = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  // Norm squared should be 55
  EXPECT_DOUBLE_EQ(NormSquared(eigen_x), 55);
}

TEST(VectorDotProd, Cpu) {
  auto eigen_x = Eigen::VectorXd(5);
  auto eigen_y = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  eigen_y << 2, 3, 4, 5, 6;
  EXPECT_DOUBLE_EQ(VectorDotProd_cpu(eigen_x, eigen_y), 70);
}

TEST(VectorDotProd, CpuOrGpu) {
  auto eigen_x = Eigen::VectorXd(5);
  auto eigen_y = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  eigen_y << 2, 3, 4, 5, 6;
  EXPECT_DOUBLE_EQ(VectorDotProd(eigen_x, eigen_y), 70);
}

#ifdef GPU
TEST(VectorDotProd, Gpu) {
  auto eigen_x = Eigen::VectorXd(5);
  auto eigen_y = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  eigen_y << 2, 3, 4, 5, 6;
  EXPECT_DOUBLE_EQ(VectorDotProd_gpu(eigen_x, eigen_y), 70);
}

TEST(NormSquared, Gpu) {
  auto eigen_x = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  // Norm squared should be 55
  EXPECT_DOUBLE_EQ(NormSquared_gpu(eigen_x), 55);
}

TEST(MatrixVectorProd, Gpu) {
  auto eigen_vec = Eigen::VectorXd(3);
  eigen_vec << 1, 1, 1;
  
  std::vector<Eigen::VectorXd> eigen_mat(3);
  for (int i = 0; i < 3; ++i) {
    Eigen::VectorXd tmp = Eigen::VectorXd(3);
    tmp << 1, 1, 1;
    eigen_mat[i] = tmp;
  }

  auto expected_result = Eigen::VectorXd(3);
  expected_result << 3, 3, 3;

  auto actual_result = MatrixVectorMultiply(eigen_mat, eigen_vec);

  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(actual_result(i), expected_result(i));
  } 
}
#endif // GPU

#endif // __TEST_MATH_UTIL_TEST_H
