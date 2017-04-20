

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "edsdca/util/math_utils.h"

namespace math_utils_tests {

TEST(NormSquared, Cpu) {
  auto eigen_x = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  // Norm squared should be 55
  EXPECT_DOUBLE_EQ(NormSquared_cpu(eigen_x), 55);

  std::vector<double> std_x{1, 2, 3, 4, 5};
  EXPECT_DOUBLE_EQ(NormSquared_cpu(std_x), 55);
}

TEST(NormSquared, CpuOrGpu) {
  auto eigen_x = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  // Norm squared should be 55
  EXPECT_DOUBLE_EQ(NormSquared(eigen_x), 55);

  std::vector<double> std_x{1, 2, 3, 4, 5};
  EXPECT_DOUBLE_EQ(NormSquared(std_x), 55);
}

TEST(VectorDotProd, Cpu) {
  auto eigen_x = Eigen::VectorXd(5);
  auto eigen_y = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  eigen_y << 2, 3, 4, 5, 6;
  EXPECT_DOUBLE_EQ(VectorDotProd_cpu(eigen_x, eigen_y), 70);

  std::vector<double> std_x{1, 2, 3, 4, 5};
  std::vector<double> std_y{2, 3, 4, 5, 6};
  EXPECT_DOUBLE_EQ(VectorDotProd_cpu(std_x, std_y), 70);

}

TEST(VectorDotProd, CpuOrGpu) {
  auto eigen_x = Eigen::VectorXd(5);
  auto eigen_y = Eigen::VectorXd(5);
  eigen_x << 1, 2, 3, 4, 5;
  eigen_y << 2, 3, 4, 5, 6;
  EXPECT_DOUBLE_EQ(VectorDotProd(eigen_x, eigen_y), 70);

  std::vector<double> std_x{1, 2, 3, 4, 5};
  std::vector<double> std_y{2, 3, 4, 5, 6};
  EXPECT_DOUBLE_EQ(VectorDotProd(std_x, std_y), 70);

}

} // namespace math_utils_tests
