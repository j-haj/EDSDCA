

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "edsdca/util/math_utils.h"

namespace math_utils_tests {

TEST(EigenMathUtilities, NormSquaredCPU) {
  auto x = Eigen::VectorXd(5);
  x << 1, 2, 3, 4, 5;
  // Norm squared should be 55
  EXPECT_EQ(NormSquared_cpu(x), 55);
}

TEST(StdMathUtilities, NormSquaredCPU) {
  std::vector<double> x{1, 2, 3, 4, 5};
  EXPECT_EQ(NormSquared_cpu(x), 55);
}

int main(int* argc, char* argv[]) {
  ::testing::InitGoogleTest(argc, argv);
  return RUN_ALL_TESTS();
}
} // namespace math_utils_tests
