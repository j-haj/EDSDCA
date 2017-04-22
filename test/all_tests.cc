
#include <gtest/gtest.h>

// Include all test suites here
#include "math_util_test.h"
#include "csv_load_test.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
