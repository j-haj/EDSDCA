
#include <gtest/gtest.h>

#include "edsdca/edsdca.h"

// Include all test suites here
#include "sdca_test.h"
#include "math_util_test.h"
#include "csv_load_test.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  dlog::DebugLogger::InitDebugLogging();  
  return RUN_ALL_TESTS();
}
