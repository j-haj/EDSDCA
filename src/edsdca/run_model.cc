/**
 * Model base file. All runs start here
 */
#include <iostream>

#include "edsdca/edsdca.h"
#include "edsdca/tools/csvloader.h"

int main(int argc, char* argv[]) {
  dlog::DebugLogger::InitDebugLogging();
  ptags::Ptag::InitPTags();
  DLOG("Running models...");

  DLOG("Attempting to instantiate an instance of CsvLoader...");
  auto loader = edsdca::tools::CsvLoader("/Users/jhaj/Documents/university-of-iowa/classes/s2017/machine-learning/EDSDCA/src/edsdca/testdata.csv");
  DLOG("CsvLoader instantiated! Attempting to load data....");
  loader.LoadData(10, 0);
  DLOG("Data loaded! Attempting to print to std::cout...");
  std::cout << loader.features() << std::endl;

  return 0;
}
