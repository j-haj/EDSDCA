/**
 * Model base file. All runs start here
 */
#include <iostream>

#include "edsdca/edsdca.h"
#include "edsdca/tools/csvloader.h"
#include "edsdca/models/sdca.h"

int main(int argc, char* argv[]) {
#ifdef CPU
  std::cout << "CPU MODE\n";
#else
  std::cout << "GPU MODE\n";
#endif

  dlog::DebugLogger::InitDebugLogging();
  ptags::Ptag::InitPTags();
  DLOG("Running models...");

  DLOG("Attempting to instantiate an instance of CsvLoader...");
  auto loader = edsdca::tools::CsvLoader("../test/data/large_test.csv");
  DLOG("CsvLoader instantiated! Attempting to load data....");
  loader.LoadData(3, 0);
  DLOG("Data loaded! Attempting to print to std::cout...");
  edsdca::models::Sdca sdca = edsdca::models::Sdca(1);
  sdca.Fit(loader.features(), loader.labels());
  return 0;
}
