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

  /*
  DLOG("Attempting to instantiate an instance of CsvLoader for australian_scale data...");
  auto loader = edsdca::tools::CsvLoader("../data/australian_scale.csv");
  DLOG("CsvLoader instantiated! Attempting to load data....");
  loader.LoadData(14, 0);
  DLOG("Data loaded! Attempting to fit model for lamba=1000");
  edsdca::models::Sdca sdca = edsdca::models::Sdca(1000);
  sdca.Fit(loader.features(), loader.labels());
  */
  DLOG("Attempting to instantiate an instance of CsvLoader for news20.binary...");
  auto loader_news = edsdca::tools::CsvLoader("../data/news20.binary.csv");
  DLOG("CsvLoader instantiated! Attempting to load data...");
  loader_news.LoadData(1355191, 0);
  DLOG("Data loaded! Attempting to fit model for lambda=1000");
  edsdca::models::Sdca sdca2 = edsdca::models::Sdca(1000);
  sdca2.Fit(loader_news.features(), loader_news.labels());

  return 0;
}
