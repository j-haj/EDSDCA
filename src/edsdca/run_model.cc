/**
 * Model base file. All runs start here
 */
#include <iostream>

#include "edsdca/edsdca.h"

int main(int argc, char* argv[]) {
  dlog::DebugLogger::InitDebugLogging();
  ptags::Ptag::InitPTags();
  DLOG("Running models...");


  return 0;
}
