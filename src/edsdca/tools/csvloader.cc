
#include "edsdca/tools/csvloader.h"


void edsdca::tools::CsvLoader::LoadData(const long num_features,
    const long label_pos) {
    
  // First read in the file
  std::ifstream ifile;
  ifile.open(this->path_, std::ios::in);
  std::string unparsed_data = edsdca::tools::FileReader::ReadStreamIntoContainer(ifile);

  // Once the file is read, parse it
  std::tie(this->labels_, this->features_) = edsdca::tools::CsvParser::Parse(unparsed_data,
      label_pos, num_features);
  ifile.close();
}

void edsdca::tools::CsvLoader::LoadDataToArray(const long num_features,
    const long label_pos) {
    // Read in file
    std::ifstream ifile;
    ifile.open(this->path_, std::ios::in);
    std::string unparsed_data = edsdca::tools::FileReader::ReadStreamIntoContainer(ifile);

    /*
    // File is loaded, parse
    std::tie(this->feature_arr_, this->label_arr_) = 
        edsdca::tools::CsvParser::ParseToStdArr(unparsed_data, label_pos, num_features);
        
    */
  }
