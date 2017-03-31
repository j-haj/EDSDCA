#ifndef __EDSDCS_LOGGING_H
#define __EDSDCA_LOGGING_H

/**
 * This is a lightweight debug logging tool. All logging functionality is
 * removed when compiled without DEBUG defined. In other words, for logging to
 * appear, the application must be build with the -DDEBUG flag
 *
 * In addition to console logging, a unique log file is created each run of the
 * parent application. THe default name for this file is
 *    "debug_log_<timestamp>.log"
 *
 * Note: this class is not synced with stdio.
 */

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <string>

#include "util/datetime.h"

#define DLOG(msg) {dlog::DebugLogger::log(__func__, msg);}

namespace dlog {


class DebugLogger {


  public:
    static std::ofstream log_file;
   
    // TODO: Add elapsed time to output
    
    /**
     * Initialize the Dlog tooling. This method simply creates a timestamped log
     * file using the given base filename and extension.
     *
     * @param log_file_name base name for the log file. Default value is
     *                      ``debug_log``
     * @param extension extension used for the log file. Default value is
     *                  ``log``
     */
    static void InitDebugLogging(const std::string log_file_name="debug_log",
                                 const std::string extension=".log") {
      std::string timestamp = util::date_time::FormattedTime::now();
      std::string filename = log_file_name + "_" + timestamp + extension;
      log_file.open(filename, std::ios::out);
      DLOG("Dlog initialization complete!");
    }

    static void log(const char* f, const std::string msg) {
#ifdef DEBUG
      std::string timestamp = util::date_time::FormattedTime::now();
      std::string function_name = "[" + std::string(f) + "]";
      std::stringstream output(timestamp);
      output << timestamp << " [DEBUG] " 
             << std::setw(15) << std::setfill(' ') << std::left 
             << function_name << "\t" << msg << "\n";
      std::cout << output.str();
      log_file << output.str();
#endif
    }

  private:
    DebugLogger() {}
};

std::ofstream dlog::DebugLogger::log_file;

}

#endif // __EDSDCA_LOGGING_H

