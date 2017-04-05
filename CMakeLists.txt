cmake_minimum_required(VERSION 2.8.9)
project(edsdca)

# If CUDA compiler found, ``CUDA_FOUND`` will be set to ``true``
find_package(CUDA)

message("\n===== PROJECT INFO ====\nCS5430 Project\nAuthors: Jeff Hajewski and Mingrui Liu\n===== END INFO =====\n\n")

# Add headers
include_directories("${PROJECT_SOURCE_DIR}/include")

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")

# Add source files
set(SOURCES 
            ${SRC_DIR}/edsdca/tools/string.cc)
set(HEADERS 
            include/ptags/ptags.h
            include/dlog/logging.h
            include/edsdca/tools/filereader.h
            include/edsdca/tools/string.h
            include/edsdca/util/datetime.h)

add_executable(edsdca_main ${SRC_DIR}/edsdca/run_model.cc ${SOURCES} ${HEADERS})
set_property(TARGET edsdca_main PROPERTY CXX_STANDARD 14)