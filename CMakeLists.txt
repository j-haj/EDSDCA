cmake_minimum_required(VERSION 2.8.9)
project(edsdca)

# If CUDA compiler found, ``CUDA_FOUND`` will be set to ``true``
find_package(CUDA)

# Try to find OpenMP. Similarly to CUDA, if found, ``OPENMP_FOUND`` will be set
# to ``true``
find_package(OpenMP)

message("\n===== PROJECT INFO ====\nCS5430 Project\nAuthors: Jeff Hajewski and Mingrui Liu\n===== END INFO =====\n\n")

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")

# Add headers
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(SYSTEM "${SRC_DIR}/eigen/")

# Add source files
set(SOURCES 
            ${SRC_DIR}/edsdca/tools/string.cc
            ${SRC_DIR}/edsdca/tools/csvparser.cc
            ${SRC_DIR}/edsdca/tools/csvloader.cc
            ${SRC_DIR}/edsdca/tools/filereader.cc)
set(HEADERS 
            include/ptags/ptags.h
            include/dlog/logging.h
            include/edsdca/tools/filereader.h
            include/edsdca/tools/string.h
            include/edsdca/tools/libsvmparser.h
            include/edsdca/tools/csvparser.h
            include/edsdca/tools/csvloader.h
            include/edsdca/util/datetime.h)

add_executable(edsdca_main ${SRC_DIR}/edsdca/run_model.cc ${SOURCES} ${HEADERS})
set_property(TARGET edsdca_main PROPERTY CXX_STANDARD 14)
