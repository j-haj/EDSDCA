cmake_minimum_required(VERSION 2.8.9)
project(edsdca)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(TEST_DIR "${PROJECT_SOURCE_DIR}/test")
set(GTEST_ROOW "${SRC_DIR}/googletest")
set(CMAKE_CXX_STANDARD 11)

# If CUDA compiler found, ``CUDA_FOUND`` will be set to ``true``
find_package(CUDA)

# Try to find OpenMP. Similarly to CUDA, if found, ``OPENMP_FOUND`` will be set
# to ``true``
find_package(OpenMP)

# Look for googletest
enable_testing()
find_package(GTest REQUIRED)

message("\n===== PROJECT INFO ====\nCS5430 Project\nAuthors: Jeff Hajewski and Mingrui Liu\n===== END INFO =====\n\n")


# Add headers
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(SYSTEM "${SRC_DIR}/eigen/")
include_directories(${GTEST_INCLUDE_DIRS})

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
            include/edsdca/util/datetime.h
            include/edsdca/util/math_utils.h)

add_executable(edsdca_main ${SRC_DIR}/edsdca/run_model.cc ${SOURCES} ${HEADERS})

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
add_executable(test_math ${TEST_DIR}/math_util_test.cc ${SOURCES} ${HEADERS})
target_link_libraries(test_math ${GTEST_BOTH_LIBRARIES})

add_executable(edsdca_read ${SRC_DIR}/read_test.cc ${SOURCES} ${HEADERS})

