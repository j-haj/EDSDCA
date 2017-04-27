cmake_minimum_required(VERSION 3.0.2)
project(edsdca)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(TEST_DIR "${PROJECT_SOURCE_DIR}/test")
set(GTEST_ROOW "${SRC_DIR}/googletest")
set(CMAKE_CXX_STANDARD 11)

# If CUDA compiler found, ``CUDA_FOUND`` will be set to ``true``
find_package(CUDA)

if (${CUDA_FOUND})
  add_definitions(-DGPU)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -DGPU")
else()
  add_definitions(-DCPU)
endif()

# Add debug flag
add_definitions(-DDEBUG)

# Try to find OpenMP. Similarly to CUDA, if found, ``OPENMP_FOUND`` will be set
# to ``true``
find_package(OpenMP)

# Look for googletest
enable_testing()
find_package(GTest REQUIRED)

# Add headers
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(SYSTEM "${SRC_DIR}/eigen/")
include_directories(${GTEST_INCLUDE_DIRS})
# Add source files
set(SOURCES 
            ${SRC_DIR}/edsdca/tools/string.cc
            ${SRC_DIR}/edsdca/tools/csvparser.cc
            ${SRC_DIR}/edsdca/tools/csvloader.cc
            ${SRC_DIR}/edsdca/tools/filereader.cc
            ${SRC_DIR}/edsdca/tools/timer.cc
            ${SRC_DIR}/edsdca/util/math_utils.cc
            ${SRC_DIR}/edsdca/util/math_utils.cu
            ${SRC_DIR}/edsdca/util/cuda_util.cu
            ${SRC_DIR}/edsdca/memory/memsync.cu
            ${SRC_DIR}/edsdca/models/sdca.cc
            ${SRC_DIR}/edsdca/loss/hingeloss.cc)
set(HEADERS 
            include/ptags/ptags.h
            include/dlog/logging.h
            include/edsdca/tools/filereader.h
            include/edsdca/tools/string.h
            include/edsdca/tools/libsvmparser.h
            include/edsdca/tools/csvparser.h
            include/edsdca/tools/csvloader.h
            include/edsdca/tools/timer.h
            include/edsdca/util/cuda_util.h
            include/edsdca/util/datetime.h
            include/edsdca/util/math_utils.h
            include/edsdca/models/sdca.h
            include/edsdca/memory/memsync.h
            include/edsdca/loss/loss.h
            include/edsdca/loss/hingeloss.h)

set(TEST_HEADERS
            test/math_util_test.h
            test/csv_load_test.h
            test/sdca_test.h)

# -----------------------------------------------------------------------------
# Executable and Linking
# -----------------------------------------------------------------------------

# Executable

#add_executable(test_all ${TEST_DIR}/all_tests.cc ${SOURCES} ${HEADERS} ${TEST_HEADERS})

# Linking
if (${CUDA_FOUND} AND ${OPENMP_FOUND})
  cuda_add_executable(edsdca_main ${SRC_DIR}/edsdca/run_model.cc ${SOURCES} ${HEADERS})
  target_link_libraries(edsdca_main ${OPENMP_LIBRARIES})
  # target_link_libraries(test_all ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES} ${OPENMP_LIBRARIES})
elseif(${CUDA_FOUND})
  cuda_add_executable(edsdca_main ${SRC_DIR}/edsdca/run_model.cc ${SOURCES} ${HEADERS})
  #target_link_libraries(test_all ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES})
else()
  add_executable(edsdca_main ${SRC_DIR}/edsdca/run_model.cc ${SOURCES} ${HEADERS})
  target_link_libraries(test_all ${GTEST_BOTH_LIBRARIES})
endif()


# -----------------------------------------------------------------------------
# Final Options
# -----------------------------------------------------------------------------
message("\n===== PROJECT INFO ====\nCS5430 Project\nAuthors: Jeff Hajewski and Mingrui Liu\n===== END INFO =====\n\n")
