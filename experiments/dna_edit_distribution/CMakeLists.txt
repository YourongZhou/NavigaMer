cmake_minimum_required(VERSION 3.16)
project(DnaEditDistribution LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# WFA2's optional internal OpenMP mode is not needed: the experiment owns the
# outer parallel loop and gives each worker an independent aligner.
set(OPENMP OFF CACHE BOOL "Enable WFA2 internal OpenMP" FORCE)
set(PROFILING OFF CACHE BOOL "Enable WFA2 profiling" FORCE)
set(ASAN OFF CACHE BOOL "Enable WFA2 ASAN" FORCE)
set(EXTRA_FLAGS OFF CACHE BOOL "Enable WFA2 extra flags" FORCE)

FetchContent_Declare(
  wfa2lib
  GIT_REPOSITORY https://github.com/smarco/WFA2-lib.git
  GIT_TAG 0db345a8fe862fd7873d3354c499da385583a65a
  GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(wfa2lib)

set_target_properties(
  wfa2 wfa2cpp wfa2cpp_static align_benchmark
  PROPERTIES EXCLUDE_FROM_ALL TRUE
)

if(TEST wfa2lib)
  set_tests_properties(wfa2lib PROPERTIES DISABLED TRUE)
endif()

find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

enable_testing()

add_library(dna_edit_distribution_core src/core.cpp src/report.cpp)
target_include_directories(
  dna_edit_distribution_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(
  dna_edit_distribution_core
  PUBLIC wfa2::wfa2_static OpenMP::OpenMP_CXX
)
target_compile_options(
  dna_edit_distribution_core
  PRIVATE -Wall -Wextra -Wpedantic
)
target_compile_definitions(
  dna_edit_distribution_core
  PRIVATE
    DNA_EDIT_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    DNA_EDIT_WFA2_VERSION="2.3.6"
    DNA_EDIT_WFA2_COMMIT="0db345a8fe862fd7873d3354c499da385583a65a"
)

add_executable(dna_edit_distribution src/main.cpp)
target_link_libraries(dna_edit_distribution PRIVATE dna_edit_distribution_core)
target_compile_options(
  dna_edit_distribution
  PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(dna_edit_distribution_tests tests/test_core.cpp)
target_link_libraries(
  dna_edit_distribution_tests
  PRIVATE dna_edit_distribution_core
)
target_compile_options(
  dna_edit_distribution_tests
  PRIVATE -Wall -Wextra -Wpedantic
)

add_test(NAME dna_edit_distribution_tests COMMAND dna_edit_distribution_tests)
add_test(
  NAME dna_edit_distribution_cli
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cli.py
    --binary ${CMAKE_BINARY_DIR}/dna_edit_distribution
)
