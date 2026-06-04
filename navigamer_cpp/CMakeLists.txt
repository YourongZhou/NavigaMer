cmake_minimum_required(VERSION 3.14)
project(NavigaMer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

# NavigaMer v8 C++ reference implementation.
add_executable(navigamer
  src/main.cpp
  src/structure.cpp
  src/tools.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
)

target_include_directories(navigamer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(navigamer PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_recall
  src/test_recall.cpp
  src/structure.cpp
  src/tools.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
)
target_include_directories(test_recall PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_recall PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_distance_bound
  src/test_distance_bound.cpp
  src/structure.cpp
  src/tools.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
)
target_include_directories(test_distance_bound PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_distance_bound PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_hierarchy_config
  src/test_hierarchy_config.cpp
  src/structure.cpp
  src/tools.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
)
add_executable(test_search_stats_bin
  src/test_search_stats.cpp
  src/structure.cpp
  src/tools.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
)
target_include_directories(test_search_stats_bin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_search_stats_bin PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_hierarchy_config PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_hierarchy_config PRIVATE OpenMP::OpenMP_CXX)

# Release optimization flags.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(navigamer PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_recall PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_distance_bound PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_hierarchy_config PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_search_stats_bin PRIVATE -O3 -DNDEBUG)
endif()
