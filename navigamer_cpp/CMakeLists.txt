cmake_minimum_required(VERSION 3.14)
project(NavigaMer LANGUAGES CXX)

option(NAVIGAMER_WITH_SEQAN3 "Enable optional SeqAn3 FM-index locator for map150" OFF)

if(NAVIGAMER_WITH_SEQAN3)
  set(CMAKE_CXX_STANDARD 23)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(OpenMP REQUIRED)
if(NAVIGAMER_WITH_SEQAN3)
  find_package(seqan3 REQUIRED)
endif()

# NavigaMer v8 C++ reference implementation.
add_executable(navigamer
  src/main.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
  src/query_benchmark.cpp
)

target_include_directories(navigamer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(navigamer PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_recall
  src/test_recall.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
target_include_directories(test_recall PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_recall PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_distance_bound
  src/test_distance_bound.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
target_include_directories(test_distance_bound PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_distance_bound PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_hierarchy_config
  src/test_hierarchy_config.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
add_executable(test_search_stats_bin
  src/test_search_stats.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
add_executable(test_map150_recall
  src/test_map150_recall.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
add_executable(test_bounded_edit_distance
  src/test_bounded_edit_distance.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
)
add_executable(test_range_join
  src/test_range_join.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
)
add_executable(test_qgram_filter
  src/test_qgram_filter.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
)
add_executable(test_build_range_equivalence
  src/test_build_range_equivalence.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
add_executable(test_mbb_rect_index
  src/test_mbb_rect_index.cpp
  src/mbb_rect_index.cpp
)
add_executable(test_mbb_filter_equivalence
  src/test_mbb_filter_equivalence.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
add_executable(test_search_qgram_prefilter
  src/test_search_qgram_prefilter.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
add_executable(test_query_benchmark_gate
  src/test_query_benchmark_gate.cpp
  src/query_benchmark.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
)
target_include_directories(test_search_stats_bin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_search_stats_bin PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_map150_recall PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_map150_recall PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_bounded_edit_distance PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_bounded_edit_distance PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_range_join PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_range_join PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_qgram_filter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_qgram_filter PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_build_range_equivalence PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_build_range_equivalence PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_mbb_rect_index PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_mbb_rect_index PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_mbb_filter_equivalence PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_mbb_filter_equivalence PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_search_qgram_prefilter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_search_qgram_prefilter PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_query_benchmark_gate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_query_benchmark_gate PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_hierarchy_config PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_hierarchy_config PRIVATE OpenMP::OpenMP_CXX)

if(NAVIGAMER_WITH_SEQAN3)
  foreach(target navigamer test_recall test_distance_bound test_hierarchy_config test_search_stats_bin test_map150_recall test_bounded_edit_distance test_qgram_filter test_range_join test_build_range_equivalence test_mbb_rect_index test_mbb_filter_equivalence test_search_qgram_prefilter test_query_benchmark_gate)
    target_compile_definitions(${target} PRIVATE NAVIGAMER_WITH_SEQAN3)
    target_link_libraries(${target} PRIVATE seqan3::seqan3)
  endforeach()
endif()

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
  target_compile_options(test_map150_recall PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_bounded_edit_distance PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_qgram_filter PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_range_join PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_build_range_equivalence PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_mbb_rect_index PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_mbb_filter_equivalence PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_search_qgram_prefilter PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_query_benchmark_gate PRIVATE -O3 -DNDEBUG)
endif()
