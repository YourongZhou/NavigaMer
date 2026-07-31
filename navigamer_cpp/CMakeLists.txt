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

set(NAVIGAMER_EDLIB_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/edlib/src/edlib.cpp
)
set(NAVIGAMER_EDLIB_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/edlib/include
)

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
  src/index_persistence.cpp
  src/sharded_index.cpp
  src/shard_router.cpp
  src/candidate_verifier.cpp
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
add_executable(test_bounded_myers
  src/test_bounded_myers.cpp
  src/structure.cpp
  src/tools.cpp
)
add_executable(test_range_join
  src/test_range_join.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
)
add_executable(test_phase1_seed_index
  src/test_phase1_seed_index.cpp
  src/phase1_seed_index.cpp
  src/structure.cpp
  src/tools.cpp
)
add_executable(test_phase2_distance_verifier
  src/test_phase2_distance_verifier.cpp
  src/phase2_distance_verifier.cpp
  src/structure.cpp
  src/tools.cpp
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
add_executable(test_build_distance_mode
  src/test_build_distance_mode.cpp
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
add_executable(test_epoch_visited
  src/test_epoch_visited.cpp
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
add_executable(test_search_graph_view
  src/test_search_graph_view.cpp
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
add_executable(test_search_distance_mode
  src/test_search_distance_mode.cpp
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
add_executable(test_build_timing_stats
  src/test_build_timing_stats.cpp
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
add_executable(test_build_scale_smoke
  src/test_build_scale_smoke.cpp
)
add_executable(test_build_progress
  src/test_build_progress.cpp
  src/build_progress.cpp
)

foreach(target navigamer test_recall test_distance_bound test_hierarchy_config test_search_stats_bin test_map150_recall test_build_range_equivalence test_build_distance_mode test_mbb_filter_equivalence test_search_qgram_prefilter test_query_benchmark_gate test_epoch_visited test_search_graph_view test_search_distance_mode test_build_timing_stats)
  target_sources(${target} PRIVATE src/simd_mbb_filter.cpp)
  target_sources(${target} PRIVATE src/phase1_seed_index.cpp)
  target_sources(${target} PRIVATE src/phase2_distance_verifier.cpp)
  target_sources(${target} PRIVATE src/build_progress.cpp)
endforeach()

foreach(target test_build_distance_mode test_query_benchmark_gate test_epoch_visited test_search_graph_view test_build_timing_stats)
  target_sources(${target} PRIVATE src/index_persistence.cpp)
endforeach()

add_executable(test_simd_mbb_filter
  src/test_simd_mbb_filter.cpp
  src/simd_mbb_filter.cpp
)

add_executable(test_simd_leaf_beacon_filter
  src/test_simd_leaf_beacon_filter.cpp
  src/simd_mbb_filter.cpp
)

add_executable(test_index_persistence
  src/test_index_persistence.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/simd_mbb_filter.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
  src/query_benchmark.cpp
  src/index_persistence.cpp
  src/phase1_seed_index.cpp
  src/phase2_distance_verifier.cpp
  src/build_progress.cpp
)

add_executable(test_sharded_index
  src/test_sharded_index.cpp
  src/structure.cpp
  src/tools.cpp
  src/qgram_filter.cpp
  src/range_join.cpp
  src/mbb_rect_index.cpp
  src/simd_mbb_filter.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
  src/experiment_utils.cpp
  src/map150.cpp
  src/query_benchmark.cpp
  src/index_persistence.cpp
  src/sharded_index.cpp
  src/shard_router.cpp
  src/phase1_seed_index.cpp
  src/phase2_distance_verifier.cpp
  src/build_progress.cpp
)

foreach(target navigamer test_recall test_distance_bound test_hierarchy_config test_search_stats_bin test_map150_recall test_bounded_edit_distance test_bounded_myers test_qgram_filter test_range_join test_phase1_seed_index test_build_range_equivalence test_build_distance_mode test_mbb_filter_equivalence test_search_qgram_prefilter test_query_benchmark_gate test_epoch_visited test_search_graph_view test_search_distance_mode test_build_timing_stats test_index_persistence test_sharded_index)
  target_sources(${target} PRIVATE ${NAVIGAMER_EDLIB_SOURCE})
  target_include_directories(${target} PRIVATE ${NAVIGAMER_EDLIB_INCLUDE})
endforeach()

target_sources(test_phase2_distance_verifier PRIVATE ${NAVIGAMER_EDLIB_SOURCE})
target_include_directories(test_phase2_distance_verifier PRIVATE ${NAVIGAMER_EDLIB_INCLUDE})

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
target_include_directories(test_bounded_myers PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_bounded_myers PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_range_join PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_range_join PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_phase1_seed_index PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_phase1_seed_index PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_phase2_distance_verifier PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_phase2_distance_verifier PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_qgram_filter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_qgram_filter PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_build_range_equivalence PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_build_range_equivalence PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_build_distance_mode PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_build_distance_mode PRIVATE OpenMP::OpenMP_CXX)
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
target_include_directories(test_epoch_visited PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_epoch_visited PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_search_graph_view PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_search_graph_view PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_search_distance_mode PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_search_distance_mode PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_build_timing_stats PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_build_timing_stats PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_build_scale_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_build_scale_smoke PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_build_progress PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_build_progress PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_hierarchy_config PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_hierarchy_config PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_simd_mbb_filter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_simd_mbb_filter PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_simd_leaf_beacon_filter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_simd_leaf_beacon_filter PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_index_persistence PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_index_persistence PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_sharded_index PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_sharded_index PRIVATE OpenMP::OpenMP_CXX)

if(NAVIGAMER_WITH_SEQAN3)
  foreach(target navigamer test_recall test_distance_bound test_hierarchy_config test_search_stats_bin test_map150_recall test_bounded_edit_distance test_bounded_myers test_qgram_filter test_range_join test_phase1_seed_index test_phase2_distance_verifier test_build_range_equivalence test_build_distance_mode test_mbb_rect_index test_mbb_filter_equivalence test_search_qgram_prefilter test_query_benchmark_gate test_epoch_visited test_search_graph_view test_search_distance_mode test_build_timing_stats test_build_scale_smoke test_simd_mbb_filter test_simd_leaf_beacon_filter test_index_persistence test_sharded_index)
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
  target_compile_options(test_bounded_myers PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_qgram_filter PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_range_join PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_phase1_seed_index PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_phase2_distance_verifier PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_build_range_equivalence PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_build_distance_mode PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_mbb_rect_index PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_mbb_filter_equivalence PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_search_qgram_prefilter PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_query_benchmark_gate PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_epoch_visited PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_search_graph_view PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_search_distance_mode PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_build_timing_stats PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_build_scale_smoke PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_build_progress PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_simd_mbb_filter PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_simd_leaf_beacon_filter PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_index_persistence PRIVATE -O3 -DNDEBUG)
  target_compile_options(test_sharded_index PRIVATE -O3 -DNDEBUG)
endif()
