CXX ?= g++
WITH_SEQAN3 ?= 0
CXXSTD ?= c++17
SEQAN3_FLAGS =
ifeq ($(WITH_SEQAN3),1)
CXXSTD = c++23
SEQAN3_FLAGS = -DNAVIGAMER_WITH_SEQAN3
endif
CXXFLAGS = -std=$(CXXSTD) -O2 -I include -I third_party/edlib/include -Wall -Wextra -fopenmp -MMD -MP $(SEQAN3_FLAGS)
LDLIBS =
EDLIB_SRC = third_party/edlib/src/edlib.cpp
EDLIB_OBJ = $(EDLIB_SRC:.cpp=.o)
EDLIB_DEP = $(EDLIB_OBJ:.o=.d)
SRC = src/structure.cpp src/tools.cpp src/qgram_filter.cpp src/range_join.cpp src/phase1_seed_index.cpp src/phase2_distance_verifier.cpp src/build_progress.cpp src/mbb_rect_index.cpp src/simd_mbb_filter.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp src/experiment_utils.cpp src/map150.cpp src/query_benchmark.cpp src/index_persistence.cpp src/candidate_verifier.cpp $(EDLIB_SRC) src/main.cpp
OBJ = $(SRC:.cpp=.o)
DEP = $(OBJ:.o=.d)
TARGET = navigamer

LIB_SRC = src/structure.cpp src/tools.cpp src/qgram_filter.cpp src/range_join.cpp src/phase1_seed_index.cpp src/phase2_distance_verifier.cpp src/build_progress.cpp src/mbb_rect_index.cpp src/simd_mbb_filter.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp src/map150.cpp src/candidate_verifier.cpp $(EDLIB_SRC)
LIB_OBJ = $(LIB_SRC:.cpp=.o)
TEST_SRC = src/test_recall.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
TEST_DEP = $(TEST_OBJ:.o=.d)
TEST_TARGET = test_recall

TEST_DIST_SRC = src/test_distance_bound.cpp
TEST_DIST_OBJ = $(TEST_DIST_SRC:.cpp=.o)
TEST_DIST_DEP = $(TEST_DIST_OBJ:.o=.d)
TEST_DIST_TARGET = test_distance_bound

TEST_HIER_SRC = src/test_hierarchy_config.cpp
TEST_HIER_OBJ = $(TEST_HIER_SRC:.cpp=.o)
TEST_HIER_DEP = $(TEST_HIER_OBJ:.o=.d)
TEST_HIER_TARGET = test_hierarchy_config

TEST_SEARCH_STATS_SRC = src/test_search_stats.cpp
TEST_SEARCH_STATS_OBJ = $(TEST_SEARCH_STATS_SRC:.cpp=.o)
TEST_SEARCH_STATS_DEP = $(TEST_SEARCH_STATS_OBJ:.o=.d)
TEST_SEARCH_STATS_TARGET = test_search_stats_bin

TEST_QUERY_PROFILE_SRC = src/test_query_profile_stats_smoke.cpp
TEST_QUERY_PROFILE_OBJ = $(TEST_QUERY_PROFILE_SRC:.cpp=.o)
TEST_QUERY_PROFILE_DEP = $(TEST_QUERY_PROFILE_OBJ:.o=.d)
TEST_QUERY_PROFILE_TARGET = test_query_profile_stats_smoke

TEST_LOCAL_ROUTER_SRC = src/test_local_router_no_false_negative.cpp
TEST_LOCAL_ROUTER_OBJ = $(TEST_LOCAL_ROUTER_SRC:.cpp=.o)
TEST_LOCAL_ROUTER_DEP = $(TEST_LOCAL_ROUTER_OBJ:.o=.d)
TEST_LOCAL_ROUTER_TARGET = test_local_router_no_false_negative

TEST_BEST_FIRST_SRC = src/test_best_first_no_false_negative.cpp
TEST_BEST_FIRST_OBJ = $(TEST_BEST_FIRST_SRC:.cpp=.o)
TEST_BEST_FIRST_DEP = $(TEST_BEST_FIRST_OBJ:.o=.d)
TEST_BEST_FIRST_TARGET = test_best_first_no_false_negative

TEST_ROUTER_HINTS_SRC = src/test_router_hints_no_false_negative.cpp
TEST_ROUTER_HINTS_OBJ = $(TEST_ROUTER_HINTS_SRC:.cpp=.o)
TEST_ROUTER_HINTS_DEP = $(TEST_ROUTER_HINTS_OBJ:.o=.d)
TEST_ROUTER_HINTS_TARGET = test_router_hints_no_false_negative

TEST_PATH_REUSE_SRC = src/test_path_reuse_no_false_negative.cpp
TEST_PATH_REUSE_OBJ = $(TEST_PATH_REUSE_SRC:.cpp=.o)
TEST_PATH_REUSE_DEP = $(TEST_PATH_REUSE_OBJ:.o=.d)
TEST_PATH_REUSE_TARGET = test_path_reuse_no_false_negative

TEST_SAFE_CHILD_ROUTER_SRC = src/test_safe_child_router_no_false_negative.cpp
TEST_SAFE_CHILD_ROUTER_OBJ = $(TEST_SAFE_CHILD_ROUTER_SRC:.cpp=.o)
TEST_SAFE_CHILD_ROUTER_DEP = $(TEST_SAFE_CHILD_ROUTER_OBJ:.o=.d)
TEST_SAFE_CHILD_ROUTER_TARGET = test_safe_child_router_no_false_negative

TEST_QUERY_PLANNER_SRC = src/test_query_planner_no_false_negative.cpp
TEST_QUERY_PLANNER_OBJ = $(TEST_QUERY_PLANNER_SRC:.cpp=.o)
TEST_QUERY_PLANNER_DEP = $(TEST_QUERY_PLANNER_OBJ:.o=.d)
TEST_QUERY_PLANNER_TARGET = test_query_planner_no_false_negative

TEST_MAP150_SRC = src/test_map150_recall.cpp
TEST_MAP150_OBJ = $(TEST_MAP150_SRC:.cpp=.o)
TEST_MAP150_DEP = $(TEST_MAP150_OBJ:.o=.d)
TEST_MAP150_TARGET = test_map150_recall

TEST_BOUNDED_SRC = src/test_bounded_edit_distance.cpp
TEST_BOUNDED_OBJ = $(TEST_BOUNDED_SRC:.cpp=.o)
TEST_BOUNDED_DEP = $(TEST_BOUNDED_OBJ:.o=.d)
TEST_BOUNDED_TARGET = test_bounded_edit_distance

TEST_BOUNDED_MYERS_SRC = src/test_bounded_myers.cpp
TEST_BOUNDED_MYERS_OBJ = $(TEST_BOUNDED_MYERS_SRC:.cpp=.o)
TEST_BOUNDED_MYERS_DEP = $(TEST_BOUNDED_MYERS_OBJ:.o=.d)
TEST_BOUNDED_MYERS_TARGET = test_bounded_myers_bin

TEST_RANGE_JOIN_SRC = src/test_range_join.cpp
TEST_RANGE_JOIN_OBJ = $(TEST_RANGE_JOIN_SRC:.cpp=.o)
TEST_RANGE_JOIN_DEP = $(TEST_RANGE_JOIN_OBJ:.o=.d)
TEST_RANGE_JOIN_TARGET = test_range_join

TEST_QGRAM_SRC = src/test_qgram_filter.cpp
TEST_QGRAM_OBJ = $(TEST_QGRAM_SRC:.cpp=.o)
TEST_QGRAM_DEP = $(TEST_QGRAM_OBJ:.o=.d)
TEST_QGRAM_TARGET = test_qgram_filter

TEST_BUILD_RANGE_SRC = src/test_build_range_equivalence.cpp
TEST_BUILD_RANGE_OBJ = $(TEST_BUILD_RANGE_SRC:.cpp=.o)
TEST_BUILD_RANGE_DEP = $(TEST_BUILD_RANGE_OBJ:.o=.d)
TEST_BUILD_RANGE_TARGET = test_build_range_equivalence

TEST_BUILD_DISTANCE_SRC = src/test_build_distance_mode.cpp
TEST_BUILD_DISTANCE_OBJ = $(TEST_BUILD_DISTANCE_SRC:.cpp=.o)
TEST_BUILD_DISTANCE_DEP = $(TEST_BUILD_DISTANCE_OBJ:.o=.d)
TEST_BUILD_DISTANCE_TARGET = test_build_distance_mode_bin

TEST_MBB_RECT_SRC = src/test_mbb_rect_index.cpp
TEST_MBB_RECT_OBJ = $(TEST_MBB_RECT_SRC:.cpp=.o)
TEST_MBB_RECT_DEP = $(TEST_MBB_RECT_OBJ:.o=.d)
TEST_MBB_RECT_TARGET = test_mbb_rect_index

TEST_MBB_FILTER_SRC = src/test_mbb_filter_equivalence.cpp
TEST_MBB_FILTER_OBJ = $(TEST_MBB_FILTER_SRC:.cpp=.o)
TEST_MBB_FILTER_DEP = $(TEST_MBB_FILTER_OBJ:.o=.d)
TEST_MBB_FILTER_TARGET = test_mbb_filter_equivalence

TEST_SEARCH_QGRAM_SRC = src/test_search_qgram_prefilter.cpp
TEST_SEARCH_QGRAM_OBJ = $(TEST_SEARCH_QGRAM_SRC:.cpp=.o)
TEST_SEARCH_QGRAM_DEP = $(TEST_SEARCH_QGRAM_OBJ:.o=.d)
TEST_SEARCH_QGRAM_TARGET = test_search_qgram_prefilter

TEST_QUERY_BENCHMARK_SRC = src/test_query_benchmark_gate.cpp
TEST_QUERY_BENCHMARK_OBJ = $(TEST_QUERY_BENCHMARK_SRC:.cpp=.o)
TEST_QUERY_BENCHMARK_DEP = $(TEST_QUERY_BENCHMARK_OBJ:.o=.d)
TEST_QUERY_BENCHMARK_TARGET = test_query_benchmark_gate

TEST_CANDIDATE_VERIFIER_SRC = src/test_candidate_verifier.cpp
TEST_CANDIDATE_VERIFIER_OBJ = $(TEST_CANDIDATE_VERIFIER_SRC:.cpp=.o)
TEST_CANDIDATE_VERIFIER_DEP = $(TEST_CANDIDATE_VERIFIER_OBJ:.o=.d)
TEST_CANDIDATE_VERIFIER_TARGET = test_candidate_verifier_bin

TEST_EPOCH_VISITED_SRC = src/test_epoch_visited.cpp
TEST_EPOCH_VISITED_OBJ = $(TEST_EPOCH_VISITED_SRC:.cpp=.o)
TEST_EPOCH_VISITED_DEP = $(TEST_EPOCH_VISITED_OBJ:.o=.d)
TEST_EPOCH_VISITED_TARGET = test_epoch_visited_bin

TEST_SEARCH_GRAPH_VIEW_SRC = src/test_search_graph_view.cpp
TEST_SEARCH_GRAPH_VIEW_OBJ = $(TEST_SEARCH_GRAPH_VIEW_SRC:.cpp=.o)
TEST_SEARCH_GRAPH_VIEW_DEP = $(TEST_SEARCH_GRAPH_VIEW_OBJ:.o=.d)
TEST_SEARCH_GRAPH_VIEW_TARGET = test_search_graph_view_bin

TEST_SEARCH_DISTANCE_MODE_SRC = src/test_search_distance_mode.cpp
TEST_SEARCH_DISTANCE_MODE_OBJ = $(TEST_SEARCH_DISTANCE_MODE_SRC:.cpp=.o)
TEST_SEARCH_DISTANCE_MODE_DEP = $(TEST_SEARCH_DISTANCE_MODE_OBJ:.o=.d)
TEST_SEARCH_DISTANCE_MODE_TARGET = test_search_distance_mode_bin

TEST_SIMD_MBB_SRC = src/test_simd_mbb_filter.cpp
TEST_SIMD_MBB_OBJ = $(TEST_SIMD_MBB_SRC:.cpp=.o)
TEST_SIMD_MBB_DEP = $(TEST_SIMD_MBB_OBJ:.o=.d)
TEST_SIMD_MBB_TARGET = test_simd_mbb_filter_bin

TEST_SIMD_LEAF_SRC = src/test_simd_leaf_beacon_filter.cpp
TEST_SIMD_LEAF_OBJ = $(TEST_SIMD_LEAF_SRC:.cpp=.o)
TEST_SIMD_LEAF_DEP = $(TEST_SIMD_LEAF_OBJ:.o=.d)
TEST_SIMD_LEAF_TARGET = test_simd_leaf_beacon_filter_bin

TEST_BUILD_TIMING_SRC = src/test_build_timing_stats.cpp
TEST_BUILD_TIMING_OBJ = $(TEST_BUILD_TIMING_SRC:.cpp=.o)
TEST_BUILD_TIMING_DEP = $(TEST_BUILD_TIMING_OBJ:.o=.d)
TEST_BUILD_TIMING_TARGET = test_build_timing_stats

TEST_BUILD_SCALE_SRC = src/test_build_scale_smoke.cpp
TEST_BUILD_SCALE_OBJ = $(TEST_BUILD_SCALE_SRC:.cpp=.o)
TEST_BUILD_SCALE_DEP = $(TEST_BUILD_SCALE_OBJ:.o=.d)
TEST_BUILD_SCALE_TARGET = test_build_scale_smoke

TEST_PHASE1_HYBRID_SRC = src/test_phase1_hybrid.cpp
TEST_PHASE1_HYBRID_OBJ = $(TEST_PHASE1_HYBRID_SRC:.cpp=.o)
TEST_PHASE1_HYBRID_DEP = $(TEST_PHASE1_HYBRID_OBJ:.o=.d)
TEST_PHASE1_HYBRID_TARGET = test_phase1_hybrid_bin

TEST_PHASE1_SEED_SRC = src/test_phase1_seed_index.cpp
TEST_PHASE1_SEED_OBJ = $(TEST_PHASE1_SEED_SRC:.cpp=.o)
TEST_PHASE1_SEED_DEP = $(TEST_PHASE1_SEED_OBJ:.o=.d)
TEST_PHASE1_SEED_TARGET = test_phase1_seed_index

TEST_INDEX_PERSISTENCE_SRC = src/test_index_persistence.cpp
TEST_INDEX_PERSISTENCE_OBJ = $(TEST_INDEX_PERSISTENCE_SRC:.cpp=.o)
TEST_INDEX_PERSISTENCE_DEP = $(TEST_INDEX_PERSISTENCE_OBJ:.o=.d)
TEST_INDEX_PERSISTENCE_TARGET = test_index_persistence_bin

TEST_PHASE2_DISTANCE_SRC = src/test_phase2_distance_verifier.cpp
TEST_PHASE2_DISTANCE_OBJ = $(TEST_PHASE2_DISTANCE_SRC:.cpp=.o)
TEST_PHASE2_DISTANCE_DEP = $(TEST_PHASE2_DISTANCE_OBJ:.o=.d)
TEST_PHASE2_DISTANCE_TARGET = test_phase2_distance_verifier_bin

TEST_BUILD_PROGRESS_SRC = src/test_build_progress.cpp
TEST_BUILD_PROGRESS_OBJ = $(TEST_BUILD_PROGRESS_SRC:.cpp=.o)
TEST_BUILD_PROGRESS_DEP = $(TEST_BUILD_PROGRESS_OBJ:.o=.d)
TEST_BUILD_PROGRESS_TARGET = test_build_progress_bin

EXP_SRC = src/experiment_utils.cpp
EXP_OBJ = $(EXP_SRC:.cpp=.o)
EXP_DEP = $(EXP_OBJ:.cpp=.d)

LIB_SRC += $(EXP_SRC) src/query_benchmark.cpp src/index_persistence.cpp
LIB_OBJ := $(LIB_SRC:.cpp=.o)

.PHONY: all clean test test_dist test_hierarchy test_search_stats test_query_profile test_local_router test_best_first test_router_hints test_path_reuse test_safe_child_router test_query_planner test_map150 test_bounded test_bounded_myers test_qgram test_range_join_check test_build_range test_build_distance_mode test_mbb_rect test_mbb_filter test_search_qgram test_query_benchmark test_candidate_verifier test_epoch_visited test_search_graph_view test_search_distance_mode test_simd_mbb_filter test_simd_leaf_beacon_filter test_build_timing_stats_check test_build_scale_smoke_check test_phase1_hybrid test_phase1_seed_index_check test_index_persistence test_phase2_distance_verifier test_build_progress test_all

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_TARGET): $(LIB_OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_DIST_TARGET): $(LIB_OBJ) $(TEST_DIST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_HIER_TARGET): $(LIB_OBJ) $(TEST_HIER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_SEARCH_STATS_TARGET): $(LIB_OBJ) $(TEST_SEARCH_STATS_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_QUERY_PROFILE_TARGET): $(LIB_OBJ) $(TEST_QUERY_PROFILE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_LOCAL_ROUTER_TARGET): $(LIB_OBJ) $(TEST_LOCAL_ROUTER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BEST_FIRST_TARGET): $(LIB_OBJ) $(TEST_BEST_FIRST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_ROUTER_HINTS_TARGET): $(LIB_OBJ) $(TEST_ROUTER_HINTS_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_PATH_REUSE_TARGET): $(LIB_OBJ) $(TEST_PATH_REUSE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_SAFE_CHILD_ROUTER_TARGET): $(LIB_OBJ) $(TEST_SAFE_CHILD_ROUTER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_QUERY_PLANNER_TARGET): $(LIB_OBJ) $(TEST_QUERY_PLANNER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_MAP150_TARGET): $(LIB_OBJ) $(TEST_MAP150_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BOUNDED_TARGET): $(LIB_OBJ) $(TEST_BOUNDED_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BOUNDED_MYERS_TARGET): src/structure.o src/tools.o $(EDLIB_OBJ) $(TEST_BOUNDED_MYERS_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_RANGE_JOIN_TARGET): $(LIB_OBJ) $(TEST_RANGE_JOIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_QGRAM_TARGET): src/structure.o src/tools.o src/qgram_filter.o $(EDLIB_OBJ) $(TEST_QGRAM_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BUILD_RANGE_TARGET): $(LIB_OBJ) $(TEST_BUILD_RANGE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BUILD_DISTANCE_TARGET): $(LIB_OBJ) $(TEST_BUILD_DISTANCE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_MBB_RECT_TARGET): src/mbb_rect_index.o $(TEST_MBB_RECT_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_MBB_FILTER_TARGET): $(LIB_OBJ) $(TEST_MBB_FILTER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_SEARCH_QGRAM_TARGET): $(LIB_OBJ) $(TEST_SEARCH_QGRAM_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_QUERY_BENCHMARK_TARGET): $(LIB_OBJ) $(TEST_QUERY_BENCHMARK_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_CANDIDATE_VERIFIER_TARGET): $(LIB_OBJ) $(TEST_CANDIDATE_VERIFIER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_EPOCH_VISITED_TARGET): $(LIB_OBJ) $(TEST_EPOCH_VISITED_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_SEARCH_GRAPH_VIEW_TARGET): $(LIB_OBJ) $(TEST_SEARCH_GRAPH_VIEW_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_SEARCH_DISTANCE_MODE_TARGET): $(LIB_OBJ) $(TEST_SEARCH_DISTANCE_MODE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_SIMD_MBB_TARGET): src/simd_mbb_filter.o $(TEST_SIMD_MBB_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_SIMD_LEAF_TARGET): src/simd_mbb_filter.o $(TEST_SIMD_LEAF_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BUILD_TIMING_TARGET): $(LIB_OBJ) $(TEST_BUILD_TIMING_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BUILD_SCALE_TARGET): $(TEST_BUILD_SCALE_OBJ) $(TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_BUILD_SCALE_OBJ)

$(TEST_PHASE1_HYBRID_TARGET): $(LIB_OBJ) $(TEST_PHASE1_HYBRID_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_PHASE1_SEED_TARGET): $(TEST_PHASE1_SEED_OBJ) src/phase1_seed_index.o src/tools.o src/structure.o $(EDLIB_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_INDEX_PERSISTENCE_TARGET): $(LIB_OBJ) $(TEST_INDEX_PERSISTENCE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_PHASE2_DISTANCE_TARGET): $(TEST_PHASE2_DISTANCE_OBJ) src/phase2_distance_verifier.o src/tools.o src/structure.o $(EDLIB_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TEST_BUILD_PROGRESS_TARGET): $(TEST_BUILD_PROGRESS_OBJ) src/build_progress.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

test_dist: $(TEST_DIST_TARGET)
	./$(TEST_DIST_TARGET)

test_hierarchy: $(TEST_HIER_TARGET)
	./$(TEST_HIER_TARGET)

test_search_stats: $(TEST_SEARCH_STATS_TARGET)
	./$(TEST_SEARCH_STATS_TARGET)

test_query_profile: $(TEST_QUERY_PROFILE_TARGET)
	./$(TEST_QUERY_PROFILE_TARGET)

test_local_router: $(TEST_LOCAL_ROUTER_TARGET)
	./$(TEST_LOCAL_ROUTER_TARGET)

test_best_first: $(TEST_BEST_FIRST_TARGET)
	./$(TEST_BEST_FIRST_TARGET)

test_router_hints: $(TEST_ROUTER_HINTS_TARGET)
	./$(TEST_ROUTER_HINTS_TARGET)

test_path_reuse: $(TEST_PATH_REUSE_TARGET)
	./$(TEST_PATH_REUSE_TARGET)

test_safe_child_router: $(TEST_SAFE_CHILD_ROUTER_TARGET)
	./$(TEST_SAFE_CHILD_ROUTER_TARGET)

test_query_planner: $(TEST_QUERY_PLANNER_TARGET)
	./$(TEST_QUERY_PLANNER_TARGET)

test_map150: $(TEST_MAP150_TARGET)
	./$(TEST_MAP150_TARGET)

test_bounded: $(TEST_BOUNDED_TARGET)
	./$(TEST_BOUNDED_TARGET)

test_bounded_myers: $(TEST_BOUNDED_MYERS_TARGET)
	./$(TEST_BOUNDED_MYERS_TARGET)

test_range_join_check: $(TEST_RANGE_JOIN_TARGET)
	./$(TEST_RANGE_JOIN_TARGET)

test_qgram: $(TEST_QGRAM_TARGET)
	./$(TEST_QGRAM_TARGET)

test_build_range: $(TEST_BUILD_RANGE_TARGET)
	./$(TEST_BUILD_RANGE_TARGET)

test_build_distance_mode: $(TEST_BUILD_DISTANCE_TARGET)
	./$(TEST_BUILD_DISTANCE_TARGET)

test_mbb_rect: $(TEST_MBB_RECT_TARGET)
	./$(TEST_MBB_RECT_TARGET)

test_mbb_filter: $(TEST_MBB_FILTER_TARGET)
	./$(TEST_MBB_FILTER_TARGET)

test_search_qgram: $(TEST_SEARCH_QGRAM_TARGET)
	./$(TEST_SEARCH_QGRAM_TARGET)

test_query_benchmark: $(TEST_QUERY_BENCHMARK_TARGET)
	./$(TEST_QUERY_BENCHMARK_TARGET)

test_candidate_verifier: $(TEST_CANDIDATE_VERIFIER_TARGET)
	./$(TEST_CANDIDATE_VERIFIER_TARGET)

test_epoch_visited: $(TEST_EPOCH_VISITED_TARGET)
	./$(TEST_EPOCH_VISITED_TARGET)

test_search_graph_view: $(TEST_SEARCH_GRAPH_VIEW_TARGET)
	./$(TEST_SEARCH_GRAPH_VIEW_TARGET)

test_search_distance_mode: $(TEST_SEARCH_DISTANCE_MODE_TARGET)
	./$(TEST_SEARCH_DISTANCE_MODE_TARGET)

test_simd_mbb_filter: $(TEST_SIMD_MBB_TARGET)
	./$(TEST_SIMD_MBB_TARGET)

test_simd_leaf_beacon_filter: $(TEST_SIMD_LEAF_TARGET)
	./$(TEST_SIMD_LEAF_TARGET)

test_build_timing_stats_check: $(TEST_BUILD_TIMING_TARGET)
	./$(TEST_BUILD_TIMING_TARGET)

test_build_scale_smoke_check: $(TEST_BUILD_SCALE_TARGET)
	./$(TEST_BUILD_SCALE_TARGET)

test_phase1_hybrid: $(TEST_PHASE1_HYBRID_TARGET)
	./$(TEST_PHASE1_HYBRID_TARGET)

test_phase1_seed_index_check: $(TEST_PHASE1_SEED_TARGET)
	./$(TEST_PHASE1_SEED_TARGET)

test_index_persistence: $(TEST_INDEX_PERSISTENCE_TARGET)
	./$(TEST_INDEX_PERSISTENCE_TARGET)

test_phase2_distance_verifier: $(TEST_PHASE2_DISTANCE_TARGET)
	./$(TEST_PHASE2_DISTANCE_TARGET)

test_build_progress: $(TEST_BUILD_PROGRESS_TARGET)
	./$(TEST_BUILD_PROGRESS_TARGET)

test_all: test test_dist test_hierarchy test_search_stats test_query_profile test_local_router test_best_first test_router_hints test_path_reuse test_safe_child_router test_query_planner test_map150 test_bounded test_bounded_myers test_qgram test_range_join_check test_build_range test_build_distance_mode test_mbb_rect test_mbb_filter test_search_qgram test_query_benchmark test_candidate_verifier test_epoch_visited test_search_graph_view test_search_distance_mode test_simd_mbb_filter test_simd_leaf_beacon_filter test_build_timing_stats_check test_build_scale_smoke_check test_phase1_hybrid test_phase1_seed_index_check test_index_persistence test_phase2_distance_verifier test_build_progress

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(LIB_OBJ) $(EDLIB_OBJ) $(TEST_OBJ) $(TEST_DIST_OBJ) $(TEST_HIER_OBJ) $(TEST_SEARCH_STATS_OBJ) $(TEST_QUERY_PROFILE_OBJ) $(TEST_LOCAL_ROUTER_OBJ) $(TEST_BEST_FIRST_OBJ) $(TEST_ROUTER_HINTS_OBJ) $(TEST_PATH_REUSE_OBJ) $(TEST_SAFE_CHILD_ROUTER_OBJ) $(TEST_QUERY_PLANNER_OBJ) $(TEST_MAP150_OBJ) $(TEST_BOUNDED_OBJ) $(TEST_BOUNDED_MYERS_OBJ) $(TEST_QGRAM_OBJ) $(TEST_RANGE_JOIN_OBJ) $(TEST_BUILD_RANGE_OBJ) $(TEST_BUILD_DISTANCE_OBJ) $(TEST_MBB_RECT_OBJ) $(TEST_MBB_FILTER_OBJ) $(TEST_SEARCH_QGRAM_OBJ) $(TEST_QUERY_BENCHMARK_OBJ) $(TEST_CANDIDATE_VERIFIER_OBJ) $(TEST_EPOCH_VISITED_OBJ) $(TEST_SEARCH_GRAPH_VIEW_OBJ) $(TEST_SEARCH_DISTANCE_MODE_OBJ) $(TEST_SIMD_MBB_OBJ) $(TEST_SIMD_LEAF_OBJ) $(TEST_BUILD_TIMING_OBJ) $(TEST_BUILD_SCALE_OBJ) $(TEST_PHASE1_HYBRID_OBJ) $(TEST_PHASE1_SEED_OBJ) $(TEST_INDEX_PERSISTENCE_OBJ) $(TEST_PHASE2_DISTANCE_OBJ) $(TEST_BUILD_PROGRESS_OBJ) $(DEP) $(EDLIB_DEP) $(TEST_DEP) $(TEST_DIST_DEP) $(TEST_HIER_DEP) $(TEST_SEARCH_STATS_DEP) $(TEST_QUERY_PROFILE_DEP) $(TEST_LOCAL_ROUTER_DEP) $(TEST_BEST_FIRST_DEP) $(TEST_ROUTER_HINTS_DEP) $(TEST_PATH_REUSE_DEP) $(TEST_SAFE_CHILD_ROUTER_DEP) $(TEST_QUERY_PLANNER_DEP) $(TEST_MAP150_DEP) $(TEST_BOUNDED_DEP) $(TEST_BOUNDED_MYERS_DEP) $(TEST_QGRAM_DEP) $(TEST_RANGE_JOIN_DEP) $(TEST_BUILD_RANGE_DEP) $(TEST_BUILD_DISTANCE_DEP) $(TEST_MBB_RECT_DEP) $(TEST_MBB_FILTER_DEP) $(TEST_SEARCH_QGRAM_DEP) $(TEST_QUERY_BENCHMARK_DEP) $(TEST_CANDIDATE_VERIFIER_DEP) $(TEST_EPOCH_VISITED_DEP) $(TEST_SEARCH_GRAPH_VIEW_DEP) $(TEST_SEARCH_DISTANCE_MODE_DEP) $(TEST_SIMD_MBB_DEP) $(TEST_SIMD_LEAF_DEP) $(TEST_BUILD_TIMING_DEP) $(TEST_BUILD_SCALE_DEP) $(TEST_PHASE1_HYBRID_DEP) $(TEST_PHASE1_SEED_DEP) $(TEST_INDEX_PERSISTENCE_DEP) $(TEST_PHASE2_DISTANCE_DEP) $(TEST_BUILD_PROGRESS_DEP) $(TARGET) $(TEST_TARGET) $(TEST_DIST_TARGET) $(TEST_HIER_TARGET) $(TEST_SEARCH_STATS_TARGET) $(TEST_QUERY_PROFILE_TARGET) $(TEST_LOCAL_ROUTER_TARGET) $(TEST_BEST_FIRST_TARGET) $(TEST_ROUTER_HINTS_TARGET) $(TEST_PATH_REUSE_TARGET) $(TEST_SAFE_CHILD_ROUTER_TARGET) $(TEST_QUERY_PLANNER_TARGET) $(TEST_MAP150_TARGET) $(TEST_BOUNDED_TARGET) $(TEST_BOUNDED_MYERS_TARGET) $(TEST_QGRAM_TARGET) $(TEST_RANGE_JOIN_TARGET) $(TEST_BUILD_RANGE_TARGET) $(TEST_BUILD_DISTANCE_TARGET) $(TEST_MBB_RECT_TARGET) $(TEST_MBB_FILTER_TARGET) $(TEST_SEARCH_QGRAM_TARGET) $(TEST_QUERY_BENCHMARK_TARGET) $(TEST_CANDIDATE_VERIFIER_TARGET) $(TEST_EPOCH_VISITED_TARGET) $(TEST_SEARCH_GRAPH_VIEW_TARGET) $(TEST_SEARCH_DISTANCE_MODE_TARGET) $(TEST_SIMD_MBB_TARGET) $(TEST_SIMD_LEAF_TARGET) $(TEST_BUILD_TIMING_TARGET) $(TEST_BUILD_SCALE_TARGET) $(TEST_PHASE1_HYBRID_TARGET) $(TEST_PHASE1_SEED_TARGET) $(TEST_INDEX_PERSISTENCE_TARGET) $(TEST_PHASE2_DISTANCE_TARGET) $(TEST_BUILD_PROGRESS_TARGET)

-include $(DEP) $(EDLIB_DEP) $(TEST_DEP) $(TEST_DIST_DEP) $(TEST_HIER_DEP) $(TEST_SEARCH_STATS_DEP) $(TEST_QUERY_PROFILE_DEP) $(TEST_LOCAL_ROUTER_DEP) $(TEST_BEST_FIRST_DEP) $(TEST_ROUTER_HINTS_DEP) $(TEST_PATH_REUSE_DEP) $(TEST_SAFE_CHILD_ROUTER_DEP) $(TEST_QUERY_PLANNER_DEP) $(TEST_MAP150_DEP) $(TEST_BOUNDED_DEP) $(TEST_BOUNDED_MYERS_DEP) $(TEST_QGRAM_DEP) $(TEST_RANGE_JOIN_DEP) $(TEST_BUILD_RANGE_DEP) $(TEST_BUILD_DISTANCE_DEP) $(TEST_MBB_RECT_DEP) $(TEST_MBB_FILTER_DEP) $(TEST_SEARCH_QGRAM_DEP) $(TEST_QUERY_BENCHMARK_DEP) $(TEST_CANDIDATE_VERIFIER_DEP) $(TEST_EPOCH_VISITED_DEP) $(TEST_SEARCH_GRAPH_VIEW_DEP) $(TEST_SEARCH_DISTANCE_MODE_DEP) $(TEST_SIMD_MBB_DEP) $(TEST_SIMD_LEAF_DEP) $(TEST_BUILD_TIMING_DEP) $(TEST_BUILD_SCALE_DEP) $(TEST_PHASE1_HYBRID_DEP) $(TEST_PHASE1_SEED_DEP) $(TEST_INDEX_PERSISTENCE_DEP) $(TEST_PHASE2_DISTANCE_DEP) $(TEST_BUILD_PROGRESS_DEP)
