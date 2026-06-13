CXX ?= g++
WITH_SEQAN3 ?= 0
CXXSTD ?= c++17
SEQAN3_FLAGS =
ifeq ($(WITH_SEQAN3),1)
CXXSTD = c++23
SEQAN3_FLAGS = -DNAVIGAMER_WITH_SEQAN3
endif
CXXFLAGS = -std=$(CXXSTD) -O2 -I include -Wall -Wextra -fopenmp -MMD -MP $(SEQAN3_FLAGS)
SRC = src/structure.cpp src/tools.cpp src/range_join.cpp src/mbb_rect_index.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp src/experiment_utils.cpp src/map150.cpp src/main.cpp
OBJ = $(SRC:.cpp=.o)
DEP = $(OBJ:.o=.d)
TARGET = navigamer

LIB_SRC = src/structure.cpp src/tools.cpp src/range_join.cpp src/mbb_rect_index.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp src/map150.cpp
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

TEST_MAP150_SRC = src/test_map150_recall.cpp
TEST_MAP150_OBJ = $(TEST_MAP150_SRC:.cpp=.o)
TEST_MAP150_DEP = $(TEST_MAP150_OBJ:.o=.d)
TEST_MAP150_TARGET = test_map150_recall

TEST_BOUNDED_SRC = src/test_bounded_edit_distance.cpp
TEST_BOUNDED_OBJ = $(TEST_BOUNDED_SRC:.cpp=.o)
TEST_BOUNDED_DEP = $(TEST_BOUNDED_OBJ:.o=.d)
TEST_BOUNDED_TARGET = test_bounded_edit_distance

TEST_RANGE_JOIN_SRC = src/test_range_join.cpp
TEST_RANGE_JOIN_OBJ = $(TEST_RANGE_JOIN_SRC:.cpp=.o)
TEST_RANGE_JOIN_DEP = $(TEST_RANGE_JOIN_OBJ:.o=.d)
TEST_RANGE_JOIN_TARGET = test_range_join

TEST_BUILD_RANGE_SRC = src/test_build_range_equivalence.cpp
TEST_BUILD_RANGE_OBJ = $(TEST_BUILD_RANGE_SRC:.cpp=.o)
TEST_BUILD_RANGE_DEP = $(TEST_BUILD_RANGE_OBJ:.o=.d)
TEST_BUILD_RANGE_TARGET = test_build_range_equivalence

TEST_MBB_RECT_SRC = src/test_mbb_rect_index.cpp
TEST_MBB_RECT_OBJ = $(TEST_MBB_RECT_SRC:.cpp=.o)
TEST_MBB_RECT_DEP = $(TEST_MBB_RECT_OBJ:.o=.d)
TEST_MBB_RECT_TARGET = test_mbb_rect_index

TEST_MBB_FILTER_SRC = src/test_mbb_filter_equivalence.cpp
TEST_MBB_FILTER_OBJ = $(TEST_MBB_FILTER_SRC:.cpp=.o)
TEST_MBB_FILTER_DEP = $(TEST_MBB_FILTER_OBJ:.o=.d)
TEST_MBB_FILTER_TARGET = test_mbb_filter_equivalence

EXP_SRC = src/experiment_utils.cpp
EXP_OBJ = $(EXP_SRC:.cpp=.o)
EXP_DEP = $(EXP_OBJ:.cpp=.d)

LIB_SRC += $(EXP_SRC)
LIB_OBJ := $(LIB_SRC:.cpp=.o)

.PHONY: all clean test test_dist test_hierarchy test_search_stats test_map150 test_bounded test_range_join_check test_build_range test_mbb_rect test_mbb_filter test_all

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_TARGET): $(LIB_OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_DIST_TARGET): $(LIB_OBJ) $(TEST_DIST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_HIER_TARGET): $(LIB_OBJ) $(TEST_HIER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_SEARCH_STATS_TARGET): $(LIB_OBJ) $(TEST_SEARCH_STATS_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_MAP150_TARGET): $(LIB_OBJ) $(TEST_MAP150_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_BOUNDED_TARGET): $(LIB_OBJ) $(TEST_BOUNDED_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_RANGE_JOIN_TARGET): $(LIB_OBJ) $(TEST_RANGE_JOIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_BUILD_RANGE_TARGET): $(LIB_OBJ) $(TEST_BUILD_RANGE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_MBB_RECT_TARGET): src/mbb_rect_index.o $(TEST_MBB_RECT_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_MBB_FILTER_TARGET): $(LIB_OBJ) $(TEST_MBB_FILTER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: $(TEST_TARGET)
	./$(TEST_TARGET)

test_dist: $(TEST_DIST_TARGET)
	./$(TEST_DIST_TARGET)

test_hierarchy: $(TEST_HIER_TARGET)
	./$(TEST_HIER_TARGET)

test_search_stats: $(TEST_SEARCH_STATS_TARGET)
	./$(TEST_SEARCH_STATS_TARGET)

test_map150: $(TEST_MAP150_TARGET)
	./$(TEST_MAP150_TARGET)

test_bounded: $(TEST_BOUNDED_TARGET)
	./$(TEST_BOUNDED_TARGET)

test_range_join_check: $(TEST_RANGE_JOIN_TARGET)
	./$(TEST_RANGE_JOIN_TARGET)

test_build_range: $(TEST_BUILD_RANGE_TARGET)
	./$(TEST_BUILD_RANGE_TARGET)

test_mbb_rect: $(TEST_MBB_RECT_TARGET)
	./$(TEST_MBB_RECT_TARGET)

test_mbb_filter: $(TEST_MBB_FILTER_TARGET)
	./$(TEST_MBB_FILTER_TARGET)

test_all: test test_dist test_hierarchy test_search_stats test_map150 test_bounded test_range_join_check test_build_range test_mbb_rect test_mbb_filter

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(LIB_OBJ) $(TEST_OBJ) $(TEST_DIST_OBJ) $(TEST_HIER_OBJ) $(TEST_SEARCH_STATS_OBJ) $(TEST_MAP150_OBJ) $(TEST_BOUNDED_OBJ) $(TEST_RANGE_JOIN_OBJ) $(TEST_BUILD_RANGE_OBJ) $(TEST_MBB_RECT_OBJ) $(TEST_MBB_FILTER_OBJ) $(DEP) $(TEST_DEP) $(TEST_DIST_DEP) $(TEST_HIER_DEP) $(TEST_SEARCH_STATS_DEP) $(TEST_MAP150_DEP) $(TEST_BOUNDED_DEP) $(TEST_RANGE_JOIN_DEP) $(TEST_BUILD_RANGE_DEP) $(TEST_MBB_RECT_DEP) $(TEST_MBB_FILTER_DEP) $(TARGET) $(TEST_TARGET) $(TEST_DIST_TARGET) $(TEST_HIER_TARGET) $(TEST_SEARCH_STATS_TARGET) $(TEST_MAP150_TARGET) $(TEST_BOUNDED_TARGET) $(TEST_RANGE_JOIN_TARGET) $(TEST_BUILD_RANGE_TARGET) $(TEST_MBB_RECT_TARGET) $(TEST_MBB_FILTER_TARGET)

-include $(DEP) $(TEST_DEP) $(TEST_DIST_DEP) $(TEST_HIER_DEP) $(TEST_SEARCH_STATS_DEP) $(TEST_MAP150_DEP) $(TEST_BOUNDED_DEP) $(TEST_RANGE_JOIN_DEP) $(TEST_BUILD_RANGE_DEP) $(TEST_MBB_RECT_DEP) $(TEST_MBB_FILTER_DEP)
