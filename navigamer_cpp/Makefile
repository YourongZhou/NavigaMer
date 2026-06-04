CXX = g++
CXXFLAGS = -std=c++17 -O2 -I include -Wall -Wextra -fopenmp -MMD -MP
SRC = src/structure.cpp src/tools.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp src/experiment_utils.cpp src/main.cpp
OBJ = $(SRC:.cpp=.o)
DEP = $(OBJ:.o=.d)
TARGET = navigamer

LIB_SRC = src/structure.cpp src/tools.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp
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

EXP_SRC = src/experiment_utils.cpp
EXP_OBJ = $(EXP_SRC:.cpp=.o)
EXP_DEP = $(EXP_OBJ:.cpp=.d)

LIB_SRC += $(EXP_SRC)
LIB_OBJ := $(LIB_SRC:.cpp=.o)

.PHONY: all clean test test_dist test_hierarchy test_search_stats test_all

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

test: $(TEST_TARGET)
	./$(TEST_TARGET)

test_dist: $(TEST_DIST_TARGET)
	./$(TEST_DIST_TARGET)

test_hierarchy: $(TEST_HIER_TARGET)
	./$(TEST_HIER_TARGET)

test_search_stats: $(TEST_SEARCH_STATS_TARGET)
	./$(TEST_SEARCH_STATS_TARGET)

test_all: test test_dist test_hierarchy test_search_stats

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(LIB_OBJ) $(TEST_OBJ) $(TEST_DIST_OBJ) $(TEST_HIER_OBJ) $(TEST_SEARCH_STATS_OBJ) $(DEP) $(TEST_DEP) $(TEST_DIST_DEP) $(TEST_HIER_DEP) $(TEST_SEARCH_STATS_DEP) $(TARGET) $(TEST_TARGET) $(TEST_DIST_TARGET) $(TEST_HIER_TARGET) $(TEST_SEARCH_STATS_TARGET)

-include $(DEP) $(TEST_DEP) $(TEST_DIST_DEP) $(TEST_HIER_DEP) $(TEST_SEARCH_STATS_DEP)
