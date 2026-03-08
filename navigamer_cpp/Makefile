CXX = g++
CXXFLAGS = -std=c++17 -O2 -I include -Wall -Wextra
SRC = src/structure.cpp src/tools.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp src/main.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = navigamer

LIB_SRC = src/structure.cpp src/tools.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp
LIB_OBJ = $(LIB_SRC:.cpp=.o)
TEST_SRC = src/test_recall.cpp
TEST_OBJ = $(TEST_SRC:.cpp=.o)
TEST_TARGET = test_recall

TEST_DIST_SRC = src/test_distance_bound.cpp
TEST_DIST_OBJ = $(TEST_DIST_SRC:.cpp=.o)
TEST_DIST_TARGET = test_distance_bound

.PHONY: all clean test test_dist test_all

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_TARGET): $(LIB_OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_DIST_TARGET): $(LIB_OBJ) $(TEST_DIST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: $(TEST_TARGET)
	./$(TEST_TARGET)

test_dist: $(TEST_DIST_TARGET)
	./$(TEST_DIST_TARGET)

test_all: test test_dist

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(TEST_DIST_OBJ) $(TARGET) $(TEST_TARGET) $(TEST_DIST_TARGET)
