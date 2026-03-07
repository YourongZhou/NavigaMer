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

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TEST_TARGET): $(LIB_OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: $(TEST_TARGET)
	./$(TEST_TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(TARGET) $(TEST_TARGET)
