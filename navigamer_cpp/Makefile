CXX = g++
CXXFLAGS = -std=c++17 -O2 -I include -Wall -Wextra
SRC = src/structure.cpp src/tools.cpp src/index_builder.cpp src/search_engine.cpp src/io_utils.cpp src/main.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = navigamer

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)
