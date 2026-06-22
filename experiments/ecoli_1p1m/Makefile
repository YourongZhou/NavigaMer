CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -fopenmp
CPPFLAGS := -Isrc -I../../navigamer_cpp/include -I../../navigamer_cpp/third_party/edlib/include
EDLIB_SRC := ../../navigamer_cpp/third_party/edlib/src/edlib.cpp

.PHONY: all clean test test_candidate_tool_cli test_reference_windows test_index_persistence

all: candidate_tool

test: test_reference_windows test_candidate_tool_cli test_index_persistence

candidate_tool: src/candidate_tool.cpp src/reference_windows.cpp src/reference_windows.hpp $(EDLIB_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/candidate_tool.cpp src/reference_windows.cpp $(EDLIB_SRC)

test_reference_windows: test_reference_windows_bin
	./test_reference_windows_bin

test_reference_windows_bin: src/test_reference_windows.cpp src/reference_windows.cpp src/reference_windows.hpp $(EDLIB_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/test_reference_windows.cpp src/reference_windows.cpp $(EDLIB_SRC)

test_candidate_tool_cli: candidate_tool src/test_candidate_tool_cli.sh
	bash src/test_candidate_tool_cli.sh ./candidate_tool

test_index_persistence: src/test_index_persistence.cpp src/index_persistence.cpp src/index_persistence.hpp src/sha256.cpp src/sha256.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o test_index_persistence_bin src/test_index_persistence.cpp src/index_persistence.cpp src/sha256.cpp
	./test_index_persistence_bin

clean:
	$(RM) candidate_tool test_reference_windows test_reference_windows_bin test_index_persistence_bin
