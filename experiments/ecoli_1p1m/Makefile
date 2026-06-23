CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -fopenmp
GIT_COMMON_DIR := $(shell git -C $(CURDIR) rev-parse --git-common-dir 2>/dev/null)
COMMON_REPO_ROOT := $(abspath $(GIT_COMMON_DIR)/..)
TENSOR_SKETCH_ROOT ?= $(COMMON_REPO_ROOT)/methods/tensor-sketch-alignment
TENSOR_SKETCH_SRC := $(TENSOR_SKETCH_ROOT)/metagraph/src
TENSOR_SKETCH_TENSOR := $(TENSOR_SKETCH_SRC)/sketch/tensor.hpp
TENSOR_SKETCH_HNSW := $(if $(wildcard $(TENSOR_SKETCH_SRC)/hnswlib/hnswlib.h),$(TENSOR_SKETCH_SRC)/hnswlib,$(TENSOR_SKETCH_ROOT)/metagraph/third_party/hnswlib/hnswlib)
HNSW_HEADER := $(TENSOR_SKETCH_HNSW)/hnswlib.h
TENSOR_SKETCH_TIMER := $(TENSOR_SKETCH_SRC)/util/timer.cpp
TENSOR_SKETCH_STRICT ?= 1

ifeq ($(wildcard $(TENSOR_SKETCH_TENSOR)),)
  ifeq ($(TENSOR_SKETCH_STRICT),1)
    $(error strict-mode: missing Tensor Sketch header $(TENSOR_SKETCH_TENSOR))
  endif
  NAVIGAMER_HAVE_TENSOR_SKETCH := 0
else ifeq ($(wildcard $(HNSW_HEADER)),)
  ifeq ($(TENSOR_SKETCH_STRICT),1)
    $(error strict-mode: missing hnswlib header $(HNSW_HEADER))
  endif
  NAVIGAMER_HAVE_TENSOR_SKETCH := 0
else
  NAVIGAMER_HAVE_TENSOR_SKETCH := 1
endif

CPPFLAGS := -Isrc -I../../navigamer_cpp/include -I../../navigamer_cpp/third_party/edlib/include \
            -I$(TENSOR_SKETCH_SRC) -I$(TENSOR_SKETCH_HNSW) \
            -DNAVIGAMER_HAVE_TENSOR_SKETCH=$(NAVIGAMER_HAVE_TENSOR_SKETCH) \
            -DNAVIGAMER_TENSOR_SKETCH_ROOT=\"$(TENSOR_SKETCH_ROOT)\"
EDLIB_SRC := ../../navigamer_cpp/third_party/edlib/src/edlib.cpp

.PHONY: all clean test test_candidate_tool_cli test_reference_windows test_index_persistence test_tensor_index

all: candidate_tool

test: test_reference_windows test_candidate_tool_cli test_index_persistence test_tensor_index

candidate_tool: src/candidate_tool.cpp src/reference_windows.cpp src/reference_windows.hpp src/tensor_index.cpp src/tensor_index.hpp src/index_persistence.cpp src/index_persistence.hpp src/sha256.cpp src/sha256.hpp $(EDLIB_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/candidate_tool.cpp src/reference_windows.cpp src/tensor_index.cpp src/index_persistence.cpp src/sha256.cpp $(EDLIB_SRC) $(TENSOR_SKETCH_TIMER)

test_reference_windows: test_reference_windows_bin
	./test_reference_windows_bin

test_reference_windows_bin: src/test_reference_windows.cpp src/reference_windows.cpp src/reference_windows.hpp $(EDLIB_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/test_reference_windows.cpp src/reference_windows.cpp $(EDLIB_SRC)

test_candidate_tool_cli: candidate_tool src/test_candidate_tool_cli.sh
	bash src/test_candidate_tool_cli.sh ./candidate_tool

test_index_persistence: src/test_index_persistence.cpp src/index_persistence.cpp src/index_persistence.hpp src/sha256.cpp src/sha256.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o test_index_persistence_bin src/test_index_persistence.cpp src/index_persistence.cpp src/sha256.cpp
	./test_index_persistence_bin

test_tensor_index: candidate_tool src/test_tensor_index.cpp src/tensor_index.cpp src/tensor_index.hpp src/index_persistence.cpp src/index_persistence.hpp src/sha256.cpp src/sha256.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o test_tensor_index_bin src/test_tensor_index.cpp src/tensor_index.cpp src/reference_windows.cpp src/index_persistence.cpp src/sha256.cpp $(TENSOR_SKETCH_TIMER)
	./test_tensor_index_bin

clean:
	$(RM) candidate_tool test_reference_windows test_reference_windows_bin test_index_persistence_bin test_tensor_index_bin
