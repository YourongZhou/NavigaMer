#pragma once

#include "index_persistence.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace tensor_index {

struct QueryHit {
  uint32_t label = 0;
  float distance = 0.0f;
};

struct TensorIndexConfig {
  std::filesystem::path reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint32_t dimension = 0;
  uint32_t seed = 0;
  uint32_t hnsw_M = 0;
  uint32_t hnsw_ef_construction = 0;
  uint32_t hnsw_ef_search = 0;
  bool exact_vectors = false;
};

struct TensorIndexSnapshot {
  IndexManifest manifest;
  std::vector<float> exact_vectors;
  std::vector<uint32_t> labels;
  uint32_t dimension = 0;
  uint32_t seed = 0;
  uint32_t hnsw_M = 0;
  uint32_t hnsw_ef_construction = 0;
  uint32_t hnsw_ef_search = 0;
  bool persist_exact_vectors = true;
};

struct TensorIndex {
  TensorIndexSnapshot snapshot;
  std::filesystem::path hnsw_path;
  std::filesystem::path exact_path;
  std::shared_ptr<hnswlib::L2Space> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
};

TensorIndex build_tensor_index(const TensorIndexConfig& config);
void save_tensor_index(const TensorIndex& index,
                       const std::filesystem::path& directory);
TensorIndex load_tensor_index(const std::filesystem::path& directory);
std::vector<QueryHit> query_tensor_index(TensorIndex& index,
                                        const std::vector<int>& query,
                                        std::size_t top_k);

}  // namespace tensor_index
