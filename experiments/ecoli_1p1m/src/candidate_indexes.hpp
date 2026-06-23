#pragma once

#include "index_persistence.hpp"
#include "occurrence_index.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

class ContiguousIndexConfig {
 public:
  std::filesystem::path reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint32_t k = 0;
};

class ContiguousIndex {
 public:
  static ContiguousIndex build(const ContiguousIndexConfig& config);
  static ContiguousIndex load(const std::filesystem::path& index_path);

  void save(const std::filesystem::path& out_dir) const;
  std::vector<uint32_t> query(std::string_view query_sequence) const;

 private:
  IndexManifest manifest_;
  OccurrenceIndex occurrence_index_;
};
