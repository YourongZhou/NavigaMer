#ifndef NAVIGAMER_SHARDED_INDEX_HPP
#define NAVIGAMER_SHARDED_INDEX_HPP

#include "index_persistence.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace navigamer {

struct IndexShardDescriptor {
  std::string path;
  std::string ref_id;
  uint32_t source_begin = 0;
  uint32_t source_end = 0;
  size_t window_count = 0;
  size_t sequence_count = 0;
  size_t world_node_count = 0;
};

struct ShardedIndexManifest {
  uint32_t format_version = 1;
  size_t window_length = 0;
  size_t stride = 0;
  size_t total_window_count = 0;
  size_t total_sequence_count = 0;
  size_t total_world_node_count = 0;
  std::string part_signature;
  std::vector<IndexShardDescriptor> shards;
};

bool is_sharded_index(const std::string& path);

void save_sharded_index_manifest(
    const std::string& path,
    const ShardedIndexManifest& manifest);

ShardedIndexManifest read_sharded_index_manifest(
    const std::string& path);

std::string resolve_index_shard_path(
    const std::string& manifest_path,
    const std::string& shard_path);

ShardedIndexManifest build_sharded_reference_index(
    const std::string& bundle_path,
    const std::string& ref_input,
    const std::string& reference_id,
    const std::string& reference_sequence,
    const std::vector<ReferenceContig>& reference_contigs,
    size_t window_length,
    size_t stride,
    size_t max_shard_windows,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config);

std::vector<LoadedIndex> load_sharded_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest);

}  // namespace navigamer

#endif  // NAVIGAMER_SHARDED_INDEX_HPP
