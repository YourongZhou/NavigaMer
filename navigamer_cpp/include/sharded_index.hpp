#ifndef NAVIGAMER_SHARDED_INDEX_HPP
#define NAVIGAMER_SHARDED_INDEX_HPP

#include "index_persistence.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace navigamer {

struct IndexedReferenceFile;

struct IndexShardDescriptor {
  uint64_t file_offset = 0;
  uint64_t file_size = 0;
  uint32_t pack_id = 0;
  uint32_t contig_id = 0;
  uint32_t source_begin = 0;
  uint32_t source_end = 0;
  uint32_t window_count = 0;
  uint32_t sequence_count = 0;
  uint32_t world_node_count = 0;
};
static_assert(sizeof(IndexShardDescriptor) <= 48,
              "shard descriptors must remain compact");

struct ShardedIndexManifest {
  uint32_t format_version = 4;
  size_t window_length = 0;
  size_t stride = 0;
  size_t total_window_count = 0;
  size_t total_sequence_count = 0;
  size_t total_world_node_count = 0;
  uint32_t router_k = 0;
  uint32_t router_window = 0;
  size_t router_entry_count = 0;
  uint64_t router_checksum = 0;
  std::string part_signature;
  std::vector<std::string> pack_paths;
  std::vector<std::string> contig_ids;
  std::vector<IndexShardDescriptor> shards;
};

struct ShardRouteSelection {
  bool enabled = false;
  std::vector<uint32_t> shard_ids;
};

// Sorted minimizer codes and bit-packed parallel shard IDs are memory-mapped
// from the router sidecar. Exact query blocks provide a no-false-negative
// necessary condition; unsupported queries conservatively disable routing.
struct ShardedSeedRouter {
  uint32_t k = 0;
  uint32_t window = 0;
  uint32_t shard_count = 0;
  uint32_t shard_id_bits = 0;
  FinalArray<uint32_t> minimizer_codes;
  FinalArray<uint8_t> packed_shard_ids;

  bool enabled() const {
    return k != 0 && window >= k && shard_count != 0 &&
           shard_id_bits != 0 && !minimizer_codes.empty() &&
           !packed_shard_ids.empty();
  }
  ShardRouteSelection select(
      std::string_view query, int tolerance) const;
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

ShardedSeedRouter load_sharded_seed_router(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest);

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
    const BuildRangeConfig& range_config,
    size_t build_jobs = 0);

ShardedIndexManifest build_sharded_reference_index(
    const std::string& bundle_path,
    const std::string& ref_input,
    const IndexedReferenceFile& reference,
    size_t window_length,
    size_t stride,
    size_t max_shard_windows,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config,
    size_t build_jobs = 0);

std::vector<LoadedIndex> load_sharded_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest);

std::vector<LoadedIndex> load_sharded_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest,
    const std::vector<uint32_t>& shard_ids);

}  // namespace navigamer

#endif  // NAVIGAMER_SHARDED_INDEX_HPP
