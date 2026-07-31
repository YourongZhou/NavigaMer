#ifndef NAVIGAMER_INDEX_PERSISTENCE_HPP
#define NAVIGAMER_INDEX_PERSISTENCE_HPP

#include "index_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace navigamer {

struct IndexBuildManifest {
  uint32_t format_version = 20;
  std::string signature;
  std::string ref_input;
  std::string reads_input;
  std::string ref_fingerprint;
  std::string reads_fingerprint;
  std::vector<int> primary_radii;
  std::vector<int> auxiliary_radii;
  std::string link_mode;
  std::string leaf_attach_mode;
  std::string leaf_attach_direction;
  std::string build_distance_mode;
  std::string phase1_candidate_mode;
  std::string range_candidate_mode;
  int range_min_seed_len = 6;
  int range_max_seed_len = 20;
  int qgram_q = 5;
  size_t auto_pigeonhole_max_candidates = 4096;
  double auto_pigeonhole_max_ratio = 0.25;
  bool auto_hybrid_on_large_candidates = true;
  size_t min_rect_index_fanout = 64;
  size_t phase1_metric_min_fanout = 12;
  size_t phase1_qgram_min_fanout = 12;
  size_t phase1_qgram_max_touched = 250000;
  bool phase2_qgram_postfilter = false;
  bool leaf_qgram_postfilter = false;
  size_t sequence_count = 0;
  size_t world_node_count = 0;
  size_t edge_count = 0;
  size_t leaf_link_count = 0;
};

struct LoadedIndex {
  BioGeometryIndexBuilder builder;
  IndexBuildManifest manifest;
};

enum class IndexLoadValidation {
  Full,
  Structural,
};

IndexBuildManifest make_index_manifest(
    const std::string& ref_input,
    const std::string& reads_input,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config);

IndexBuildManifest make_reference_window_index_manifest(
    const std::string& ref_input,
    size_t actual_prefix_length,
    int window_size,
    int stride,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config);

IndexBuildManifest read_index_manifest(const std::string& path);

bool index_matches_manifest(
    const std::string& path,
    const IndexBuildManifest& expected,
    IndexBuildManifest* stored = nullptr,
    std::string* reason = nullptr);

void save_index(const std::string& path,
                const BioGeometryIndexBuilder& builder,
                const IndexBuildManifest& manifest);

LoadedIndex load_index(
    const std::string& path,
    IndexLoadValidation validation =
        IndexLoadValidation::Full);

}  // namespace navigamer

#endif  // NAVIGAMER_INDEX_PERSISTENCE_HPP
