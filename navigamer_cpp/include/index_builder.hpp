#ifndef NAVIGAMER_INDEX_BUILDER_HPP
#define NAVIGAMER_INDEX_BUILDER_HPP

#include "structure.hpp"
#include "tools.hpp"
#include "range_join.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace navigamer {

struct HierarchyConfig {
  std::vector<int> primary_radii;
  std::vector<int> auxiliary_radii;

  HierarchyConfig() = default;
  explicit HierarchyConfig(std::vector<int> primary_radii_in);
  HierarchyConfig(std::vector<int> primary_radii_in, std::vector<int> auxiliary_radii_in);

  int num_primary_layers() const;
  int num_auxiliary_layers() const;
  int num_expanded_layers() const;
  void validate() const;
};

enum class BuildRangeMode {
  Full,
  Indexed,
};

const char* build_range_mode_name(BuildRangeMode mode);
BuildRangeMode parse_build_range_mode(const std::string& value);

enum class BuildDistanceMode {
  DP,
  Edlib,
  Auto,
};

const char* build_distance_mode_name(BuildDistanceMode mode);
BuildDistanceMode parse_build_distance_mode(const std::string& value);

struct BuildRangeConfig {
  BuildRangeMode link_mode = BuildRangeMode::Indexed;
  BuildRangeMode leaf_attach_mode = BuildRangeMode::Indexed;
  BuildDistanceMode distance_mode = BuildDistanceMode::DP;
  RangeJoinConfig range_join;
  size_t min_rect_index_fanout = 64;
};

struct SearchGraphView {
  std::vector<std::shared_ptr<WorldNode>> nodes;
  std::vector<std::shared_ptr<BioSequence>> leaves;

  std::vector<NodeId> child_ids;
  std::vector<uint32_t> child_begin;
  std::vector<uint32_t> child_end;

  std::vector<LeafId> leaf_ids;
  std::vector<uint32_t> leaf_begin;
  std::vector<uint32_t> leaf_end;

  std::vector<int32_t> mbb_lo;
  std::vector<int32_t> mbb_hi;
  std::vector<uint32_t> mbb_begin;
  std::vector<uint32_t> mbb_dim;
  std::vector<LeafId> beacon_ids;
  std::vector<uint32_t> beacon_begin;
  std::vector<uint32_t> beacon_end;

  std::vector<int32_t> leaf_beacon_dists;
  std::vector<uint32_t> leaf_beacon_begin;
  std::vector<uint32_t> leaf_beacon_dim;
};

class BioGeometryIndexBuilder {
 public:
  BioGeometryIndexBuilder();
  BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw);
  explicit BioGeometryIndexBuilder(const HierarchyConfig& config);
  BioGeometryIndexBuilder(const HierarchyConfig& config,
                          const BuildRangeConfig& range_config);

  void build(const std::vector<std::shared_ptr<BioSequence>>& raw_sequences);

  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_sequences;

  struct Statistics {
    size_t added_sequences = 0;
    size_t unique_sequences = 0;
    size_t deduplicated = 0;
    size_t created_auxiliary_nodes = 0;
    std::vector<size_t> created_primary_nodes;
    double compression_ratio = 0.0;
    double dag_redundancy = 0.0;
    size_t phase2_total_possible_pairs = 0;
    size_t phase2_candidate_pairs = 0;
    size_t phase2_exact_distance_calls = 0;
    size_t phase2_edges_added = 0;
    size_t phase2_full_scan_fallback_count = 0;
    size_t phase2_pigeonhole_queries = 0;
    size_t phase2_qgram_queries = 0;
    size_t phase2_hybrid_queries = 0;
    size_t phase2_qgram_candidate_pairs = 0;
    size_t phase2_qgram_pruned_by_l1 = 0;
    size_t phase2_length_pruned_pairs = 0;
    size_t phase2_required_shared_nonpositive_count = 0;
    size_t phase2_auto_pigeonhole_accepted = 0;
    size_t phase2_auto_pigeonhole_rejected_large_candidates = 0;
    size_t phase2_auto_qgram_invoked = 0;
    size_t phase2_auto_hybrid_invoked = 0;
    size_t phase2_auto_final_candidate_pairs = 0;
    double phase2_auto_candidate_ratio_sum = 0.0;
    double phase2_auto_candidate_ratio_avg = 0.0;
    double phase2_candidate_reduction_ratio = 0.0;
    double phase2_exact_distance_reduction_ratio = 0.0;
    size_t total_possible_leaf_pairs = 0;
    size_t leaf_candidate_pairs = 0;
    size_t leaf_exact_distance_calls = 0;
    size_t leaf_attachments_added = 0;
    size_t leaf_full_scan_fallback_count = 0;
    size_t leaf_pigeonhole_queries = 0;
    size_t leaf_qgram_queries = 0;
    size_t leaf_hybrid_queries = 0;
    size_t leaf_qgram_candidate_pairs = 0;
    size_t leaf_qgram_pruned_by_l1 = 0;
    size_t leaf_length_pruned_pairs = 0;
    size_t leaf_required_shared_nonpositive_count = 0;
    size_t leaf_auto_pigeonhole_accepted = 0;
    size_t leaf_auto_pigeonhole_rejected_large_candidates = 0;
    size_t leaf_auto_qgram_invoked = 0;
    size_t leaf_auto_hybrid_invoked = 0;
    size_t leaf_auto_final_candidate_pairs = 0;
    double leaf_auto_candidate_ratio_sum = 0.0;
    double leaf_auto_candidate_ratio_avg = 0.0;
    double leaf_candidate_reduction_ratio = 0.0;
    double leaf_exact_distance_reduction_ratio = 0.0;
  };
  Statistics get_statistics() const;
  const HierarchyConfig& hierarchy_config() const { return hierarchy_; }
  const BuildRangeConfig& build_range_config() const { return range_config_; }
  int num_primary_layers() const { return hierarchy_.num_primary_layers(); }
  int num_expanded_layers() const { return hierarchy_.num_expanded_layers(); }
  int coarsest_primary_layer_index() const { return 0; }
  int finest_primary_layer_index() const { return std::max(0, num_primary_layers() - 1); }
  const std::vector<std::shared_ptr<WorldNode>>& primary_layer(int idx) const;
  const std::vector<std::vector<std::shared_ptr<WorldNode>>>& primary_layers() const {
    return primary_layers_;
  }
  size_t num_world_nodes() const { return world_node_count_; }
  size_t num_sequences() const { return sequence_count_; }
  bool validate_integer_ids() const;
  const SearchGraphView& search_graph_view() const { return search_graph_view_; }
  bool validate_search_graph_view() const;

  std::vector<std::shared_ptr<WorldNode>> find_neighbors(
      const BioSequence& query_seq,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      int radius) const;

 private:
  Statistics stats_;
  HierarchyConfig hierarchy_;
  BuildRangeConfig range_config_;
  size_t world_node_count_ = 0;
  size_t sequence_count_ = 0;
  SearchGraphView search_graph_view_;
  std::vector<int> expanded_radii_;
  std::vector<std::vector<std::shared_ptr<WorldNode>>> extended_layers_;
  std::vector<std::vector<std::shared_ptr<WorldNode>>> primary_layers_;

  std::vector<std::shared_ptr<BioSequence>> deduplicate(
      const std::vector<std::shared_ptr<BioSequence>>& raw);

  void phase1_build_extended_sketch(const std::vector<std::shared_ptr<BioSequence>>& unique_seqs);
  void phase2_inter_tier_rebinding();
  void phase3_collapse_and_compute_mbb();

  void attach_leaves(const std::vector<std::shared_ptr<BioSequence>>& unique_seqs);
  void assign_integer_ids();
  void build_search_graph_view();

  void print_summary() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_INDEX_BUILDER_HPP
