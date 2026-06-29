#ifndef NAVIGAMER_SEARCH_ENGINE_HPP
#define NAVIGAMER_SEARCH_ENGINE_HPP

#include "index_builder.hpp"
#include "qgram_filter.hpp"
#include "simd_mbb_filter.hpp"
#include "structure.hpp"
#include "tools.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace navigamer {

enum class MBBFilterMode {
  Scan,
  RectIndex,
};

enum class VisitedMode {
  StringSet,
  Epoch,
};

enum class GraphViewMode {
  Original,
  Flat,
};

const char* mbb_filter_mode_name(MBBFilterMode mode);
MBBFilterMode parse_mbb_filter_mode(const std::string& value);
const char* visited_mode_name(VisitedMode mode);
VisitedMode parse_visited_mode(const std::string& value);
const char* graph_view_mode_name(GraphViewMode mode);
GraphViewMode parse_graph_view_mode(const std::string& value);

struct SearchConfig {
  MBBFilterMode mbb_filter_mode = MBBFilterMode::Scan;
  VisitedMode visited_mode = VisitedMode::Epoch;
  GraphViewMode graph_view_mode = GraphViewMode::Flat;
  SimdMode simd_mode = SimdMode::Auto;
  DistanceMode distance_mode = DistanceMode::Myers;
  bool search_qgram_prefilter = false;
  bool search_prefetch = false;
  bool trace_paths = false;
  int search_qgram_q = 5;
};

struct SearchStats {
  size_t world_access_count = 0;
  size_t node_access_count = 0;
  size_t edge_access_count = 0;
  size_t anchor_distance_count = 0;
  size_t bound_check_count = 0;
  size_t candidate_count = 0;
  size_t candidate_verify_count = 0;
  size_t dist_calc_count = 0;
  size_t leaf_verify_count = 0;
  std::vector<size_t> layer_breakdown;
  size_t beacon_prune_count = 0;
  size_t candidate_count_for_prune = 0;
  size_t mbb_scan_child_checks = 0;
  size_t mbb_rect_index_queries = 0;
  size_t mbb_rect_candidate_children = 0;
  size_t mbb_rect_fallback_count = 0;
  size_t mbb_filter_parent_count = 0;
  size_t mbb_surviving_child_count = 0;
  size_t path_contained_step_count = 0;
  size_t path_overlap_step_count = 0;
  size_t path_uncovered_step_count = 0;
  size_t mbb_scalar_checks = 0;
  size_t mbb_simd_batches = 0;
  size_t mbb_simd_fallbacks = 0;
  size_t leaf_beacon_check_count = 0;
  size_t leaf_beacon_scalar_checks = 0;
  size_t leaf_beacon_simd_batches = 0;
  size_t leaf_beacon_simd_fallbacks = 0;
  size_t mbb_check_count = 0;
  size_t leaf_exact_distance_call_count = 0;
  size_t center_exact_distance_call_count = 0;
  size_t visited_check_count = 0;
  size_t visited_hit_count = 0;
  size_t center_distance_calls_after_mbb = 0;
  bool search_qgram_prefilter_enabled = false;
  bool search_prefetch_enabled = false;
  bool trace_paths_enabled = false;
  int search_qgram_q = 0;
  size_t search_qgram_signature_build_count = 0;
  size_t search_qgram_signature_missing_count = 0;
  size_t search_qgram_checks = 0;
  size_t search_qgram_pruned_children = 0;
  size_t search_qgram_passed_children = 0;
  size_t center_distance_calls_before_qgram = 0;
  size_t center_distance_calls_after_qgram = 0;
  size_t result_count = 0;
  std::vector<NodeId> world_trace;
  std::vector<LeafId> leaf_trace;

  SearchStats() = default;
  explicit SearchStats(size_t num_layers) : layer_breakdown(num_layers, 0) {}

  double pruning_rate() const {
    if (candidate_count_for_prune == 0) return 0.0;
    return static_cast<double>(beacon_prune_count) / candidate_count_for_prune;
  }

  double qgram_prune_ratio() const {
    if (search_qgram_checks == 0) return 0.0;
    return static_cast<double>(search_qgram_pruned_children) /
           search_qgram_checks;
  }

  void record_path_step(size_t overlap_count, bool contained) {
    if (contained) {
      path_contained_step_count++;
    } else if (overlap_count == 0) {
      path_uncovered_step_count++;
    } else {
      path_overlap_step_count++;
    }
  }

  std::string query_path_class() const {
    if (path_uncovered_step_count > 0) return "uncovered";
    if (path_overlap_step_count > 0) return "overlap";
    if (path_contained_step_count > 0) return "contained";
    return "unclassified";
  }
};

struct SearchScratch {
  std::vector<uint32_t> visited_epoch;
  uint32_t current_epoch = 0;
  std::vector<std::shared_ptr<WorldNode>> frontier;
  std::vector<std::shared_ptr<WorldNode>> next_frontier;
  std::vector<std::shared_ptr<WorldNode>> mbb_candidates;
  std::vector<std::shared_ptr<WorldNode>> verified_children;

  void begin_query(size_t node_count);
  bool is_visited(NodeId id) const;
  bool mark_visited(NodeId id);
};

class BioGeometrySearchEngine {
 public:
  explicit BioGeometrySearchEngine(
      const BioGeometryIndexBuilder& index,
      const SearchConfig& config = SearchConfig{});

  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_adaptive(const BioSequence& query_seq, int tolerance);

  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_greedy(const BioSequence& query_seq, int tolerance);

  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_exhaustive(const BioSequence& query_seq, int tolerance);

  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_brute_force(const BioSequence& query_seq, int tolerance,
                     const std::vector<std::shared_ptr<BioSequence>>& all_sequences);

 private:
  const BioGeometryIndexBuilder& index_;
  SearchConfig config_;
  std::unordered_map<std::string, QGramSignature> world_qgram_signatures_;

  bool mbb_prunable_row(const std::vector<MBB>& row, const std::vector<int>& V_Q,
                        int tolerance) const;
  bool leaf_beacon_prunable_row(const std::vector<int>& row,
                                const std::vector<int>& V_Q,
                                int tolerance) const;
  std::vector<int> compute_query_beacon_distances(
      const std::shared_ptr<WorldNode>& node,
      const BioSequence& query_seq,
      SearchStats& stats) const;
  std::vector<std::shared_ptr<WorldNode>> scan_mbb_surviving_children(
      const std::shared_ptr<WorldNode>& node,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats) const;
  std::vector<std::shared_ptr<WorldNode>> get_mbb_surviving_children(
      const std::shared_ptr<WorldNode>& node,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats) const;
  void verify_leaf_candidates(
      const std::shared_ptr<WorldNode>& node,
      const BioSequence& query_seq,
      int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      SearchStats& stats) const;
  void verify_leaf_candidates_view(
      NodeId node_id,
      const BioSequence& query_seq,
      int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      SearchStats& stats) const;

  void process_node_adaptive(
      const std::shared_ptr<WorldNode>& node, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats,
      const QGramSignature* query_qgram_signature) const;
  void process_node_adaptive_epoch(
      const std::shared_ptr<WorldNode>& node, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      SearchScratch& scratch,
      SearchStats& stats,
      const QGramSignature* query_qgram_signature) const;

  void search_layer_adaptive(
      const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats,
      bool after_mbb_filter,
      const QGramSignature* query_qgram_signature) const;
  void search_layer_adaptive_epoch(
      const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      SearchScratch& scratch,
      SearchStats& stats,
      bool after_mbb_filter,
      const QGramSignature* query_qgram_signature) const;

  bool flat_is_visited(
      NodeId node_id,
      const std::unordered_set<std::string>* visited_nodes,
      const SearchScratch* scratch) const;
  bool flat_mark_visited(
      NodeId node_id,
      std::unordered_set<std::string>* visited_nodes,
      SearchScratch* scratch) const;
  std::vector<NodeId> get_mbb_surviving_child_ids_view(
      NodeId node_id,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats) const;
  void process_node_adaptive_view(
      NodeId node_id, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>* visited_nodes,
      SearchScratch* scratch,
      SearchStats& stats,
      const QGramSignature* query_qgram_signature) const;
  void search_layer_adaptive_view(
      const std::vector<NodeId>& candidates, int layer_id,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>* visited_nodes,
      SearchScratch* scratch,
      SearchStats& stats,
      bool after_mbb_filter,
      const QGramSignature* query_qgram_signature) const;

  void traverse_exhaustive(
      const std::shared_ptr<WorldNode>& node, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats) const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_SEARCH_ENGINE_HPP
