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

// Query-side optimization contract:
// - RouterHint may be incomplete or wrong and may affect only ordering,
//   warm-starts, or candidate priority.
// - SafeBound may prune only when impossibility is conservatively proven.
// - ExactVerifier is the final authority for returned hits.
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
  bool query_profile = false;
  bool path_reuse_enabled = false;
  int near_query_max_neighbor_edit_distance = 8;
  double near_query_min_qgram_jaccard = 0.35;
  bool router_hint_enabled = false;
  int router_hint_qgram_q = 5;
  int router_hint_minimizer_k = 4;
  int router_hint_minimizer_w = 8;
  bool local_router_enabled = false;
  size_t local_router_max_anchors = 4;
  size_t local_router_max_children = 64;
  std::string local_router_score_mode = "anchor-envelope";
  bool best_first_enabled = false;
  bool safe_child_router_enabled = false;
  size_t safe_child_router_min_fanout = 64;
  size_t safe_child_router_max_candidates = 4096;
  double safe_child_router_max_ratio = 0.5;
  int safe_child_router_min_seed_len = 8;
  std::string safe_child_router_mode = "auto";
  bool safe_child_router_validate = false;
  bool proximal_oracle_enabled = false;
  bool query_planner_enabled = false;
  size_t planner_direct_verify_max_candidates = 32;
  size_t planner_router_min_fanout = 64;
  size_t planner_safe_child_router_min_fanout = 64;
  bool planner_allow_direct_qgram_verify = true;
};

struct SearchStats {
  bool query_profile_enabled = false;
  size_t query_count = 0;
  double query_total_ms = 0.0;
  double router_lookup_ms = 0.0;
  double anchor_distance_ms = 0.0;
  double mbb_filter_ms = 0.0;
  double child_bound_ms = 0.0;
  double center_distance_ms = 0.0;
  double best_first_queue_ms = 0.0;
  double leaf_collect_ms = 0.0;
  double leaf_mbb_filter_ms = 0.0;
  double leaf_verify_ms = 0.0;
  double result_dedup_ms = 0.0;
  double path_reuse_ms = 0.0;
  size_t world_access_count = 0;
  size_t node_access_count = 0;
  size_t edge_access_count = 0;
  size_t child_edge_considered_count = 0;
  size_t child_mbb_pruned_count = 0;
  size_t child_safe_bound_pruned_count = 0;
  size_t anchor_distance_count = 0;
  size_t center_distance_count = 0;
  size_t bound_check_count = 0;
  size_t frontier_max_size = 0;
  size_t frontier_total_pushed = 0;
  size_t contained_fastpath_count = 0;
  size_t overlap_fallback_count = 0;
  size_t leaf_world_count = 0;
  size_t raw_candidate_count = 0;
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
  size_t router_hint_invoked_count = 0;
  size_t router_qgram_signature_build_count = 0;
  size_t router_minimizer_signature_build_count = 0;
  size_t router_qgram_ranked_count = 0;
  size_t router_minimizer_ranked_count = 0;
  size_t router_pigeonhole_query_count = 0;
  size_t router_candidate_count = 0;
  size_t router_candidate_hit_count = 0;
  size_t router_fallback_count = 0;
  size_t unsafe_hint_ignored_count = 0;
  size_t path_reuse_attempt_count = 0;
  size_t path_reuse_hit_count = 0;
  size_t near_query_reuse_attempt_count = 0;
  size_t near_query_reuse_hit_count = 0;
	  size_t near_query_triangle_pruned_count = 0;
	  size_t near_query_center_distance_reused_count = 0;
	  size_t near_query_bound_fallback_count = 0;
	  size_t near_query_direct_verify_count = 0;
	  size_t near_query_leaf_triangle_pruned_count = 0;
	  size_t near_query_leaf_distance_reused_count = 0;
	  size_t near_query_leaf_bound_fallback_count = 0;
	  bool query_similarity_schedule_enabled = false;
  size_t query_similarity_cluster_count = 0;
  double query_similarity_mean_neighbor_distance = 0.0;
  size_t anchor_cache_hit_count = 0;
  size_t child_shortlist_reuse_hit_count = 0;
  size_t child_shortlist_cache_hit_count = 0;
  size_t safe_child_candidate_cache_hit_count = 0;
  size_t productive_world_reuse_hit_count = 0;
  size_t local_router_enabled_count = 0;
  size_t local_router_invoked_count = 0;
  size_t local_router_empty_count = 0;
  size_t local_router_shortlist_child_count = 0;
  size_t local_router_remaining_child_count = 0;
  size_t local_router_first_hit_rank = 0;
  size_t local_router_hit_in_topk_count = 0;
  size_t local_router_fallback_count = 0;
  size_t best_first_enabled_count = 0;
  size_t best_first_invoked_count = 0;
  size_t best_first_reordered_count = 0;
  size_t best_first_bound_candidate_count = 0;
  double safe_child_router_build_ms = 0.0;
  double safe_child_router_query_ms = 0.0;
  size_t safe_child_router_invoked_count = 0;
  size_t safe_child_router_skipped_low_fanout_count = 0;
  size_t safe_child_router_fallback_count = 0;
  size_t safe_child_router_candidate_count = 0;
  double safe_child_router_candidate_ratio_sum = 0.0;
  size_t safe_child_router_pruned_by_not_candidate_count = 0;
  size_t safe_child_router_exact_verify_count = 0;
  size_t safe_child_router_exact_pruned_count = 0;
  size_t safe_child_router_center_distance_reused_count = 0;
  size_t child_count_before_router = 0;
  size_t post_mbb_survivor_count = 0;
  size_t safe_router_candidate_count = 0;
  double candidate_ratio_to_all_children = 0.0;
  double candidate_ratio_to_post_mbb_survivors = 0.0;
  size_t children_actually_processed = 0;
  size_t center_checks_saved = 0;
  size_t planner_invoked_count = 0;
  size_t planner_strategy_baseline_count = 0;
  size_t planner_strategy_direct_qgram_count = 0;
  size_t planner_strategy_router_count = 0;
  size_t planner_strategy_safe_child_router_count = 0;
  size_t planner_strategy_path_reuse_count = 0;
  size_t planner_near_reuse_enabled_count = 0;
  size_t planner_near_reuse_disabled_count = 0;
  size_t planner_fallback_count = 0;
  double planner_decision_ms = 0.0;
  bool planner_disable_router_stack = false;
  bool planner_disable_router_ordering = false;
  std::vector<std::string> proximal_actual_anchor_node_ids;
  std::vector<std::string> proximal_frontier_node_ids;
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

  std::vector<std::string> debug_safe_child_router_candidate_ids(
      const std::string& parent_node_id,
      const BioSequence& query_seq,
      int tolerance,
      bool* used_router) const;

 private:
  struct ParentRouterHintIndex {
    ExactRangeJoinIndex range_index;
    std::unordered_map<std::string, size_t> child_item_ids_by_node_id;
  };

  struct ParentSafeChildRouterIndex {
    struct RadiusBucket {
      int radius = 0;
      ExactRangeJoinIndex range_index;
    };
    size_t child_count = 0;
    int max_child_radius = 0;
    std::vector<RadiusBucket> radius_buckets;
  };

  const BioGeometryIndexBuilder& index_;
  SearchConfig config_;
  std::unordered_map<int, std::unordered_map<std::string, QGramSignature>>
      world_qgram_signatures_by_q_;
  std::unordered_map<std::string, std::vector<uint64_t>>
      world_minimizer_signatures_;
  std::unordered_map<std::string, ParentRouterHintIndex>
      parent_router_hint_indexes_;
  std::unordered_map<std::string, ParentSafeChildRouterIndex>
      parent_safe_child_router_indexes_;
  double safe_child_router_build_ms_ = 0.0;

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
  std::vector<size_t> safe_child_router_candidate_indices(
      const std::shared_ptr<WorldNode>& node,
      const BioSequence& query_seq,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats,
      bool* used_router) const;
  std::vector<std::shared_ptr<WorldNode>> scan_mbb_surviving_child_indices(
      const std::shared_ptr<WorldNode>& node,
      const std::vector<size_t>& child_indices,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats) const;
  std::vector<std::shared_ptr<WorldNode>> rank_children_with_router_hints(
      const std::shared_ptr<WorldNode>& node,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      const BioSequence& query_seq,
      int tolerance,
      SearchStats& stats,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;
  std::vector<std::shared_ptr<WorldNode>> rank_children_with_local_router(
      const std::shared_ptr<WorldNode>& node,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      const std::vector<int>& query_beacon_dists,
      SearchStats& stats) const;
  std::vector<std::shared_ptr<WorldNode>> rank_children_with_best_first(
      const std::shared_ptr<WorldNode>& node,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats) const;
  std::vector<std::shared_ptr<WorldNode>> apply_path_reuse_order(
      const std::shared_ptr<WorldNode>& parent,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
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
      const QGramSignature* query_qgram_signature,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;
  void process_node_adaptive_epoch(
      const std::shared_ptr<WorldNode>& node, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      SearchScratch& scratch,
      SearchStats& stats,
      const QGramSignature* query_qgram_signature,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;

  void search_layer_adaptive(
      const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats,
      bool after_mbb_filter,
      const QGramSignature* query_qgram_signature,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;
  void search_layer_adaptive_epoch(
      const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      SearchScratch& scratch,
      SearchStats& stats,
      bool after_mbb_filter,
      const QGramSignature* query_qgram_signature,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;

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
  std::vector<NodeId> scan_mbb_surviving_child_ids_view(
      NodeId node_id,
      const std::vector<size_t>& child_offsets,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats) const;
  std::vector<NodeId> rank_child_ids_with_router_hints_view(
      NodeId node_id,
      const std::vector<NodeId>& candidates,
      const BioSequence& query_seq,
      int tolerance,
      SearchStats& stats,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;
  std::vector<NodeId> rank_child_ids_with_local_router_view(
      NodeId node_id,
      const std::vector<NodeId>& candidates,
      const std::vector<int>& query_beacon_dists,
      SearchStats& stats) const;
  std::vector<NodeId> rank_child_ids_with_best_first_view(
      NodeId node_id,
      const std::vector<NodeId>& candidates,
      const std::vector<int>& query_beacon_dists,
      int tolerance,
      SearchStats& stats) const;
  std::vector<NodeId> apply_path_reuse_order_view(
      NodeId node_id,
      const std::vector<NodeId>& candidates,
      SearchStats& stats) const;
  bool near_query_triangle_prunes_center(
      const std::string& node_id,
      int tau,
      SearchStats& stats) const;
  int compute_center_distance_for_search(
      const BioSequence& query_seq,
      const std::string& node_id,
      const std::string& center_sequence,
      int tau,
      bool after_mbb_filter,
      bool* cache_hit = nullptr) const;
  void process_node_adaptive_view(
      NodeId node_id, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>* visited_nodes,
      SearchScratch* scratch,
      SearchStats& stats,
      const QGramSignature* query_qgram_signature,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;
  void search_layer_adaptive_view(
      const std::vector<NodeId>& candidates, int layer_id,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>* visited_nodes,
      SearchScratch* scratch,
      SearchStats& stats,
      bool after_mbb_filter,
      const QGramSignature* query_qgram_signature,
      const QGramSignature* router_qgram_signature,
      const std::vector<uint64_t>* router_minimizers) const;

  void traverse_exhaustive(
      const std::shared_ptr<WorldNode>& node, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats) const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_SEARCH_ENGINE_HPP
