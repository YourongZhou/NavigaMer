#ifndef NAVIGAMER_SEARCH_ENGINE_HPP
#define NAVIGAMER_SEARCH_ENGINE_HPP

#include "index_builder.hpp"
#include "structure.hpp"
#include "tools.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace navigamer {

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

  SearchStats() = default;
  explicit SearchStats(size_t num_layers) : layer_breakdown(num_layers, 0) {}

  double pruning_rate() const {
    if (candidate_count_for_prune == 0) return 0.0;
    return static_cast<double>(beacon_prune_count) / candidate_count_for_prune;
  }
};

class BioGeometrySearchEngine {
 public:
  explicit BioGeometrySearchEngine(const BioGeometryIndexBuilder& index);

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

  bool mbb_prunable_row(const std::vector<MBB>& row, const std::vector<int>& V_Q,
                        int tolerance) const;
  bool leaf_beacon_prunable_row(const std::vector<int>& row,
                                const std::vector<int>& V_Q,
                                int tolerance) const;
  std::vector<int> compute_query_beacon_distances(
      const std::shared_ptr<WorldNode>& node,
      const BioSequence& query_seq,
      SearchStats& stats) const;
  void verify_leaf_candidates(
      const std::shared_ptr<WorldNode>& node,
      const BioSequence& query_seq,
      int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      SearchStats& stats) const;

  void process_node_adaptive(
      const std::shared_ptr<WorldNode>& node, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats) const;

  void search_layer_adaptive(
      const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats) const;

  void traverse_exhaustive(
      const std::shared_ptr<WorldNode>& node, int current_layer,
      const BioSequence& query_seq, int tolerance,
      std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
      std::unordered_set<std::string>& visited_nodes,
      SearchStats& stats) const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_SEARCH_ENGINE_HPP
