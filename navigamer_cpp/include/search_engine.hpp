#ifndef NAVIGAMER_SEARCH_ENGINE_HPP
#define NAVIGAMER_SEARCH_ENGINE_HPP

#include "structure.hpp"
#include "index_builder.hpp"
#include "tools.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>

namespace navigamer {

struct SearchStats {
  size_t node_access_count = 0;
  size_t dist_calc_count = 0;
  size_t leaf_verify_count = 0;
  size_t layer_breakdown[4] = {0, 0, 0, 0};  // LW, MW, SW
  size_t beacon_prune_count = 0;
  size_t candidate_count_for_prune = 0;

  double pruning_rate() const {
    if (candidate_count_for_prune == 0) return 0.0;
    return static_cast<double>(beacon_prune_count) / candidate_count_for_prune;
  }
};

class BioGeometrySearchEngine {
 public:
  explicit BioGeometrySearchEngine(const BioGeometryIndexBuilder& index);

  // 自适应搜索 (v7): 0 FN + Sibling Pruning + Beacon Pruning
  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_adaptive(const BioSequence& query_seq, int tolerance);

  // 贪婪单路径
  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_greedy(const BioSequence& query_seq, int tolerance);

  // 穷举全路径，保证 0 FN
  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_exhaustive(const BioSequence& query_seq, int tolerance);

  // 暴力全量扫描 (ground truth)
  std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
  search_brute_force(const BioSequence& query_seq, int tolerance,
                    const std::vector<std::shared_ptr<BioSequence>>& all_sequences);

 private:
  const BioGeometryIndexBuilder& index_;
  std::vector<std::shared_ptr<WorldNode>> layer_beacons_[4];

  BioSequence center_seq(const std::shared_ptr<WorldNode>& node) const;
  std::vector<int> compute_query_beacon_dists(const BioSequence& query_seq,
                                              int layer_id, SearchStats& stats) const;
  bool beacon_prunable(const std::shared_ptr<WorldNode>& child,
                       const std::vector<int>& V_Q, int tolerance) const;

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
