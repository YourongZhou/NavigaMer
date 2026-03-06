#include "search_engine.hpp"
#include <algorithm>
#include <limits>

namespace navigamer {

BioGeometrySearchEngine::BioGeometrySearchEngine(const BioGeometryIndexBuilder& index)
    : index_(index) {
  for (int i = 0; i < 4; ++i)
    layer_beacons_[i] = index_.layer_beacons[i];
}

BioSequence BioGeometrySearchEngine::center_seq(const std::shared_ptr<WorldNode>& node) const {
  return BioSequence("_center", node->get_center_sequence());
}

std::vector<int> BioGeometrySearchEngine::compute_query_beacon_dists(
    const BioSequence& query_seq, int layer_id, SearchStats& stats) const {
  const auto& beacons = layer_beacons_[layer_id];
  if (beacons.empty()) return {};
  std::vector<int> V_Q;
  for (const auto& b : beacons) {
    int d = compute_distance(query_seq.seq, b->get_center_sequence());
    stats.dist_calc_count++;
    V_Q.push_back(d);
  }
  return V_Q;
}

bool BioGeometrySearchEngine::beacon_prunable(const std::shared_ptr<WorldNode>& child,
                                             const std::vector<int>& V_Q,
                                             int tolerance) const {
  if (child->beacon_dists.size() != V_Q.size()) return false;
  int lb = 0;
  for (size_t i = 0; i < V_Q.size(); ++i) {
    int v = std::abs(V_Q[i] - child->beacon_dists[i]);
    if (v > lb) lb = v;
  }
  return lb > child->radius + tolerance;
}

void BioGeometrySearchEngine::process_node_adaptive(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats) const {
  if (current_layer == 1) {
    for (const auto& child : node->child_leaves) {
      int leaf_dist = compute_distance(query_seq.seq, child->seq);
      stats.dist_calc_count++;
      stats.leaf_verify_count++;
      if (leaf_dist <= tolerance)
        unique_results[child->id] = child;
    }
    return;
  }

  std::vector<std::shared_ptr<WorldNode>> world_children = node->child_nodes;
  if (world_children.empty()) return;

  int child_layer = current_layer - 1;
  const auto& beacons = layer_beacons_[current_layer];
  std::vector<std::shared_ptr<WorldNode>> surviving;

  if (!beacons.empty()) {
    std::vector<int> V_Q = compute_query_beacon_dists(query_seq, current_layer, stats);
    if (!V_Q.empty()) {
      for (const auto& c : world_children) {
        stats.candidate_count_for_prune++;
        if (beacon_prunable(c, V_Q, tolerance)) {
          stats.beacon_prune_count++;
          continue;
        }
        surviving.push_back(c);
      }
    } else {
      surviving = world_children;
    }
  } else {
    surviving = world_children;
  }

  search_layer_adaptive(surviving, child_layer, query_seq, tolerance,
                        unique_results, visited_nodes, stats);
}

void BioGeometrySearchEngine::search_layer_adaptive(
    const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats) const {
  std::shared_ptr<WorldNode> contained_node;
  std::vector<std::shared_ptr<WorldNode>> overlap_nodes;

  for (const auto& node : candidates) {
    if (visited_nodes.count(node->node_id)) continue;

    int d = compute_distance(query_seq.seq, node->get_center_sequence());
    stats.dist_calc_count++;
    stats.node_access_count++;
    if (layer_id >= 1 && layer_id <= 3) stats.layer_breakdown[layer_id]++;

    if (d > node->radius + tolerance) continue;

    if (d + tolerance <= node->radius) {
      contained_node = node;
      break;
    }
    overlap_nodes.push_back(node);
  }

  if (contained_node) {
    visited_nodes.insert(contained_node->node_id);
    process_node_adaptive(contained_node, layer_id, query_seq, tolerance,
                          unique_results, visited_nodes, stats);
  } else {
    for (const auto& node : overlap_nodes) {
      if (visited_nodes.count(node->node_id)) continue;
      visited_nodes.insert(node->node_id);
      process_node_adaptive(node, layer_id, query_seq, tolerance,
                            unique_results, visited_nodes, stats);
    }
  }
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_adaptive(const BioSequence& query_seq, int tolerance) {
  SearchStats stats;
  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_results;
  std::unordered_set<std::string> visited_nodes;

  search_layer_adaptive(index_.layers[3], 3, query_seq, tolerance,
                        unique_results, visited_nodes, stats);

  std::vector<std::shared_ptr<BioSequence>> out;
  for (const auto& p : unique_results) out.push_back(p.second);
  return {out, stats};
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_greedy(const BioSequence& query_seq, int tolerance) {
  SearchStats stats;
  std::vector<std::shared_ptr<WorldNode>> current = index_.layers[3];

  for (int layer_id = 3; layer_id >= 1; --layer_id) {
    std::shared_ptr<WorldNode> best_node;
    int min_dist = std::numeric_limits<int>::max();

    for (const auto& node : current) {
      int d = compute_distance(query_seq.seq, node->get_center_sequence());
      stats.dist_calc_count++;
      stats.node_access_count++;
      if (layer_id >= 1 && layer_id <= 3) stats.layer_breakdown[layer_id]++;
      if (d <= node->radius + tolerance && d < min_dist) {
        min_dist = d;
        best_node = node;
      }
    }

    if (!best_node) return {{}, stats};

    if (layer_id == 1) {
      std::vector<std::shared_ptr<BioSequence>> results = best_node->child_leaves;
      return {results, stats};
    }

    std::vector<int> V_Q = compute_query_beacon_dists(query_seq, layer_id, stats);
    current.clear();
    for (const auto& child : best_node->child_nodes) {
      if (!V_Q.empty() && child->beacon_dists.size() == V_Q.size()) {
        stats.candidate_count_for_prune++;
        if (beacon_prunable(child, V_Q, tolerance)) {
          stats.beacon_prune_count++;
          continue;
        }
      }
      current.push_back(child);
    }
  }
  return {{}, stats};
}

void BioGeometrySearchEngine::traverse_exhaustive(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats) const {
  if (visited_nodes.count(node->node_id)) return;
  visited_nodes.insert(node->node_id);

  int dist = compute_distance(query_seq.seq, node->get_center_sequence());
  stats.dist_calc_count++;
  stats.node_access_count++;
  if (current_layer >= 1 && current_layer <= 3) stats.layer_breakdown[current_layer]++;

  if (dist > node->radius + tolerance) return;

  for (const auto& child : node->child_nodes)
    traverse_exhaustive(child, current_layer - 1, query_seq, tolerance,
                       unique_results, visited_nodes, stats);

  if (current_layer == 1) {
    for (const auto& child : node->child_leaves) {
      int leaf_dist = compute_distance(query_seq.seq, child->seq);
      stats.dist_calc_count++;
      stats.leaf_verify_count++;
      if (leaf_dist <= tolerance)
        unique_results[child->id] = child;
    }
  }
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_exhaustive(const BioSequence& query_seq, int tolerance) {
  SearchStats stats;
  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_results;
  std::unordered_set<std::string> visited_nodes;

  for (const auto& lw_node : index_.layers[3])
    traverse_exhaustive(lw_node, 3, query_seq, tolerance,
                        unique_results, visited_nodes, stats);

  std::vector<std::shared_ptr<BioSequence>> out;
  for (const auto& p : unique_results) out.push_back(p.second);
  return {out, stats};
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_brute_force(
    const BioSequence& query_seq, int tolerance,
    const std::vector<std::shared_ptr<BioSequence>>& all_sequences) {
  SearchStats stats;
  std::vector<std::shared_ptr<BioSequence>> results;
  for (const auto& seq : all_sequences) {
    int d = compute_distance(query_seq.seq, seq->seq);
    stats.dist_calc_count++;
    stats.leaf_verify_count++;
    if (d <= tolerance) results.push_back(seq);
  }
  return {results, stats};
}

}  // namespace navigamer
