#include "search_engine.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <omp.h>

namespace navigamer {

BioGeometrySearchEngine::BioGeometrySearchEngine(const BioGeometryIndexBuilder& index)
    : index_(index) {}

bool BioGeometrySearchEngine::mbb_prunable_row(const std::vector<MBB>& row,
                                               const std::vector<int>& V_Q,
                                               int tolerance) const {
  if (row.size() != V_Q.size()) return false;
  for (size_t i = 0; i < V_Q.size(); ++i) {
    int q_b = V_Q[i];
    if (q_b < row[i].min_dist - tolerance || q_b > row[i].max_dist + tolerance)
      return true;
  }
  return false;
}

bool BioGeometrySearchEngine::leaf_beacon_prunable_row(const std::vector<int>& row,
                                                       const std::vector<int>& V_Q,
                                                       int tolerance) const {
  if (row.size() != V_Q.size()) return false;
  for (size_t i = 0; i < V_Q.size(); ++i) {
    if (std::abs(V_Q[i] - row[i]) > tolerance) return true;
  }
  return false;
}

std::vector<int> BioGeometrySearchEngine::compute_query_beacon_distances(
    const std::shared_ptr<WorldNode>& node,
    const BioSequence& query_seq,
    SearchStats& stats) const {
  std::vector<int> V_Q;
  V_Q.reserve(node->beacons.size());
  for (const auto& b : node->beacons) {
    if (!b) {
      V_Q.push_back(0);
      continue;
    }
    V_Q.push_back(compute_distance(query_seq.seq, b->seq));
    stats.dist_calc_count++;
  }
  return V_Q;
}

void BioGeometrySearchEngine::verify_leaf_candidates(
    const std::shared_ptr<WorldNode>& node,
    const BioSequence& query_seq,
    int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    SearchStats& stats) const {
  std::vector<int> V_Q;
  const bool has_leaf_sieve =
      !node->beacons.empty() &&
      node->leaf_beacon_dists.size() == node->child_leaves.size();
  if (has_leaf_sieve) V_Q = compute_query_beacon_distances(node, query_seq, stats);

  for (size_t li = 0; li < node->child_leaves.size(); ++li) {
    if (has_leaf_sieve && node->leaf_beacon_dists[li].size() == V_Q.size()) {
      stats.candidate_count_for_prune++;
      if (leaf_beacon_prunable_row(node->leaf_beacon_dists[li], V_Q, tolerance)) {
        stats.beacon_prune_count++;
        continue;
      }
    }

    const auto& child = node->child_leaves[li];
    int leaf_dist = compute_distance(query_seq.seq, child->seq);
    stats.dist_calc_count++;
    stats.leaf_verify_count++;
    if (leaf_dist <= tolerance)
      unique_results[child->id] = child;
  }
}

void BioGeometrySearchEngine::process_node_adaptive(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats) const {
  if (current_layer == 1) {
    verify_leaf_candidates(node, query_seq, tolerance, unique_results, stats);
    return;
  }

  std::vector<std::shared_ptr<WorldNode>> world_children = node->child_nodes;
  if (world_children.empty()) return;

  int child_layer = current_layer - 1;
  std::vector<std::shared_ptr<WorldNode>> surviving;

  if (!node->beacons.empty()) {
    std::vector<int> V_Q = compute_query_beacon_distances(node, query_seq, stats);

    bool mbb_ok =
        V_Q.size() == node->beacons.size() &&
        node->child_beacon_mbbs.size() == world_children.size();
    if (mbb_ok) {
      for (size_t ci = 0; ci < world_children.size(); ++ci) {
        if (node->child_beacon_mbbs[ci].size() != V_Q.size()) {
          mbb_ok = false;
          break;
        }
      }
    }

    if (mbb_ok) {
      for (size_t ci = 0; ci < world_children.size(); ++ci) {
        stats.candidate_count_for_prune++;
        if (mbb_prunable_row(node->child_beacon_mbbs[ci], V_Q, tolerance)) {
          stats.beacon_prune_count++;
          continue;
        }
        surviving.push_back(world_children[ci]);
      }
    } else {
      surviving = std::move(world_children);
    }
  } else {
    surviving = std::move(world_children);
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
      std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_results;
      verify_leaf_candidates(best_node, query_seq, tolerance, unique_results, stats);

      std::vector<std::shared_ptr<BioSequence>> results;
      for (const auto& p : unique_results) results.push_back(p.second);
      return {results, stats};
    }

    std::vector<int> V_Q = compute_query_beacon_distances(best_node, query_seq, stats);

    current.clear();
    for (size_t ci = 0; ci < best_node->child_nodes.size(); ++ci) {
      const auto& child = best_node->child_nodes[ci];
      if (!V_Q.empty() && ci < best_node->child_beacon_mbbs.size() &&
          best_node->child_beacon_mbbs[ci].size() == V_Q.size()) {
        stats.candidate_count_for_prune++;
        if (mbb_prunable_row(best_node->child_beacon_mbbs[ci], V_Q, tolerance)) {
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
    verify_leaf_candidates(node, query_seq, tolerance, unique_results, stats);
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
  std::vector<std::vector<std::shared_ptr<BioSequence>>> thread_results;

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    #pragma omp single
    thread_results.resize(nthreads);

    int tid = omp_get_thread_num();
    #pragma omp for schedule(dynamic, 64)
    for (size_t i = 0; i < all_sequences.size(); ++i) {
      int d = compute_distance(query_seq.seq, all_sequences[i]->seq);
      if (d <= tolerance)
        thread_results[tid].push_back(all_sequences[i]);
    }
  }

  std::vector<std::shared_ptr<BioSequence>> results;
  for (auto& tr : thread_results)
    for (auto& r : tr)
      results.push_back(std::move(r));

  stats.dist_calc_count = all_sequences.size();
  stats.leaf_verify_count = all_sequences.size();
  return {results, stats};
}

}  // namespace navigamer
