#include "search_engine.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace navigamer {

const char* mbb_filter_mode_name(MBBFilterMode mode) {
  return mode == MBBFilterMode::Scan ? "scan" : "rect";
}

MBBFilterMode parse_mbb_filter_mode(const std::string& value) {
  if (value == "scan") return MBBFilterMode::Scan;
  if (value == "rect") return MBBFilterMode::RectIndex;
  throw std::invalid_argument("MBB filter mode must be scan or rect");
}

BioGeometrySearchEngine::BioGeometrySearchEngine(
    const BioGeometryIndexBuilder& index, const SearchConfig& config)
    : index_(index), config_(config) {
  if (!config_.search_qgram_prefilter || config_.search_qgram_q <= 0) return;
  for (const auto& layer : index_.primary_layers()) {
    for (const auto& node : layer) {
      if (!node || !node->center_ptr ||
          world_qgram_signatures_.count(node->node_id)) {
        continue;
      }
      world_qgram_signatures_.emplace(
          node->node_id,
          compute_qgram_signature(node->center_ptr->seq, config_.search_qgram_q));
    }
  }
}

bool BioGeometrySearchEngine::mbb_prunable_row(const std::vector<MBB>& row,
                                               const std::vector<int>& V_Q,
                                               int tolerance) const {
  if (row.size() != V_Q.size()) return false;
  for (size_t i = 0; i < V_Q.size(); ++i) {
    int q_b = V_Q[i];
    if (q_b < row[i].min_dist - tolerance || q_b > row[i].max_dist + tolerance) {
      return true;
    }
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
  std::vector<int> dists;
  dists.reserve(node->beacons.size());
  for (const auto& beacon : node->beacons) {
    if (!beacon) {
      dists.push_back(0);
      continue;
    }
    dists.push_back(compute_distance(query_seq.seq, beacon->seq));
    stats.anchor_distance_count++;
    stats.dist_calc_count++;
  }
  return dists;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::scan_mbb_surviving_children(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
    SearchStats& stats) const {
  std::vector<std::shared_ptr<WorldNode>> surviving;
  const auto& children = node->child_nodes;
  bool mbb_ok =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == node->beacons.size() &&
      node->child_beacon_mbbs.size() == children.size();
  if (mbb_ok) {
    for (const auto& row : node->child_beacon_mbbs) {
      if (row.size() != query_beacon_dists.size()) {
        mbb_ok = false;
        break;
      }
    }
  }

  if (!mbb_ok) {
    surviving = children;
  } else {
    surviving.reserve(children.size());
    for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
      stats.edge_access_count++;
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      stats.mbb_scan_child_checks++;
      if (mbb_prunable_row(
              node->child_beacon_mbbs[child_idx], query_beacon_dists, tolerance)) {
        stats.beacon_prune_count++;
        continue;
      }
      surviving.push_back(children[child_idx]);
    }
  }
  stats.mbb_surviving_child_count += surviving.size();
  return surviving;
}

std::vector<std::shared_ptr<WorldNode>>
BioGeometrySearchEngine::get_mbb_surviving_children(
    const std::shared_ptr<WorldNode>& node,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
    SearchStats& stats) const {
  stats.mbb_filter_parent_count++;
  if (config_.mbb_filter_mode == MBBFilterMode::Scan) {
    return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
  }

  const auto& children = node->child_nodes;
  bool index_ok =
      node->mbb_rect_index &&
      node->mbb_rect_index->size() == children.size() &&
      node->mbb_rect_index->dim() == query_beacon_dists.size() &&
      !query_beacon_dists.empty() &&
      node->child_beacon_mbbs.size() == children.size();
  if (index_ok) {
    for (const auto& row : node->child_beacon_mbbs) {
      if (row.size() != query_beacon_dists.size()) {
        index_ok = false;
        break;
      }
    }
  }
  if (!index_ok) {
    stats.mbb_rect_fallback_count++;
    return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
  }

  try {
    std::vector<int> q_lo;
    std::vector<int> q_hi;
    q_lo.reserve(query_beacon_dists.size());
    q_hi.reserve(query_beacon_dists.size());
    for (int distance : query_beacon_dists) {
      q_lo.push_back(distance - tolerance);
      q_hi.push_back(distance + tolerance);
    }

    stats.mbb_rect_index_queries++;
    auto child_ids = node->mbb_rect_index->query_intersect(q_lo, q_hi);
    std::vector<bool> seen(children.size(), false);
    std::vector<std::shared_ptr<WorldNode>> surviving;
    surviving.reserve(child_ids.size());
    for (uint32_t child_id : child_ids) {
      if (child_id >= children.size() || seen[child_id]) {
        stats.mbb_rect_fallback_count++;
        return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
      }
      seen[child_id] = true;
      surviving.push_back(children[child_id]);
    }

    stats.edge_access_count += children.size();
    stats.candidate_count_for_prune += children.size();
    stats.bound_check_count += children.size();
    stats.beacon_prune_count += children.size() - surviving.size();
    stats.mbb_rect_candidate_children += surviving.size();
    stats.mbb_surviving_child_count += surviving.size();
    return surviving;
  } catch (...) {
    stats.mbb_rect_fallback_count++;
    return scan_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
  }
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

  for (size_t leaf_idx = 0; leaf_idx < node->child_leaves.size(); ++leaf_idx) {
    stats.node_access_count++;
    if (has_leaf_sieve && node->leaf_beacon_dists[leaf_idx].size() == V_Q.size()) {
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      if (leaf_beacon_prunable_row(node->leaf_beacon_dists[leaf_idx], V_Q, tolerance)) {
        stats.beacon_prune_count++;
        continue;
      }
    }

    const auto& child = node->child_leaves[leaf_idx];
    stats.candidate_count++;
    stats.candidate_verify_count++;
    int leaf_dist = compute_distance(query_seq.seq, child->seq);
    stats.dist_calc_count++;
    stats.leaf_verify_count++;
    if (leaf_dist <= tolerance) unique_results[child->id] = child;
  }
}

void BioGeometrySearchEngine::process_node_adaptive(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats,
    const QGramSignature* query_qgram_signature) const {
  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates(node, query_seq, tolerance, unique_results, stats);
    return;
  }

  if (node->child_nodes.empty()) return;

  int child_layer = current_layer + 1;
  std::vector<int> V_Q;
  if (!node->beacons.empty()) {
    V_Q = compute_query_beacon_distances(node, query_seq, stats);
  }
  auto surviving = get_mbb_surviving_children(node, V_Q, tolerance, stats);

  search_layer_adaptive(surviving, child_layer, query_seq, tolerance,
                        unique_results, visited_nodes, stats, true,
                        query_qgram_signature);
}

void BioGeometrySearchEngine::search_layer_adaptive(
    const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>& visited_nodes,
    SearchStats& stats,
    bool after_mbb_filter,
    const QGramSignature* query_qgram_signature) const {
  std::shared_ptr<WorldNode> contained_node;
  std::vector<std::shared_ptr<WorldNode>> overlap_nodes;

  for (const auto& node : candidates) {
    if (visited_nodes.count(node->node_id)) continue;

    const int tau = node->radius + tolerance;
    if (after_mbb_filter) {
      stats.center_distance_calls_after_mbb++;
      stats.center_distance_calls_before_qgram++;
      if (stats.search_qgram_prefilter_enabled) {
        auto signature_it = world_qgram_signatures_.find(node->node_id);
        if (!query_qgram_signature ||
            !query_qgram_signature->safe_for_pruning ||
            signature_it == world_qgram_signatures_.end() ||
            !signature_it->second.safe_for_pruning) {
          stats.search_qgram_signature_missing_count++;
        } else {
          stats.search_qgram_checks++;
          if (qgram_can_prune_edit_distance(
                  *query_qgram_signature, signature_it->second, tau)) {
            stats.search_qgram_pruned_children++;
            continue;
          }
          stats.search_qgram_passed_children++;
        }
      }
      stats.center_distance_calls_after_qgram++;
    }
    int dist = after_mbb_filter
                   ? compute_distance_bounded(
                         query_seq.seq, node->get_center_sequence(), tau)
                   : compute_distance(query_seq.seq, node->get_center_sequence());
    stats.dist_calc_count++;
    stats.world_access_count++;
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
      stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
    }

    if (dist > tau) continue;

    if (dist + tolerance <= node->radius) {
      contained_node = node;
      break;
    }
    overlap_nodes.push_back(node);
  }

  if (contained_node) {
    visited_nodes.insert(contained_node->node_id);
    process_node_adaptive(contained_node, layer_id, query_seq, tolerance,
                          unique_results, visited_nodes, stats,
                          query_qgram_signature);
  } else {
    for (const auto& node : overlap_nodes) {
      if (visited_nodes.count(node->node_id)) continue;
      visited_nodes.insert(node->node_id);
      process_node_adaptive(node, layer_id, query_seq, tolerance,
                            unique_results, visited_nodes, stats,
                            query_qgram_signature);
    }
  }
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_adaptive(const BioSequence& query_seq, int tolerance) {
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  stats.search_qgram_prefilter_enabled =
      config_.search_qgram_prefilter && config_.search_qgram_q > 0;
  stats.search_qgram_q =
      stats.search_qgram_prefilter_enabled ? config_.search_qgram_q : 0;
  stats.search_qgram_signature_build_count =
      stats.search_qgram_prefilter_enabled ? world_qgram_signatures_.size() : 0;
  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_results;
  std::unordered_set<std::string> visited_nodes;
  QGramSignature query_qgram_signature;
  const QGramSignature* query_qgram_signature_ptr = nullptr;
  if (stats.search_qgram_prefilter_enabled) {
    query_qgram_signature =
        compute_qgram_signature(query_seq.seq, config_.search_qgram_q);
    query_qgram_signature_ptr = &query_qgram_signature;
  }

  search_layer_adaptive(index_.primary_layer(index_.coarsest_primary_layer_index()),
                        index_.coarsest_primary_layer_index(), query_seq, tolerance,
                        unique_results, visited_nodes, stats, false,
                        query_qgram_signature_ptr);

  std::vector<std::shared_ptr<BioSequence>> out;
  for (const auto& entry : unique_results) out.push_back(entry.second);
  stats.result_count = out.size();
  return {out, stats};
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_greedy(const BioSequence& query_seq, int tolerance) {
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  std::vector<std::shared_ptr<WorldNode>> current =
      index_.primary_layer(index_.coarsest_primary_layer_index());

  for (int layer_id = index_.coarsest_primary_layer_index();
       layer_id <= index_.finest_primary_layer_index(); ++layer_id) {
    std::shared_ptr<WorldNode> best_node;
    int min_dist = std::numeric_limits<int>::max();

    for (const auto& node : current) {
      int dist = compute_distance(query_seq.seq, node->get_center_sequence());
      stats.dist_calc_count++;
      stats.world_access_count++;
      if (layer_id >= 0 && static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
        stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
      }
      if (dist <= node->radius + tolerance && dist < min_dist) {
        min_dist = dist;
        best_node = node;
      }
    }

    if (!best_node) return {{}, stats};

    if (layer_id == index_.finest_primary_layer_index()) {
      std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_results;
      verify_leaf_candidates(best_node, query_seq, tolerance, unique_results, stats);

      std::vector<std::shared_ptr<BioSequence>> results;
      for (const auto& entry : unique_results) results.push_back(entry.second);
      return {results, stats};
    }

    std::vector<int> V_Q = compute_query_beacon_distances(best_node, query_seq, stats);
    current.clear();
    for (size_t child_idx = 0; child_idx < best_node->child_nodes.size(); ++child_idx) {
      const auto& child = best_node->child_nodes[child_idx];
      stats.edge_access_count++;
      if (!V_Q.empty() && child_idx < best_node->child_beacon_mbbs.size() &&
          best_node->child_beacon_mbbs[child_idx].size() == V_Q.size()) {
        stats.candidate_count_for_prune++;
        stats.bound_check_count++;
        if (mbb_prunable_row(best_node->child_beacon_mbbs[child_idx], V_Q, tolerance)) {
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
  stats.world_access_count++;
  if (current_layer >= 0 && static_cast<size_t>(current_layer) < stats.layer_breakdown.size()) {
    stats.layer_breakdown[static_cast<size_t>(current_layer)]++;
  }

  if (dist > node->radius + tolerance) return;

  for (const auto& child : node->child_nodes) {
    stats.edge_access_count++;
    traverse_exhaustive(child, current_layer + 1, query_seq, tolerance,
                        unique_results, visited_nodes, stats);
  }

  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates(node, query_seq, tolerance, unique_results, stats);
  }
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_exhaustive(const BioSequence& query_seq, int tolerance) {
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_results;
  std::unordered_set<std::string> visited_nodes;

  for (const auto& node : index_.primary_layer(index_.coarsest_primary_layer_index())) {
    traverse_exhaustive(node, index_.coarsest_primary_layer_index(), query_seq, tolerance,
                        unique_results, visited_nodes, stats);
  }

  std::vector<std::shared_ptr<BioSequence>> out;
  for (const auto& entry : unique_results) out.push_back(entry.second);
  return {out, stats};
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_brute_force(
    const BioSequence& query_seq, int tolerance,
    const std::vector<std::shared_ptr<BioSequence>>& all_sequences) {
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  std::vector<std::vector<std::shared_ptr<BioSequence>>> thread_results;

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    #pragma omp single
    thread_results.resize(static_cast<size_t>(nthreads));

    int tid = omp_get_thread_num();
    #pragma omp for schedule(dynamic, 64)
    for (size_t i = 0; i < all_sequences.size(); ++i) {
      int dist = compute_distance(query_seq.seq, all_sequences[i]->seq);
      if (dist <= tolerance) thread_results[static_cast<size_t>(tid)].push_back(all_sequences[i]);
    }
  }

  std::vector<std::shared_ptr<BioSequence>> results;
  for (auto& thread_vec : thread_results) {
    for (auto& result : thread_vec) results.push_back(std::move(result));
  }

  stats.dist_calc_count = all_sequences.size();
  stats.leaf_verify_count = all_sequences.size();
  return {results, stats};
}

}  // namespace navigamer
