#include "search_engine.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace navigamer {

namespace {

inline void prefetch_read(const void* ptr) {
  if (!ptr) return;
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(ptr, 0, 1);
#else
  (void)ptr;
#endif
}

template <typename T>
void prefetch_vector_data(const std::vector<T>& values) {
  if (!values.empty()) prefetch_read(values.data());
}

void prefetch_sequence(const std::shared_ptr<BioSequence>& sequence) {
  if (!sequence) return;
  prefetch_read(sequence.get());
  if (!sequence->seq.empty()) prefetch_read(sequence->seq.data());
}

void prefetch_world_node(const std::shared_ptr<WorldNode>& node) {
  if (!node) return;
  prefetch_read(node.get());
  prefetch_sequence(node->center_ptr);
  prefetch_vector_data(node->child_nodes);
  prefetch_vector_data(node->child_leaves);
  prefetch_vector_data(node->child_beacon_mbbs);
  prefetch_vector_data(node->leaf_beacon_dists);
}

}  // namespace

const char* mbb_filter_mode_name(MBBFilterMode mode) {
  return mode == MBBFilterMode::Scan ? "scan" : "rect";
}

MBBFilterMode parse_mbb_filter_mode(const std::string& value) {
  if (value == "scan") return MBBFilterMode::Scan;
  if (value == "rect") return MBBFilterMode::RectIndex;
  throw std::invalid_argument("MBB filter mode must be scan or rect");
}

const char* visited_mode_name(VisitedMode mode) {
  return mode == VisitedMode::StringSet ? "string" : "epoch";
}

VisitedMode parse_visited_mode(const std::string& value) {
  if (value == "string") return VisitedMode::StringSet;
  if (value == "epoch") return VisitedMode::Epoch;
  throw std::invalid_argument("visited mode must be string or epoch");
}

const char* graph_view_mode_name(GraphViewMode mode) {
  return mode == GraphViewMode::Original ? "original" : "flat";
}

GraphViewMode parse_graph_view_mode(const std::string& value) {
  if (value == "original") return GraphViewMode::Original;
  if (value == "flat") return GraphViewMode::Flat;
  throw std::invalid_argument("graph view mode must be original or flat");
}

void SearchScratch::begin_query(size_t node_count) {
  if (visited_epoch.size() != node_count) {
    visited_epoch.assign(node_count, 0);
    current_epoch = 0;
  }
  if (current_epoch == std::numeric_limits<uint32_t>::max()) {
    std::fill(visited_epoch.begin(), visited_epoch.end(), 0);
    current_epoch = 1;
  } else {
    current_epoch++;
  }
  frontier.clear();
  next_frontier.clear();
  mbb_candidates.clear();
  verified_children.clear();
}

bool SearchScratch::mark_visited(NodeId id) {
  if (id >= visited_epoch.size()) {
    throw std::out_of_range("visited node id is outside scratch epoch array");
  }
  if (visited_epoch[id] == current_epoch) return false;
  visited_epoch[id] = current_epoch;
  return true;
}

bool SearchScratch::is_visited(NodeId id) const {
  if (id >= visited_epoch.size()) {
    throw std::out_of_range("visited node id is outside scratch epoch array");
  }
  return visited_epoch[id] == current_epoch;
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
      if (config_.search_prefetch && child_idx + 1 < children.size()) {
        prefetch_world_node(children[child_idx + 1]);
        prefetch_vector_data(node->child_beacon_mbbs[child_idx + 1]);
      }
      stats.edge_access_count++;
      stats.mbb_check_count++;
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      stats.mbb_scan_child_checks++;
      stats.mbb_scalar_checks++;
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
    stats.mbb_check_count += children.size();
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
      stats.leaf_beacon_check_count++;
      stats.candidate_count_for_prune++;
      stats.bound_check_count++;
      stats.leaf_beacon_scalar_checks++;
      if (leaf_beacon_prunable_row(node->leaf_beacon_dists[leaf_idx], V_Q, tolerance)) {
        stats.beacon_prune_count++;
        continue;
      }
    }

    const auto& child = node->child_leaves[leaf_idx];
    stats.candidate_count++;
    stats.candidate_verify_count++;
    stats.leaf_exact_distance_call_count++;
    int leaf_dist = compute_distance(query_seq.seq, child->seq);
    stats.dist_calc_count++;
    stats.leaf_verify_count++;
    if (config_.trace_paths && child->sequence_id != INVALID_LEAF_ID) {
      stats.leaf_trace.push_back(child->sequence_id);
    }
    if (leaf_dist <= tolerance) unique_results[child->id] = child;
  }
}

void BioGeometrySearchEngine::verify_leaf_candidates_view(
    NodeId node_id,
    const BioSequence& query_seq,
    int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    SearchStats& stats) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.nodes.size()) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto& node = view.nodes[node_id];
  if (!node) throw std::runtime_error("view node id has no node pointer");

  const uint32_t leaf_begin = view.leaf_begin[node_id];
  const uint32_t leaf_end = view.leaf_end[node_id];
  const size_t leaf_count = leaf_end - leaf_begin;

  std::vector<int> V_Q;
  const bool has_leaf_sieve =
      !node->beacons.empty() &&
      view.leaf_beacon_dim[node_id] == node->beacons.size() &&
      view.leaf_beacon_dim[node_id] > 0 &&
      view.leaf_beacon_begin[node_id] +
              view.leaf_beacon_dim[node_id] * leaf_count <=
          view.leaf_beacon_dists.size();
  if (has_leaf_sieve) V_Q = compute_query_beacon_distances(node, query_seq, stats);

  std::vector<uint32_t> survivor_offsets;
  const bool leaf_sieve_ready =
      has_leaf_sieve && view.leaf_beacon_dim[node_id] == V_Q.size();
  if (leaf_sieve_ready) {
    std::vector<int32_t> query32;
    query32.reserve(V_Q.size());
    for (int distance : V_Q) {
      query32.push_back(static_cast<int32_t>(distance));
    }

    LeafBeaconFilterSimdStats simd_stats;
    const uint32_t offset = view.leaf_beacon_begin[node_id];
    if (config_.search_prefetch) {
      prefetch_read(view.leaf_beacon_dists.data() + offset);
      prefetch_read(view.leaf_ids.data() + leaf_begin);
    }
    survivor_offsets = filter_leaf_beacon_survivors(
        view.leaf_beacon_dists.data() + offset,
        leaf_count,
        view.leaf_beacon_dim[node_id],
        query32.data(),
        static_cast<int32_t>(tolerance),
        config_.simd_mode,
        &simd_stats);

    stats.node_access_count += leaf_count;
    stats.leaf_beacon_check_count += leaf_count;
    stats.candidate_count_for_prune += leaf_count;
    stats.bound_check_count += leaf_count;
    stats.beacon_prune_count += leaf_count - survivor_offsets.size();
    stats.leaf_beacon_scalar_checks += simd_stats.scalar_checks;
    stats.leaf_beacon_simd_batches += simd_stats.simd_batches;
    stats.leaf_beacon_simd_fallbacks += simd_stats.simd_fallbacks;
  } else {
    survivor_offsets.reserve(leaf_count);
    for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
      if (config_.search_prefetch && leaf_idx + 1 < leaf_count) {
        prefetch_read(&view.leaf_ids[leaf_begin + leaf_idx + 1]);
      }
      survivor_offsets.push_back(static_cast<uint32_t>(leaf_idx));
    }
  }

  for (size_t survivor_idx = 0; survivor_idx < survivor_offsets.size();
       ++survivor_idx) {
    const uint32_t leaf_offset = survivor_offsets[survivor_idx];
    if (leaf_offset >= leaf_count) {
      throw std::runtime_error("SIMD leaf beacon filter returned offset out of range");
    }
    if (config_.search_prefetch && survivor_idx + 1 < survivor_offsets.size()) {
      const uint32_t next_offset = survivor_offsets[survivor_idx + 1];
      if (next_offset < leaf_count) {
        const LeafId next_leaf_id = view.leaf_ids[leaf_begin + next_offset];
        if (next_leaf_id < view.leaves.size()) {
          prefetch_sequence(view.leaves[next_leaf_id]);
        }
      }
    }
    if (!leaf_sieve_ready) stats.node_access_count++;
    const LeafId leaf_id = view.leaf_ids[leaf_begin + leaf_offset];
    if (leaf_id >= view.leaves.size() || !view.leaves[leaf_id]) {
      throw std::runtime_error("view leaf id has no sequence pointer");
    }
    const auto& child = view.leaves[leaf_id];
    stats.candidate_count++;
    stats.candidate_verify_count++;
    stats.leaf_exact_distance_call_count++;
    int leaf_dist = compute_distance(query_seq.seq, child->seq);
    stats.dist_calc_count++;
    stats.leaf_verify_count++;
    if (config_.trace_paths) {
      stats.leaf_trace.push_back(leaf_id);
    }
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
    stats.visited_check_count++;
    if (visited_nodes.count(node->node_id)) {
      stats.visited_hit_count++;
      continue;
    }

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
    stats.center_exact_distance_call_count++;
    int dist = after_mbb_filter
                   ? compute_distance_bounded_with_mode(
                         query_seq.seq, node->get_center_sequence(), tau,
                         config_.distance_mode)
                   : compute_distance(query_seq.seq, node->get_center_sequence());
    stats.dist_calc_count++;
    stats.world_access_count++;
    if (config_.trace_paths && node->integer_id != INVALID_NODE_ID) {
      stats.world_trace.push_back(node->integer_id);
    }
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

  stats.record_path_step(overlap_nodes.size(), static_cast<bool>(contained_node));
  if (contained_node) {
    visited_nodes.insert(contained_node->node_id);
    process_node_adaptive(contained_node, layer_id, query_seq, tolerance,
                          unique_results, visited_nodes, stats,
                          query_qgram_signature);
  } else {
    for (const auto& node : overlap_nodes) {
      stats.visited_check_count++;
      if (visited_nodes.count(node->node_id)) {
        stats.visited_hit_count++;
        continue;
      }
      visited_nodes.insert(node->node_id);
      process_node_adaptive(node, layer_id, query_seq, tolerance,
                            unique_results, visited_nodes, stats,
                            query_qgram_signature);
    }
  }
}

void BioGeometrySearchEngine::process_node_adaptive_epoch(
    const std::shared_ptr<WorldNode>& node, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    SearchScratch& scratch,
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

  search_layer_adaptive_epoch(surviving, child_layer, query_seq, tolerance,
                              unique_results, scratch, stats, true,
                              query_qgram_signature);
}

void BioGeometrySearchEngine::search_layer_adaptive_epoch(
    const std::vector<std::shared_ptr<WorldNode>>& candidates, int layer_id,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    SearchScratch& scratch,
    SearchStats& stats,
    bool after_mbb_filter,
    const QGramSignature* query_qgram_signature) const {
  std::shared_ptr<WorldNode> contained_node;
  std::vector<std::shared_ptr<WorldNode>> overlap_nodes;

  for (const auto& node : candidates) {
    stats.visited_check_count++;
    if (scratch.is_visited(node->integer_id)) {
      stats.visited_hit_count++;
      continue;
    }

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
    stats.center_exact_distance_call_count++;
    int dist = after_mbb_filter
                   ? compute_distance_bounded_with_mode(
                         query_seq.seq, node->get_center_sequence(), tau,
                         config_.distance_mode)
                   : compute_distance(query_seq.seq, node->get_center_sequence());
    stats.dist_calc_count++;
    stats.world_access_count++;
    if (config_.trace_paths && node->integer_id != INVALID_NODE_ID) {
      stats.world_trace.push_back(node->integer_id);
    }
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

  stats.record_path_step(overlap_nodes.size(), static_cast<bool>(contained_node));
  if (contained_node) {
    scratch.mark_visited(contained_node->integer_id);
    process_node_adaptive_epoch(contained_node, layer_id, query_seq, tolerance,
                                unique_results, scratch, stats,
                                query_qgram_signature);
  } else {
    for (const auto& node : overlap_nodes) {
      stats.visited_check_count++;
      if (!scratch.mark_visited(node->integer_id)) {
        stats.visited_hit_count++;
        continue;
      }
      process_node_adaptive_epoch(node, layer_id, query_seq, tolerance,
                                  unique_results, scratch, stats,
                                  query_qgram_signature);
    }
  }
}

bool BioGeometrySearchEngine::flat_is_visited(
    NodeId node_id,
    const std::unordered_set<std::string>* visited_nodes,
    const SearchScratch* scratch) const {
  if (scratch) return scratch->is_visited(node_id);
  const auto& view = index_.search_graph_view();
  if (!visited_nodes || node_id >= view.nodes.size() || !view.nodes[node_id]) {
    throw std::runtime_error("invalid flat visited state");
  }
  return visited_nodes->count(view.nodes[node_id]->node_id) != 0;
}

bool BioGeometrySearchEngine::flat_mark_visited(
    NodeId node_id,
    std::unordered_set<std::string>* visited_nodes,
    SearchScratch* scratch) const {
  if (scratch) return scratch->mark_visited(node_id);
  const auto& view = index_.search_graph_view();
  if (!visited_nodes || node_id >= view.nodes.size() || !view.nodes[node_id]) {
    throw std::runtime_error("invalid flat visited state");
  }
  return visited_nodes->insert(view.nodes[node_id]->node_id).second;
}

std::vector<NodeId> BioGeometrySearchEngine::get_mbb_surviving_child_ids_view(
    NodeId node_id,
    const std::vector<int>& query_beacon_dists,
    int tolerance,
    SearchStats& stats) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.nodes.size() || !view.nodes[node_id]) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto& node = view.nodes[node_id];
  if (config_.mbb_filter_mode == MBBFilterMode::RectIndex) {
    auto surviving_nodes =
        get_mbb_surviving_children(node, query_beacon_dists, tolerance, stats);
    std::vector<NodeId> surviving;
    surviving.reserve(surviving_nodes.size());
    for (const auto& child : surviving_nodes) {
      if (!child || child->integer_id == INVALID_NODE_ID) {
        throw std::runtime_error("rect MBB returned child without integer id");
      }
      surviving.push_back(child->integer_id);
    }
    return surviving;
  }
  stats.mbb_filter_parent_count++;

  const uint32_t child_begin = view.child_begin[node_id];
  const uint32_t child_end = view.child_end[node_id];
  const size_t child_count = child_end - child_begin;
  const size_t dim = view.mbb_dim[node_id];
  const size_t mbb_begin = view.mbb_begin[node_id];
  const bool mbb_ok =
      !query_beacon_dists.empty() &&
      query_beacon_dists.size() == node->beacons.size() &&
      query_beacon_dists.size() == dim &&
      mbb_begin + dim * child_count <= view.mbb_lo.size() &&
      mbb_begin + dim * child_count <= view.mbb_hi.size();

  std::vector<NodeId> surviving;
  if (!mbb_ok) {
    surviving.reserve(child_count);
    for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
      if (config_.search_prefetch && child_idx + 1 < child_count) {
        prefetch_read(&view.child_ids[child_begin + child_idx + 1]);
      }
      surviving.push_back(view.child_ids[child_begin + child_idx]);
    }
  } else {
    std::vector<int32_t> query32;
    query32.reserve(query_beacon_dists.size());
    for (int distance : query_beacon_dists) {
      query32.push_back(static_cast<int32_t>(distance));
    }

    MBBFilterSimdStats simd_stats;
    if (config_.search_prefetch) {
      prefetch_read(view.mbb_lo.data() + mbb_begin);
      prefetch_read(view.mbb_hi.data() + mbb_begin);
      prefetch_read(view.child_ids.data() + child_begin);
    }
    auto survivor_offsets = filter_mbb_survivors(
        view.mbb_lo.data() + mbb_begin,
        view.mbb_hi.data() + mbb_begin,
        child_count,
        dim,
        query32.data(),
        static_cast<int32_t>(tolerance),
        config_.simd_mode,
        &simd_stats);

    stats.edge_access_count += child_count;
    stats.mbb_check_count += child_count;
    stats.candidate_count_for_prune += child_count;
    stats.bound_check_count += child_count;
    stats.mbb_scan_child_checks += child_count;
    stats.beacon_prune_count += child_count - survivor_offsets.size();
    stats.mbb_scalar_checks += simd_stats.scalar_checks;
    stats.mbb_simd_batches += simd_stats.simd_batches;
    stats.mbb_simd_fallbacks += simd_stats.simd_fallbacks;

    surviving.reserve(survivor_offsets.size());
    for (size_t survivor_idx = 0; survivor_idx < survivor_offsets.size();
         ++survivor_idx) {
      const uint32_t child_offset = survivor_offsets[survivor_idx];
      if (child_offset >= child_count) {
        throw std::runtime_error("SIMD MBB filter returned child offset out of range");
      }
      if (config_.search_prefetch && survivor_idx + 1 < survivor_offsets.size()) {
        const uint32_t next_offset = survivor_offsets[survivor_idx + 1];
        if (next_offset < child_count) {
          const NodeId next_id = view.child_ids[child_begin + next_offset];
          if (next_id < view.nodes.size()) prefetch_world_node(view.nodes[next_id]);
        }
      }
      surviving.push_back(view.child_ids[child_begin + child_offset]);
    }
  }
  stats.mbb_surviving_child_count += surviving.size();
  return surviving;
}

void BioGeometrySearchEngine::process_node_adaptive_view(
    NodeId node_id, int current_layer,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>* visited_nodes,
    SearchScratch* scratch,
    SearchStats& stats,
    const QGramSignature* query_qgram_signature) const {
  const auto& view = index_.search_graph_view();
  if (node_id >= view.nodes.size() || !view.nodes[node_id]) {
    throw std::out_of_range("view node id is outside search graph view");
  }
  const auto& node = view.nodes[node_id];

  if (current_layer == index_.finest_primary_layer_index()) {
    verify_leaf_candidates_view(node_id, query_seq, tolerance, unique_results, stats);
    return;
  }

  if (view.child_begin[node_id] == view.child_end[node_id]) return;

  int child_layer = current_layer + 1;
  std::vector<int> V_Q;
  if (!node->beacons.empty()) {
    V_Q = compute_query_beacon_distances(node, query_seq, stats);
  }
  auto surviving = get_mbb_surviving_child_ids_view(node_id, V_Q, tolerance, stats);

  search_layer_adaptive_view(surviving, child_layer, query_seq, tolerance,
                             unique_results, visited_nodes, scratch, stats, true,
                             query_qgram_signature);
}

void BioGeometrySearchEngine::search_layer_adaptive_view(
    const std::vector<NodeId>& candidates, int layer_id,
    const BioSequence& query_seq, int tolerance,
    std::unordered_map<std::string, std::shared_ptr<BioSequence>>& unique_results,
    std::unordered_set<std::string>* visited_nodes,
    SearchScratch* scratch,
    SearchStats& stats,
    bool after_mbb_filter,
    const QGramSignature* query_qgram_signature) const {
  const auto& view = index_.search_graph_view();
  NodeId contained_node = INVALID_NODE_ID;
  std::vector<NodeId> overlap_nodes;

  for (size_t candidate_idx = 0; candidate_idx < candidates.size();
       ++candidate_idx) {
    const NodeId node_id = candidates[candidate_idx];
    if (node_id >= view.nodes.size() || !view.nodes[node_id]) {
      throw std::out_of_range("view node id is outside search graph view");
    }
    if (config_.search_prefetch && candidate_idx + 1 < candidates.size()) {
      const NodeId next_id = candidates[candidate_idx + 1];
      if (next_id < view.nodes.size()) prefetch_world_node(view.nodes[next_id]);
    }
    const auto& node = view.nodes[node_id];
    stats.visited_check_count++;
    if (flat_is_visited(node_id, visited_nodes, scratch)) {
      stats.visited_hit_count++;
      continue;
    }

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
    stats.center_exact_distance_call_count++;
    int dist = after_mbb_filter
                   ? compute_distance_bounded_with_mode(
                         query_seq.seq, node->get_center_sequence(), tau,
                         config_.distance_mode)
                   : compute_distance(query_seq.seq, node->get_center_sequence());
    stats.dist_calc_count++;
    stats.world_access_count++;
    if (config_.trace_paths) {
      stats.world_trace.push_back(node_id);
    }
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < stats.layer_breakdown.size()) {
      stats.layer_breakdown[static_cast<size_t>(layer_id)]++;
    }

    if (dist > tau) continue;

    if (dist + tolerance <= node->radius) {
      contained_node = node_id;
      break;
    }
    overlap_nodes.push_back(node_id);
  }

  stats.record_path_step(overlap_nodes.size(),
                         contained_node != INVALID_NODE_ID);
  if (contained_node != INVALID_NODE_ID) {
    flat_mark_visited(contained_node, visited_nodes, scratch);
    process_node_adaptive_view(contained_node, layer_id, query_seq, tolerance,
                               unique_results, visited_nodes, scratch, stats,
                               query_qgram_signature);
  } else {
    for (NodeId node_id : overlap_nodes) {
      stats.visited_check_count++;
      if (!flat_mark_visited(node_id, visited_nodes, scratch)) {
        stats.visited_hit_count++;
        continue;
      }
      process_node_adaptive_view(node_id, layer_id, query_seq, tolerance,
                                 unique_results, visited_nodes, scratch, stats,
                                 query_qgram_signature);
    }
  }
}

std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
BioGeometrySearchEngine::search_adaptive(const BioSequence& query_seq, int tolerance) {
  SearchStats stats(static_cast<size_t>(index_.num_primary_layers()));
  stats.search_prefetch_enabled = config_.search_prefetch;
  stats.trace_paths_enabled = config_.trace_paths;
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

  if (config_.graph_view_mode == GraphViewMode::Original &&
      config_.visited_mode == VisitedMode::StringSet) {
    search_layer_adaptive(index_.primary_layer(index_.coarsest_primary_layer_index()),
                          index_.coarsest_primary_layer_index(), query_seq, tolerance,
                          unique_results, visited_nodes, stats, false,
                          query_qgram_signature_ptr);
  } else if (config_.graph_view_mode == GraphViewMode::Original) {
    thread_local SearchScratch scratch;
    scratch.begin_query(index_.num_world_nodes());
    search_layer_adaptive_epoch(
        index_.primary_layer(index_.coarsest_primary_layer_index()),
        index_.coarsest_primary_layer_index(), query_seq, tolerance,
        unique_results, scratch, stats, false, query_qgram_signature_ptr);
  } else {
    std::vector<NodeId> top_candidates;
    const auto& top_layer =
        index_.primary_layer(index_.coarsest_primary_layer_index());
    top_candidates.reserve(top_layer.size());
    for (const auto& node : top_layer) {
      if (!node || node->integer_id == INVALID_NODE_ID) {
        throw std::runtime_error("top-level search node has no integer id");
      }
      top_candidates.push_back(node->integer_id);
    }

    if (config_.visited_mode == VisitedMode::StringSet) {
      search_layer_adaptive_view(top_candidates,
                                 index_.coarsest_primary_layer_index(),
                                 query_seq, tolerance, unique_results,
                                 &visited_nodes, nullptr, stats, false,
                                 query_qgram_signature_ptr);
    } else {
      thread_local SearchScratch scratch;
      scratch.begin_query(index_.num_world_nodes());
      search_layer_adaptive_view(top_candidates,
                                 index_.coarsest_primary_layer_index(),
                                 query_seq, tolerance, unique_results,
                                 nullptr, &scratch, stats, false,
                                 query_qgram_signature_ptr);
    }
  }

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
