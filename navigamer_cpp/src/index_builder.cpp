#include "index_builder.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <omp.h>

namespace navigamer {

namespace {

std::shared_ptr<WorldNode> nearest_in_cover(
    const std::vector<std::shared_ptr<WorldNode>>& cover,
    const std::shared_ptr<BioSequence>& sequence,
    BuildDistanceMode distance_mode) {
  std::shared_ptr<WorldNode> best;
  int best_dist = INT_MAX;
  for (const auto& node : cover) {
    if (!node->center_ptr) continue;
    int dist = distance_mode == BuildDistanceMode::Edlib
                   ? compute_distance_edlib(sequence->seq, node->center_ptr->seq)
                   : compute_distance(sequence->seq, node->center_ptr->seq);
    if (dist < best_dist) {
      best_dist = dist;
      best = node;
    }
  }
  return best;
}

std::vector<int> make_auxiliary_radii(const std::vector<int>& primary_radii) {
  std::vector<int> out;
  if (primary_radii.size() < 2) return out;
  out.reserve(primary_radii.size() - 1);
  for (size_t i = 0; i + 1 < primary_radii.size(); ++i) {
    out.push_back(std::max(1, (primary_radii[i] + primary_radii[i + 1]) / 2));
  }
  return out;
}

std::vector<int> leaf_beacon_distances(
    const std::shared_ptr<BioSequence>& leaf,
    const std::vector<std::shared_ptr<BioSequence>>& beacons,
    int center_dist,
    BuildDistanceMode distance_mode) {
  std::vector<int> dists;
  dists.reserve(beacons.size());
  for (size_t i = 0; i < beacons.size(); ++i) {
    if (!beacons[i]) {
      dists.push_back(0);
    } else if (i == 0) {
      dists.push_back(center_dist);
    } else {
      dists.push_back(distance_mode == BuildDistanceMode::Edlib
                          ? compute_distance_edlib(leaf->seq, beacons[i]->seq)
                          : compute_distance(leaf->seq, beacons[i]->seq));
    }
  }
  return dists;
}

int build_distance(const std::string& a, const std::string& b,
                   BuildDistanceMode mode) {
  return mode == BuildDistanceMode::Edlib ? compute_distance_edlib(a, b)
                                          : compute_distance(a, b);
}

int build_distance_bounded(const std::string& a, const std::string& b, int tau,
                           BuildDistanceMode mode) {
  return mode == BuildDistanceMode::Edlib
             ? compute_distance_bounded_edlib(a, b, tau)
             : compute_distance_bounded_dp(a, b, tau);
}

std::vector<int> build_expanded_radii(const HierarchyConfig& config) {
  std::vector<int> expanded;
  expanded.reserve(static_cast<size_t>(config.num_expanded_layers()));
  for (int i = 0; i < config.num_primary_layers(); ++i) {
    expanded.push_back(config.primary_radii[static_cast<size_t>(i)]);
    if (i < config.num_auxiliary_layers()) {
      expanded.push_back(config.auxiliary_radii[static_cast<size_t>(i)]);
    }
  }
  return expanded;
}

bool expanded_layer_is_primary(int expanded_layer_idx) {
  return expanded_layer_idx % 2 == 0;
}

int expanded_to_primary_index(int expanded_layer_idx) {
  return expanded_layer_idx / 2;
}

void reset_node_metadata(const std::shared_ptr<WorldNode>& node,
                         int expanded_layer_idx,
                         bool is_primary,
                         int primary_layer_idx) {
  node->expanded_layer_index = expanded_layer_idx;
  node->is_primary = is_primary;
  node->primary_layer_index = primary_layer_idx;
}

double reduction_ratio(size_t before, size_t after) {
  if (before == 0) return 0.0;
  return 1.0 - static_cast<double>(after) / static_cast<double>(before);
}

void validate_range_config(const BuildRangeConfig& config) {
  if (config.range_join.min_seed_len <= 0) {
    throw std::invalid_argument("range-join min seed length must be positive");
  }
  if (config.range_join.max_seed_len < config.range_join.min_seed_len) {
    throw std::invalid_argument(
        "range-join max seed length must be at least min seed length");
  }
  if (config.range_join.qgram_q <= 0) {
    throw std::invalid_argument("range-join q-gram length must be positive");
  }
  if (!std::isfinite(config.range_join.auto_pigeonhole_max_ratio) ||
      config.range_join.auto_pigeonhole_max_ratio < 0.0 ||
      config.range_join.auto_pigeonhole_max_ratio > 1.0) {
    throw std::invalid_argument(
        "auto pigeonhole max ratio must be finite and in [0, 1]");
  }
  if (config.min_rect_index_fanout == 0) {
    throw std::invalid_argument("minimum rectangle-index fanout must be positive");
  }
}

}  // namespace

HierarchyConfig::HierarchyConfig(std::vector<int> primary_radii_in)
    : primary_radii(std::move(primary_radii_in)),
      auxiliary_radii(make_auxiliary_radii(primary_radii)) {
  validate();
}

HierarchyConfig::HierarchyConfig(std::vector<int> primary_radii_in,
                                 std::vector<int> auxiliary_radii_in)
    : primary_radii(std::move(primary_radii_in)),
      auxiliary_radii(std::move(auxiliary_radii_in)) {
  validate();
}

int HierarchyConfig::num_primary_layers() const {
  return static_cast<int>(primary_radii.size());
}

int HierarchyConfig::num_auxiliary_layers() const {
  return static_cast<int>(auxiliary_radii.size());
}

int HierarchyConfig::num_expanded_layers() const {
  if (primary_radii.empty()) return 0;
  return static_cast<int>(primary_radii.size() * 2 - 1);
}

void HierarchyConfig::validate() const {
  if (primary_radii.size() < 2) {
    throw std::invalid_argument("HierarchyConfig requires at least two primary radii");
  }
  if (auxiliary_radii.size() != primary_radii.size() - 1) {
    throw std::invalid_argument("HierarchyConfig auxiliary_radii must have size K-1");
  }
  for (size_t i = 0; i + 1 < primary_radii.size(); ++i) {
    if (primary_radii[i] <= primary_radii[i + 1]) {
      throw std::invalid_argument("HierarchyConfig primary_radii must be strictly decreasing");
    }
    int aux = auxiliary_radii[i];
    if (aux <= 0) {
      throw std::invalid_argument("HierarchyConfig auxiliary radii must be positive");
    }
    if (aux > primary_radii[i] || aux < primary_radii[i + 1]) {
      throw std::invalid_argument("HierarchyConfig auxiliary radius must lie between adjacent primary radii");
    }
  }
}

const char* build_range_mode_name(BuildRangeMode mode) {
  return mode == BuildRangeMode::Full ? "full" : "indexed";
}

BuildRangeMode parse_build_range_mode(const std::string& value) {
  if (value == "full") return BuildRangeMode::Full;
  if (value == "indexed") return BuildRangeMode::Indexed;
  throw std::invalid_argument("build range mode must be full or indexed");
}

const char* build_distance_mode_name(BuildDistanceMode mode) {
  switch (mode) {
    case BuildDistanceMode::DP:
      return "dp";
    case BuildDistanceMode::Edlib:
      return "edlib";
    case BuildDistanceMode::Auto:
      return "auto";
  }
  return "dp";
}

BuildDistanceMode parse_build_distance_mode(const std::string& value) {
  if (value == "dp") return BuildDistanceMode::DP;
  if (value == "edlib") return BuildDistanceMode::Edlib;
  if (value == "auto") return BuildDistanceMode::Auto;
  throw std::invalid_argument("build distance mode must be dp, edlib, or auto");
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder()
    : stats_{},
      hierarchy_(HierarchyConfig({R_LW, R_MW, R_SW})),
      range_config_{},
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())) {
  stats_.created_primary_nodes.assign(static_cast<size_t>(hierarchy_.num_primary_layers()), 0);
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw)
    : BioGeometryIndexBuilder(HierarchyConfig({r_lw, r_mw, r_sw})) {}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(const HierarchyConfig& config)
    : BioGeometryIndexBuilder(config, BuildRangeConfig{}) {}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(
    const HierarchyConfig& config, const BuildRangeConfig& range_config)
    : stats_{},
      hierarchy_(config),
      range_config_(range_config),
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())) {
  validate_range_config(range_config_);
  stats_.created_primary_nodes.assign(static_cast<size_t>(hierarchy_.num_primary_layers()), 0);
}

const std::vector<std::shared_ptr<WorldNode>>& BioGeometryIndexBuilder::primary_layer(int idx) const {
  return primary_layers_.at(static_cast<size_t>(idx));
}

bool BioGeometryIndexBuilder::validate_integer_ids() const {
  if (world_node_count_ == 0 && !primary_layers_.empty()) return false;
  std::vector<bool> seen_nodes(world_node_count_, false);
  size_t visited_nodes = 0;
  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id >= world_node_count_) return false;
      if (seen_nodes[node->integer_id]) return false;
      seen_nodes[node->integer_id] = true;
      visited_nodes++;
      for (const auto& child : node->child_nodes) {
        if (!child || child->integer_id >= world_node_count_) return false;
      }
      for (const auto& leaf : node->child_leaves) {
        if (!leaf || leaf->sequence_id >= sequence_count_) return false;
      }
    }
  }
  if (visited_nodes != world_node_count_) return false;
  for (bool seen : seen_nodes) {
    if (!seen) return false;
  }

  if (unique_sequences.size() != sequence_count_) return false;
  std::vector<bool> seen_sequences(sequence_count_, false);
  for (const auto& entry : unique_sequences) {
    const auto& sequence = entry.second;
    if (!sequence || sequence->sequence_id >= sequence_count_) return false;
    if (seen_sequences[sequence->sequence_id]) return false;
    seen_sequences[sequence->sequence_id] = true;
  }
  for (bool seen : seen_sequences) {
    if (!seen) return false;
  }
  return true;
}

bool BioGeometryIndexBuilder::validate_search_graph_view() const {
  const auto& view = search_graph_view_;
  if (!validate_integer_ids()) return false;
  if (view.nodes.size() != world_node_count_ ||
      view.leaves.size() != sequence_count_ ||
      view.child_begin.size() != world_node_count_ ||
      view.child_end.size() != world_node_count_ ||
      view.leaf_begin.size() != world_node_count_ ||
      view.leaf_end.size() != world_node_count_ ||
      view.mbb_begin.size() != world_node_count_ ||
      view.mbb_dim.size() != world_node_count_ ||
      view.beacon_begin.size() != world_node_count_ ||
      view.beacon_end.size() != world_node_count_ ||
      view.leaf_beacon_begin.size() != world_node_count_ ||
      view.leaf_beacon_dim.size() != world_node_count_) {
    return false;
  }

  for (const auto& entry : unique_sequences) {
    const auto& sequence = entry.second;
    if (!sequence || sequence->sequence_id >= view.leaves.size()) return false;
    if (view.leaves[sequence->sequence_id] != sequence) return false;
  }

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id >= view.nodes.size()) return false;
      const NodeId node_id = node->integer_id;
      if (view.nodes[node_id] != node) return false;

      if (view.child_end[node_id] < view.child_begin[node_id] ||
          view.leaf_end[node_id] < view.leaf_begin[node_id] ||
          view.beacon_end[node_id] < view.beacon_begin[node_id]) {
        return false;
      }
      if (view.child_end[node_id] > view.child_ids.size() ||
          view.leaf_end[node_id] > view.leaf_ids.size() ||
          view.beacon_end[node_id] > view.beacon_ids.size()) {
        return false;
      }

      const uint32_t child_begin = view.child_begin[node_id];
      if (view.child_end[node_id] - child_begin != node->child_nodes.size()) {
        return false;
      }
      for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
        const auto& child = node->child_nodes[child_idx];
        if (!child || view.child_ids[child_begin + child_idx] != child->integer_id) {
          return false;
        }
      }

      const uint32_t leaf_begin = view.leaf_begin[node_id];
      if (view.leaf_end[node_id] - leaf_begin != node->child_leaves.size()) {
        return false;
      }
      for (size_t leaf_idx = 0; leaf_idx < node->child_leaves.size(); ++leaf_idx) {
        const auto& leaf = node->child_leaves[leaf_idx];
        if (!leaf || view.leaf_ids[leaf_begin + leaf_idx] != leaf->sequence_id) {
          return false;
        }
      }

      const uint32_t beacon_begin = view.beacon_begin[node_id];
      if (view.beacon_end[node_id] - beacon_begin != node->beacons.size()) {
        return false;
      }
      for (size_t beacon_idx = 0; beacon_idx < node->beacons.size(); ++beacon_idx) {
        const auto& beacon = node->beacons[beacon_idx];
        if (!beacon || view.beacon_ids[beacon_begin + beacon_idx] !=
                           beacon->sequence_id) {
          return false;
        }
      }

      const size_t child_count = node->child_nodes.size();
      const size_t mbb_dim = view.mbb_dim[node_id];
      const size_t mbb_begin = view.mbb_begin[node_id];
      if (mbb_dim != node->beacons.size()) return false;
      if (mbb_begin + mbb_dim * child_count > view.mbb_lo.size() ||
          mbb_begin + mbb_dim * child_count > view.mbb_hi.size()) {
        return false;
      }
      if (!node->child_beacon_mbbs.empty() &&
          node->child_beacon_mbbs.size() != child_count) {
        return false;
      }
      for (size_t child_idx = 0; child_idx < node->child_beacon_mbbs.size(); ++child_idx) {
        if (node->child_beacon_mbbs[child_idx].size() != mbb_dim) return false;
        for (size_t dim = 0; dim < mbb_dim; ++dim) {
          const size_t flat = mbb_begin + dim * child_count + child_idx;
          if (view.mbb_lo[flat] != node->child_beacon_mbbs[child_idx][dim].min_dist ||
              view.mbb_hi[flat] != node->child_beacon_mbbs[child_idx][dim].max_dist) {
            return false;
          }
        }
      }

      const size_t leaf_count = node->child_leaves.size();
      const size_t leaf_dim = view.leaf_beacon_dim[node_id];
      const size_t leaf_beacon_begin = view.leaf_beacon_begin[node_id];
      if (leaf_dim != node->beacons.size()) return false;
      if (leaf_beacon_begin + leaf_dim * leaf_count >
          view.leaf_beacon_dists.size()) {
        return false;
      }
      if (!node->leaf_beacon_dists.empty() &&
          node->leaf_beacon_dists.size() != leaf_count) {
        return false;
      }
      for (size_t leaf_idx = 0; leaf_idx < node->leaf_beacon_dists.size(); ++leaf_idx) {
        if (node->leaf_beacon_dists[leaf_idx].size() != leaf_dim) return false;
        for (size_t dim = 0; dim < leaf_dim; ++dim) {
          const size_t flat = leaf_beacon_begin + dim * leaf_count + leaf_idx;
          if (view.leaf_beacon_dists[flat] !=
              node->leaf_beacon_dists[leaf_idx][dim]) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

std::vector<std::shared_ptr<WorldNode>> BioGeometryIndexBuilder::find_neighbors(
    const BioSequence& query_seq,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    int radius) const {
  std::vector<std::shared_ptr<WorldNode>> result;
  for (const auto& node : candidates) {
    if (!node->center_ptr) continue;
    int dist = build_distance(query_seq.seq, node->center_ptr->seq,
                              range_config_.distance_mode);
    if (dist <= radius) result.push_back(node);
  }
  return result;
}

std::vector<std::shared_ptr<BioSequence>> BioGeometryIndexBuilder::deduplicate(
    const std::vector<std::shared_ptr<BioSequence>>& raw) {
  std::unordered_map<std::string, std::shared_ptr<BioSequence>> seq_map;
  for (const auto& seq : raw) {
    stats_.added_sequences++;
    auto it = seq_map.find(seq->seq);
    if (it != seq_map.end()) {
      for (const auto& occ : seq->ref_positions) {
        it->second->add_occurrence(occ.ref_id, occ.start, occ.end, occ.strand);
      }
      if (seq->ref_positions.empty() && it->second->ref_positions.empty()) {
        it->second->add_occurrence(seq->id, 0, static_cast<int>(seq->seq.size()), "+");
      }
      if (!it->second->bwt_interval.valid() && seq->bwt_interval.valid()) {
        it->second->set_bwt_interval(seq->bwt_interval.start, seq->bwt_interval.end);
      }
      stats_.deduplicated++;
    } else {
      seq_map[seq->seq] = seq;
    }
  }

  unique_sequences.clear();
  for (const auto& entry : seq_map) unique_sequences[entry.second->id] = entry.second;
  stats_.unique_sequences = seq_map.size();

  std::vector<std::shared_ptr<BioSequence>> out;
  out.reserve(seq_map.size());
  for (const auto& entry : seq_map) out.push_back(entry.second);
  return out;
}

void BioGeometryIndexBuilder::phase1_build_extended_sketch(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  extended_layers_.assign(static_cast<size_t>(hierarchy_.num_expanded_layers()),
                          std::vector<std::shared_ptr<WorldNode>>());

  for (const auto& sequence : unique_seqs) {
    std::shared_ptr<WorldNode> parent;
    for (int layer_idx = 0; layer_idx < hierarchy_.num_expanded_layers(); ++layer_idx) {
      std::vector<std::shared_ptr<WorldNode>> cover;
      if (layer_idx == 0) {
        for (const auto& node : extended_layers_[0]) {
          if (!node->center_ptr) continue;
          if (build_distance(sequence->seq, node->center_ptr->seq,
                             range_config_.distance_mode) <=
              expanded_radii_[static_cast<size_t>(layer_idx)]) {
            cover.push_back(node);
          }
        }
      } else if (parent) {
        for (const auto& node : parent->child_nodes) {
          if (!node->center_ptr) continue;
          if (build_distance(sequence->seq, node->center_ptr->seq,
                             range_config_.distance_mode) <=
              expanded_radii_[static_cast<size_t>(layer_idx)]) {
            cover.push_back(node);
          }
        }
      }

      if (cover.empty()) {
        auto new_node = std::make_shared<WorldNode>(
            sequence, expanded_radii_[static_cast<size_t>(layer_idx)], layer_idx);
        const bool is_primary = expanded_layer_is_primary(layer_idx);
        const int primary_idx = is_primary ? expanded_to_primary_index(layer_idx) : -1;
        reset_node_metadata(new_node, layer_idx, is_primary, primary_idx);
        extended_layers_[static_cast<size_t>(layer_idx)].push_back(new_node);
        if (parent) parent->child_nodes.push_back(new_node);
        cover.push_back(new_node);
        if (is_primary) {
          stats_.created_primary_nodes[static_cast<size_t>(primary_idx)]++;
        } else {
          stats_.created_auxiliary_nodes++;
        }
      }

      parent = nearest_in_cover(cover, sequence, range_config_.distance_mode);
      if (!parent) break;
    }
  }
}

void BioGeometryIndexBuilder::phase2_inter_tier_rebinding() {
  for (int layer_idx = 0; layer_idx + 1 < hierarchy_.num_expanded_layers(); ++layer_idx) {
    auto& parents = extended_layers_[static_cast<size_t>(layer_idx)];
    auto& children = extended_layers_[static_cast<size_t>(layer_idx + 1)];
    stats_.phase2_total_possible_pairs += parents.size() * children.size();
    for (auto& parent : parents) parent->child_nodes.clear();

    if (range_config_.link_mode == BuildRangeMode::Full) {
      for (auto& parent : parents) {
        if (!parent->center_ptr) continue;
        for (auto& child : children) {
          if (!child->center_ptr) continue;
          stats_.phase2_candidate_pairs++;
          stats_.phase2_exact_distance_calls++;
          int dist = build_distance(parent->center_ptr->seq,
                                    child->center_ptr->seq,
                                    range_config_.distance_mode);
          if (dist <= parent->radius + child->radius) {
            parent->child_nodes.push_back(child);
            stats_.phase2_edges_added++;
          }
        }
      }
      continue;
    }

    std::vector<RangeJoinItem> items;
    items.reserve(parents.size());
    int max_parent_radius = 0;
    for (size_t parent_idx = 0; parent_idx < parents.size(); ++parent_idx) {
      if (!parents[parent_idx]->center_ptr) continue;
      items.push_back({parent_idx, parents[parent_idx]->center_ptr->seq});
      max_parent_radius = std::max(max_parent_radius, parents[parent_idx]->radius);
    }
    ExactRangeJoinIndex parent_index(range_config_.range_join);
    parent_index.build(items);

    for (auto& child : children) {
      if (!child->center_ptr) continue;
      auto candidates =
          parent_index.query(child->center_ptr->seq, max_parent_radius + child->radius);
      stats_.phase2_candidate_pairs += candidates.candidate_item_ids.size();
      if (candidates.used_full_scan) stats_.phase2_full_scan_fallback_count++;
      if (candidates.mode_used == RangeCandidateMode::PigeonholeOnly) {
        stats_.phase2_pigeonhole_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::QGramOnly) {
        stats_.phase2_qgram_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::Hybrid) {
        stats_.phase2_hybrid_queries++;
      }
      stats_.phase2_qgram_candidate_pairs += candidates.qgram_candidate_count;
      stats_.phase2_qgram_pruned_by_l1 += candidates.qgram_pruned_by_l1;
      stats_.phase2_length_pruned_pairs += candidates.length_filtered_items;
      stats_.phase2_required_shared_nonpositive_count +=
          candidates.required_shared_nonpositive;
      stats_.phase2_auto_pigeonhole_accepted +=
          candidates.auto_pigeonhole_accepted;
      stats_.phase2_auto_pigeonhole_rejected_large_candidates +=
          candidates.auto_pigeonhole_rejected_large_candidates;
      stats_.phase2_auto_qgram_invoked += candidates.auto_qgram_invoked;
      stats_.phase2_auto_hybrid_invoked += candidates.auto_hybrid_invoked;
      stats_.phase2_auto_final_candidate_pairs +=
          candidates.auto_final_candidate_pairs;
      stats_.phase2_auto_candidate_ratio_sum +=
          candidates.auto_candidate_ratio_sum;
      for (size_t parent_idx : candidates.candidate_item_ids) {
        auto& parent = parents[parent_idx];
        int tau = parent->radius + child->radius;
        if (std::llabs(
                static_cast<long long>(parent->center_ptr->seq.size()) -
                static_cast<long long>(child->center_ptr->seq.size())) > tau) {
          stats_.phase2_length_pruned_pairs++;
          continue;
        }
        stats_.phase2_exact_distance_calls++;
        int dist = build_distance_bounded(
            parent->center_ptr->seq, child->center_ptr->seq, tau,
            range_config_.distance_mode);
        if (dist <= tau) {
          parent->child_nodes.push_back(child);
          stats_.phase2_edges_added++;
        }
      }
    }
  }
}

void BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb() {
  primary_layers_.assign(static_cast<size_t>(hierarchy_.num_primary_layers()),
                         std::vector<std::shared_ptr<WorldNode>>());

  for (int primary_idx = 0; primary_idx < hierarchy_.num_primary_layers(); ++primary_idx) {
    auto& target_layer = primary_layers_[static_cast<size_t>(primary_idx)];
    target_layer = extended_layers_[static_cast<size_t>(primary_idx * 2)];
    for (auto& node : target_layer) {
      reset_node_metadata(node, primary_idx * 2, true, primary_idx);
      node->beacons.clear();
      node->child_beacon_mbbs.clear();
      node->mbb_rect_index.reset();
      node->leaf_beacon_dists.clear();
    }
  }

  for (int primary_idx = 0; primary_idx < hierarchy_.num_primary_layers(); ++primary_idx) {
    auto& current_layer = primary_layers_[static_cast<size_t>(primary_idx)];
    const bool is_finest = (primary_idx == finest_primary_layer_index());
    for (auto& node : current_layer) {
      if (is_finest) {
        node->child_nodes.clear();
        continue;
      }

      std::vector<std::shared_ptr<WorldNode>> auxiliary_nodes = node->child_nodes;
      node->beacons.reserve(auxiliary_nodes.size());
      for (const auto& aux : auxiliary_nodes) {
        if (aux && aux->center_ptr) node->beacons.push_back(aux->center_ptr);
      }

      std::vector<std::shared_ptr<WorldNode>> direct_children;
      for (const auto& aux : auxiliary_nodes) {
        if (!aux) continue;
        for (const auto& child : aux->child_nodes) {
          if (std::find(direct_children.begin(), direct_children.end(), child) ==
              direct_children.end()) {
            direct_children.push_back(child);
          }
        }
      }
      node->child_nodes = std::move(direct_children);

      node->child_beacon_mbbs.assign(node->child_nodes.size(), {});
      for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
        auto& child = node->child_nodes[child_idx];
        if (!child->center_ptr) continue;
        node->child_beacon_mbbs[child_idx].reserve(node->beacons.size());
        for (const auto& beacon : node->beacons) {
          if (!beacon) continue;
          int dist = build_distance(child->center_ptr->seq, beacon->seq,
                                    range_config_.distance_mode);
          MBB mbb;
          mbb.min_dist = std::max(0, dist - child->radius);
          mbb.max_dist = dist + child->radius;
          node->child_beacon_mbbs[child_idx].push_back(mbb);
        }
      }

      if (node->child_nodes.size() >= range_config_.min_rect_index_fanout &&
          node->child_nodes.size() <= std::numeric_limits<uint32_t>::max() &&
          !node->beacons.empty() &&
          node->child_beacon_mbbs.size() == node->child_nodes.size()) {
        try {
          std::vector<MBBRectIndex::Rect> rects;
          rects.reserve(node->child_nodes.size());
          bool valid = true;
          for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
            const auto& row = node->child_beacon_mbbs[child_idx];
            if (row.size() != node->beacons.size()) {
              valid = false;
              break;
            }
            MBBRectIndex::Rect rect;
            rect.child_id = static_cast<uint32_t>(child_idx);
            rect.lo.reserve(row.size());
            rect.hi.reserve(row.size());
            for (const auto& mbb : row) {
              rect.lo.push_back(mbb.min_dist);
              rect.hi.push_back(mbb.max_dist);
            }
            rects.push_back(std::move(rect));
          }
          if (valid) {
            auto rect_index = std::make_shared<MBBRectIndex>();
            rect_index->build(rects);
            if (rect_index->size() == node->child_nodes.size() &&
                rect_index->dim() == node->beacons.size()) {
              node->mbb_rect_index = std::move(rect_index);
            }
          }
        } catch (...) {
          node->mbb_rect_index.reset();
        }
      }
    }
  }
}

void BioGeometryIndexBuilder::attach_leaves(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  stats_.total_possible_leaf_pairs = finest_layer.size() * unique_seqs.size();
  for (auto& node : finest_layer) {
    node->child_leaves.clear();
    node->beacons.clear();
    node->leaf_beacon_dists.clear();
    if (node->center_ptr) node->beacons.push_back(node->center_ptr);
  }

  if (range_config_.leaf_attach_mode == BuildRangeMode::Full) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t layer_idx = 0; layer_idx < finest_layer.size(); ++layer_idx) {
      auto& node = finest_layer[layer_idx];
      std::string center = node->get_center_sequence();
      for (const auto& seq : unique_seqs) {
        int dist = build_distance(center, seq->seq, range_config_.distance_mode);
        if (dist <= node->radius) {
          node->child_leaves.push_back(seq);
          node->leaf_beacon_dists.push_back(
              leaf_beacon_distances(seq, node->beacons, dist,
                                    range_config_.distance_mode));
        }
      }
      node->data_count = static_cast<int>(node->child_leaves.size());
    }
    stats_.leaf_candidate_pairs = stats_.total_possible_leaf_pairs;
    stats_.leaf_exact_distance_calls = stats_.total_possible_leaf_pairs;
  } else {
    std::vector<RangeJoinItem> items;
    items.reserve(finest_layer.size());
    int max_radius = 0;
    for (size_t world_idx = 0; world_idx < finest_layer.size(); ++world_idx) {
      const auto& world = finest_layer[world_idx];
      if (!world->center_ptr) continue;
      items.push_back({world_idx, world->center_ptr->seq});
      max_radius = std::max(max_radius, world->radius);
    }
    ExactRangeJoinIndex world_index(range_config_.range_join);
    world_index.build(items);
    for (const auto& seq : unique_seqs) {
      auto candidates = world_index.query(seq->seq, max_radius);
      stats_.leaf_candidate_pairs += candidates.candidate_item_ids.size();
      if (candidates.used_full_scan) stats_.leaf_full_scan_fallback_count++;
      if (candidates.mode_used == RangeCandidateMode::PigeonholeOnly) {
        stats_.leaf_pigeonhole_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::QGramOnly) {
        stats_.leaf_qgram_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::Hybrid) {
        stats_.leaf_hybrid_queries++;
      }
      stats_.leaf_qgram_candidate_pairs += candidates.qgram_candidate_count;
      stats_.leaf_qgram_pruned_by_l1 += candidates.qgram_pruned_by_l1;
      stats_.leaf_length_pruned_pairs += candidates.length_filtered_items;
      stats_.leaf_required_shared_nonpositive_count +=
          candidates.required_shared_nonpositive;
      stats_.leaf_auto_pigeonhole_accepted +=
          candidates.auto_pigeonhole_accepted;
      stats_.leaf_auto_pigeonhole_rejected_large_candidates +=
          candidates.auto_pigeonhole_rejected_large_candidates;
      stats_.leaf_auto_qgram_invoked += candidates.auto_qgram_invoked;
      stats_.leaf_auto_hybrid_invoked += candidates.auto_hybrid_invoked;
      stats_.leaf_auto_final_candidate_pairs +=
          candidates.auto_final_candidate_pairs;
      stats_.leaf_auto_candidate_ratio_sum +=
          candidates.auto_candidate_ratio_sum;
      for (size_t world_idx : candidates.candidate_item_ids) {
        auto& world = finest_layer[world_idx];
        if (std::llabs(static_cast<long long>(seq->seq.size()) -
                       static_cast<long long>(world->center_ptr->seq.size())) >
            world->radius) {
          stats_.leaf_length_pruned_pairs++;
          continue;
        }
        stats_.leaf_exact_distance_calls++;
        int dist = build_distance_bounded(seq->seq, world->center_ptr->seq,
                                          world->radius,
                                          range_config_.distance_mode);
        if (dist <= world->radius) {
          world->child_leaves.push_back(seq);
          world->leaf_beacon_dists.push_back(
              leaf_beacon_distances(seq, world->beacons, dist,
                                    range_config_.distance_mode));
        }
      }
    }
    for (auto& node : finest_layer) {
      node->data_count = static_cast<int>(node->child_leaves.size());
    }
  }

  size_t total_links = 0;
  for (const auto& node : finest_layer) total_links += node->child_leaves.size();
  stats_.leaf_attachments_added = total_links;
  double avg_links =
      finest_layer.empty() ? 0.0 : static_cast<double>(total_links) / finest_layer.size();
  std::cerr << "    Attached " << total_links << " leaf links to finest primary layer"
            << " (avg " << avg_links << " per node)\n";
}

void BioGeometryIndexBuilder::assign_integer_ids() {
  world_node_count_ = 0;
  sequence_count_ = 0;

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (node) node->integer_id = INVALID_NODE_ID;
    }
  }

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id != INVALID_NODE_ID) continue;
      if (world_node_count_ > static_cast<size_t>(INVALID_NODE_ID - 1)) {
        throw std::runtime_error("too many world nodes for 32-bit NodeId");
      }
      node->integer_id = static_cast<NodeId>(world_node_count_++);
    }
  }

  std::vector<std::shared_ptr<BioSequence>> sequences;
  sequences.reserve(unique_sequences.size());
  for (const auto& entry : unique_sequences) {
    if (entry.second) {
      entry.second->sequence_id = INVALID_LEAF_ID;
      sequences.push_back(entry.second);
    }
  }
  std::sort(sequences.begin(), sequences.end(),
            [](const std::shared_ptr<BioSequence>& left,
               const std::shared_ptr<BioSequence>& right) {
              return left->id < right->id;
            });
  for (const auto& sequence : sequences) {
    if (sequence_count_ > static_cast<size_t>(INVALID_LEAF_ID - 1)) {
      throw std::runtime_error("too many sequences for 32-bit LeafId");
    }
    sequence->sequence_id = static_cast<LeafId>(sequence_count_++);
  }
}

void BioGeometryIndexBuilder::build_search_graph_view() {
  auto to_u32 = [](size_t value, const char* field) -> uint32_t {
    if (value > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(std::string(field) + " exceeds 32-bit view range");
    }
    return static_cast<uint32_t>(value);
  };

  SearchGraphView view;
  view.nodes.assign(world_node_count_, nullptr);
  view.leaves.assign(sequence_count_, nullptr);
  view.child_begin.assign(world_node_count_, 0);
  view.child_end.assign(world_node_count_, 0);
  view.leaf_begin.assign(world_node_count_, 0);
  view.leaf_end.assign(world_node_count_, 0);
  view.mbb_begin.assign(world_node_count_, 0);
  view.mbb_dim.assign(world_node_count_, 0);
  view.beacon_begin.assign(world_node_count_, 0);
  view.beacon_end.assign(world_node_count_, 0);
  view.leaf_beacon_begin.assign(world_node_count_, 0);
  view.leaf_beacon_dim.assign(world_node_count_, 0);

  for (const auto& entry : unique_sequences) {
    const auto& sequence = entry.second;
    if (!sequence || sequence->sequence_id >= sequence_count_) {
      throw std::runtime_error("cannot build search graph view with invalid leaf id");
    }
    view.leaves[sequence->sequence_id] = sequence;
  }

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id >= world_node_count_) {
        throw std::runtime_error("cannot build search graph view with invalid node id");
      }
      const NodeId node_id = node->integer_id;
      view.nodes[node_id] = node;

      view.child_begin[node_id] = to_u32(view.child_ids.size(), "child_ids");
      for (const auto& child : node->child_nodes) {
        if (!child || child->integer_id >= world_node_count_) {
          throw std::runtime_error("cannot build search graph view with invalid child id");
        }
        view.child_ids.push_back(child->integer_id);
      }
      view.child_end[node_id] = to_u32(view.child_ids.size(), "child_ids");

      view.leaf_begin[node_id] = to_u32(view.leaf_ids.size(), "leaf_ids");
      for (const auto& leaf : node->child_leaves) {
        if (!leaf || leaf->sequence_id >= sequence_count_) {
          throw std::runtime_error("cannot build search graph view with invalid leaf id");
        }
        view.leaf_ids.push_back(leaf->sequence_id);
      }
      view.leaf_end[node_id] = to_u32(view.leaf_ids.size(), "leaf_ids");

      view.beacon_begin[node_id] = to_u32(view.beacon_ids.size(), "beacon_ids");
      for (const auto& beacon : node->beacons) {
        if (!beacon || beacon->sequence_id >= sequence_count_) {
          throw std::runtime_error("cannot build search graph view with invalid beacon id");
        }
        view.beacon_ids.push_back(beacon->sequence_id);
      }
      view.beacon_end[node_id] = to_u32(view.beacon_ids.size(), "beacon_ids");

      const size_t child_count = node->child_nodes.size();
      const size_t mbb_dim = node->beacons.size();
      view.mbb_begin[node_id] = to_u32(view.mbb_lo.size(), "mbb arrays");
      view.mbb_dim[node_id] = to_u32(mbb_dim, "mbb_dim");
      const size_t mbb_cells = child_count * mbb_dim;
      view.mbb_lo.resize(view.mbb_lo.size() + mbb_cells, 0);
      view.mbb_hi.resize(view.mbb_hi.size() + mbb_cells, 0);
      if (!node->child_beacon_mbbs.empty()) {
        if (node->child_beacon_mbbs.size() != child_count) {
          throw std::runtime_error("child MBB rows are not aligned with child nodes");
        }
        for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
          if (node->child_beacon_mbbs[child_idx].size() != mbb_dim) {
            throw std::runtime_error("child MBB row dimension mismatch");
          }
          for (size_t dim = 0; dim < mbb_dim; ++dim) {
            const size_t flat = view.mbb_begin[node_id] +
                                dim * child_count + child_idx;
            view.mbb_lo[flat] =
                static_cast<int32_t>(node->child_beacon_mbbs[child_idx][dim].min_dist);
            view.mbb_hi[flat] =
                static_cast<int32_t>(node->child_beacon_mbbs[child_idx][dim].max_dist);
          }
        }
      }

      const size_t leaf_count = node->child_leaves.size();
      const size_t leaf_dim = node->beacons.size();
      view.leaf_beacon_begin[node_id] =
          to_u32(view.leaf_beacon_dists.size(), "leaf_beacon_dists");
      view.leaf_beacon_dim[node_id] = to_u32(leaf_dim, "leaf_beacon_dim");
      const size_t leaf_cells = leaf_count * leaf_dim;
      view.leaf_beacon_dists.resize(view.leaf_beacon_dists.size() + leaf_cells, 0);
      if (!node->leaf_beacon_dists.empty()) {
        if (node->leaf_beacon_dists.size() != leaf_count) {
          throw std::runtime_error("leaf beacon rows are not aligned with leaves");
        }
        for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
          if (node->leaf_beacon_dists[leaf_idx].size() != leaf_dim) {
            throw std::runtime_error("leaf beacon row dimension mismatch");
          }
          for (size_t dim = 0; dim < leaf_dim; ++dim) {
            const size_t flat = view.leaf_beacon_begin[node_id] +
                                dim * leaf_count + leaf_idx;
            view.leaf_beacon_dists[flat] =
                static_cast<int32_t>(node->leaf_beacon_dists[leaf_idx][dim]);
          }
        }
      }
    }
  }

  search_graph_view_ = std::move(view);
}

void BioGeometryIndexBuilder::print_summary() const {
  Statistics stats = get_statistics();
  std::cerr << "  Construction range modes: links="
            << build_range_mode_name(range_config_.link_mode)
            << " leaves=" << build_range_mode_name(range_config_.leaf_attach_mode)
            << " seeds=" << range_config_.range_join.min_seed_len
            << ".." << range_config_.range_join.max_seed_len
            << " candidates="
            << range_candidate_mode_name(range_config_.range_join.candidate_mode)
            << " qgram_q=" << range_config_.range_join.qgram_q
            << " auto_max_candidates="
            << range_config_.range_join.auto_pigeonhole_max_candidates
            << " auto_max_ratio="
            << range_config_.range_join.auto_pigeonhole_max_ratio
            << " auto_hybrid="
            << (range_config_.range_join.auto_hybrid_on_large_candidates
                    ? "true"
                    : "false")
            << "\n";
  std::cerr << "  Phase2 range join: possible=" << stats.phase2_total_possible_pairs
            << " candidates=" << stats.phase2_candidate_pairs
            << " exact_calls=" << stats.phase2_exact_distance_calls
            << " edges=" << stats.phase2_edges_added
            << " fallbacks=" << stats.phase2_full_scan_fallback_count
            << " pigeonhole_queries=" << stats.phase2_pigeonhole_queries
            << " qgram_queries=" << stats.phase2_qgram_queries
            << " hybrid_queries=" << stats.phase2_hybrid_queries
            << " qgram_candidates=" << stats.phase2_qgram_candidate_pairs
            << " qgram_l1_pruned=" << stats.phase2_qgram_pruned_by_l1
            << " length_pruned=" << stats.phase2_length_pruned_pairs
            << " required_shared_nonpositive="
            << stats.phase2_required_shared_nonpositive_count
            << " auto_pigeonhole_accepted="
            << stats.phase2_auto_pigeonhole_accepted
            << " auto_pigeonhole_rejected_large="
            << stats.phase2_auto_pigeonhole_rejected_large_candidates
            << " auto_qgram_invoked=" << stats.phase2_auto_qgram_invoked
            << " auto_hybrid_invoked=" << stats.phase2_auto_hybrid_invoked
            << " auto_final_candidates="
            << stats.phase2_auto_final_candidate_pairs
            << " auto_avg_candidate_ratio="
            << stats.phase2_auto_candidate_ratio_avg
            << " candidate_reduction=" << (stats.phase2_candidate_reduction_ratio * 100.0)
            << "% exact_reduction=" << (stats.phase2_exact_distance_reduction_ratio * 100.0)
            << "%\n";
  std::cerr << "  Leaf range join: possible=" << stats.total_possible_leaf_pairs
            << " candidates=" << stats.leaf_candidate_pairs
            << " exact_calls=" << stats.leaf_exact_distance_calls
            << " attachments=" << stats.leaf_attachments_added
            << " fallbacks=" << stats.leaf_full_scan_fallback_count
            << " pigeonhole_queries=" << stats.leaf_pigeonhole_queries
            << " qgram_queries=" << stats.leaf_qgram_queries
            << " hybrid_queries=" << stats.leaf_hybrid_queries
            << " qgram_candidates=" << stats.leaf_qgram_candidate_pairs
            << " qgram_l1_pruned=" << stats.leaf_qgram_pruned_by_l1
            << " length_pruned=" << stats.leaf_length_pruned_pairs
            << " required_shared_nonpositive="
            << stats.leaf_required_shared_nonpositive_count
            << " auto_pigeonhole_accepted="
            << stats.leaf_auto_pigeonhole_accepted
            << " auto_pigeonhole_rejected_large="
            << stats.leaf_auto_pigeonhole_rejected_large_candidates
            << " auto_qgram_invoked=" << stats.leaf_auto_qgram_invoked
            << " auto_hybrid_invoked=" << stats.leaf_auto_hybrid_invoked
            << " auto_final_candidates="
            << stats.leaf_auto_final_candidate_pairs
            << " auto_avg_candidate_ratio="
            << stats.leaf_auto_candidate_ratio_avg
            << " candidate_reduction=" << (stats.leaf_candidate_reduction_ratio * 100.0)
            << "% exact_reduction=" << (stats.leaf_exact_distance_reduction_ratio * 100.0)
            << "%\n";
  std::cerr << "  Primary layers: " << num_primary_layers() << "\n";
  for (int layer_idx = 0; layer_idx < num_primary_layers(); ++layer_idx) {
    std::cerr << "    W" << layer_idx
              << " radius=" << hierarchy_.primary_radii[static_cast<size_t>(layer_idx)]
              << " nodes=" << primary_layers_[static_cast<size_t>(layer_idx)].size() << "\n";
  }

  const auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  if (stats_.unique_sequences > 0 && !finest_layer.empty()) {
    double compression =
        1.0 - static_cast<double>(finest_layer.size()) / stats_.unique_sequences;
    std::cerr << "  Finest-layer compression: " << (compression * 100.0) << "% ("
              << stats_.unique_sequences << " unique -> " << finest_layer.size() << " nodes)\n";
  }

  for (int layer_idx = 0; layer_idx + 1 < num_primary_layers(); ++layer_idx) {
    const auto& layer = primary_layers_[static_cast<size_t>(layer_idx)];
    size_t total_edges = 0;
    for (const auto& node : layer) total_edges += node->child_nodes.size();
    double avg_edges = layer.empty() ? 0.0 : static_cast<double>(total_edges) / layer.size();
    std::cerr << "  Avg W" << layer_idx << " -> W" << (layer_idx + 1)
              << " edges: " << avg_edges << "\n";
  }
}

void BioGeometryIndexBuilder::build(
    const std::vector<std::shared_ptr<BioSequence>>& raw_sequences) {
  stats_ = Statistics{};
  stats_.created_primary_nodes.assign(static_cast<size_t>(num_primary_layers()), 0);
  unique_sequences.clear();
  world_node_count_ = 0;
  sequence_count_ = 0;
  search_graph_view_ = SearchGraphView{};
  primary_layers_.assign(static_cast<size_t>(num_primary_layers()),
                         std::vector<std::shared_ptr<WorldNode>>());
  extended_layers_.clear();

  std::cerr << "[Build generalized hierarchy] Starting for " << raw_sequences.size()
            << " sequences...\n";
  std::cerr << "  Phase 0: Deduplicating sequences...\n";
  auto unique_seqs = deduplicate(raw_sequences);
  std::cerr << "    " << raw_sequences.size() << " -> " << unique_seqs.size() << " unique ("
            << stats_.deduplicated << " merged)\n";

  std::cerr << "  Phase 1: Extended hierarchy sketch (top-down)...\n";
  std::cerr << "    Primary radii: ";
  for (size_t i = 0; i < hierarchy_.primary_radii.size(); ++i) {
    if (i) std::cerr << ",";
    std::cerr << hierarchy_.primary_radii[i];
  }
  std::cerr << "\n";
  std::cerr << "    Auxiliary radii: ";
  for (size_t i = 0; i < hierarchy_.auxiliary_radii.size(); ++i) {
    if (i) std::cerr << ",";
    std::cerr << hierarchy_.auxiliary_radii[i];
  }
  std::cerr << "\n";
  phase1_build_extended_sketch(unique_seqs);

  std::cerr << "    Expanded layers:";
  for (size_t i = 0; i < extended_layers_.size(); ++i) {
    std::cerr << " L" << i << "=" << extended_layers_[i].size();
  }
  std::cerr << "\n";

  std::cerr << "  Phase 2: Inter-tier rebinding (DAG)...\n";
  phase2_inter_tier_rebinding();

  std::cerr << "  Phase 3: Collapse auxiliary tiers + MBB...\n";
  phase3_collapse_and_compute_mbb();

  std::cerr << "  Phase 4: Leaf attachment...\n";
  attach_leaves(unique_seqs);
  assign_integer_ids();
  build_search_graph_view();

  std::cerr << "[Build generalized hierarchy] Completed.\n";
  print_summary();
}

BioGeometryIndexBuilder::Statistics BioGeometryIndexBuilder::get_statistics() const {
  Statistics stats = stats_;
  stats.phase2_candidate_reduction_ratio =
      reduction_ratio(stats.phase2_total_possible_pairs, stats.phase2_candidate_pairs);
  stats.phase2_exact_distance_reduction_ratio =
      reduction_ratio(stats.phase2_total_possible_pairs, stats.phase2_exact_distance_calls);
  stats.leaf_candidate_reduction_ratio =
      reduction_ratio(stats.total_possible_leaf_pairs, stats.leaf_candidate_pairs);
  stats.leaf_exact_distance_reduction_ratio =
      reduction_ratio(stats.total_possible_leaf_pairs, stats.leaf_exact_distance_calls);
  const size_t phase2_auto_ratio_count =
      stats.phase2_auto_pigeonhole_accepted +
      stats.phase2_auto_pigeonhole_rejected_large_candidates;
  if (phase2_auto_ratio_count > 0) {
    stats.phase2_auto_candidate_ratio_avg =
        stats.phase2_auto_candidate_ratio_sum /
        static_cast<double>(phase2_auto_ratio_count);
  }
  const size_t leaf_auto_ratio_count =
      stats.leaf_auto_pigeonhole_accepted +
      stats.leaf_auto_pigeonhole_rejected_large_candidates;
  if (leaf_auto_ratio_count > 0) {
    stats.leaf_auto_candidate_ratio_avg =
        stats.leaf_auto_candidate_ratio_sum /
        static_cast<double>(leaf_auto_ratio_count);
  }
  const auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  if (stats.unique_sequences > 0 && !finest_layer.empty()) {
    stats.compression_ratio =
        1.0 - static_cast<double>(finest_layer.size()) / stats.unique_sequences;
  }

  if (num_primary_layers() >= 2) {
    size_t total_edges = 0;
    size_t total_nodes = 0;
    for (int layer_idx = 0; layer_idx + 1 < num_primary_layers(); ++layer_idx) {
      const auto& layer = primary_layers_[static_cast<size_t>(layer_idx)];
      total_nodes += layer.size();
      for (const auto& node : layer) total_edges += node->child_nodes.size();
    }
    if (total_nodes > 0) {
      stats.dag_redundancy =
          (static_cast<double>(total_edges) / static_cast<double>(total_nodes) - 1.0) * 100.0;
    }
  }
  return stats;
}

}  // namespace navigamer
