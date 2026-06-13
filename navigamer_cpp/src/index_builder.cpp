#include "index_builder.hpp"
#include <algorithm>
#include <climits>
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
    const std::shared_ptr<BioSequence>& sequence) {
  std::shared_ptr<WorldNode> best;
  int best_dist = INT_MAX;
  for (const auto& node : cover) {
    if (!node->center_ptr) continue;
    int dist = compute_distance(sequence->seq, node->center_ptr->seq);
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
    int center_dist) {
  std::vector<int> dists;
  dists.reserve(beacons.size());
  for (size_t i = 0; i < beacons.size(); ++i) {
    if (!beacons[i]) {
      dists.push_back(0);
    } else if (i == 0) {
      dists.push_back(center_dist);
    } else {
      dists.push_back(compute_distance(leaf->seq, beacons[i]->seq));
    }
  }
  return dists;
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

std::vector<std::shared_ptr<WorldNode>> BioGeometryIndexBuilder::find_neighbors(
    const BioSequence& query_seq,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    int radius) const {
  std::vector<std::shared_ptr<WorldNode>> result;
  for (const auto& node : candidates) {
    if (!node->center_ptr) continue;
    int dist = compute_distance(query_seq.seq, node->center_ptr->seq);
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
          if (compute_distance(sequence->seq, node->center_ptr->seq) <=
              expanded_radii_[static_cast<size_t>(layer_idx)]) {
            cover.push_back(node);
          }
        }
      } else if (parent) {
        for (const auto& node : parent->child_nodes) {
          if (!node->center_ptr) continue;
          if (compute_distance(sequence->seq, node->center_ptr->seq) <=
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

      parent = nearest_in_cover(cover, sequence);
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
          int dist = compute_distance(parent->center_ptr->seq, child->center_ptr->seq);
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
        int dist = compute_distance_bounded(
            parent->center_ptr->seq, child->center_ptr->seq, tau);
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
          int dist = compute_distance(child->center_ptr->seq, beacon->seq);
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
        int dist = compute_distance(center, seq->seq);
        if (dist <= node->radius) {
          node->child_leaves.push_back(seq);
          node->leaf_beacon_dists.push_back(
              leaf_beacon_distances(seq, node->beacons, dist));
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
      for (size_t world_idx : candidates.candidate_item_ids) {
        auto& world = finest_layer[world_idx];
        if (std::llabs(static_cast<long long>(seq->seq.size()) -
                       static_cast<long long>(world->center_ptr->seq.size())) >
            world->radius) {
          stats_.leaf_length_pruned_pairs++;
          continue;
        }
        stats_.leaf_exact_distance_calls++;
        int dist =
            compute_distance_bounded(seq->seq, world->center_ptr->seq, world->radius);
        if (dist <= world->radius) {
          world->child_leaves.push_back(seq);
          world->leaf_beacon_dists.push_back(
              leaf_beacon_distances(seq, world->beacons, dist));
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

void BioGeometryIndexBuilder::print_summary() const {
  Statistics stats = get_statistics();
  std::cerr << "  Construction range modes: links="
            << build_range_mode_name(range_config_.link_mode)
            << " leaves=" << build_range_mode_name(range_config_.leaf_attach_mode)
            << " seeds=" << range_config_.range_join.min_seed_len
            << ".." << range_config_.range_join.max_seed_len
            << " candidates="
            << range_candidate_mode_name(range_config_.range_join.candidate_mode)
            << " qgram_q=" << range_config_.range_join.qgram_q << "\n";
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
