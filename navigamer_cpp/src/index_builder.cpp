#include "index_builder.hpp"
#include <algorithm>
#include <climits>
#include <iostream>
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

BioGeometryIndexBuilder::BioGeometryIndexBuilder()
    : hierarchy_(HierarchyConfig({R_LW, R_MW, R_SW})),
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())),
      stats_{} {
  stats_.created_primary_nodes.assign(static_cast<size_t>(hierarchy_.num_primary_layers()), 0);
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw)
    : BioGeometryIndexBuilder(HierarchyConfig({r_lw, r_mw, r_sw})) {}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(const HierarchyConfig& config)
    : hierarchy_(config),
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())),
      stats_{} {
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
    int parent_radius = expanded_radii_[static_cast<size_t>(layer_idx)];
    int child_radius = expanded_radii_[static_cast<size_t>(layer_idx + 1)];
    for (auto& parent : extended_layers_[static_cast<size_t>(layer_idx)]) {
      if (!parent->center_ptr) continue;
      parent->child_nodes.clear();
      std::string parent_seq = parent->center_ptr->seq;
      for (auto& child : extended_layers_[static_cast<size_t>(layer_idx + 1)]) {
        if (!child->center_ptr) continue;
        int dist = compute_distance(parent_seq, child->center_ptr->seq);
        if (dist <= parent_radius + child_radius) {
          parent->child_nodes.push_back(child);
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
    }
  }
}

void BioGeometryIndexBuilder::attach_leaves(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  #pragma omp parallel for schedule(dynamic)
  for (size_t layer_idx = 0; layer_idx < finest_layer.size(); ++layer_idx) {
    auto& node = finest_layer[layer_idx];
    node->child_leaves.clear();
    node->beacons.clear();
    node->leaf_beacon_dists.clear();
    if (node->center_ptr) node->beacons.push_back(node->center_ptr);

    std::string center = node->get_center_sequence();
    for (const auto& seq : unique_seqs) {
      int dist = compute_distance(center, seq->seq);
      if (dist <= node->radius) {
        node->child_leaves.push_back(seq);
        node->leaf_beacon_dists.push_back(leaf_beacon_distances(seq, node->beacons, dist));
      }
    }
    node->data_count = static_cast<int>(node->child_leaves.size());
  }

  size_t total_links = 0;
  for (const auto& node : finest_layer) total_links += node->child_leaves.size();
  double avg_links =
      finest_layer.empty() ? 0.0 : static_cast<double>(total_links) / finest_layer.size();
  std::cerr << "    Attached " << total_links << " leaf links to finest primary layer"
            << " (avg " << avg_links << " per node)\n";
}

void BioGeometryIndexBuilder::print_summary() const {
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
