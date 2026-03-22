#include "index_builder.hpp"
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <omp.h>

namespace navigamer {

namespace {

std::shared_ptr<WorldNode> nearest_in_cover(
    const std::vector<std::shared_ptr<WorldNode>>& W_cover,
    const std::shared_ptr<BioSequence>& u) {
  std::shared_ptr<WorldNode> best;
  int best_d = INT_MAX;
  for (const auto& w : W_cover) {
    if (!w->center_ptr) continue;
    int d = compute_distance(u->seq, w->center_ptr->seq);
    if (d < best_d) {
      best_d = d;
      best = w;
    }
  }
  return best;
}

void bump_created_stats(BioGeometryIndexBuilder::Statistics& st, int extended_tier) {
  if (extended_tier == 0) st.created_nodes[3]++;
  else if (extended_tier == 2) st.created_nodes[2]++;
  else if (extended_tier == 4) st.created_nodes[1]++;
  else st.created_nodes[0]++;
}

}  // namespace

BioGeometryIndexBuilder::BioGeometryIndexBuilder() {
  radius_config[0] = 0;
  radius_config[1] = R_SW;
  radius_config[2] = R_MW;
  radius_config[3] = R_LW;
  for (int i = 0; i < 4; ++i) layers[i].clear();
  int r_lw = radius_config[3];
  int r_mw = radius_config[2];
  int r_sw = radius_config[1];
  int r_int1 = std::max(1, (r_lw + r_mw) / 2);
  int r_int2 = std::max(1, (r_mw + r_sw) / 2);
  extended_radii_ = {r_lw, r_int1, r_mw, r_int2, r_sw};
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw) {
  radius_config[0] = 0;
  radius_config[1] = r_sw;
  radius_config[2] = r_mw;
  radius_config[3] = r_lw;
  for (int i = 0; i < 4; ++i) layers[i].clear();
  int r_int1 = std::max(1, (r_lw + r_mw) / 2);
  int r_int2 = std::max(1, (r_mw + r_sw) / 2);
  extended_radii_ = {r_lw, r_int1, r_mw, r_int2, r_sw};
}

std::vector<std::shared_ptr<WorldNode>> BioGeometryIndexBuilder::find_neighbors(
    const BioSequence& query_seq,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    int radius) const {
  std::vector<std::shared_ptr<WorldNode>> result;
  for (const auto& node : candidates) {
    if (!node->center_ptr) continue;
    int d = compute_distance(query_seq.seq, node->center_ptr->seq);
    if (d <= radius) result.push_back(node);
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
      for (const auto& occ : seq->ref_positions)
        it->second->add_occurrence(occ.ref_id, occ.start, occ.end, occ.strand);
      if (seq->ref_positions.empty() && it->second->ref_positions.empty())
        it->second->add_occurrence(seq->id, 0, static_cast<int>(seq->seq.size()), "+");
      if (!it->second->bwt_interval.valid() && seq->bwt_interval.valid())
        it->second->set_bwt_interval(seq->bwt_interval.start, seq->bwt_interval.end);
      stats_.deduplicated++;
    } else {
      seq_map[seq->seq] = seq;
    }
  }
  unique_sequences.clear();
  for (const auto& p : seq_map) unique_sequences[p.second->id] = p.second;
  stats_.unique_sequences = seq_map.size();
  std::vector<std::shared_ptr<BioSequence>> out;
  for (const auto& p : seq_map) out.push_back(p.second);
  return out;
}

void BioGeometryIndexBuilder::phase1_build_extended_sketch(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  extended_layers_.assign(5, std::vector<std::shared_ptr<WorldNode>>());

  for (const auto& u : unique_seqs) {
    std::shared_ptr<WorldNode> parent;
    for (int l = 0; l < 5; ++l) {
      std::vector<std::shared_ptr<WorldNode>> W_cover;
      if (l == 0) {
        for (auto& w : extended_layers_[0]) {
          if (!w->center_ptr) continue;
          if (compute_distance(u->seq, w->center_ptr->seq) <= extended_radii_[static_cast<size_t>(l)])
            W_cover.push_back(w);
        }
      } else {
        for (auto& w : parent->child_nodes) {
          if (!w->center_ptr) continue;
          if (compute_distance(u->seq, w->center_ptr->seq) <= extended_radii_[static_cast<size_t>(l)])
            W_cover.push_back(w);
        }
      }

      if (W_cover.empty()) {
        auto w_new = std::make_shared<WorldNode>(u, extended_radii_[static_cast<size_t>(l)], l);
        extended_layers_[static_cast<size_t>(l)].push_back(w_new);
        if (parent) parent->child_nodes.push_back(w_new);
        W_cover.push_back(w_new);
        bump_created_stats(stats_, l);
      }
      parent = nearest_in_cover(W_cover, u);
      if (!parent) break;
    }
  }
}

void BioGeometryIndexBuilder::phase2_inter_tier_rebinding() {
  for (int l = 0; l < 4; ++l) {
    int r_p = extended_radii_[static_cast<size_t>(l)];
    int r_c = extended_radii_[static_cast<size_t>(l + 1)];
    for (auto& p : extended_layers_[static_cast<size_t>(l)]) {
      if (!p->center_ptr) continue;
      p->child_nodes.clear();
      std::string p_seq = p->center_ptr->seq;
      for (auto& c : extended_layers_[static_cast<size_t>(l + 1)]) {
        if (!c->center_ptr) continue;
        int d = compute_distance(p_seq, c->center_ptr->seq);
        if (d <= r_p + r_c) p->child_nodes.push_back(c);
      }
    }
  }
}

void BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb() {
  layers[3] = extended_layers_[0];
  layers[2] = extended_layers_[2];
  layers[1] = extended_layers_[4];

  auto collapse_primary = [&](int primary_ext_idx, int primary_layer_id) {
    for (auto& w : extended_layers_[static_cast<size_t>(primary_ext_idx)]) {
      w->layer = primary_layer_id;
      w->beacons.clear();
      w->child_beacon_mbbs.clear();

      std::vector<std::shared_ptr<WorldNode>> int_nodes = w->child_nodes;

      for (auto& v : int_nodes) {
        if (v && v->center_ptr) w->beacons.push_back(v->center_ptr);
      }

      std::vector<std::shared_ptr<WorldNode>> direct_children;
      for (auto& v : int_nodes) {
        if (!v) continue;
        for (auto& c : v->child_nodes) {
          if (std::find(direct_children.begin(), direct_children.end(), c) == direct_children.end())
            direct_children.push_back(c);
        }
      }
      w->child_nodes = std::move(direct_children);

      w->child_beacon_mbbs.assign(w->child_nodes.size(), {});
      for (size_t j = 0; j < w->child_nodes.size(); ++j) {
        auto& child = w->child_nodes[j];
        if (!child->center_ptr) continue;
        w->child_beacon_mbbs[j].reserve(w->beacons.size());
        for (auto& b : w->beacons) {
          if (!b) continue;
          int d_cb = compute_distance(child->center_ptr->seq, b->seq);
          MBB mbb;
          mbb.min_dist = std::max(0, d_cb - child->radius);
          mbb.max_dist = d_cb + child->radius;
          w->child_beacon_mbbs[j].push_back(mbb);
        }
      }
    }
  };

  collapse_primary(0, 3);
  collapse_primary(2, 2);

  for (auto& w : layers[1]) {
    w->layer = 1;
    w->beacons.clear();
    w->child_beacon_mbbs.clear();
  }

  extended_layers_.clear();
}

void BioGeometryIndexBuilder::attach_leaves(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  #pragma omp parallel for schedule(dynamic)
  for (size_t si = 0; si < layers[1].size(); ++si) {
    auto& sw = layers[1][si];
    std::string center = sw->get_center_sequence();
    for (const auto& seq : unique_seqs) {
      int d = compute_distance(center, seq->seq);
      if (d <= sw->radius) sw->child_leaves.push_back(seq);
    }
    sw->data_count = static_cast<int>(sw->child_leaves.size());
  }
  size_t total_links = 0;
  for (const auto& sw : layers[1]) total_links += sw->child_leaves.size();
  double avg = layers[1].empty() ? 0 : static_cast<double>(total_links) / layers[1].size();
  std::cerr << "    Attached " << total_links << " leaf-SW links (avg " << avg << " per SW)\n";
}

void BioGeometryIndexBuilder::print_summary() const {
  std::cerr << "  Layer 1 (SW): " << layers[1].size() << " nodes\n";
  std::cerr << "  Layer 2 (MW): " << layers[2].size() << " nodes\n";
  std::cerr << "  Layer 3 (LW): " << layers[3].size() << " nodes\n";
  if (!layers[1].empty() && !layers[2].empty()) {
    size_t total_refs = 0;
    for (const auto& mw : layers[2]) total_refs += mw->child_nodes.size();
    double avg_parents = static_cast<double>(total_refs) / layers[1].size();
    std::cerr << "  Avg parents per SW: " << avg_parents << "\n";
  }
  if (stats_.unique_sequences > 0 && !layers[1].empty()) {
    double compression = 1.0 - static_cast<double>(layers[1].size()) / stats_.unique_sequences;
    std::cerr << "  Compression: " << (compression * 100) << "% ("
              << stats_.unique_sequences << " unique -> " << layers[1].size() << " SW)\n";
  }
}

void BioGeometryIndexBuilder::build(
    const std::vector<std::shared_ptr<BioSequence>>& raw_sequences) {
  std::cerr << "[Build v8 Top-down + Intermediate Collapse] Starting for " << raw_sequences.size()
            << " sequences...\n";
  std::cerr << "  Phase 0: Deduplicating sequences...\n";
  auto unique_seqs = deduplicate(raw_sequences);
  std::cerr << "    " << raw_sequences.size() << " -> " << unique_seqs.size() << " unique ("
            << stats_.deduplicated << " merged)\n";

  std::cerr << "  Phase 1: Extended hierarchy sketch (top-down)...\n";
  std::cerr << "    Radii LW,INT1,MW,INT2,SW = " << extended_radii_[0] << "," << extended_radii_[1] << ","
            << extended_radii_[2] << "," << extended_radii_[3] << "," << extended_radii_[4] << "\n";
  phase1_build_extended_sketch(unique_seqs);
  std::cerr << "    Ext layers: "
            << extended_layers_[0].size() << " LW, " << extended_layers_[1].size() << " I1, "
            << extended_layers_[2].size() << " MW, " << extended_layers_[3].size() << " I2, "
            << extended_layers_[4].size() << " SW\n";

  std::cerr << "  Phase 2: Inter-tier rebinding (DAG)...\n";
  phase2_inter_tier_rebinding();

  std::cerr << "  Phase 3: Collapse intermediate tiers + MBB...\n";
  phase3_collapse_and_compute_mbb();

  std::cerr << "  Phase 4: Leaf attachment...\n";
  attach_leaves(unique_seqs);

  std::cerr << "[Build v8] Completed.\n";
  print_summary();
}

BioGeometryIndexBuilder::Statistics BioGeometryIndexBuilder::get_statistics() const {
  Statistics s = stats_;
  size_t sw = layers[1].size(), mw = layers[2].size();
  if (stats_.unique_sequences > 0 && sw > 0)
    s.compression_ratio = 1.0 - static_cast<double>(sw) / stats_.unique_sequences;
  if (sw > 0 && mw > 0) {
    size_t total_sw_refs = 0;
    for (const auto& n : layers[2]) total_sw_refs += n->child_nodes.size();
    double avg = static_cast<double>(total_sw_refs) / sw;
    s.dag_redundancy = (avg - 1.0) * 100.0;
  }
  return s;
}

}  // namespace navigamer
