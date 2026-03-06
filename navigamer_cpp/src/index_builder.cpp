#include "index_builder.hpp"
#include <iostream>
#include <unordered_set>
#include <random>
#include <algorithm>

namespace navigamer {

BioGeometryIndexBuilder::BioGeometryIndexBuilder() {
  for (int i = 0; i < 4; ++i) {
    layers[i].clear();
    layer_beacons[i].clear();
  }
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

std::vector<std::shared_ptr<WorldNode>> BioGeometryIndexBuilder::build_layer_sparse(
    const std::vector<std::shared_ptr<BioSequence>>& items,
    int radius, int layer_level, const std::string& label) {
  std::vector<std::shared_ptr<WorldNode>> nodes;
  std::vector<size_t> indices(items.size());
  for (size_t i = 0; i < items.size(); ++i) indices[i] = i;
  shuffle_indices(indices, static_cast<unsigned>(std::random_device{}()));

  for (size_t idx = 0; idx < indices.size(); ++idx) {
    size_t i = indices[idx];
    const auto& item = items[i];
    bool covered = false;
    for (const auto& node : nodes) {
      if (!node->center_ptr) continue;
      int d = compute_distance(item->seq, node->center_ptr->seq);
      if (d <= radius) { covered = true; break; }
    }
    if (!covered) {
      auto node = std::make_shared<WorldNode>(item, radius, layer_level);
      nodes.push_back(node);
      stats_.created_nodes[layer_level]++;
    }
    if ((idx + 1) % 500 == 0 && items.size() > 500)
      std::cerr << "    " << label << ": scanned " << (idx+1) << "/" << items.size()
                << ", nodes=" << nodes.size() << "\r";
  }
  if (items.size() > 500) std::cerr << "\n";
  return nodes;
}

std::vector<std::shared_ptr<WorldNode>> BioGeometryIndexBuilder::build_layer_sparse_from_nodes(
    const std::vector<std::shared_ptr<WorldNode>>& items,
    int radius, int layer_level, const std::string& label) {
  std::vector<std::shared_ptr<WorldNode>> nodes;
  std::vector<size_t> indices(items.size());
  for (size_t i = 0; i < items.size(); ++i) indices[i] = i;
  shuffle_indices(indices, static_cast<unsigned>(std::random_device{}()));

  for (size_t idx = 0; idx < indices.size(); ++idx) {
    size_t i = indices[idx];
    const auto& item = items[i];
    std::string center = item->get_center_sequence();
    bool covered = false;
    for (const auto& node : nodes) {
      if (!node->center_ptr) continue;
      int d = compute_distance(center, node->center_ptr->seq);
      if (d <= radius) { covered = true; break; }
    }
    if (!covered) {
      auto node = std::make_shared<WorldNode>(item->center_ptr, radius, layer_level);
      nodes.push_back(node);
      stats_.created_nodes[layer_level]++;
    }
    if ((idx + 1) % 500 == 0 && items.size() > 500)
      std::cerr << "    " << label << ": scanned " << (idx+1) << "/" << items.size()
                << ", nodes=" << nodes.size() << "\r";
  }
  if (items.size() > 500) std::cerr << "\n";
  return nodes;
}

void BioGeometryIndexBuilder::build_skeleton(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  layers[1] = build_layer_sparse(unique_seqs, R_SW, 1, "SW");
  layers[2] = build_layer_sparse_from_nodes(layers[1], R_MW, 2, "MW");
  layers[3] = build_layer_sparse_from_nodes(layers[2], R_LW, 3, "LW");
}

void BioGeometryIndexBuilder::wire_overlap(
    std::vector<std::shared_ptr<WorldNode>>& parents,
    const std::vector<std::shared_ptr<WorldNode>>& children) {
  for (auto& parent : parents) {
    if (!parent->center_ptr) continue;
    std::string p_seq = parent->center_ptr->seq;
    for (const auto& child : children) {
      if (!child->center_ptr) continue;
      int d = compute_distance(p_seq, child->center_ptr->seq);
      if (d <= parent->radius + child->radius)
        parent->child_nodes.push_back(child);
    }
  }
}

void BioGeometryIndexBuilder::dense_wiring() {
  auto& sw = layers[1];
  auto& mw = layers[2];
  auto& lw = layers[3];

  std::cerr << "    Wiring " << sw.size() << " SW -> " << mw.size() << " MW...\n";
  for (auto& n : mw) n->child_nodes.clear();
  wire_overlap(mw, sw);

  std::cerr << "    Wiring " << mw.size() << " MW -> " << lw.size() << " LW...\n";
  for (auto& n : lw) n->child_nodes.clear();
  wire_overlap(lw, mw);

  std::vector<std::shared_ptr<WorldNode>> mw2, lw2;
  for (const auto& n : mw) if (!n->child_nodes.empty()) mw2.push_back(n);
  for (const auto& n : lw) if (!n->child_nodes.empty()) lw2.push_back(n);
  layers[2] = std::move(mw2);
  layers[3] = std::move(lw2);
  std::cerr << "    After cleanup: MW=" << layers[2].size() << ", LW=" << layers[3].size() << "\n";
}

void BioGeometryIndexBuilder::inject_beacons() {
  const int K = BEACON_COUNT;
  if (!layers[3].empty()) {
    size_t k = std::min(static_cast<size_t>(K), layers[3].size());
    auto idx = farthest_point_sampling(layers[3], k);
    for (size_t i : idx) layer_beacons[3].push_back(layers[3][i]);
    for (auto& mw : layers[2]) {
      for (const auto& b : layer_beacons[3]) {
        int d = compute_distance(mw->get_center_sequence(), b->get_center_sequence());
        mw->beacon_dists.push_back(d);
      }
    }
    std::cerr << "    LW beacons: " << layer_beacons[3].size()
              << ", MW nodes have beacon_dists (len=" << layer_beacons[3].size() << ")\n";
  }
  if (!layers[2].empty()) {
    size_t k = std::min(static_cast<size_t>(K), layers[2].size());
    auto idx = farthest_point_sampling(layers[2], k);
    for (size_t i : idx) layer_beacons[2].push_back(layers[2][i]);
    for (auto& sw : layers[1]) {
      for (const auto& b : layer_beacons[2]) {
        int d = compute_distance(sw->get_center_sequence(), b->get_center_sequence());
        sw->beacon_dists.push_back(d);
      }
    }
    std::cerr << "    MW beacons: " << layer_beacons[2].size()
              << ", SW nodes have beacon_dists (len=" << layer_beacons[2].size() << ")\n";
  }
}

void BioGeometryIndexBuilder::attach_leaves(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  size_t total_links = 0;
  for (auto& sw : layers[1]) {
    std::string center = sw->get_center_sequence();
    for (const auto& seq : unique_seqs) {
      int d = compute_distance(center, seq->seq);
      if (d <= sw->radius) {
        sw->child_leaves.push_back(seq);
        total_links++;
      }
    }
    sw->data_count = static_cast<int>(sw->child_leaves.size());
  }
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
  std::cerr << "[Build v7 Multilateration] Starting for " << raw_sequences.size() << " sequences...\n";
  std::cerr << "  Phase 0: Deduplicating sequences...\n";
  auto unique_seqs = deduplicate(raw_sequences);
  std::cerr << "    " << raw_sequences.size() << " -> " << unique_seqs.size() << " unique ("
            << stats_.deduplicated << " merged)\n";

  std::cerr << "  Phase 1: Skeleton generation (sparse selection)...\n";
  build_skeleton(unique_seqs);
  std::cerr << "    SW=" << layers[1].size() << ", MW=" << layers[2].size()
            << ", LW=" << layers[3].size() << "\n";

  std::cerr << "  Phase 2: Dense wiring (exhaustive overlap)...\n";
  dense_wiring();

  std::cerr << "  Phase 3: Beacon injection (FPS)...\n";
  inject_beacons();

  std::cerr << "  Phase 4: Leaf attachment...\n";
  attach_leaves(unique_seqs);

  std::cerr << "[Build v7] Completed.\n";
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
