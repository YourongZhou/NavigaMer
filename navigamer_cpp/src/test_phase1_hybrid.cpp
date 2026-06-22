#include "index_builder.hpp"
#include "search_engine.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using SequencePtr = std::shared_ptr<navigamer::BioSequence>;
using LinkMap = std::map<std::string, std::set<std::string>>;

std::string random_dna(size_t length, std::mt19937& gen) {
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> pick(0, 3);
  std::string out;
  out.reserve(length);
  for (size_t i = 0; i < length; ++i) out.push_back(bases[pick(gen)]);
  return out;
}

std::string mutate(std::string value, int edits, std::mt19937& gen) {
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> base(0, 3);
  for (int i = 0; i < edits; ++i) {
    std::uniform_int_distribution<size_t> pos(0, value.size() - 1);
    value[pos(gen)] = bases[base(gen)];
  }
  return value;
}

std::vector<SequencePtr> make_sequences() {
  std::mt19937 gen(4242);
  std::vector<SequencePtr> sequences;
  for (size_t i = 0; i < 160; ++i) {
    sequences.push_back(std::make_shared<navigamer::BioSequence>(
        "seq_" + std::to_string(i), random_dna(90, gen)));
  }
  for (size_t cluster = 0; cluster < 8; ++cluster) {
    const std::string center = random_dna(90, gen);
    for (int edit = 0; edit < 6; ++edit) {
      sequences.push_back(std::make_shared<navigamer::BioSequence>(
          "near_" + std::to_string(cluster) + "_" + std::to_string(edit),
          mutate(center, edit, gen)));
    }
  }
  sequences.push_back(std::make_shared<navigamer::BioSequence>(
      "ambiguous", "ACGTACGTNNNNACGTACGTNNNNACGTACGTNNNNACGTACGT"));
  return sequences;
}

std::vector<SequencePtr> make_sliding_sequences() {
  std::mt19937 gen(7731);
  const std::string reference = random_dna(420, gen);
  std::vector<SequencePtr> sequences;
  for (size_t start = 0; start + 90 <= reference.size(); ++start) {
    sequences.push_back(std::make_shared<navigamer::BioSequence>(
        "window_" + std::to_string(start), reference.substr(start, 90)));
  }
  return sequences;
}

LinkMap primary_edges(const navigamer::BioGeometryIndexBuilder& builder) {
  LinkMap edges;
  for (int layer_idx = 0; layer_idx + 1 < builder.num_primary_layers();
       ++layer_idx) {
    for (const auto& parent : builder.primary_layer(layer_idx)) {
      auto& children = edges[parent->get_center_sequence()];
      for (const auto& child : parent->child_nodes) {
        children.insert(child->get_center_sequence());
      }
    }
  }
  return edges;
}

LinkMap leaf_links(const navigamer::BioGeometryIndexBuilder& builder) {
  LinkMap links;
  for (const auto& world :
       builder.primary_layer(builder.finest_primary_layer_index())) {
    auto& leaves = links[world->get_center_sequence()];
    for (const auto& leaf : world->child_leaves) leaves.insert(leaf->seq);
  }
  return links;
}

std::set<std::string> adaptive_hits(
    const navigamer::BioGeometryIndexBuilder& builder,
    const std::string& sequence, int tau) {
  navigamer::BioGeometrySearchEngine engine(builder);
  navigamer::BioSequence query("query", sequence);
  auto [hits, stats] = engine.search_adaptive(query, tau);
  std::set<std::string> out;
  for (const auto& hit : hits) out.insert(hit->seq);
  return out;
}

navigamer::BuildRangeConfig phase1_scan_config() {
  navigamer::BuildRangeConfig config;
  config.phase1_candidate_mode = navigamer::Phase1CandidateMode::Scan;
  config.range_join.qgram_q = 5;
  return config;
}

navigamer::BuildRangeConfig phase1_hybrid_config() {
  navigamer::BuildRangeConfig config;
  config.phase1_candidate_mode = navigamer::Phase1CandidateMode::Hybrid;
  config.phase1_metric_min_fanout = 4;
  config.phase1_qgram_min_fanout = 10;
  config.phase1_qgram_max_touched = 100000;
  config.range_join.qgram_q = 5;
  return config;
}

}  // namespace

int main() {
  const auto sequences = make_sequences();
  const navigamer::HierarchyConfig hierarchy({8, 4, 2});

  navigamer::BioGeometryIndexBuilder scan_builder(
      hierarchy, phase1_scan_config());
  navigamer::BioGeometryIndexBuilder hybrid_builder(
      hierarchy, phase1_hybrid_config());
  scan_builder.build(sequences);
  hybrid_builder.build(sequences);

  assert(primary_edges(scan_builder) == primary_edges(hybrid_builder));
  assert(leaf_links(scan_builder) == leaf_links(hybrid_builder));
  for (size_t idx : {size_t{0}, size_t{7}, size_t{31}, size_t{95},
                     size_t{150}, size_t{170}}) {
    assert(adaptive_hits(scan_builder, sequences[idx]->seq, 2) ==
           adaptive_hits(hybrid_builder, sequences[idx]->seq, 2));
  }

  const auto scan_stats = scan_builder.get_statistics();
  const auto hybrid_stats = hybrid_builder.get_statistics();
  assert(scan_stats.phase1_scan_queries > 0);
  assert(scan_stats.phase1_metric_index_queries == 0);
  assert(scan_stats.phase1_qgram_index_queries == 0);

  assert(hybrid_stats.phase1_scan_queries > 0);
  assert(hybrid_stats.phase1_metric_index_queries > 0);
  assert(hybrid_stats.phase1_pigeonhole_queries > 0);
  assert(hybrid_stats.phase1_seed_posting_entries_visited > 0);
  assert(hybrid_stats.phase1_pigeonhole_candidates > 0);
  assert(hybrid_stats.phase1_total_possible_pairs >
         hybrid_stats.phase1_candidate_pairs);
  assert(hybrid_stats.phase1_candidate_pairs ==
         hybrid_stats.phase1_exact_distance_calls);
  assert(hybrid_stats.phase1_candidate_pairs <=
         hybrid_stats.phase1_total_possible_pairs);
  assert(hybrid_stats.phase1_cover_misses ==
         scan_stats.phase1_cover_misses);
  assert(hybrid_stats.phase1_best_cover_hits ==
         scan_stats.phase1_best_cover_hits);

  const auto sliding_sequences = make_sliding_sequences();
  const navigamer::HierarchyConfig sliding_hierarchy({20, 10, 4});
  navigamer::BioGeometryIndexBuilder sliding_scan_builder(
      sliding_hierarchy, phase1_scan_config());
  navigamer::BioGeometryIndexBuilder sliding_hybrid_builder(
      sliding_hierarchy, phase1_hybrid_config());
  sliding_scan_builder.build(sliding_sequences);
  sliding_hybrid_builder.build(sliding_sequences);

  assert(sliding_scan_builder.primary_layer(0).front()->get_center_sequence() ==
         sliding_sequences.front()->seq);
  assert(primary_edges(sliding_scan_builder) ==
         primary_edges(sliding_hybrid_builder));
  assert(leaf_links(sliding_scan_builder) ==
         leaf_links(sliding_hybrid_builder));
  const auto sliding_scan_stats = sliding_scan_builder.get_statistics();
  const auto sliding_hybrid_stats = sliding_hybrid_builder.get_statistics();
  assert(sliding_hybrid_stats.phase1_hint_checks > 0);
  assert(sliding_hybrid_stats.phase1_hint_hits > 0);
  assert(sliding_hybrid_stats.phase1_exact_distance_calls <
         sliding_scan_stats.phase1_exact_distance_calls);

  std::cout << "phase1 hybrid tests passed\n";
  return 0;
}
