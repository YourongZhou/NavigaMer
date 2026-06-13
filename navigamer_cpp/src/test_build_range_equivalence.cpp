#include "index_builder.hpp"
#include "search_engine.hpp"

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

std::string random_dna(size_t length, std::mt19937& gen) {
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> pick(0, 3);
  std::string out;
  out.reserve(length);
  for (size_t i = 0; i < length; ++i) out.push_back(bases[pick(gen)]);
  return out;
}

std::string mutate(std::string value, int edits, std::mt19937& gen) {
  std::uniform_int_distribution<size_t> pos(0, value.size() - 1);
  for (int i = 0; i < edits; ++i) {
    size_t p = pos(gen);
    value[p] = value[p] == 'A' ? 'C' : 'A';
  }
  return value;
}

std::vector<SequencePtr> make_sequences(size_t length) {
  std::mt19937 gen(1234);
  std::vector<SequencePtr> sequences;
  for (int cluster = 0; cluster < 12; ++cluster) {
    std::string center = random_dna(length, gen);
    for (int member = 0; member < 8; ++member) {
      sequences.push_back(std::make_shared<navigamer::BioSequence>(
          "s_" + std::to_string(cluster) + "_" + std::to_string(member),
          mutate(center, member, gen)));
    }
  }
  return sequences;
}

using LinkMap = std::map<std::string, std::set<std::string>>;

LinkMap primary_edges(const navigamer::BioGeometryIndexBuilder& builder) {
  LinkMap edges;
  for (int layer_idx = 0; layer_idx + 1 < builder.num_primary_layers(); ++layer_idx) {
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

std::set<std::string> hit_sequences(
    const navigamer::BioGeometryIndexBuilder& builder,
    const navigamer::BioSequence& query, int tau) {
  navigamer::BioGeometrySearchEngine engine(builder);
  auto [hits, stats] = engine.search_adaptive(query, tau);
  std::set<std::string> result;
  for (const auto& hit : hits) result.insert(hit->seq);
  return result;
}

navigamer::BuildRangeConfig full_config() {
  navigamer::BuildRangeConfig config;
  config.link_mode = navigamer::BuildRangeMode::Full;
  config.leaf_attach_mode = navigamer::BuildRangeMode::Full;
  return config;
}

navigamer::BuildRangeConfig indexed_config(navigamer::RangeCandidateMode mode) {
  navigamer::BuildRangeConfig config;
  config.range_join.candidate_mode = mode;
  config.range_join.qgram_q = 5;
  return config;
}

void assert_equivalent(
    const navigamer::BioGeometryIndexBuilder& full_builder,
    const navigamer::BioGeometryIndexBuilder& indexed_builder,
    const std::vector<SequencePtr>& sequences) {
  assert(primary_edges(full_builder) == primary_edges(indexed_builder));
  assert(leaf_links(full_builder) == leaf_links(indexed_builder));

  std::mt19937 gen(99);
  for (size_t i = 0; i < std::min<size_t>(25, sequences.size()); ++i) {
    navigamer::BioSequence query(
        "q_" + std::to_string(i), mutate(sequences[i]->seq, 2, gen));
    assert(hit_sequences(full_builder, query, 2) ==
           hit_sequences(indexed_builder, query, 2));
  }
}

}  // namespace

int main() {
  auto sequences = make_sequences(100);
  sequences.push_back(
      std::make_shared<navigamer::BioSequence>("ambiguous", "AACNNGTACN"));
  navigamer::HierarchyConfig hierarchy({20, 10, 4});

  navigamer::BioGeometryIndexBuilder full_builder(hierarchy, full_config());
  navigamer::BioGeometryIndexBuilder qgram_builder(
      hierarchy, indexed_config(navigamer::RangeCandidateMode::QGramOnly));
  navigamer::BioGeometryIndexBuilder hybrid_builder(
      hierarchy, indexed_config(navigamer::RangeCandidateMode::Hybrid));
  navigamer::BioGeometryIndexBuilder auto_builder(
      hierarchy, indexed_config(navigamer::RangeCandidateMode::Auto));
  full_builder.build(sequences);
  qgram_builder.build(sequences);
  hybrid_builder.build(sequences);
  auto_builder.build(sequences);

  assert_equivalent(full_builder, qgram_builder, sequences);
  assert_equivalent(full_builder, hybrid_builder, sequences);
  assert_equivalent(full_builder, auto_builder, sequences);

  auto full_stats = full_builder.get_statistics();
  auto indexed_stats = qgram_builder.get_statistics();
  assert(full_stats.phase2_exact_distance_calls ==
         full_stats.phase2_total_possible_pairs);
  assert(full_stats.leaf_exact_distance_calls ==
         full_stats.total_possible_leaf_pairs);
  assert(indexed_stats.phase2_candidate_pairs <=
         indexed_stats.phase2_total_possible_pairs);
  assert(indexed_stats.phase2_exact_distance_calls <=
         indexed_stats.phase2_total_possible_pairs);
  assert(indexed_stats.leaf_candidate_pairs <=
         indexed_stats.total_possible_leaf_pairs);
  assert(indexed_stats.leaf_exact_distance_calls <=
         indexed_stats.total_possible_leaf_pairs);
  assert(full_stats.phase2_edges_added == indexed_stats.phase2_edges_added);
  assert(full_stats.leaf_attachments_added ==
         indexed_stats.leaf_attachments_added);
  assert(indexed_stats.phase2_qgram_queries > 0);
  assert(indexed_stats.leaf_qgram_queries > 0);
  assert(indexed_stats.phase2_qgram_candidate_pairs >=
         indexed_stats.phase2_candidate_pairs);
  assert(indexed_stats.leaf_qgram_candidate_pairs >=
         indexed_stats.leaf_candidate_pairs);

  auto hybrid_stats = hybrid_builder.get_statistics();
  assert(hybrid_stats.phase2_hybrid_queries > 0);
  assert(hybrid_stats.leaf_hybrid_queries > 0);
  assert(hybrid_stats.phase2_exact_distance_calls <=
         hybrid_stats.phase2_candidate_pairs);
  assert(hybrid_stats.leaf_exact_distance_calls <=
         hybrid_stats.leaf_candidate_pairs);

  auto short_sequences = make_sequences(20);
  navigamer::BioGeometryIndexBuilder fallback_builder(
      navigamer::HierarchyConfig({10, 5}),
      indexed_config(navigamer::RangeCandidateMode::PigeonholeOnly));
  fallback_builder.build(short_sequences);
  auto fallback_stats = fallback_builder.get_statistics();
  assert(fallback_stats.phase2_full_scan_fallback_count > 0);
  assert(fallback_stats.leaf_full_scan_fallback_count > 0);

  std::cout << "build range equivalence tests passed\n";
  return 0;
}
