#include "index_builder.hpp"
#include "search_engine.hpp"

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

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

std::vector<SequencePtr> make_mutated_window_dataset() {
  std::mt19937 gen(5678);
  std::vector<SequencePtr> sequences;
  for (int cluster = 0; cluster < 18; ++cluster) {
    std::string center = random_dna(90, gen);
    for (int member = 0; member < 9; ++member) {
      sequences.push_back(std::make_shared<navigamer::BioSequence>(
          "w_" + std::to_string(cluster) + "_" + std::to_string(member),
          mutate(center, member % 7, gen)));
    }
  }
  return sequences;
}

using LinkMap = std::map<std::string, std::set<std::string>>;
using OrderedLinkMap = std::map<std::string, std::vector<std::string>>;

LinkMap primary_edges(const navigamer::BioGeometryIndexBuilder& builder) {
  LinkMap edges;
  const auto& view = builder.search_graph_view();
  for (int layer_idx = 0; layer_idx + 1 < builder.num_primary_layers(); ++layer_idx) {
    const size_t layer = static_cast<size_t>(layer_idx);
    for (uint32_t parent_id = view.layer_begin[layer];
         parent_id < view.layer_end[layer]; ++parent_id) {
      const auto& parent = view.node_records[parent_id];
      auto& children =
          edges[view.sequences[parent.center_sequence_id].seq];
      for (uint32_t offset = 0; offset < parent.child_count; ++offset) {
        const auto& child =
            view.node_records[view.child_ids[parent.child_begin + offset]];
        children.insert(view.sequences[child.center_sequence_id].seq);
      }
    }
  }
  return edges;
}

OrderedLinkMap ordered_primary_edges(
    const navigamer::BioGeometryIndexBuilder& builder) {
  OrderedLinkMap edges;
  const auto& view = builder.search_graph_view();
  for (int layer_idx = 0; layer_idx + 1 < builder.num_primary_layers();
       ++layer_idx) {
    const size_t layer = static_cast<size_t>(layer_idx);
    for (uint32_t parent_id = view.layer_begin[layer];
         parent_id < view.layer_end[layer]; ++parent_id) {
      const auto& parent = view.node_records[parent_id];
      std::string key =
          std::to_string(layer_idx) + ":" +
          view.sequences[parent.center_sequence_id].seq;
      auto& children = edges[key];
      for (uint32_t offset = 0; offset < parent.child_count; ++offset) {
        const auto& child =
            view.node_records[view.child_ids[parent.child_begin + offset]];
        children.push_back(view.sequences[child.center_sequence_id].seq);
      }
    }
  }
  return edges;
}

LinkMap leaf_links(const navigamer::BioGeometryIndexBuilder& builder) {
  LinkMap links;
  const auto& view = builder.search_graph_view();
  const size_t layer =
      static_cast<size_t>(builder.finest_primary_layer_index());
  for (uint32_t world_id = view.layer_begin[layer];
       world_id < view.layer_end[layer]; ++world_id) {
    const auto& world = view.node_records[world_id];
    auto& leaves = links[view.sequences[world.center_sequence_id].seq];
    for (uint32_t offset = 0; offset < world.leaf_count; ++offset) {
      leaves.insert(
          view.sequences[view.leaf_ids[world.leaf_begin + offset]].seq);
    }
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

navigamer::BuildRangeConfig selective_auto_config() {
  auto config = indexed_config(navigamer::RangeCandidateMode::Auto);
  config.range_join.auto_pigeonhole_max_candidates = 0;
  config.range_join.auto_pigeonhole_max_ratio = 0.0;
  config.range_join.auto_hybrid_on_large_candidates = true;
  return config;
}

navigamer::BuildRangeConfig indexed_leaf_direction_config(
    navigamer::LeafAttachDirection direction) {
  auto config = indexed_config(navigamer::RangeCandidateMode::QGramOnly);
  config.leaf_attach_direction = direction;
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

OrderedLinkMap build_and_collect_edges(
    const std::vector<SequencePtr>& sequences,
    const navigamer::BuildRangeConfig& config) {
  navigamer::HierarchyConfig hierarchy({24, 12, 5});
  navigamer::BioGeometryIndexBuilder builder(hierarchy, config);
  builder.build(sequences);
  return ordered_primary_edges(builder);
}

struct Phase3BuildResult {
  std::vector<std::string> signature;
  navigamer::BioGeometryIndexBuilder::Statistics stats;
};

Phase3BuildResult build_and_collect_phase3(
    const std::vector<SequencePtr>& sequences,
    const navigamer::BuildRangeConfig& config) {
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({24, 12, 5}), config);
  builder.build(sequences);

  Phase3BuildResult result;
  result.stats = builder.get_statistics();
  const auto& view = builder.search_graph_view();
  for (int layer_idx = 0; layer_idx < builder.num_primary_layers();
       ++layer_idx) {
    const size_t layer = static_cast<size_t>(layer_idx);
    for (uint32_t parent_id = view.layer_begin[layer];
         parent_id < view.layer_end[layer]; ++parent_id) {
      const auto& parent = view.node_records[parent_id];
      std::ostringstream row;
      row << layer_idx << ':'
          << view.sequences[parent.center_sequence_id].seq << "|b=";
      for (uint32_t offset = 0; offset < parent.beacon_count; ++offset) {
        row << view.sequences[
                   view.beacon_ids[parent.beacon_begin + offset]]
                   .seq
            << ';';
      }
      row << "|c=";
      for (uint32_t offset = 0; offset < parent.child_count; ++offset) {
        const auto& child =
            view.node_records[view.child_ids[parent.child_begin + offset]];
        row << view.sequences[child.center_sequence_id].seq << ';';
      }
      row << "|m=";
      for (uint32_t child = 0; child < parent.child_count; ++child) {
        for (uint32_t dim = 0; dim < parent.beacon_count; ++dim) {
          const size_t flat = parent.mbb_begin +
                              static_cast<size_t>(dim) *
                                  parent.child_count +
                              child;
          row << view.mbb_lo[flat] << ',' << view.mbb_hi[flat] << ';';
        }
        row << '/';
      }
      row << "|r="
          << (parent.child_count >= config.min_rect_index_fanout ? 1 : 0);
      result.signature.push_back(row.str());
    }
  }
  return result;
}

void test_indexed_parallel_phase2_matches_single_thread() {
  auto sequences = make_mutated_window_dataset();

  navigamer::BuildRangeConfig config;
  config.link_mode = navigamer::BuildRangeMode::Indexed;
  config.leaf_attach_mode = navigamer::BuildRangeMode::Indexed;
  config.leaf_attach_direction = navigamer::LeafAttachDirection::WorldToSeq;
  config.range_join.candidate_mode = navigamer::RangeCandidateMode::Auto;

  const int original_threads = omp_get_max_threads();
  omp_set_num_threads(1);
  auto single = build_and_collect_edges(sequences, config);

  omp_set_num_threads(4);
  auto parallel = build_and_collect_edges(sequences, config);

  omp_set_num_threads(original_threads);
  assert(single == parallel);
}

void test_parallel_phase3_matches_single_thread() {
  auto sequences = make_mutated_window_dataset();

  navigamer::BuildRangeConfig config;
  config.min_rect_index_fanout = 1;

  const int original_threads = omp_get_max_threads();
  omp_set_num_threads(1);
  auto single = build_and_collect_phase3(sequences, config);

  omp_set_num_threads(4);
  auto parallel = build_and_collect_phase3(sequences, config);

  omp_set_num_threads(original_threads);
  assert(single.signature == parallel.signature);
  assert(single.stats.phase3_parallel_threads == 1);
  assert(parallel.stats.phase3_parallel_threads >= 2);
}

void test_phase2_qgram_postfilter_matches_full_construction() {
  auto sequences = make_sequences(100);
  navigamer::HierarchyConfig hierarchy({20, 10, 4});

  navigamer::BuildRangeConfig config;
  config.link_mode = navigamer::BuildRangeMode::Indexed;
  config.leaf_attach_mode = navigamer::BuildRangeMode::Full;
  config.range_join.candidate_mode = navigamer::RangeCandidateMode::FullScan;
  config.range_join.qgram_q = 3;
  config.phase2_qgram_postfilter = true;

  navigamer::BioGeometryIndexBuilder full_builder(hierarchy, full_config());
  navigamer::BioGeometryIndexBuilder filtered_builder(hierarchy, config);
  full_builder.build(sequences);
  filtered_builder.build(sequences);

  assert_equivalent(full_builder, filtered_builder, sequences);
  assert(filtered_builder.get_statistics().phase2_qgram_pruned_by_l1 > 0);
}

void test_leaf_qgram_postfilter_matches_unfiltered_construction() {
  auto sequences = make_mutated_window_dataset();
  navigamer::HierarchyConfig hierarchy({24, 12, 5});

  navigamer::BuildRangeConfig unfiltered;
  unfiltered.link_mode = navigamer::BuildRangeMode::Indexed;
  unfiltered.leaf_attach_mode = navigamer::BuildRangeMode::Indexed;
  unfiltered.leaf_attach_direction = navigamer::LeafAttachDirection::WorldToSeq;
  unfiltered.range_join.candidate_mode = navigamer::RangeCandidateMode::Auto;
  unfiltered.range_join.qgram_q = 5;
  unfiltered.leaf_qgram_postfilter = false;

  auto filtered = unfiltered;
  filtered.leaf_qgram_postfilter = true;

  navigamer::BioGeometryIndexBuilder unfiltered_builder(
      hierarchy, unfiltered);
  navigamer::BioGeometryIndexBuilder filtered_builder(
      hierarchy, filtered);
  unfiltered_builder.build(sequences);
  filtered_builder.build(sequences);

  assert(primary_edges(unfiltered_builder) == primary_edges(filtered_builder));
  assert(leaf_links(unfiltered_builder) == leaf_links(filtered_builder));
  assert(filtered_builder.get_statistics().leaf_attachments_added ==
         unfiltered_builder.get_statistics().leaf_attachments_added);
  assert(filtered_builder.get_statistics().leaf_qgram_pruned_by_l1 > 0);
  assert(filtered_builder.get_statistics().leaf_exact_distance_calls <
         unfiltered_builder.get_statistics().leaf_exact_distance_calls);
}

}  // namespace

int main() {
  test_indexed_parallel_phase2_matches_single_thread();
  test_parallel_phase3_matches_single_thread();
  test_phase2_qgram_postfilter_matches_full_construction();
  test_leaf_qgram_postfilter_matches_unfiltered_construction();

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
      hierarchy, selective_auto_config());
  navigamer::BioGeometryIndexBuilder seq_to_world_builder(
      hierarchy,
      indexed_leaf_direction_config(navigamer::LeafAttachDirection::SeqToWorld));
  navigamer::BioGeometryIndexBuilder world_to_seq_builder(
      hierarchy,
      indexed_leaf_direction_config(navigamer::LeafAttachDirection::WorldToSeq));
  full_builder.build(sequences);
  qgram_builder.build(sequences);
  hybrid_builder.build(sequences);
  auto_builder.build(sequences);
  seq_to_world_builder.build(sequences);
  world_to_seq_builder.build(sequences);

  assert_equivalent(full_builder, qgram_builder, sequences);
  assert_equivalent(full_builder, hybrid_builder, sequences);
  assert_equivalent(full_builder, auto_builder, sequences);
  assert_equivalent(full_builder, seq_to_world_builder, sequences);
  assert_equivalent(full_builder, world_to_seq_builder, sequences);

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

  auto auto_stats = auto_builder.get_statistics();
  assert(auto_stats.phase2_auto_pigeonhole_rejected_large_candidates > 0);
  assert(auto_stats.phase2_auto_qgram_invoked > 0);
  assert(auto_stats.phase2_pigeonhole_early_abort_count > 0);
  assert(auto_stats.phase2_auto_hybrid_invoked == 0);
  assert(auto_stats.phase2_auto_final_candidate_pairs ==
         auto_stats.phase2_candidate_pairs);
  assert(auto_stats.leaf_auto_pigeonhole_rejected_large_candidates > 0);
  assert(auto_stats.leaf_auto_qgram_invoked > 0);
  assert(auto_stats.leaf_pigeonhole_early_abort_count > 0);
  assert(auto_stats.leaf_auto_hybrid_invoked == 0);
  assert(auto_stats.leaf_auto_final_candidate_pairs ==
         auto_stats.leaf_candidate_pairs);
  assert(auto_stats.leaf_seed_candidate_pairs_before_length_filter > 0);
  assert(auto_stats.leaf_range_final_candidate_pairs ==
         auto_stats.leaf_candidate_pairs);

  auto seq_to_world_stats = seq_to_world_builder.get_statistics();
  auto world_to_seq_stats = world_to_seq_builder.get_statistics();
  assert(seq_to_world_stats.leaf_attach_direction_used ==
         navigamer::LeafAttachDirection::SeqToWorld);
  assert(world_to_seq_stats.leaf_attach_direction_used ==
         navigamer::LeafAttachDirection::WorldToSeq);
  assert(seq_to_world_stats.leaf_attachments_added ==
         world_to_seq_stats.leaf_attachments_added);

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
