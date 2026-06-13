#include "index_builder.hpp"
#include "mbb_rect_index.hpp"
#include "search_engine.hpp"

#include <cassert>
#include <iostream>
#include <limits>
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

std::vector<SequencePtr> make_sequences() {
  std::mt19937 gen(777);
  std::vector<SequencePtr> sequences;
  for (int cluster = 0; cluster < 16; ++cluster) {
    std::string center = random_dna(60, gen);
    for (int member = 0; member < 10; ++member) {
      sequences.push_back(std::make_shared<navigamer::BioSequence>(
          "s_" + std::to_string(cluster) + "_" + std::to_string(member),
          mutate(center, member % 6, gen)));
    }
  }
  return sequences;
}

std::set<std::string> ids(const std::vector<SequencePtr>& hits) {
  std::set<std::string> out;
  for (const auto& hit : hits) out.insert(hit->id);
  return out;
}

navigamer::SearchConfig search_config(navigamer::MBBFilterMode mode) {
  navigamer::SearchConfig config;
  config.mbb_filter_mode = mode;
  return config;
}

void assert_scan_rect_and_brute_force_equivalent(
    navigamer::BioGeometryIndexBuilder& builder,
    const std::vector<SequencePtr>& sequences,
    bool expect_rect_query,
    bool expect_fallback) {
  navigamer::BioGeometrySearchEngine scan_engine(
      builder, search_config(navigamer::MBBFilterMode::Scan));
  navigamer::BioGeometrySearchEngine rect_engine(
      builder, search_config(navigamer::MBBFilterMode::RectIndex));
  std::vector<SequencePtr> unique_sequences;
  for (const auto& entry : builder.unique_sequences) {
    unique_sequences.push_back(entry.second);
  }

  std::mt19937 gen(999);
  size_t rect_queries = 0;
  size_t rect_fallbacks = 0;
  size_t scan_checks = 0;
  for (size_t i = 0; i < 30; ++i) {
    navigamer::BioSequence query(
        "q_" + std::to_string(i), mutate(sequences[i]->seq, 2, gen));
    auto [scan_hits, scan_stats] = scan_engine.search_adaptive(query, 2);
    auto [rect_hits, rect_stats] = rect_engine.search_adaptive(query, 2);
    auto [bf_hits, bf_stats] =
        rect_engine.search_brute_force(query, 2, unique_sequences);
    (void)bf_stats;

    assert(ids(scan_hits) == ids(rect_hits));
    assert(ids(rect_hits) == ids(bf_hits));
    assert(scan_stats.mbb_rect_index_queries == 0);
    assert(scan_stats.mbb_rect_fallback_count == 0);
    assert(scan_stats.mbb_scan_child_checks >= scan_stats.mbb_surviving_child_count);
    assert(rect_stats.center_distance_calls_after_mbb <=
           rect_stats.mbb_surviving_child_count);

    rect_queries += rect_stats.mbb_rect_index_queries;
    rect_fallbacks += rect_stats.mbb_rect_fallback_count;
    scan_checks += scan_stats.mbb_scan_child_checks;
  }
  assert(scan_checks > 0);
  assert((rect_queries > 0) == expect_rect_query);
  assert((rect_fallbacks > 0) == expect_fallback);
}

}  // namespace

int main() {
  auto sequences = make_sequences();
  navigamer::HierarchyConfig hierarchy({24, 12, 5});

  navigamer::BuildRangeConfig indexed_config;
  indexed_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder indexed_builder(hierarchy, indexed_config);
  indexed_builder.build(sequences);
  assert_scan_rect_and_brute_force_equivalent(
      indexed_builder, sequences, true, false);

  navigamer::BuildRangeConfig low_fanout_config;
  low_fanout_config.min_rect_index_fanout = std::numeric_limits<size_t>::max();
  navigamer::BioGeometryIndexBuilder low_fanout_builder(hierarchy, low_fanout_config);
  low_fanout_builder.build(sequences);
  assert_scan_rect_and_brute_force_equivalent(
      low_fanout_builder, sequences, false, true);

  navigamer::BioGeometryIndexBuilder missing_builder(hierarchy, indexed_config);
  missing_builder.build(sequences);
  for (int layer = 0; layer < missing_builder.finest_primary_layer_index(); ++layer) {
    for (const auto& parent : missing_builder.primary_layer(layer)) {
      parent->mbb_rect_index.reset();
    }
  }
  assert_scan_rect_and_brute_force_equivalent(
      missing_builder, sequences, false, true);

  navigamer::BioGeometryIndexBuilder mismatch_builder(hierarchy, indexed_config);
  mismatch_builder.build(sequences);
  for (int layer = 0; layer < mismatch_builder.finest_primary_layer_index(); ++layer) {
    for (const auto& parent : mismatch_builder.primary_layer(layer)) {
      if (!parent->child_nodes.empty()) {
        parent->mbb_rect_index = std::make_shared<navigamer::MBBRectIndex>();
        std::vector<int> lo(parent->beacons.size() + 1, 0);
        std::vector<int> hi(parent->beacons.size() + 1, 1000);
        parent->mbb_rect_index->build({{0, std::move(lo), std::move(hi)}});
      }
    }
  }
  assert_scan_rect_and_brute_force_equivalent(
      mismatch_builder, sequences, false, true);

  std::cout << "MBB filter equivalence tests passed\n";
  return 0;
}
