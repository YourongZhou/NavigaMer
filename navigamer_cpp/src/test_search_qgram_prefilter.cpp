#include "index_builder.hpp"
#include "search_engine.hpp"

#include <cassert>
#include <iostream>
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
  std::mt19937 gen(321);
  std::vector<SequencePtr> sequences;
  for (int cluster = 0; cluster < 30; ++cluster) {
    std::string center = random_dna(250, gen);
    for (int member = 0; member < 8; ++member) {
      sequences.push_back(std::make_shared<navigamer::BioSequence>(
          "s_" + std::to_string(cluster) + "_" + std::to_string(member),
          mutate(center, member % 4, gen)));
    }
  }
  return sequences;
}

std::set<std::string> ids(const std::vector<SequencePtr>& hits) {
  std::set<std::string> out;
  for (const auto& hit : hits) out.insert(hit->id);
  return out;
}

navigamer::SearchConfig config(
    navigamer::MBBFilterMode mode, bool qgram_enabled, int q = 5) {
  navigamer::SearchConfig out;
  out.mbb_filter_mode = mode;
  out.search_qgram_prefilter = qgram_enabled;
  out.search_qgram_q = q;
  return out;
}

void disable_mbb_pruning(navigamer::BioGeometryIndexBuilder& builder) {
  for (int layer = 0; layer < builder.finest_primary_layer_index(); ++layer) {
    for (const auto& parent : builder.primary_layer(layer)) {
      parent->child_nodes = builder.primary_layer(layer + 1);
      parent->beacons.clear();
      parent->child_beacon_mbbs.clear();
      parent->mbb_rect_index.reset();
    }
  }
}

void assert_four_modes_equal(
    navigamer::BioGeometryIndexBuilder& builder,
    const navigamer::BioSequence& query,
    int tolerance,
    bool expect_unsafe_fallback,
    bool* saw_pruning) {
  constexpr int search_q = 3;
  navigamer::BioGeometrySearchEngine scan_off(
      builder, config(navigamer::MBBFilterMode::Scan, false, search_q));
  navigamer::BioGeometrySearchEngine scan_on(
      builder, config(navigamer::MBBFilterMode::Scan, true, search_q));
  navigamer::BioGeometrySearchEngine rect_off(
      builder, config(navigamer::MBBFilterMode::RectIndex, false, search_q));
  navigamer::BioGeometrySearchEngine rect_on(
      builder, config(navigamer::MBBFilterMode::RectIndex, true, search_q));

  auto [scan_off_hits, scan_off_stats] =
      scan_off.search_adaptive(query, tolerance);
  auto [scan_on_hits, scan_on_stats] =
      scan_on.search_adaptive(query, tolerance);
  auto [rect_off_hits, rect_off_stats] =
      rect_off.search_adaptive(query, tolerance);
  auto [rect_on_hits, rect_on_stats] =
      rect_on.search_adaptive(query, tolerance);

  assert(ids(scan_off_hits) == ids(scan_on_hits));
  assert(ids(scan_on_hits) == ids(rect_off_hits));
  assert(ids(rect_off_hits) == ids(rect_on_hits));

  assert(!scan_off_stats.search_qgram_prefilter_enabled);
  assert(scan_off_stats.search_qgram_checks == 0);
  assert(scan_off_stats.center_distance_calls_before_qgram ==
         scan_off_stats.center_distance_calls_after_qgram);
  assert(scan_off_stats.center_distance_calls_after_mbb ==
         scan_off_stats.center_distance_calls_before_qgram);

  for (const auto* stats : {&scan_on_stats, &rect_on_stats}) {
    assert(stats->search_qgram_prefilter_enabled);
    assert(stats->search_qgram_q == search_q);
    assert(stats->search_qgram_signature_build_count > 0);
    assert(stats->center_distance_calls_after_qgram <=
           stats->center_distance_calls_before_qgram);
    assert(stats->center_distance_calls_after_mbb ==
           stats->center_distance_calls_before_qgram);
    assert(stats->search_qgram_passed_children +
               stats->search_qgram_pruned_children ==
           stats->search_qgram_checks);
    if (expect_unsafe_fallback) {
      assert(stats->search_qgram_signature_missing_count > 0);
      assert(stats->search_qgram_checks == 0);
    }
    if (stats->search_qgram_pruned_children > 0) *saw_pruning = true;
  }
}

}  // namespace

int main() {
  auto sequences = make_sequences();
  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({28, 14, 5}), build_config);
  builder.build(sequences);
  disable_mbb_pruning(builder);

  bool saw_pruning = false;
  for (size_t i = 0; i < 12; ++i) {
    navigamer::BioSequence query("q_" + std::to_string(i), sequences[i]->seq);
    assert_four_modes_equal(builder, query, 2, false, &saw_pruning);
  }
  assert(saw_pruning);

  navigamer::BioSequence ambiguous("ambiguous", sequences[0]->seq);
  ambiguous.seq[10] = 'N';
  assert_four_modes_equal(builder, ambiguous, 2, true, &saw_pruning);

  navigamer::BioGeometrySearchEngine disabled_q_engine(
      builder, config(navigamer::MBBFilterMode::Scan, true, 0));
  auto [disabled_q_hits, disabled_q_stats] =
      disabled_q_engine.search_adaptive(ambiguous, 2);
  assert(!disabled_q_hits.empty());
  assert(!disabled_q_stats.search_qgram_prefilter_enabled);
  assert(disabled_q_stats.search_qgram_checks == 0);
  assert(disabled_q_stats.center_distance_calls_before_qgram ==
         disabled_q_stats.center_distance_calls_after_qgram);

  std::cout << "search qgram prefilter tests passed\n";
  return 0;
}
