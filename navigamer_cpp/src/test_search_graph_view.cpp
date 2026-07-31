#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

std::vector<std::shared_ptr<navigamer::BioSequence>> build_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a0", "ACGTACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("a1", "ACGTACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("a2", "ACGTACGTACGTACGTACAA"),
      std::make_shared<navigamer::BioSequence>("b0", "TTTTACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("b1", "TTTTACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("c0", "GGGGACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("c1", "GGGGACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("d0", "CCCCACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("d1", "CCCCACGTACGTACGTACGA"),
  };
}

void assert_view_equivalent_to_original() {
  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 10, 3}), build_config);
  builder.build(build_sequences());
  assert(builder.validate_integer_ids());
  assert(builder.validate_search_graph_view());

  const auto& view = builder.search_graph_view();
  assert(view.node_records.size() == builder.num_world_nodes());
  assert(view.sequences.size() == builder.num_sequences());
  assert(view.layer_begin.size() ==
         static_cast<size_t>(builder.num_primary_layers()));
  assert(view.layer_end.size() ==
         static_cast<size_t>(builder.num_primary_layers()));
  for (size_t sequence_id = 0; sequence_id < view.sequences.size();
       ++sequence_id) {
    assert(view.sequences.records[sequence_id].sequence_id == sequence_id);
  }
  for (size_t layer = 0; layer < view.layer_begin.size(); ++layer) {
    assert(view.layer_begin[layer] <= view.layer_end[layer]);
    for (uint32_t node_id = view.layer_begin[layer];
         node_id < view.layer_end[layer]; ++node_id) {
      const auto& record = view.node_records[node_id];
      const uint32_t link_count = view.link_count(node_id);
      const uint32_t beacon_count = view.beacon_count(node_id);
      assert(record.center_sequence_id < view.sequences.size());
      if (layer + 1 == view.layer_begin.size()) {
        assert(record.leaf_begin() + link_count <=
               view.leaf_ids.size());
        assert(record.leaf_begin() +
                   static_cast<size_t>(link_count) * beacon_count <=
               view.leaf_beacon_dists.size());
      } else {
        assert(record.child_begin() + link_count <=
               (view.child_ids_are_delta16(node_id)
                    ? view.child_id_deltas16.size()
                    : view.child_ids.size()));
        assert(beacon_count <= link_count);
        assert(record.mbb_begin +
                   static_cast<size_t>(link_count) * beacon_count <=
               view.child_beacon_dists.size());
      }
    }
  }
}

std::set<navigamer::LeafId> ids(
    const navigamer::SearchResult& hits) {
  return {hits.begin(), hits.end()};
}

void assert_flat_search_matches_original() {
  assert(navigamer::parse_graph_view_mode("original") ==
         navigamer::GraphViewMode::Original);
  assert(navigamer::parse_graph_view_mode("flat") ==
         navigamer::GraphViewMode::Flat);
  assert(std::string(navigamer::graph_view_mode_name(
             navigamer::GraphViewMode::Original)) == "original");
  assert(std::string(navigamer::graph_view_mode_name(
             navigamer::GraphViewMode::Flat)) == "flat");

  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 10, 3}), build_config);
  builder.build(build_sequences());

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTACGTACGTACAA"),
      navigamer::BioSequence("q2", "TTTTACGTACGTACGTACGT"),
      navigamer::BioSequence("q3", "AAAAGGGGAAAAGGGGAAAA"),
  };

  for (navigamer::MBBFilterMode mbb_mode :
       {navigamer::MBBFilterMode::Scan, navigamer::MBBFilterMode::RectIndex}) {
    for (bool qgram_enabled : {false, true}) {
      for (navigamer::VisitedMode visited_mode :
           {navigamer::VisitedMode::StringSet, navigamer::VisitedMode::Epoch}) {
        navigamer::SearchConfig original_config;
        original_config.mbb_filter_mode = mbb_mode;
        original_config.search_qgram_prefilter = qgram_enabled;
        original_config.search_qgram_q = 3;
        original_config.visited_mode = visited_mode;
        original_config.graph_view_mode = navigamer::GraphViewMode::Original;
        original_config.simd_mode = navigamer::SimdMode::Scalar;

        navigamer::BioGeometrySearchEngine original_engine(builder, original_config);

        for (navigamer::SimdMode simd_mode :
             {navigamer::SimdMode::Scalar,
              navigamer::SimdMode::Auto,
              navigamer::SimdMode::AVX2}) {
          navigamer::SearchConfig flat_config = original_config;
          flat_config.graph_view_mode = navigamer::GraphViewMode::Flat;
          flat_config.simd_mode = simd_mode;
          navigamer::BioGeometrySearchEngine flat_engine(builder, flat_config);

          for (const auto& query : queries) {
            auto [original_hits, original_stats] =
                original_engine.search_adaptive(query, 2);
            auto [flat_hits, flat_stats] = flat_engine.search_adaptive(query, 2);
            assert(ids(original_hits) == ids(flat_hits));
            assert(original_stats.result_count == flat_stats.result_count);
          }
        }
      }
    }
  }
}

void assert_max_byte_distance_is_recall_safe() {
  auto sequences =
      std::vector<std::shared_ptr<navigamer::BioSequence>>{
          std::make_shared<navigamer::BioSequence>(
              "a", std::string(255, 'A')),
          std::make_shared<navigamer::BioSequence>(
              "c", std::string(255, 'C')),
          std::make_shared<navigamer::BioSequence>(
              "t", std::string(255, 'T')),
      };
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({255, 254}));
  builder.build(std::move(sequences));
  const auto& view = builder.search_graph_view();
  assert(std::find(
             view.child_beacon_dists.begin(),
             view.child_beacon_dists.end(),
             static_cast<uint8_t>(255)) !=
         view.child_beacon_dists.end());

  for (navigamer::SimdMode simd_mode :
       {navigamer::SimdMode::Scalar,
        navigamer::SimdMode::Auto,
        navigamer::SimdMode::AVX2}) {
    navigamer::SearchConfig config;
    config.simd_mode = simd_mode;
    navigamer::BioGeometrySearchEngine engine(builder, config);
    for (char base : {'A', 'C', 'T'}) {
      navigamer::BioSequence query(
          "q", std::string(255, base));
      auto [adaptive, adaptive_stats] =
          engine.search_adaptive(query, 0);
      auto [brute_force, brute_stats] =
          engine.search_brute_force(query, 0);
      (void)adaptive_stats;
      (void)brute_stats;
      assert(ids(adaptive) == ids(brute_force));
      assert(adaptive.size() == 1);
    }
  }
}

void assert_all_beacon_id_encodings_are_exact() {
  navigamer::SearchGraphView view;
  view.node_records.resize(4);
  view.beacon_deltas8 = {-120};
  view.beacon_deltas16 = {30000};
  view.beacon_ids32 = {4000000000U};

  auto& delta8 = view.node_records[0];
  delta8.center_sequence_id = 200;
  delta8.beacon_begin = 0;
  view.set_node_counts(
      0, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  assert(view.beacon_sequence_id(0, 0) == 80);

  auto& delta16 = view.node_records[1];
  delta16.center_sequence_id = 1000;
  delta16.beacon_begin = 0;
  view.set_node_counts(
      1, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::Delta16);
  assert(view.beacon_sequence_id(1, 0) == 31000);

  auto& absolute32 = view.node_records[2];
  absolute32.center_sequence_id = 0;
  absolute32.beacon_begin = 0;
  view.set_node_counts(
      2, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::Absolute32);
  assert(view.beacon_sequence_id(2, 0) == 4000000000U);

  auto& implicit = view.node_records[3];
  implicit.center_sequence_id = 123456789U;
  implicit.beacon_begin = 0;
  view.set_node_counts(
      3, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  assert(view.beacon_sequence_id(3, 0) == 123456789U);
}

void assert_node_count_overflow_is_exact() {
  navigamer::SearchGraphView view;
  view.node_records.resize(2);
  view.set_node_counts(
      0, navigamer::WorldNodeRecord::LINK_COUNT_MASK,
      navigamer::WorldNodeRecord::COUNT_OVERFLOW_CODE - 1,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  assert(!view.node_records[0].counts_overflow());
  assert(view.link_count(0) ==
         navigamer::WorldNodeRecord::LINK_COUNT_MASK);
  assert(view.beacon_count(0) ==
         navigamer::WorldNodeRecord::COUNT_OVERFLOW_CODE - 1);

  const uint32_t large_link_count =
      navigamer::WorldNodeRecord::LINK_COUNT_MASK + 1;
  const uint32_t large_beacon_count =
      navigamer::WorldNodeRecord::COUNT_OVERFLOW_CODE + 17;
  view.set_node_counts(
      1, large_link_count, large_beacon_count,
      navigamer::WorldNodeRecord::BeaconStorage::Absolute32);
  assert(view.node_records[1].counts_overflow());
  assert(view.node_count_overflows.size() == 1);
  assert(view.link_count(1) == large_link_count);
  assert(view.beacon_count(1) == large_beacon_count);
  assert(view.node_records[1].beacon_storage() ==
         navigamer::WorldNodeRecord::BeaconStorage::Absolute32);
}

void assert_child_id_encodings_are_exact() {
  navigamer::SearchGraphView view;
  view.node_records.resize(2);
  view.child_delta16_node_bits.assign(1, 0);
  view.child_id_deltas16 = {0, UINT16_MAX};
  view.child_ids = {70000, UINT32_MAX - 1};

  view.node_records[0].link_begin = 0;
  view.set_node_counts(
      0, 2, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  view.set_child_ids_delta16(0);
  assert(view.child_id(0, 0) == 1);
  assert(view.child_id(0, 1) == 65536);

  view.node_records[1].link_begin = 0;
  view.set_node_counts(
      1, 2, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  assert(!view.child_ids_are_delta16(1));
  assert(view.child_id(1, 0) == 70000);
  assert(view.child_id(1, 1) == UINT32_MAX - 1);
  assert(view.edge_count() == 4);
}

}  // namespace

int main() {
  assert_view_equivalent_to_original();
  assert_flat_search_matches_original();
  assert_max_byte_distance_is_recall_safe();
  assert_all_beacon_id_encodings_are_exact();
  assert_node_count_overflow_is_exact();
  assert_child_id_encodings_are_exact();
  std::cout << "search graph view tests passed\n";
  return 0;
}
