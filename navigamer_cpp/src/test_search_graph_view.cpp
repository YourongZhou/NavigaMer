#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
  assert(view.beacon_begins_valid(view.layer_begin.back()));
  for (size_t sequence_id = 0; sequence_id < view.sequences.size();
       ++sequence_id) {
    assert(view.sequences.records[sequence_id].sequence_id == sequence_id);
  }
  assert(view.child_base_ids_valid(view.layer_begin.back()));
  assert(view.implicit_contiguous_child_ranges);
  for (size_t layer = 0; layer < view.layer_begin.size(); ++layer) {
    assert(view.layer_begin[layer] <= view.layer_end[layer]);
    for (uint32_t node_id = view.layer_begin[layer];
         node_id < view.layer_end[layer]; ++node_id) {
      const auto& record = view.node_records[node_id];
      const uint32_t link_count = view.link_count(node_id);
      const uint32_t beacon_count = view.beacon_count(node_id);
      assert(view.center_sequence_id(node_id) < view.sequences.size());
      if (layer + 1 == view.layer_begin.size()) {
        assert(view.leaf_mbb_bits(node_id) >= 1);
        assert(view.leaf_mbb_bits(node_id) <= 8);
        assert(view.leaf_mbb_range_valid(node_id));
      } else {
        assert(beacon_count <= 3);
        assert(view.child_mbb_bits(node_id) >= 1);
        assert(view.child_mbb_bits(node_id) <= 8);
        const uint32_t expected_bin_width =
            layer + 2 == view.layer_begin.size()
                ? navigamer::SearchGraphView::
                      FINE_CHILD_MBB_BIN_WIDTH
                : navigamer::SearchGraphView::
                      COARSE_CHILD_MBB_BIN_WIDTH;
        assert(view.child_mbb_bin_width(node_id) ==
               expected_bin_width);
        if (view.child_ids_are_base_delta8(node_id)) {
          assert(view.child_begin(node_id, record) +
                     view.child_base_byte_count() +
                     link_count <=
                 view.child_id_base_deltas8.size());
        } else if (view.child_ids_are_packed_delta(node_id)) {
          assert(view.child_begin(node_id, record) <=
                 view.child_id_base_deltas8.size());
          assert(view.packed_child_byte_count(node_id) <=
                 view.child_id_base_deltas8.size() -
                     view.child_begin(node_id, record));
        } else if (view.child_ids_are_delta16(node_id)) {
          assert(view.child_begin(node_id, record) + link_count <=
                 view.child_id_deltas16.size());
        } else {
          assert(view.child_begin(node_id, record) + link_count <=
                 view.child_ids.size());
        }
        assert(beacon_count <= link_count);
        assert(view.child_mbb_range_valid(node_id));
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
  bool found_max_distance = false;
  for (uint32_t node_id = view.layer_begin.front();
       node_id < view.layer_begin.back(); ++node_id) {
    const size_t cells =
        static_cast<size_t>(view.child_count(node_id)) *
        view.beacon_count(node_id);
    for (size_t cell = 0; cell < cells; ++cell) {
      found_max_distance =
          found_max_distance ||
          view.child_beacon_distance(node_id, cell) >=
              255 - navigamer::SearchGraphView::child_mbb_quantization_error(
                        view.child_mbb_bin_width(node_id));
    }
  }
  assert(found_max_distance);

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
  view.layer_begin = {0, 1, 2, 3};
  view.layer_end = {1, 2, 3, 4};
  view.initialize_center_sequence_ids({0, 0, 0, 0});
  view.beacon_delta_bits = 16;
  view.beacon_id_bytes = {
      0x88, 0x60, 0xea, 0x00, 0x28, 0x6b, 0xee};
  view.initialize_beacon_begins(3, 3, 3);
  view.set_beacon_begin(0, 0, 0);
  view.set_beacon_begin(1, 1, 0);
  view.set_beacon_begin(2, 3, 0);

  view.set_center_sequence_id(0, 0, 200);
  view.set_node_counts(
      0, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  assert(view.beacon_sequence_id(0, 0) == 80);

  view.set_center_sequence_id(1, 1, 1000);
  view.set_node_counts(
      1, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
  assert(view.beacon_sequence_id(1, 0) == 31000);

  view.set_center_sequence_id(2, 2, 0);
  view.set_node_counts(
      2, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::Absolute32);
  assert(view.beacon_sequence_id(2, 0) == 4000000000U);

  view.set_center_sequence_id(3, 3, 123456789U);
  view.set_node_counts(
      3, 0, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  assert(view.beacon_sequence_id(3, 0) == 123456789U);

  for (uint32_t bits = 1; bits <= 32; ++bits) {
    navigamer::SearchGraphView packed_view;
    packed_view.node_records.resize(1);
    packed_view.layer_begin = {0};
    packed_view.layer_end = {1};
    packed_view.initialize_center_sequence_ids({0});
    constexpr navigamer::LeafId kCenter = uint32_t{1} << 31;
    packed_view.set_center_sequence_id(0, 0, kCenter);
    packed_view.beacon_delta_bits = static_cast<uint8_t>(bits);
    const uint32_t zigzag =
        bits == 32 ? UINT32_MAX : (uint32_t{1} << bits) - 1;
    packed_view.beacon_id_bytes.resize((bits + 7) / 8);
    for (size_t byte = 0; byte < packed_view.beacon_id_bytes.size(); ++byte) {
      packed_view.beacon_id_bytes[byte] =
          static_cast<uint8_t>(zigzag >> (byte * 8));
    }
    packed_view.initialize_beacon_begins(1, 0, 0);
    packed_view.set_beacon_begin(0, 0, 0);
    packed_view.set_node_counts(
        0, 0, 1,
        navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
    const int64_t delta =
        -static_cast<int64_t>((zigzag >> 1) + 1);
    assert(packed_view.beacon_sequence_id(0, 0) ==
           static_cast<navigamer::LeafId>(
               static_cast<int64_t>(kCenter) + delta));
  }
}

void assert_all_center_id_widths_are_exact() {
  constexpr navigamer::NodeId kNodeCount = 16;
  for (uint32_t bits = 0; bits <= 32; ++bits) {
    navigamer::SearchGraphView view;
    const navigamer::NodeId node_count = bits == 0 ? 1 : kNodeCount;
    view.node_records.resize(node_count);
    view.layer_begin = {0};
    view.layer_end = {node_count};
    view.initialize_center_sequence_ids({static_cast<uint8_t>(bits)});
    assert(view.center_id_delta_bits.front() == bits);
    const uint64_t mask =
        bits == 32 ? UINT32_MAX : bits == 0 ? 0 : (uint64_t{1} << bits) - 1;
    const uint32_t divisor = node_count > 1 ? node_count - 1 : 1;
    for (navigamer::NodeId node_id = 0; node_id < node_count;
         ++node_id) {
      const navigamer::LeafId center_id =
          static_cast<navigamer::LeafId>(
              mask * node_id / divisor);
      view.set_center_sequence_id(node_id, 0, center_id);
      assert(view.center_sequence_id(node_id) == center_id);
    }
    assert(view.center_sequence_ids_valid());
  }
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

void assert_child_mbb_layout_is_exact() {
  navigamer::WorldNodeRecord node;
  for (uint32_t bits = 1; bits <= 8; ++bits) {
    node.set_child_mbb_layout(
        navigamer::WorldNodeRecord::CHILD_MBB_BEGIN_MASK, bits);
    assert(node.child_mbb_begin() ==
           navigamer::WorldNodeRecord::CHILD_MBB_BEGIN_MASK);
    assert(node.child_mbb_bits() == bits);
  }

  bool saw_offset_overflow = false;
  try {
    node.set_child_mbb_layout(
        navigamer::WorldNodeRecord::CHILD_MBB_BEGIN_MASK + 1, 8);
  } catch (const std::length_error&) {
    saw_offset_overflow = true;
  }
  assert(saw_offset_overflow);

  for (uint32_t invalid_bits : {0U, 9U}) {
    bool saw_invalid_width = false;
    try {
      node.set_child_mbb_layout(0, invalid_bits);
    } catch (const std::invalid_argument&) {
      saw_invalid_width = true;
    }
    assert(saw_invalid_width);
  }
}

void assert_leaf_mbb_layout_and_values_are_exact() {
  navigamer::WorldNodeRecord layout;
  for (uint32_t bits = 1; bits <= 8; ++bits) {
    layout.set_leaf_mbb_layout(
        navigamer::WorldNodeRecord::CHILD_MBB_BEGIN_MASK, bits);
    assert(layout.leaf_mbb_begin() ==
           navigamer::WorldNodeRecord::CHILD_MBB_BEGIN_MASK);
    assert(layout.leaf_mbb_bits() == bits);
  }

  bool saw_offset_overflow = false;
  try {
    layout.set_leaf_mbb_layout(
        navigamer::WorldNodeRecord::CHILD_MBB_BEGIN_MASK + 1, 8);
  } catch (const std::length_error&) {
    saw_offset_overflow = true;
  }
  assert(saw_offset_overflow);

  for (uint32_t invalid_bits : {0U, 9U}) {
    bool saw_invalid_width = false;
    try {
      layout.set_leaf_mbb_layout(0, invalid_bits);
    } catch (const std::invalid_argument&) {
      saw_invalid_width = true;
    }
    assert(saw_invalid_width);
  }

  constexpr size_t value_count = 19;
  for (uint32_t bits = 1; bits <= 8; ++bits) {
    navigamer::SearchGraphView view;
    view.node_records.resize(1);
    view.layer_begin = {0};
    view.layer_end = {1};
    view.set_node_counts(
        0, value_count, 1,
        navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
    view.node_records[0].set_leaf_mbb_layout(0, bits);
    const uint32_t mask = (uint32_t{1} << bits) - 1;
    const size_t byte_count = (value_count * bits + 7) / 8;
    view.leaf_beacon_dists.assign(byte_count, 0);
    for (size_t value_idx = 0; value_idx < value_count; ++value_idx) {
      const uint32_t value =
          static_cast<uint32_t>(value_idx * 37 + 11) & mask;
      const size_t bit_offset = value_idx * bits;
      const size_t byte_offset = bit_offset >> 3;
      const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
      view.leaf_beacon_dists[byte_offset] |=
          static_cast<uint8_t>(value << shift);
      if (shift + bits > 8) {
        view.leaf_beacon_dists[byte_offset + 1] |=
            static_cast<uint8_t>(value >> (8 - shift));
      }
    }
    assert(view.leaf_mbb_bits(0) == bits);
    assert(view.leaf_mbb_byte_count(0) == byte_count);
    assert(view.leaf_mbb_range_valid(0));
    for (size_t value_idx = 0; value_idx < value_count; ++value_idx) {
      const uint8_t expected = static_cast<uint8_t>(
          static_cast<uint32_t>(value_idx * 37 + 11) & mask);
      assert(view.leaf_beacon_distance(0, value_idx) == expected);
    }
  }
}

void assert_dense_leaf_ternary_mbb_is_exact() {
  navigamer::SearchGraphView view;
  view.node_records.resize(2);
  view.layer_begin = {0};
  view.layer_end = {2};
  view.dense_leaf_mbb_ternary = true;
  view.dense_leaf_mbb_values = {0, 2, 4};
  view.set_node_counts(
      0, 5, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  view.set_node_counts(
      1, 3, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  view.node_records[0].set_leaf_mbb_layout(0, 1);
  view.node_records[1].set_leaf_mbb_layout(0, 1);
  // Codes [0,1,2,1,0] and [2,0,1], least-significant code first.
  view.leaf_beacon_dists = {48, 11};

  assert(view.leaf_mbb_begin(0) == 0);
  assert(view.leaf_mbb_begin(1) == 1);
  assert(view.leaf_mbb_byte_count(0) == 1);
  assert(view.leaf_mbb_byte_count(1) == 1);
  assert(view.leaf_mbb_range_valid(0));
  assert(view.leaf_mbb_range_valid(1));
  const std::array<uint8_t, 5> expected_first = {0, 2, 4, 2, 0};
  for (size_t idx = 0; idx < expected_first.size(); ++idx) {
    assert(view.leaf_beacon_distance(0, idx) == expected_first[idx]);
  }
  const std::array<uint8_t, 3> expected_second = {4, 0, 2};
  for (size_t idx = 0; idx < expected_second.size(); ++idx) {
    assert(view.leaf_beacon_distance(1, idx) == expected_second[idx]);
  }
  navigamer::LeafBeaconFilterSimdStats stats;
  const auto survivors = navigamer::filter_dense_leaf_ternary_survivors(
      view.leaf_beacon_dists[0], 5, 2, 0,
      view.dense_leaf_mbb_values, &stats);
  assert((survivors == std::vector<uint32_t>{1, 3}));
  assert(stats.scalar_checks == 5);
}

void assert_child_id_encodings_are_exact() {
  navigamer::SearchGraphView view;
  view.node_records.resize(4);
  view.layer_begin = {4};
  view.initialize_child_base_ids(4, 4997);
  const navigamer::NodeId child_base = 1000;
  view.append_child_base_id(0, child_base);
  view.child_id_base_deltas8.push_back(0);
  view.child_id_base_deltas8.push_back(255);
  const size_t packed_begin = view.child_id_base_deltas8.size();
  const navigamer::NodeId packed_base = 5000;
  view.append_child_base_id(2, packed_base);
  const size_t packed_payload_begin =
      packed_begin + view.child_base_byte_count();
  view.child_id_base_deltas8.resize(packed_payload_begin + 3);
  const std::array<uint32_t, 2> packed_values = {0, 2047};
  for (size_t offset = 0; offset < packed_values.size(); ++offset) {
    const size_t bit_offset = offset * 11;
    const size_t byte_offset = bit_offset >> 3;
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    const uint32_t value = packed_values[offset];
    view.child_id_base_deltas8[packed_payload_begin + byte_offset] |=
        static_cast<uint8_t>(value << shift);
    if (shift + 11 > 8) {
      view.child_id_base_deltas8[
          packed_payload_begin + byte_offset + 1] |=
          static_cast<uint8_t>(value >> (8 - shift));
    }
    if (shift + 11 > 16) {
      view.child_id_base_deltas8[
          packed_payload_begin + byte_offset + 2] |=
          static_cast<uint8_t>(value >> (16 - shift));
    }
  }
  view.child_id_deltas16 = {0, UINT16_MAX};
  view.child_ids = {70000, UINT32_MAX - 1};

  view.node_records[0].set_link_begin_value(0);
  view.set_node_counts(
      0, 2, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  view.set_child_ids_base_delta8(0);
  assert(view.child_id(0, 0) == 1000);
  assert(view.child_id(0, 1) == 1255);

  view.node_records[1].set_link_begin_value(0);
  view.set_node_counts(
      1, 2, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  view.set_child_ids_delta16(1);
  assert(view.child_id(1, 0) == 2);
  assert(view.child_id(1, 1) == 65537);

  view.node_records[2].set_link_storage(
      navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
  view.node_records[2].set_packed_child_layout(packed_begin, 11);
  view.set_node_counts(
      2, 2, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  assert(view.child_ids_are_packed_delta(2));
  assert(view.child_id(2, 0) == 5000);
  assert(view.child_id(2, 1) == 7047);

  view.node_records[3].set_link_begin_value(0);
  view.set_node_counts(
      3, 2, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  assert(!view.child_ids_are_base_delta8(3));
  assert(!view.child_ids_are_delta16(3));
  assert(!view.child_ids_are_packed_delta(3));
  assert(view.child_id(3, 0) == 70000);
  assert(view.child_id(3, 1) == UINT32_MAX - 1);
  assert(view.edge_count() == 8);
}

void assert_all_child_base_widths_are_exact() {
  constexpr navigamer::NodeId kCount = 19;
  for (uint32_t bits = 1; bits <= 32; ++bits) {
    const uint64_t mask =
        bits == 32 ? std::numeric_limits<uint32_t>::max()
                   : (uint64_t{1} << bits) - 1;
    navigamer::SearchGraphView view;
    view.initialize_child_base_ids(kCount, mask);
    assert(view.child_base_forward_delta_bytes == (bits + 7) / 8);
    assert(view.child_base_ids_valid(kCount));

    std::array<navigamer::NodeId, kCount> expected{};
    std::array<size_t, kCount> begins{};
    for (navigamer::NodeId node_id = 0; node_id < kCount; ++node_id) {
      const uint64_t maximum_valid_delta =
          std::numeric_limits<navigamer::NodeId>::max() -
          static_cast<uint64_t>(node_id) - 1;
      uint64_t delta =
          (static_cast<uint64_t>(node_id + 1) * 2654435761ULL) & mask;
      if (node_id == 0) delta = std::min(mask, maximum_valid_delta);
      delta = std::min(delta, maximum_valid_delta);
      expected[node_id] = static_cast<navigamer::NodeId>(
          static_cast<uint64_t>(node_id) + 1 + delta);
      begins[node_id] = view.child_id_base_deltas8.size();
      view.append_child_base_id(node_id, expected[node_id]);
    }
    assert(view.child_id_base_deltas8.size() ==
           kCount * view.child_base_byte_count());
    for (navigamer::NodeId node_id = 0; node_id < kCount; ++node_id) {
      assert(view.child_base_id(
                 node_id,
                 view.child_id_base_deltas8.data() + begins[node_id],
                 view.child_base_byte_count()) ==
             expected[node_id]);
    }
  }
}

void assert_all_packed_child_widths_are_exact() {
  constexpr uint32_t kCount = 19;
  constexpr navigamer::NodeId kBase = 100000;
  navigamer::SearchGraphView view;
  view.node_records.resize(1);
  view.layer_begin = {1};
  view.initialize_child_base_ids(1, kBase - 1);
  view.node_records[0].set_link_storage(
      navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
  view.set_node_counts(
      0, kCount, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);

  for (uint32_t bits = 1; bits <
       navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS; ++bits) {
    const uint32_t mask = (uint32_t{1} << bits) - 1;
    const size_t payload_bytes =
        (static_cast<size_t>(kCount) * bits + 7) / 8;
    view.child_id_base_deltas8.clear();
    view.append_child_base_id(0, kBase);
    view.child_id_base_deltas8.resize(
        view.child_base_byte_count() + payload_bytes);
    view.node_records[0].set_packed_child_layout(0, bits);

    for (uint32_t offset = 0; offset < kCount; ++offset) {
      const uint32_t value =
          offset == 0 ? 0
                      : offset + 1 == kCount
                            ? mask
                            : (offset * 2654435761u) & mask;
      const size_t bit_offset = static_cast<size_t>(offset) * bits;
      const size_t byte_offset = bit_offset >> 3;
      const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
      const size_t payload_begin = view.child_base_byte_count();
      view.child_id_base_deltas8[payload_begin + byte_offset] |=
          static_cast<uint8_t>(value << shift);
      if (shift + bits > 8) {
        view.child_id_base_deltas8[payload_begin + byte_offset + 1] |=
            static_cast<uint8_t>(value >> (8 - shift));
      }
      if (shift + bits > 16) {
        view.child_id_base_deltas8[payload_begin + byte_offset + 2] |=
            static_cast<uint8_t>(value >> (16 - shift));
      }
      assert(view.child_id(0, offset) == kBase + value);
    }
  }

  view.child_id_base_deltas8.clear();
  view.append_child_base_id(0, kBase);
  view.node_records[0].set_packed_child_layout(
      0, navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS);
  assert(view.packed_child_byte_count(0) == view.child_base_byte_count());
  for (uint32_t offset = 0; offset < kCount; ++offset) {
    assert(view.child_id(0, offset) == kBase + offset);
  }

  constexpr uint32_t kLargeContiguousCount =
      static_cast<uint32_t>(
          navigamer::SearchGraphView::CONTIGUOUS_CHILD_OFFSET_TABLE_SIZE +
          1);
  view.set_node_counts(
      0, kLargeContiguousCount, 0,
      navigamer::WorldNodeRecord::BeaconStorage::Delta8);
  assert(view.packed_child_byte_count(0) == view.child_base_byte_count());
  assert(view.child_id(0, 0) == kBase);
  assert(view.child_id(0, kLargeContiguousCount - 1) ==
         kBase + kLargeContiguousCount - 1);
}

void assert_leaf_id_encodings_are_exact() {
  navigamer::SearchGraphView view;
  view.node_records.resize(3);
  view.layer_begin = {0, 1, 2};
  view.layer_end = {1, 2, 3};
  view.initialize_center_sequence_ids({0, 0, 0});
  view.leaf_id_deltas8 = {-120, 127};
  view.leaf_id_deltas16 = {-30000, 30000};
  view.leaf_ids = {17, UINT32_MAX - 1};

  auto delta8 = view.node_records[0];
  view.set_center_sequence_id(0, 0, 200);
  delta8.set_link_begin_value(0);
  delta8.set_link_storage(
      navigamer::WorldNodeRecord::LinkStorage::Delta8);
  view.set_node_counts(
      0, 2, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  assert(view.leaf_id(0, 0) == 80);
  assert(view.leaf_id(0, 1) == 327);

  auto delta16 = view.node_records[1];
  view.set_center_sequence_id(1, 1, 40000);
  delta16.set_link_begin_value(0);
  delta16.set_link_storage(
      navigamer::WorldNodeRecord::LinkStorage::Delta16);
  view.set_node_counts(
      1, 2, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  assert(view.leaf_id(1, 0) == 10000);
  assert(view.leaf_id(1, 1) == 70000);

  auto absolute32 = view.node_records[2];
  view.set_center_sequence_id(2, 2, 0);
  absolute32.set_link_begin_value(0);
  view.set_node_counts(
      2, 2, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  assert(view.leaf_id(2, 0) == 17);
  assert(view.leaf_id(2, 1) == UINT32_MAX - 1);

  constexpr uint32_t packed_count = 19;
  for (uint32_t bits = 1; bits <= 16; ++bits) {
    navigamer::SearchGraphView packed_view;
    packed_view.node_records.resize(1);
    packed_view.layer_begin = {0};
    packed_view.layer_end = {1};
    packed_view.initialize_center_sequence_ids({0});
    auto packed = packed_view.node_records[0];
    packed_view.set_center_sequence_id(0, 0, 100000);
    packed.set_link_storage(
        navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    packed.set_packed_leaf_layout(0, bits);
    packed_view.set_node_counts(
        0, packed_count, 1,
        navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
    const uint32_t mask = (uint32_t{1} << bits) - 1;
    packed_view.leaf_id_deltas8.assign(
        (packed_count * bits + 7) / 8, 0);
    for (uint32_t offset = 0; offset < packed_count; ++offset) {
      const uint32_t zigzag = (offset * 7919 + 17) & mask;
      const size_t bit_offset = static_cast<size_t>(offset) * bits;
      const size_t byte_offset = bit_offset >> 3;
      const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
      const auto or_byte = [&](size_t index, uint8_t value) {
        packed_view.leaf_id_deltas8[index] = static_cast<int8_t>(
            static_cast<uint8_t>(packed_view.leaf_id_deltas8[index]) |
            value);
      };
      or_byte(byte_offset, static_cast<uint8_t>(zigzag << shift));
      if (shift + bits > 8) {
        or_byte(byte_offset + 1,
                static_cast<uint8_t>(zigzag >> (8 - shift)));
      }
      if (shift + bits > 16) {
        or_byte(byte_offset + 2,
                static_cast<uint8_t>(zigzag >> (16 - shift)));
      }
      const int64_t delta =
          (zigzag & 1) != 0
              ? -static_cast<int64_t>((zigzag >> 1) + 1)
              : static_cast<int64_t>(zigzag >> 1);
      assert(packed_view.leaf_id(0, offset) ==
             static_cast<navigamer::LeafId>(100000 + delta));
    }
    assert(packed.leaf_begin() == 0);
    assert(packed.packed_leaf_bits() == bits);
    assert(packed_view.packed_leaf_byte_count(0) ==
           packed_view.leaf_id_deltas8.size());
  }

  navigamer::WorldNodeRecord packed_layout;
  bool saw_offset_overflow = false;
  try {
    packed_layout.set_packed_leaf_layout(
        navigamer::WorldNodeRecord::PACKED_CHILD_BEGIN_MASK + 1, 8);
  } catch (const std::length_error&) {
    saw_offset_overflow = true;
  }
  assert(saw_offset_overflow);
  for (uint32_t invalid_bits : {0U, 17U}) {
    bool saw_invalid_width = false;
    try {
      packed_layout.set_packed_leaf_layout(0, invalid_bits);
    } catch (const std::invalid_argument&) {
      saw_invalid_width = true;
    }
    assert(saw_invalid_width);
  }
}

void assert_compact_node_layout_is_exact() {
  const auto layout = navigamer::PackedWorldNodeLayout::compact(
      1094080, 3657858, 1501);
  assert(layout.link_begin_bits == 21);
  assert(layout.mbb_begin_bits == 22);
  assert(layout.link_count_bits == 11);
  assert(layout.record_bytes == 9);
  assert(layout.valid());

  navigamer::PackedWorldNodeArray records;
  records.initialize(2, layout);
  {
    auto node = records[0];
    node.set_packed_child_layout(1094080, 15);
    node.set_link_storage(
        navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    node.set_inline_counts(
        1501, 10,
        navigamer::WorldNodeRecord::BeaconStorage::Absolute32);
    node.set_child_mbb_layout(3657858, 7);
  }
  {
    const auto node = records[0];
    assert(node.child_begin() == 1094080);
    assert(node.packed_child_bits() == 15);
    assert(node.link_storage() ==
           navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    assert(!node.counts_overflow());
    assert(node.inline_link_count_or_overflow_index() == 1501);
    assert(node.inline_beacon_count() == 10);
    assert(node.beacon_storage() ==
           navigamer::WorldNodeRecord::BeaconStorage::Absolute32);
    assert(node.child_mbb_begin() == 3657858);
    assert(node.child_mbb_bits() == 7);
  }
  {
    auto node = records[1];
    node.set_count_overflow(
        1501,
        navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
    assert(node.counts_overflow());
    assert(node.inline_link_count_or_overflow_index() == 1501);
  }
  assert(records.bytes().size() == 18);

  const auto widest = navigamer::PackedWorldNodeLayout::compact(
      navigamer::WorldNodeRecord::PACKED_CHILD_BEGIN_MASK,
      navigamer::WorldNodeRecord::CHILD_MBB_BEGIN_MASK,
      navigamer::WorldNodeRecord::LINK_COUNT_MASK);
  assert(widest.record_bytes == sizeof(navigamer::WorldNodeRecord));

  const auto implicit_leaf_layout =
      navigamer::PackedWorldNodeLayout::compact(
          100, 0, 10, true, true, 3, true);
  assert(implicit_leaf_layout.mbb_begin_bits == 0);
  assert(implicit_leaf_layout.has_implicit_packed_link_fields());
  assert(implicit_leaf_layout.has_implicit_center_beacon_storage());
  assert(implicit_leaf_layout.record_bytes == 3);
  assert(implicit_leaf_layout.valid());
  navigamer::PackedWorldNodeArray implicit_leaf_records;
  implicit_leaf_records.initialize(1, implicit_leaf_layout);
  auto implicit_leaf = implicit_leaf_records[0];
  implicit_leaf.set_packed_leaf_layout(0, 3);
  implicit_leaf.set_link_storage(
      navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
  implicit_leaf.set_inline_counts(
      10, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  implicit_leaf.set_leaf_mbb_layout(0, 3);
  assert(implicit_leaf.leaf_begin() == 0);
  assert(implicit_leaf.packed_leaf_bits() == 3);
  assert(implicit_leaf.link_storage() ==
         navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
  assert(implicit_leaf.inline_link_count_or_overflow_index() == 10);
  assert(implicit_leaf.inline_beacon_count() == 1);
  assert(implicit_leaf.beacon_storage() ==
         navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  assert(implicit_leaf.leaf_mbb_begin() == 0);
  assert(implicit_leaf.leaf_mbb_bits() == 3);

  const auto implicit_leaf_count_layout =
      navigamer::PackedWorldNodeLayout::compact(
          0, 0, 10, true, true, 3, true, true, true);
  assert(implicit_leaf_count_layout.has_implicit_one_beacon_count());
  assert(implicit_leaf_count_layout.record_bytes == 1);
  assert(implicit_leaf_count_layout.valid());
  navigamer::PackedWorldNodeArray implicit_leaf_count_records;
  implicit_leaf_count_records.initialize(1, implicit_leaf_count_layout);
  auto implicit_leaf_count = implicit_leaf_count_records[0];
  implicit_leaf_count.set_packed_leaf_layout(0, 3);
  implicit_leaf_count.set_link_storage(
      navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
  implicit_leaf_count.set_inline_counts(
      10, 1,
      navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  implicit_leaf_count.set_leaf_mbb_layout(0, 3);
  assert(implicit_leaf_count.inline_beacon_count() == 1);
  assert(!implicit_leaf_count.counts_overflow());
  bool saw_implicit_count_overflow = false;
  try {
    implicit_leaf_count.set_count_overflow(
        0, navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
  } catch (const std::length_error&) {
    saw_implicit_count_overflow = true;
  }
  assert(saw_implicit_count_overflow);

  const auto implicit_child_layout =
      navigamer::PackedWorldNodeLayout::compact(
          0, 300, 20, false, true,
          navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS);
  assert(implicit_child_layout.record_bytes == 3);
  assert(implicit_child_layout.has_implicit_packed_link_fields());
  assert(!implicit_child_layout.has_implicit_center_beacon_storage());
  navigamer::PackedWorldNodeArray implicit_child_records;
  implicit_child_records.initialize(1, implicit_child_layout);
  auto implicit_child = implicit_child_records[0];
  implicit_child.set_packed_child_layout(
      0, navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS);
  implicit_child.set_link_storage(
      navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
  implicit_child.set_inline_counts(
      20, 2,
      navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
  implicit_child.set_child_mbb_layout(300, 3);
  assert(implicit_child.packed_child_bits() ==
         navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS);
  assert(implicit_child.link_storage() ==
         navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
  assert(implicit_child.beacon_storage() ==
         navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
}

void assert_implicit_leaf_link_begins_are_exact() {
  constexpr uint32_t kLeafCount = 10;
  constexpr uint32_t kPackedBits = 3;
  const auto layout = navigamer::PackedWorldNodeLayout::compact(
      0, 0, 8, true, true, kPackedBits, true, true, true);
  assert(layout.has_implicit_link_begin());
  assert(layout.has_implicit_one_beacon_count());
  assert(layout.record_bytes == 1);

  navigamer::SearchGraphView view;
  view.node_records.initialize(kLeafCount, layout, layout, 0);
  view.layer_begin = {0};
  view.layer_end = {kLeafCount};
  view.initialize_center_sequence_ids({0}, 1000);
  view.initialize_leaf_link_begin_blocks(kLeafCount);
  view.implicit_leaf_mbb_offsets = true;

  uint32_t expected_begin = 0;
  for (uint32_t node_id = 0; node_id < kLeafCount; ++node_id) {
    const uint32_t link_count = node_id % 5 + 1;
    view.set_center_sequence_id(node_id, 0, 1000);
    if (node_id % 8 == 0) {
      view.set_leaf_link_begin(node_id, expected_begin);
    }
    auto node = view.node_records[node_id];
    node.set_packed_leaf_layout(0, kPackedBits);
    node.set_link_storage(
        navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    node.set_inline_counts(
        link_count, 1,
        navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
    node.set_leaf_mbb_layout(0, kPackedBits);

    assert(view.leaf_link_begin(node_id, node) == expected_begin);
    assert(view.leaf_mbb_begin(node_id, node) == expected_begin);
    expected_begin += (link_count * kPackedBits + 7) / 8;
  }
  view.leaf_id_deltas8.assign(expected_begin, 0);
  view.leaf_beacon_dists.assign(expected_begin, 0);
  assert(view.leaf_link_begins_valid());
  for (uint32_t node_id = 0; node_id < kLeafCount; ++node_id) {
    assert(view.leaf_id(node_id, 0) == 1000);
    assert(view.leaf_mbb_range_valid(node_id));
  }
}

void assert_implicit_consecutive_leaf_ids_are_exact() {
  const auto layout = navigamer::PackedWorldNodeLayout::compact(
      0, 0, 5, false, true, 3, true, false, true);
  assert(!layout.has_implicit_link_begin());
  assert(layout.has_implicit_packed_link_fields());

  navigamer::SearchGraphView view;
  view.node_records.initialize(2, layout, layout, 0);
  view.layer_begin = {0};
  view.layer_end = {2};
  view.initialize_center_sequence_ids({4}, 9);
  view.implicit_consecutive_leaf_ids = true;
  view.implicit_consecutive_leaf_radius = 2;
  for (uint32_t node_id = 0; node_id < 2; ++node_id) {
    const auto center = node_id == 0 ? 0U : 9U;
    auto node = view.node_records[node_id];
    view.set_center_sequence_id(node_id, 0, center);
    node.set_packed_leaf_layout(0, 3);
    node.set_link_storage(
        navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    node.set_inline_counts(
        3, 1, navigamer::WorldNodeRecord::BeaconStorage::ImplicitCenter);
    node.set_leaf_mbb_layout(0, 1);
  }
  assert(view.leaf_id_deltas8.empty());
  assert(view.leaf_link_begin_blocks.empty());
  assert(view.leaf_link_begins_valid());
  assert(view.packed_leaf_byte_count(0) == 0);
  assert(view.packed_leaf_byte_count(1) == 0);
  assert(view.leaf_id(0, 0) == 0);
  assert(view.leaf_id(0, 1) == 1);
  assert(view.leaf_id(0, 2) == 2);
  assert(view.leaf_id(1, 0) == 7);
  assert(view.leaf_id(1, 1) == 8);
  assert(view.leaf_id(1, 2) == 9);
}

void assert_dense_beacon_patterns_are_exact() {
  const auto layout = navigamer::PackedWorldNodeLayout::compact(0, 0, 1);
  navigamer::SearchGraphView view;
  view.node_records.initialize(3, layout, layout, 2);
  view.layer_begin = {0, 2};
  view.layer_end = {2, 3};
  view.initialize_center_sequence_ids({4, 0}, 20);
  view.set_center_sequence_id(0, 0, 4);
  view.set_center_sequence_id(1, 0, 10);
  view.dense_beacon_patterns = true;
  view.dense_beacon_pattern_count = 2;
  view.dense_beacon_pattern_deltas[0] = -2;
  view.dense_beacon_pattern_deltas[1] = 0;
  view.dense_beacon_pattern_deltas[2] = 2;
  view.dense_beacon_pattern_deltas[3] = -4;
  view.dense_beacon_pattern_deltas[4] = 0;
  view.dense_beacon_pattern_deltas[5] = 4;
  view.beacon_id_bytes = {0x10};
  for (uint32_t node_id = 0; node_id < 2; ++node_id) {
    auto node = view.node_records[node_id];
    node.set_link_storage(
        navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    node.set_inline_counts(
        1, 3, navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
  }
  assert(view.beacon_begins_valid(2));
  assert(view.beacon_begin(0) == 0);
  assert(view.beacon_begin(1) == 0);
  assert(view.beacon_sequence_id(0, 0) == 2);
  assert(view.beacon_sequence_id(0, 1) == 4);
  assert(view.beacon_sequence_id(0, 2) == 6);
  assert(view.beacon_sequence_id(1, 0) == 6);
  assert(view.beacon_sequence_id(1, 1) == 10);
  assert(view.beacon_sequence_id(1, 2) == 14);
}

void assert_dense_child_mbb_widths_are_exact() {
  const auto layout = navigamer::PackedWorldNodeLayout::compact(
      0, 300, 20, false, true,
      navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS, false,
      true, false, true, 7);
  assert(layout.has_implicit_dense_child_fields());
  assert(layout.record_bytes == 2);
  assert(layout.valid());

  navigamer::SearchGraphView view;
  view.node_records.initialize(3, layout, layout, 2);
  view.layer_begin = {0, 2};
  view.layer_end = {2, 3};
  view.implicit_child_mbb_widths = true;
  view.implicit_child_mbb_exception_bits = 3;
  view.child_mbb_width_exceptions = {0x02};
  for (uint32_t node_id = 0; node_id < 2; ++node_id) {
    auto node = view.node_records[node_id];
    node.set_packed_child_layout(
        0, navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS);
    node.set_link_storage(
        navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    node.set_inline_counts(
        20, 3, navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
    node.set_child_mbb_layout(node_id == 0 ? 300 : 100,
                              node_id == 0 ? 7 : 3);
  }
  assert(view.node_records[0].child_mbb_bits() == 7);
  assert(view.node_records[1].child_mbb_bits() == 7);
  assert(view.child_mbb_bits(0) == 7);
  assert(view.child_mbb_bits(1) == 3);
  assert(view.node_records[0].child_mbb_begin() == 300);
  assert(view.node_records[1].child_mbb_begin() == 100);
  assert(view.node_records.bytes().size() == 6);
}

void assert_compact_child_base_blocks_are_exact() {
  const auto layout = navigamer::PackedWorldNodeLayout::compact(
      0, 0, 1, false, true,
      navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS, false,
      true);
  navigamer::SearchGraphView view;
  view.node_records.initialize(3, layout, layout, 2);
  view.layer_begin = {0, 2};
  view.layer_end = {2, 3};
  view.child_base_forward_delta_bytes = 2;
  view.implicit_contiguous_child_ranges = true;
  view.compact_child_base_blocks = true;
  view.child_base_block_bases = {100};
  view.append_child_base_id(0, 103);
  view.append_child_base_id(1, 108);
  for (uint32_t node_id = 0; node_id < 2; ++node_id) {
    auto node = view.node_records[node_id];
    node.set_packed_child_layout(
        0, navigamer::SearchGraphView::CONTIGUOUS_CHILD_RANGE_BITS);
    node.set_link_storage(
        navigamer::WorldNodeRecord::LinkStorage::PackedDelta);
    node.set_inline_counts(
        1, 0, navigamer::WorldNodeRecord::BeaconStorage::PackedDelta);
  }
  assert(view.child_base_ids_valid(2));
  assert(view.child_base_byte_count() == 1);
  assert(view.child_id(0, 0) == 103);
  assert(view.child_id(1, 0) == 108);
  assert(view.packed_child_byte_count(0) == 1);
  assert(view.packed_child_byte_count(1) == 1);
}

void assert_child_and_leaf_node_layouts_are_independent() {
  const auto child_layout = navigamer::PackedWorldNodeLayout::compact(
      1094080, 3657858, 1501);
  const auto leaf_layout = navigamer::PackedWorldNodeLayout::compact(
      100, 200, 10);
  assert(child_layout.record_bytes == 9);
  assert(leaf_layout.record_bytes == 5);

  navigamer::PackedWorldNodeArray records;
  records.initialize(4, child_layout, leaf_layout, 2);
  records[0].set_child_mbb_layout(3657858, 7);
  records[1].set_child_mbb_layout(1234567, 4);
  records[2].set_leaf_mbb_layout(200, 3);
  records[3].set_leaf_mbb_layout(17, 2);

  assert(records[0].child_mbb_begin() == 3657858);
  assert(records[1].child_mbb_begin() == 1234567);
  assert(records[2].leaf_mbb_begin() == 200);
  assert(records[3].leaf_mbb_begin() == 17);
  assert(records.record_data(1) - records.record_data(0) == 9);
  assert(records.record_data(2) - records.record_data(0) == 18);
  assert(records.record_data(3) - records.record_data(2) == 5);
  assert(records.bytes().size() == 28);
}

void assert_all_beacon_begin_widths_are_exact() {
  for (uint32_t maximum :
       {uint32_t{0}, uint32_t{1}, uint32_t{255}, uint32_t{70000},
        std::numeric_limits<uint32_t>::max()}) {
    const std::array<uint32_t, 7> values = {
        0, maximum / 6, maximum / 3, maximum / 2,
        maximum / 2, (maximum / 2) + maximum / 8,
        (maximum / 2) + maximum / 4};
    for (uint8_t block_size : {uint8_t{2}, uint8_t{4}, uint8_t{8},
                               uint8_t{16}, uint8_t{32}}) {
      uint32_t maximum_block_delta = 0;
      for (size_t idx = 0; idx < values.size(); ++idx) {
        const size_t block_begin = idx - idx % block_size;
        maximum_block_delta = std::max(
            maximum_block_delta, values[idx] - values[block_begin]);
      }
      navigamer::SearchGraphView view;
      view.initialize_beacon_begins(
          values.size(), maximum, maximum_block_delta, block_size);
      assert(view.beacon_begins_valid(values.size()));
      for (size_t idx = 0; idx < values.size(); ++idx) {
        const size_t block_begin = idx - idx % block_size;
        view.set_beacon_begin(
            static_cast<navigamer::NodeId>(idx), values[idx],
            values[block_begin]);
      }
      for (size_t idx = 0; idx < values.size(); ++idx) {
        assert(view.beacon_begin(
                   static_cast<navigamer::NodeId>(idx)) == values[idx]);
      }
    }
  }
}

}  // namespace

int main() {
  assert_view_equivalent_to_original();
  assert_flat_search_matches_original();
  assert_max_byte_distance_is_recall_safe();
  assert_all_center_id_widths_are_exact();
  assert_all_beacon_id_encodings_are_exact();
  assert_node_count_overflow_is_exact();
  assert_child_mbb_layout_is_exact();
  assert_leaf_mbb_layout_and_values_are_exact();
  assert_dense_leaf_ternary_mbb_is_exact();
  assert_child_id_encodings_are_exact();
  assert_all_child_base_widths_are_exact();
  assert_all_packed_child_widths_are_exact();
  assert_leaf_id_encodings_are_exact();
  assert_compact_node_layout_is_exact();
  assert_implicit_leaf_link_begins_are_exact();
  assert_implicit_consecutive_leaf_ids_are_exact();
  assert_dense_beacon_patterns_are_exact();
  assert_dense_child_mbb_widths_are_exact();
  assert_compact_child_base_blocks_are_exact();
  assert_child_and_leaf_node_layouts_are_independent();
  assert_all_beacon_begin_widths_are_exact();
  std::cout << "search graph view tests passed\n";
  return 0;
}
