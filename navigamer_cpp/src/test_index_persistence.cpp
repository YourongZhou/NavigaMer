#include "index_builder.hpp"
#include "index_persistence.hpp"
#include "io_utils.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using SequencePtr = std::shared_ptr<navigamer::BioSequence>;

std::vector<SequencePtr> make_sequences() {
  auto a0 = std::make_shared<navigamer::BioSequence>("a0", "ACGTACGTACGTACGTACGT");
  a0->add_occurrence("ref", 10, 30, "+");
  a0->set_bwt_interval(100, 101);

  auto a1 = std::make_shared<navigamer::BioSequence>("a1", "ACGTACGTACGTACGTACGA");
  a1->add_occurrence("ref", 20, 40, "+");
  a1->set_bwt_interval(200, 201);

  auto dup = std::make_shared<navigamer::BioSequence>("a0_dup", "ACGTACGTACGTACGTACGT");
  dup->add_occurrence("ref", 90, 110, "-");

  return {
      a0,
      a1,
      dup,
      std::make_shared<navigamer::BioSequence>("b0", "TTTTACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("b1", "TTTTACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("c0", "GGGGACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("c1", "GGGGACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("d0", "CCCCACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("d1", "CCCCACGTACGTACGTACGA"),
  };
}

std::set<navigamer::LeafId> sequence_ids(
    const navigamer::SearchResult& hits) {
  return {hits.begin(), hits.end()};
}

void assert_loaded_search_matches_built() {
  const std::string path = "/tmp/navigamer_test_index_persistence.navidx";
  std::remove(path.c_str());

  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::HierarchyConfig hierarchy({20, 10, 3});
  navigamer::BioGeometryIndexBuilder built(hierarchy, build_config);
  built.build(make_sequences());

  navigamer::IndexBuildManifest manifest =
      navigamer::make_index_manifest("ref", "reads", hierarchy, build_config);
  navigamer::save_index(path, built, manifest);

  navigamer::LoadedIndex loaded = navigamer::load_index(path);
  assert(loaded.builder.validate_integer_ids());
  assert(loaded.builder.validate_search_graph_view());
  const auto& built_mbb =
      built.search_graph_view().child_beacon_dists;
  const auto& loaded_mbb =
      loaded.builder.search_graph_view().child_beacon_dists;
  assert(loaded.builder.search_graph_view().center_id_block_begins ==
         built.search_graph_view().center_id_block_begins);
  assert(loaded.builder.search_graph_view().center_id_delta_begins ==
         built.search_graph_view().center_id_delta_begins);
  assert(loaded.builder.search_graph_view().center_id_delta_bits ==
         built.search_graph_view().center_id_delta_bits);
  assert(loaded.builder.search_graph_view().center_id_block_bases_16bit ==
         built.search_graph_view().center_id_block_bases_16bit);
  if (built.search_graph_view().center_id_block_bases_16bit) {
    assert(std::equal(
        built.search_graph_view().center_id_block_bases16.begin(),
        built.search_graph_view().center_id_block_bases16.end(),
        loaded.builder.search_graph_view().center_id_block_bases16.begin()));
  } else {
    assert(std::equal(
        built.search_graph_view().center_id_block_bases.begin(),
        built.search_graph_view().center_id_block_bases.end(),
        loaded.builder.search_graph_view().center_id_block_bases.begin()));
  }
  assert(std::equal(
      built.search_graph_view().center_id_block_deltas.begin(),
      built.search_graph_view().center_id_block_deltas.end(),
      loaded.builder.search_graph_view().center_id_block_deltas.begin()));
  assert(loaded.builder.search_graph_view().node_records.child_layout() ==
         built.search_graph_view().node_records.child_layout());
  assert(loaded.builder.search_graph_view().node_records.leaf_layout() ==
         built.search_graph_view().node_records.leaf_layout());
  assert(loaded.builder.search_graph_view().node_records.finest_node_begin() ==
         built.search_graph_view().node_records.finest_node_begin());
  assert(loaded.builder.search_graph_view().node_records.bytes() ==
         built.search_graph_view().node_records.bytes());
  assert(loaded.builder.search_graph_view().child_base_forward_delta_bytes ==
         built.search_graph_view().child_base_forward_delta_bytes);
  assert(loaded.builder.search_graph_view().compact_child_base_blocks ==
         built.search_graph_view().compact_child_base_blocks);
  assert(loaded.builder.search_graph_view().child_base_block_bases ==
         built.search_graph_view().child_base_block_bases);
  assert(loaded.builder.search_graph_view().implicit_child_mbb_widths ==
         built.search_graph_view().implicit_child_mbb_widths);
  assert(loaded.builder.search_graph_view().implicit_child_mbb_exception_bits ==
         built.search_graph_view().implicit_child_mbb_exception_bits);
  assert(loaded.builder.search_graph_view().child_mbb_width_exceptions ==
         built.search_graph_view().child_mbb_width_exceptions);
  assert(loaded.builder.search_graph_view().interned_child_mbb_payloads ==
         built.search_graph_view().interned_child_mbb_payloads);
  assert(loaded.builder.search_graph_view().implicit_leaf_mbb_offsets ==
         built.search_graph_view().implicit_leaf_mbb_offsets);
  assert(loaded.builder.search_graph_view().implicit_consecutive_leaf_ids ==
         built.search_graph_view().implicit_consecutive_leaf_ids);
  assert(loaded.builder.search_graph_view().implicit_consecutive_leaf_radius ==
         built.search_graph_view().implicit_consecutive_leaf_radius);
  assert(loaded.builder.search_graph_view().dense_beacon_patterns ==
         built.search_graph_view().dense_beacon_patterns);
  assert(loaded.builder.search_graph_view().dense_beacon_pattern_count ==
         built.search_graph_view().dense_beacon_pattern_count);
  assert(loaded.builder.search_graph_view().dense_beacon_pattern_deltas ==
         built.search_graph_view().dense_beacon_pattern_deltas);
  assert(loaded.builder.search_graph_view().dense_leaf_mbb_ternary ==
         built.search_graph_view().dense_leaf_mbb_ternary);
  assert(loaded.builder.search_graph_view().dense_leaf_mbb_values ==
         built.search_graph_view().dense_leaf_mbb_values);
  assert(loaded.builder.search_graph_view().leaf_link_begin_blocks ==
         built.search_graph_view().leaf_link_begin_blocks);
  assert(loaded_mbb.size() == built_mbb.size());
  assert(std::equal(
      built_mbb.begin(), built_mbb.end(), loaded_mbb.begin()));
  for (navigamer::NodeId node_id = 0;
       node_id < built.search_graph_view().layer_begin.back(); ++node_id) {
    assert(built.search_graph_view().child_mbb_bits(node_id) ==
           loaded.builder.search_graph_view().child_mbb_bits(node_id));
  }
  assert(loaded.builder.search_graph_view().beacon_begin_blocks ==
         built.search_graph_view().beacon_begin_blocks);
  assert(loaded.builder.search_graph_view().beacon_begin_base_bits ==
         built.search_graph_view().beacon_begin_base_bits);
  assert(loaded.builder.search_graph_view().beacon_begin_delta_bits ==
         built.search_graph_view().beacon_begin_delta_bits);
#if defined(__unix__) || defined(__APPLE__)
  const auto assert_mapped = [](const auto& array) {
    if (!array.empty()) {
      assert(array.is_mapped());
      assert((reinterpret_cast<uintptr_t>(array.data()) & 63) == 0);
    }
  };
  const auto& loaded_view = loaded.builder.search_graph_view();
  assert_mapped(loaded_view.node_records);
  if (loaded_view.center_id_block_bases_16bit) {
    assert_mapped(loaded_view.center_id_block_bases16);
  } else {
    assert_mapped(loaded_view.center_id_block_bases);
  }
  assert_mapped(loaded_view.center_id_block_deltas);
  assert_mapped(loaded_view.node_count_overflows);
  assert_mapped(loaded_view.child_id_base_deltas8);
  assert_mapped(loaded_view.child_base_block_bases);
  assert_mapped(loaded_view.child_id_deltas16);
  assert_mapped(loaded_view.child_ids);
  assert_mapped(loaded_view.child_mbb_width_exceptions);
  assert_mapped(loaded_view.leaf_id_deltas8);
  assert_mapped(loaded_view.leaf_id_deltas16);
  assert_mapped(loaded_view.leaf_ids);
  assert_mapped(loaded_view.beacon_id_bytes);
  assert_mapped(loaded_view.beacon_begin_blocks);
  assert_mapped(loaded_view.child_beacon_dists);
  assert_mapped(loaded_view.leaf_beacon_dists);
#endif
  assert(loaded.manifest.signature == manifest.signature);
  assert(loaded.manifest.primary_radii == manifest.primary_radii);

  const navigamer::BioSequence* restored = nullptr;
  for (const auto& sequence : loaded.builder.sequence_store().records) {
    if (sequence.seq == "ACGTACGTACGTACGTACGT") {
      restored = &sequence;
      break;
    }
  }
  assert(restored != nullptr);
  assert(restored->ref_positions.size() == 2);
  assert(restored->bwt_interval.start == 100);
  assert(restored->bwt_interval.end == 101);

  navigamer::SearchConfig scan_config;
  scan_config.mbb_filter_mode = navigamer::MBBFilterMode::Scan;
  navigamer::BioGeometrySearchEngine built_scan(built, scan_config);
  navigamer::BioGeometrySearchEngine loaded_scan(loaded.builder, scan_config);

  navigamer::SearchConfig rect_config;
  rect_config.mbb_filter_mode = navigamer::MBBFilterMode::RectIndex;
  navigamer::BioGeometrySearchEngine loaded_rect(loaded.builder, rect_config);

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGTACGTACGT"),
      navigamer::BioSequence("q1", "TTTTACGTACGTACGTACGA"),
      navigamer::BioSequence("q2", "AAAAGGGGAAAAGGGGAAAA"),
  };

  for (const auto& query : queries) {
    auto [built_hits, built_stats] = built_scan.search_adaptive(query, 2);
    auto [loaded_hits, loaded_stats] = loaded_scan.search_adaptive(query, 2);
    auto [rect_hits, rect_stats] = loaded_rect.search_adaptive(query, 2);
    (void)built_stats;
    (void)loaded_stats;
    (void)rect_stats;
    assert(sequence_ids(built_hits) == sequence_ids(loaded_hits));
    assert(sequence_ids(built_hits) == sequence_ids(rect_hits));
  }

  const std::string copied_path =
      "/tmp/navigamer_test_mapped_copy.navidx";
  std::remove(copied_path.c_str());
  navigamer::save_index(
      copied_path, loaded.builder, loaded.manifest);
  auto copied = navigamer::load_index(copied_path);
  assert(copied.builder.validate_integer_ids());
  assert(copied.builder.validate_search_graph_view());
  assert(std::equal(
      copied.builder.search_graph_view().child_beacon_dists.begin(),
      copied.builder.search_graph_view().child_beacon_dists.end(),
      built_mbb.begin()));

  std::remove(path.c_str());
  std::remove(copied_path.c_str());
}

void assert_graph_payload_uses_shared_manifest() {
  const std::string full_path =
      "/tmp/navigamer_test_full_index.navidx";
  const std::string payload_path =
      "/tmp/navigamer_test_graph_payload.navpayload";
  std::remove(full_path.c_str());
  std::remove(payload_path.c_str());

  navigamer::BuildRangeConfig build_config;
  navigamer::HierarchyConfig hierarchy({20, 10, 3});
  navigamer::BioGeometryIndexBuilder built(hierarchy, build_config);
  built.build(make_sequences());
  const auto manifest = navigamer::make_index_manifest(
      "payload-ref", "payload-reads", hierarchy, build_config);

  navigamer::save_index(full_path, built, manifest);
  navigamer::save_index_payload(payload_path, built);
  const auto file_size = [](const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    assert(in);
    return static_cast<uint64_t>(in.tellg());
  };
  assert(file_size(payload_path) < file_size(full_path));

  auto loaded = navigamer::load_index_payload(
      payload_path, manifest);
  assert(loaded.manifest.signature == manifest.signature);
  assert(loaded.manifest.sequence_count == built.num_sequences());
  assert(loaded.manifest.world_node_count == built.num_world_nodes());
  assert(loaded.manifest.edge_count ==
         built.search_graph_view().edge_count());
  assert(loaded.manifest.leaf_link_count ==
         built.search_graph_view().leaf_link_count());
  assert(loaded.builder.validate_search_graph_view());

  navigamer::BioGeometrySearchEngine built_search(built);
  navigamer::BioGeometrySearchEngine loaded_search(loaded.builder);
  const navigamer::BioSequence query(
      "payload-query", "ACGTACGTACGTACGTACGT");
  assert(sequence_ids(built_search.search_adaptive(query, 2).first) ==
         sequence_ids(loaded_search.search_adaptive(query, 2).first));

  auto invalid_manifest = manifest;
  invalid_manifest.signature.front() =
      invalid_manifest.signature.front() == '0' ? '1' : '0';
  bool invalid_rejected = false;
  try {
    (void)navigamer::load_index_payload(
        payload_path, invalid_manifest);
  } catch (const std::runtime_error&) {
    invalid_rejected = true;
  }
  assert(invalid_rejected);

  std::remove(full_path.c_str());
  std::remove(payload_path.c_str());
}

void assert_manifest_matching_detects_reusable_index() {
  const std::string path = "/tmp/navigamer_test_index_manifest.navidx";
  std::remove(path.c_str());

  navigamer::BuildRangeConfig build_config;
  navigamer::HierarchyConfig hierarchy({20, 10, 3});
  navigamer::BioGeometryIndexBuilder built(hierarchy, build_config);
  built.build(make_sequences());

  navigamer::IndexBuildManifest manifest =
      navigamer::make_index_manifest("same-ref", "same-reads", hierarchy, build_config);
  navigamer::save_index(path, built, manifest);

  navigamer::IndexBuildManifest stored;
  std::string reason;
  assert(navigamer::index_matches_manifest(path, manifest, &stored, &reason));
  assert(stored.signature == manifest.signature);
  assert(reason.empty());

  navigamer::IndexBuildManifest changed =
      navigamer::make_index_manifest("same-ref", "same-reads",
                                     navigamer::HierarchyConfig({24, 12, 3}),
                                     build_config);
  assert(!navigamer::index_matches_manifest(path, changed, &stored, &reason));
  assert(!reason.empty());

  navigamer::BuildRangeConfig no_progress_config = build_config;
  no_progress_config.progress_interval_seconds = 0;
  navigamer::IndexBuildManifest no_progress_manifest =
      navigamer::make_index_manifest("same-ref", "same-reads", hierarchy,
                                     no_progress_config);
  assert(no_progress_manifest.signature == manifest.signature);

  std::remove(path.c_str());
}

void assert_reference_window_manifest_tracks_slicing_parameters() {
  navigamer::BuildRangeConfig build_config;
  navigamer::HierarchyConfig hierarchy({12, 6, 2});
  const auto manifest = navigamer::make_reference_window_index_manifest(
      "ACGTACGTACGT", 12, 6, 1, hierarchy, build_config);
  assert(manifest.reads_input ==
         "reference-windows:v1;prefix=12;window=6;stride=1");

  const auto changed_prefix =
      navigamer::make_reference_window_index_manifest(
          "ACGTACGTACGT", 11, 6, 1, hierarchy, build_config);
  const auto changed_window =
      navigamer::make_reference_window_index_manifest(
          "ACGTACGTACGT", 12, 7, 1, hierarchy, build_config);
  const auto changed_stride =
      navigamer::make_reference_window_index_manifest(
          "ACGTACGTACGT", 12, 6, 2, hierarchy, build_config);
  assert(changed_prefix.signature != manifest.signature);
  assert(changed_window.signature != manifest.signature);
  assert(changed_stride.signature != manifest.signature);
}

void assert_reference_backed_index_round_trip() {
  const std::string path =
      "/tmp/navigamer_test_reference_backed.navidx";
  std::remove(path.c_str());
  const std::string reference =
      "ACGTGCTAGCTAGGATCCGATGCTTACGATCGGCTAACGT"
      "TTGACCGTACGATGGCATTCGACTAGCTTGACCTAGGCTA";
  constexpr size_t kWindowLength = 16;

  navigamer::BuildRangeConfig build_config;
  navigamer::HierarchyConfig hierarchy({10, 5, 2});
  navigamer::BioGeometryIndexBuilder built(hierarchy, build_config);
  built.build_reference_windows(
      "synthetic_ref", reference, kWindowLength, 1);
  assert(built.sequence_store().reference_backed);
  assert(built.sequence_store().records.empty());
  assert(built.sequence_store().reference_positions_global_linear ||
         !built.sequence_store().reference_position_blocks.empty());

  auto manifest = navigamer::make_reference_window_index_manifest(
      reference, reference.size(), static_cast<int>(kWindowLength), 1,
      hierarchy, build_config);
  navigamer::save_index(path, built, manifest);
  navigamer::LoadedIndex loaded = navigamer::load_index(path);
  const auto& built_store = built.sequence_store();
  const auto& loaded_store = loaded.builder.sequence_store();
  assert(loaded_store.reference_backed);
  assert(loaded_store.reference_id == "synthetic_ref");
  assert(loaded_store.reference_view() == reference);
  assert(loaded_store.fixed_sequence_length == kWindowLength);
  assert(loaded_store.size() == built_store.size());
  for (size_t sequence_idx = 0;
       sequence_idx < built_store.size(); ++sequence_idx) {
    const auto id = static_cast<navigamer::LeafId>(sequence_idx);
    assert(loaded_store.sequence(id) == built_store.sequence(id));
    assert(loaded_store.source_position(id) ==
           built_store.source_position(id));
    assert(loaded_store.sa_interval(id).start ==
           built_store.sa_interval(id).start);
    assert(loaded_store.sa_interval(id).end ==
           built_store.sa_interval(id).end);
  }

  navigamer::BioGeometrySearchEngine engine(loaded.builder);
  navigamer::BioSequence query(
      "q_reference", reference.substr(7, kWindowLength));
  const auto [brute_hits, brute_stats] =
      engine.search_brute_force(query, 0);
  const auto [adaptive_hits, adaptive_stats] =
      engine.search_adaptive(query, 0);
  (void)brute_stats;
  (void)adaptive_stats;
  assert(!brute_hits.empty());
  assert(sequence_ids(adaptive_hits) == sequence_ids(brute_hits));

  navigamer::LoadedIndex structurally_loaded =
      navigamer::load_index(
          path,
          navigamer::IndexLoadValidation::Structural);
  navigamer::BioGeometrySearchEngine structural_engine(
      structurally_loaded.builder);
  const auto [structural_hits, structural_stats] =
      structural_engine.search_adaptive(query, 0);
  (void)structural_stats;
  assert(sequence_ids(structural_hits) ==
         sequence_ids(brute_hits));

  std::remove(path.c_str());
}

void assert_multicontig_invalid_base_and_occurrence_round_trip() {
  const std::string fasta_path =
      "/tmp/navigamer_test_multicontig.fa";
  const std::string index_path =
      "/tmp/navigamer_test_multicontig.navidx";
  {
    std::ofstream fasta(fasta_path);
    fasta << ">chr1 description\n"
          << "aaaaccccgggg\n"
          << ">chr2\n"
          << "ttttaaaaNaaaacccc\n";
  }
  auto reference = navigamer::load_reference_genome(fasta_path);
  assert(reference.id == "chr1");
  assert(reference.sequence ==
         "AAAACCCCGGGGTTTTAAAANAAAACCCC");
  assert(reference.contigs.size() == 2);
  assert(reference.contigs[0].id == "chr1");
  assert(reference.contigs[0].begin == 0);
  assert(reference.contigs[0].end == 12);
  assert(reference.contigs[1].id == "chr2");
  assert(reference.contigs[1].begin == 12);
  assert(reference.contigs[1].end == reference.sequence.size());

  navigamer::BuildRangeConfig build_config;
  navigamer::HierarchyConfig hierarchy({8, 4, 2});
  navigamer::BioGeometryIndexBuilder built(hierarchy, build_config);
  built.build_reference_windows(
      reference.id, reference.sequence, 4, 2, reference.contigs);
  const auto stats = built.get_statistics();
  assert(stats.added_sequences == 12);
  assert(stats.invalid_reference_windows == 2);
  assert(built.validate_integer_ids());

  const auto& store = built.sequence_store();
  navigamer::LeafId aaaa_id = navigamer::INVALID_LEAF_ID;
  for (size_t sequence_idx = 0; sequence_idx < store.size();
       ++sequence_idx) {
    const auto id = static_cast<navigamer::LeafId>(sequence_idx);
    const auto sequence = store.sequence(id);
    assert(sequence.find('N') == std::string_view::npos);
    if (sequence == "AAAA") aaaa_id = id;
    const size_t global_start = store.source_position(id);
    const auto& contig = store.contig_for_position(global_start);
    assert(global_start + store.fixed_sequence_length <= contig.end);
  }
  assert(aaaa_id != navigamer::INVALID_LEAF_ID);
  assert(store.identifier(aaaa_id) == "chr1_0");
  assert(store.occurrence_positions(aaaa_id) ==
         std::vector<uint32_t>({0, 16, 21}));

  auto manifest = navigamer::make_reference_window_index_manifest(
      fasta_path, reference.sequence.size(), 4, 2,
      hierarchy, build_config);
  navigamer::save_index(index_path, built, manifest);
  auto loaded = navigamer::load_index(index_path);
  const auto& loaded_store = loaded.builder.sequence_store();
  assert(loaded.manifest.format_version == 58);
  assert(loaded_store.reference_contigs.size() == 2);
  assert(loaded_store.singleton_occurrences ==
         store.singleton_occurrences);
  assert(loaded_store.occurrence_groups ==
         store.occurrence_groups);
  assert(loaded_store.grouped_occurrence_positions ==
         store.grouped_occurrence_positions);
  assert(loaded_store.occurrence_positions(aaaa_id) ==
         std::vector<uint32_t>({0, 16, 21}));
#if defined(__unix__) || defined(__APPLE__)
  if (loaded_store.reference_positions_global_linear) {
    assert(loaded_store.reference_position_blocks.empty());
    assert(loaded_store.reference_position_payload.empty());
  } else {
    assert(loaded_store.reference_position_blocks.is_mapped());
    if (!loaded_store.reference_position_payload.empty()) {
      assert(loaded_store.reference_position_payload.is_mapped());
    }
  }
  if (!loaded_store.singleton_occurrences.empty()) {
    assert(loaded_store.singleton_occurrences.is_mapped());
  }
  if (!loaded_store.occurrence_groups.empty()) {
    assert(loaded_store.occurrence_groups.is_mapped());
  }
  if (!loaded_store.grouped_occurrence_positions.empty()) {
    assert(loaded_store.grouped_occurrence_positions.is_mapped());
  }
#endif

  navigamer::BioGeometrySearchEngine engine(loaded.builder);
  navigamer::BioSequence query("q", "AAAA");
  const auto [adaptive, adaptive_stats] =
      engine.search_adaptive(query, 0);
  const auto [brute_force, brute_stats] =
      engine.search_brute_force(query, 0);
  (void)adaptive_stats;
  (void)brute_stats;
  assert(sequence_ids(adaptive) == sequence_ids(brute_force));
  assert(sequence_ids(adaptive).count(aaaa_id) == 1);

  navigamer::BioGeometryIndexBuilder unsampled_first(
      navigamer::HierarchyConfig({8, 4, 2}), build_config);
  unsampled_first.build_reference_windows(
      "order", "CAAAACAAAA", 4, 2);
  navigamer::LeafId ordered_id = navigamer::INVALID_LEAF_ID;
  for (size_t sequence_idx = 0;
       sequence_idx < unsampled_first.sequence_store().size();
       ++sequence_idx) {
    const auto id = static_cast<navigamer::LeafId>(sequence_idx);
    if (unsampled_first.sequence_store().sequence(id) == "AAAA") {
      ordered_id = id;
      break;
    }
  }
  assert(ordered_id != navigamer::INVALID_LEAF_ID);
  assert(unsampled_first.sequence_store().source_position(ordered_id) == 6);
  assert(unsampled_first.sequence_store().occurrence_positions(ordered_id) ==
         std::vector<uint32_t>({1, 6}));

  navigamer::BioGeometryIndexBuilder grouped(
      navigamer::HierarchyConfig({8, 4, 2}), build_config);
  grouped.build_reference_windows(
      "grouped", "CAAAACAAAACAAAACAAAA", 4, 2);
  navigamer::LeafId grouped_id = navigamer::INVALID_LEAF_ID;
  for (size_t sequence_idx = 0;
       sequence_idx < grouped.sequence_store().size(); ++sequence_idx) {
    const auto id = static_cast<navigamer::LeafId>(sequence_idx);
    if (grouped.sequence_store().sequence(id) == "AAAA") {
      grouped_id = id;
      break;
    }
  }
  assert(grouped_id != navigamer::INVALID_LEAF_ID);
  assert(grouped.sequence_store().occurrence_positions(grouped_id) ==
         std::vector<uint32_t>({1, 6, 11, 16}));
  assert(grouped.sequence_store().additional_occurrence_count(grouped_id) ==
         3);
  assert(std::none_of(
      grouped.sequence_store().singleton_occurrences.begin(),
      grouped.sequence_store().singleton_occurrences.end(),
      [&](const navigamer::ReferenceOccurrence& occurrence) {
        return occurrence.sequence_id == grouped_id;
      }));

  std::remove(fasta_path.c_str());
  std::remove(index_path.c_str());
}

void assert_reference_encoding_is_exact() {
  const std::string index_path =
      "/tmp/navigamer_test_raw_reference.navidx";
  const std::string packed_index_path =
      "/tmp/navigamer_test_two_bit_reference.navidx";
  std::remove(index_path.c_str());
  std::remove(packed_index_path.c_str());

  constexpr size_t kChunkBoundary = size_t{1} << 20;
  std::string reference(kChunkBoundary + 37, 'C');
  reference.replace(0, 4, "ACGT");
  reference[kChunkBoundary - 1] = 'N';
  reference[kChunkBoundary] = 'R';
  reference[kChunkBoundary + 1] = 'a';

  navigamer::BuildRangeConfig build_config;
  navigamer::HierarchyConfig hierarchy({8, 4, 2});
  navigamer::BioGeometryIndexBuilder built(hierarchy, build_config);
  built.build_reference_windows(
      "mapped", reference, 4, reference.size());
  auto manifest = navigamer::make_reference_window_index_manifest(
      reference, reference.size(), 4,
      static_cast<int>(reference.size()), hierarchy, build_config);
  assert(manifest.ref_input.size() < reference.size());
  navigamer::save_index(index_path, built, manifest);

  const auto loaded = navigamer::load_index(index_path);
  const auto& loaded_store = loaded.builder.sequence_store();
  assert(loaded_store.reference_sequence == reference);
  assert(loaded_store.mapped_reference_sequence.empty());
  assert(loaded_store.reference_view() == reference);
  assert(loaded_store.reference_view().data() ==
         loaded_store.reference_sequence.data());
  assert(loaded.builder.validate_integer_ids());
  const auto reconstructed_manifest =
      navigamer::make_reference_window_index_manifest(
          loaded.manifest.ref_input, reference.size(), 4,
          static_cast<int>(reference.size()), hierarchy, build_config);
  assert(reconstructed_manifest.signature == loaded.manifest.signature);

  std::string two_bit_reference = reference;
  two_bit_reference[kChunkBoundary - 1] = 'C';
  two_bit_reference[kChunkBoundary] = 'C';
  two_bit_reference[kChunkBoundary + 1] = 'C';
  navigamer::BioGeometryIndexBuilder packed(hierarchy, build_config);
  packed.build_reference_windows(
      "packed", two_bit_reference, 4, two_bit_reference.size());
  const auto packed_manifest = navigamer::make_reference_window_index_manifest(
      two_bit_reference, two_bit_reference.size(), 4,
      static_cast<int>(two_bit_reference.size()), hierarchy, build_config);
  navigamer::save_index(packed_index_path, packed, packed_manifest);
  const auto packed_loaded = navigamer::load_index(packed_index_path);
  assert(packed_loaded.builder.sequence_store().reference_view() ==
         two_bit_reference);
  const auto file_size = [](const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    assert(in);
    return static_cast<uint64_t>(in.tellg());
  };
  assert(file_size(packed_index_path) + reference.size() / 2 <
         file_size(index_path));
  std::remove(index_path.c_str());
  std::remove(packed_index_path.c_str());
}

std::string deterministic_dna(size_t length, uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> base(0, 3);
  static constexpr char kBases[] = {'A', 'C', 'G', 'T'};
  std::string sequence(length, 'A');
  for (char& value : sequence) value = kBases[base(generator)];
  return sequence;
}

void assert_compact_child_base_blocks_round_trip() {
  const std::string path = "/tmp/navigamer_test_compact_child_bases.navidx";
  std::remove(path.c_str());
  const std::string reference = deterministic_dna(10000, 0x91e10da5U);
  constexpr size_t kWindowLength = 150;
  navigamer::HierarchyConfig hierarchy({8, 4, 2});
  navigamer::BuildRangeConfig build_config;
  navigamer::BioGeometryIndexBuilder built(hierarchy, build_config);
  built.build_reference_windows(
      "compact_child_bases", reference, kWindowLength, 1);
  const auto& built_view = built.search_graph_view();
  assert(built_view.compact_child_base_blocks);
  assert(!built_view.child_base_block_bases.empty());

  auto manifest = navigamer::make_reference_window_index_manifest(
      reference, reference.size(), static_cast<int>(kWindowLength), 1,
      hierarchy, build_config);
  navigamer::save_index(path, built, manifest);
  const auto loaded = navigamer::load_index(path);
  const auto& loaded_view = loaded.builder.search_graph_view();
  assert(loaded_view.compact_child_base_blocks);
  assert(loaded_view.child_base_block_bases ==
         built_view.child_base_block_bases);
  assert(loaded_view.child_id_base_deltas8 ==
         built_view.child_id_base_deltas8);
  for (navigamer::NodeId node_id = 0;
       node_id < built_view.node_records.finest_node_begin();
       ++node_id) {
    assert(built_view.child_count(node_id) ==
           loaded_view.child_count(node_id));
    for (uint32_t child = 0;
         child < built_view.child_count(node_id); ++child) {
      assert(built_view.child_id(node_id, child) ==
             loaded_view.child_id(node_id, child));
    }
  }
  std::remove(path.c_str());
}

std::string sparse_reference(const std::vector<size_t>& positions) {
  constexpr size_t kWindowLength = 16;
  std::string reference(positions.back() + kWindowLength, 'N');
  for (size_t idx = 0; idx < positions.size(); ++idx) {
    const std::string sequence =
        deterministic_dna(kWindowLength, static_cast<uint32_t>(idx + 91));
    reference.replace(positions[idx], kWindowLength, sequence);
  }
  return reference;
}

void assert_reference_position_encodings_are_exact() {
  constexpr size_t kWindowLength = 16;
  navigamer::BuildRangeConfig build_config;
  const navigamer::HierarchyConfig hierarchy({8, 4, 2});

  const auto build_store = [&](const std::string& reference)
      -> navigamer::BioGeometryIndexBuilder {
    navigamer::BioGeometryIndexBuilder builder(hierarchy, build_config);
    builder.build_reference_windows(
        "position_encoding", reference, kWindowLength, 1);
    assert(builder.validate_integer_ids());
    return builder;
  };

  const std::string linear_reference = deterministic_dna(400, 1001);
  auto linear = build_store(linear_reference);
  const auto& linear_store = linear.sequence_store();
  assert(linear_store.size() > navigamer::kReferencePositionBlockSize);
  assert(linear_store.reference_positions_global_linear);
  assert(linear_store.reference_position_begin == 0);
  assert(linear_store.reference_position_step == 1);
  assert(linear_store.reference_position_blocks.empty());
  assert(linear_store.reference_position_payload.empty());
  for (size_t idx = 0; idx < linear_store.size(); ++idx) {
    assert(linear_store.source_position(static_cast<navigamer::LeafId>(idx)) ==
           idx);
  }
  const std::string linear_path =
      "/tmp/navigamer_test_position_linear.navidx";
  auto linear_manifest = navigamer::make_reference_window_index_manifest(
      linear_reference, linear_reference.size(), kWindowLength, 1,
      hierarchy, build_config);
  navigamer::save_index(linear_path, linear, linear_manifest);
  const auto loaded_linear = navigamer::load_index(linear_path);
  const auto& loaded_linear_store = loaded_linear.builder.sequence_store();
  assert(loaded_linear_store.reference_positions_global_linear);
  assert(loaded_linear_store.reference_position_begin == 0);
  assert(loaded_linear_store.reference_position_step == 1);
  assert(loaded_linear_store.reference_position_blocks.empty());
  assert(loaded_linear_store.reference_position_payload.empty());
  for (size_t idx = 0; idx < linear_store.size(); ++idx) {
    const auto id = static_cast<navigamer::LeafId>(idx);
    assert(loaded_linear_store.source_position(id) ==
           linear_store.source_position(id));
    assert(loaded_linear_store.sequence(id) == linear_store.sequence(id));
  }
  std::remove(linear_path.c_str());

  const std::string stride_linear_reference = deterministic_dna(528, 1002);
  navigamer::BioGeometryIndexBuilder stride_linear(hierarchy, build_config);
  stride_linear.build_reference_windows(
      "position_encoding_stride", stride_linear_reference,
      kWindowLength, 2);
  const auto& stride_linear_store = stride_linear.sequence_store();
  assert(stride_linear_store.size() == 257);
  assert(stride_linear_store.reference_positions_global_linear);
  assert(stride_linear_store.reference_position_begin == 0);
  assert(stride_linear_store.reference_position_step == 2);
  assert(stride_linear_store.reference_position_blocks.empty());
  for (size_t idx = 0; idx < stride_linear_store.size(); ++idx) {
    assert(stride_linear_store.source_position(
               static_cast<navigamer::LeafId>(idx)) ==
           idx * 2);
  }

  std::string bitset_reference = deterministic_dna(500, 2002);
  for (size_t pos = 99; pos < bitset_reference.size(); pos += 100) {
    bitset_reference[pos] = 'N';
  }
  auto bitset = build_store(bitset_reference);
  const auto& bitset_store = bitset.sequence_store();
  assert(bitset_store.size() > navigamer::kReferencePositionBlockSize);
  assert(!bitset_store.reference_positions_global_linear);
  assert(bitset_store.reference_position_blocks[0].encoding ==
         navigamer::ReferencePositionEncoding::Bitset);
  uint32_t previous = 0;
  for (size_t idx = 0; idx < bitset_store.size(); ++idx) {
    const uint32_t position = static_cast<uint32_t>(
        bitset_store.source_position(static_cast<navigamer::LeafId>(idx)));
    assert(idx == 0 || position > previous);
    assert(bitset_store.sequence(static_cast<navigamer::LeafId>(idx)).find('N') ==
           std::string_view::npos);
    previous = position;
  }

  const auto assert_sparse_encoding = [&](
      const std::vector<size_t>& positions,
      navigamer::ReferencePositionEncoding expected) {
    auto sparse = build_store(sparse_reference(positions));
    const auto& store = sparse.sequence_store();
    assert(store.size() == positions.size());
    assert(!store.reference_positions_global_linear);
    assert(store.reference_position_blocks.size() == 1);
    assert(store.reference_position_blocks[0].encoding == expected);
    for (size_t idx = 0; idx < positions.size(); ++idx) {
      assert(store.source_position(static_cast<navigamer::LeafId>(idx)) ==
             positions[idx]);
    }
  };
  assert_sparse_encoding(
      {0, 50, 200}, navigamer::ReferencePositionEncoding::Delta8);
  assert_sparse_encoding(
      {0, 1000, 3000}, navigamer::ReferencePositionEncoding::Delta16);
  assert_sparse_encoding(
      {0, 70000, 140000},
      navigamer::ReferencePositionEncoding::Absolute32);

  const std::string path = "/tmp/navigamer_test_position_bitset.navidx";
  auto manifest = navigamer::make_reference_window_index_manifest(
      bitset_reference, bitset_reference.size(), kWindowLength, 1,
      hierarchy, build_config);
  navigamer::save_index(path, bitset, manifest);
  const auto loaded = navigamer::load_index(path);
  const auto& loaded_store = loaded.builder.sequence_store();
  assert(loaded_store.size() == bitset_store.size());
  for (size_t idx = 0; idx < bitset_store.size(); ++idx) {
    const auto id = static_cast<navigamer::LeafId>(idx);
    assert(loaded_store.source_position(id) == bitset_store.source_position(id));
    assert(loaded_store.sequence(id) == bitset_store.sequence(id));
  }
  std::remove(path.c_str());
}

}  // namespace

int main() {
  assert_loaded_search_matches_built();
  assert_graph_payload_uses_shared_manifest();
  assert_manifest_matching_detects_reusable_index();
  assert_reference_window_manifest_tracks_slicing_parameters();
  assert_reference_backed_index_round_trip();
  assert_multicontig_invalid_base_and_occurrence_round_trip();
  assert_reference_encoding_is_exact();
  assert_reference_position_encodings_are_exact();
  assert_compact_child_base_blocks_round_trip();
  std::cout << "index persistence tests passed\n";
  return 0;
}
