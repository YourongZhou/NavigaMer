#include "index_builder.hpp"
#include "index_persistence.hpp"
#include "io_utils.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
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
  assert(loaded_mbb.size() == built_mbb.size());
  assert(std::equal(
      built_mbb.begin(), built_mbb.end(), loaded_mbb.begin()));
#if defined(__unix__) || defined(__APPLE__)
  const auto assert_mapped = [](const auto& array) {
    if (!array.empty()) assert(array.is_mapped());
  };
  const auto& loaded_view = loaded.builder.search_graph_view();
  assert_mapped(loaded_view.node_records);
  assert_mapped(loaded_view.child_ids);
  assert_mapped(loaded_view.leaf_ids);
  assert_mapped(loaded_view.beacon_deltas8);
  assert_mapped(loaded_view.beacon_deltas16);
  assert_mapped(loaded_view.beacon_ids32);
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
  assert(!built.sequence_store().reference_records.empty());

  auto manifest = navigamer::make_reference_window_index_manifest(
      reference, reference.size(), static_cast<int>(kWindowLength), 1,
      hierarchy, build_config);
  navigamer::save_index(path, built, manifest);
  navigamer::LoadedIndex loaded = navigamer::load_index(path);
  const auto& built_store = built.sequence_store();
  const auto& loaded_store = loaded.builder.sequence_store();
  assert(loaded_store.reference_backed);
  assert(loaded_store.reference_id == "synthetic_ref");
  assert(loaded_store.reference_sequence == reference);
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
  assert(loaded.manifest.format_version == 16);
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
  assert(loaded_store.reference_records.is_mapped());
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

void assert_chunked_reference_encoding_is_exact() {
  const std::string index_path =
      "/tmp/navigamer_test_chunked_reference.navidx";
  std::remove(index_path.c_str());

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
      "chunked", reference, 4, reference.size());
  auto manifest = navigamer::make_reference_window_index_manifest(
      reference, reference.size(), 4,
      static_cast<int>(reference.size()), hierarchy, build_config);
  assert(manifest.ref_input.size() < reference.size());
  navigamer::save_index(index_path, built, manifest);

  std::ifstream stored(index_path, std::ios::binary | std::ios::ate);
  assert(stored);
  assert(static_cast<size_t>(stored.tellg()) < reference.size());

  const auto loaded = navigamer::load_index(index_path);
  assert(loaded.builder.sequence_store().reference_sequence == reference);
  assert(loaded.builder.validate_integer_ids());
  const auto reconstructed_manifest =
      navigamer::make_reference_window_index_manifest(
          loaded.manifest.ref_input, reference.size(), 4,
          static_cast<int>(reference.size()), hierarchy, build_config);
  assert(reconstructed_manifest.signature == loaded.manifest.signature);
  std::remove(index_path.c_str());
}

}  // namespace

int main() {
  assert_loaded_search_matches_built();
  assert_manifest_matching_detects_reusable_index();
  assert_reference_window_manifest_tracks_slicing_parameters();
  assert_reference_backed_index_round_trip();
  assert_multicontig_invalid_base_and_occurrence_round_trip();
  assert_chunked_reference_encoding_is_exact();
  std::cout << "index persistence tests passed\n";
  return 0;
}
