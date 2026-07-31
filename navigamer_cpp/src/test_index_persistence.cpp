#include "index_builder.hpp"
#include "index_persistence.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <cassert>
#include <cstdio>
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

std::set<std::string> ids(const navigamer::SearchResult& hits) {
  std::set<std::string> out;
  for (const auto& hit : hits) out.insert(hit->id);
  return out;
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
    assert(ids(built_hits) == ids(loaded_hits));
    assert(ids(built_hits) == ids(rect_hits));
  }

  std::remove(path.c_str());
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

}  // namespace

int main() {
  assert_loaded_search_matches_built();
  assert_manifest_matching_detects_reusable_index();
  assert_reference_window_manifest_tracks_slicing_parameters();
  std::cout << "index persistence tests passed\n";
  return 0;
}
