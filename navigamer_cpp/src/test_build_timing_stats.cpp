#include "index_builder.hpp"
#include "structure.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<std::shared_ptr<navigamer::BioSequence>> make_sequences() {
  std::vector<std::shared_ptr<navigamer::BioSequence>> sequences;
  const std::vector<std::string> centers = {
      "ACGTACGTACGTACGTACGTACGTACGTACGT",
      "TTTTACGTACGTACGTACGTACGTACGTACGT",
      "GGGGACGTACGTACGTACGTACGTACGTACGT",
      "CCCCACGTACGTACGTACGTACGTACGTACGT",
  };
  int id = 0;
  for (const auto& center : centers) {
    for (int edit = 0; edit < 4; ++edit) {
      std::string seq = center;
      if (edit > 0) seq[static_cast<size_t>(edit)] = 'A';
      sequences.push_back(std::make_shared<navigamer::BioSequence>(
          "seq_" + std::to_string(id++), seq));
    }
  }
  return sequences;
}

void assert_nonnegative(double value) {
  assert(value >= 0.0);
}

}  // namespace

int main() {
  navigamer::BuildRangeConfig config;
  assert(!config.phase2_qgram_postfilter);
  assert(config.progress_interval_seconds == 600);
  config.min_rect_index_fanout = 1;
  config.range_join.candidate_mode = navigamer::RangeCandidateMode::Auto;
  config.range_join.qgram_q = 3;

  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({16, 8, 3}), config);
  builder.build(make_sequences());

  auto stats = builder.get_statistics();
  assert(stats.total_build_ms > 0.0);

  assert_nonnegative(stats.phase0_dedup_ms);
  assert_nonnegative(stats.phase1_sketch_ms);
  assert(stats.phase1_cover_candidate_scans > 0);
  assert(stats.phase1_exact_distance_calls > 0);
  assert(stats.phase1_exact_distance_calls <=
         stats.phase1_cover_candidate_scans);
  assert(stats.phase1_best_cover_hits > 0);
  assert(stats.phase1_cover_misses > 0);
  assert(stats.phase1_length_pruned_candidates <=
         stats.phase1_cover_candidate_scans);
  assert(stats.phase1_pigeonhole_candidates <=
         stats.phase1_total_possible_pairs);
  assert(stats.phase1_hint_hits <= stats.phase1_hint_checks);
  assert(stats.phase1_pigeonhole_queries == 0 ||
         stats.phase1_seed_posting_entries_visited > 0 ||
         stats.phase1_pigeonhole_fallbacks > 0);
  assert_nonnegative(stats.phase2_rebinding_ms);
  assert_nonnegative(stats.phase3_mbb_ms);
  assert_nonnegative(stats.phase4_attach_ms);
  assert_nonnegative(stats.assign_ids_ms);
  assert_nonnegative(stats.graph_view_ms);
  assert_nonnegative(stats.print_summary_ms);

  const double measured_sum =
      stats.phase0_dedup_ms + stats.phase1_sketch_ms +
      stats.phase2_rebinding_ms + stats.phase3_mbb_ms +
      stats.phase4_attach_ms + stats.assign_ids_ms + stats.graph_view_ms;
  assert(measured_sum <= stats.total_build_ms * 1.2);

  assert_nonnegative(stats.phase2_index_build_ms);
  assert_nonnegative(stats.phase2_candidate_query_ms);
  assert_nonnegative(stats.phase2_exact_verify_ms);
  assert_nonnegative(stats.phase2_candidate_query_worker_ms);
  assert_nonnegative(stats.phase2_exact_verify_worker_ms);
  assert_nonnegative(stats.phase2_edge_insert_ms);
  assert(stats.phase2_distance_batches > 0);

  assert_nonnegative(stats.phase3_collect_beacons_ms);
  assert_nonnegative(stats.phase3_collapse_children_ms);
  assert_nonnegative(stats.phase3_child_mbb_distance_ms);
  assert_nonnegative(stats.phase3_rect_index_build_ms);

  assert_nonnegative(stats.leaf_index_build_ms);
  assert_nonnegative(stats.leaf_candidate_query_ms);
  assert_nonnegative(stats.leaf_exact_verify_ms);
  assert_nonnegative(stats.leaf_tuple_emit_ms);
  assert_nonnegative(stats.leaf_tuple_merge_sort_ms);
  assert_nonnegative(stats.leaf_populate_ms);
  assert_nonnegative(stats.leaf_beacon_distance_ms);

  assert_nonnegative(stats.range_posting_lookup_ms);
  assert_nonnegative(stats.range_seed_union_ms);
  assert_nonnegative(stats.range_length_filter_ms);
  assert_nonnegative(stats.range_qgram_query_ms);
  assert_nonnegative(stats.range_hybrid_intersection_ms);
  assert_nonnegative(stats.range_full_scan_ms);

  std::cout << "build timing stats tests passed\n";
  return 0;
}
