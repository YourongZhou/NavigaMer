#include "experiment_utils.hpp"
#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace {

std::vector<std::shared_ptr<navigamer::BioSequence>> build_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a", "AAAAAA"),
      std::make_shared<navigamer::BioSequence>("b", "AAAATA"),
      std::make_shared<navigamer::BioSequence>("c", "AAATTA"),
      std::make_shared<navigamer::BioSequence>("d", "TTTTTT"),
  };
}

}  // namespace

int main() {
  {
    auto radii = navigamer::generate_geometric_radius_schedule(4, 8, 0.5);
    assert((radii == std::vector<int>{64, 32, 16, 8}));
  }

  {
    auto radii = navigamer::generate_geometric_radius_schedule(4, 8, 0.7);
    assert((radii == std::vector<int>{23, 16, 11, 8}));
    assert(navigamer::join_radius_schedule(radii) == "23|16|11|8");
  }

  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 12, 6}));
  builder.build(build_sequences());
  navigamer::BioGeometrySearchEngine engine(builder);
  navigamer::BioSequence query("q", "AAAATA");

  auto [results, stats] = engine.search_adaptive(query, 2);
  assert(!results.empty());
  assert(stats.world_access_count > 0);
  assert(stats.node_access_count > 0);
  assert(stats.edge_access_count > 0);
  assert(stats.anchor_distance_count > 0);
  assert(stats.bound_check_count > 0);
  assert(stats.candidate_count >= stats.candidate_verify_count);
  assert(stats.candidate_verify_count == stats.leaf_verify_count);
  assert(stats.result_count == results.size());
  assert(!stats.search_qgram_prefilter_enabled);
  assert(stats.search_qgram_checks == 0);
  assert(stats.center_distance_calls_before_qgram ==
         stats.center_distance_calls_after_qgram);
  assert(stats.leaf_beacon_check_count > 0);
  assert(stats.mbb_check_count == stats.edge_access_count);
  assert(stats.visited_check_count > 0);
  assert(stats.visited_hit_count <= stats.visited_check_count);
  assert(stats.leaf_exact_distance_call_count == stats.leaf_verify_count);
  assert(stats.center_exact_distance_call_count ==
         stats.world_access_count);

  navigamer::SearchConfig qgram_config;
  qgram_config.search_qgram_prefilter = true;
  qgram_config.search_qgram_q = 3;
  navigamer::BioGeometrySearchEngine qgram_engine(builder, qgram_config);
  auto [qgram_results, qgram_stats] = qgram_engine.search_adaptive(query, 2);
  assert(qgram_stats.result_count == qgram_results.size());
  assert(qgram_stats.search_qgram_prefilter_enabled);
  assert(qgram_stats.search_qgram_q == 3);
  assert(qgram_stats.search_qgram_signature_build_count > 0);
  assert(qgram_stats.center_distance_calls_after_qgram <=
         qgram_stats.center_distance_calls_before_qgram);
  assert(qgram_stats.leaf_beacon_check_count > 0);
  assert(qgram_stats.mbb_check_count == qgram_stats.edge_access_count);
  assert(qgram_stats.visited_check_count > 0);
  assert(qgram_stats.visited_hit_count <= qgram_stats.visited_check_count);
  assert(qgram_stats.leaf_exact_distance_call_count ==
         qgram_stats.leaf_verify_count);
  assert(qgram_stats.center_exact_distance_call_count ==
         qgram_stats.world_access_count);

  std::cout << "ALL PASSED\n";
  return 0;
}
