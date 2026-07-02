#include "experiment_utils.hpp"
#include "index_builder.hpp"
#include "query_benchmark.hpp"
#include "search_engine.hpp"
#include "structure.hpp"
#include <cassert>
#include <fstream>
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
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 12, 6}));
  builder.build(build_sequences());
  navigamer::BioSequence query("q", "AAAATA");

  navigamer::SearchConfig profile_off;
  profile_off.query_profile = false;
  navigamer::BioGeometrySearchEngine off_engine(builder, profile_off);
  auto [off_results, off_stats] = off_engine.search_adaptive(query, 2);
  assert(!off_results.empty());
  assert(off_stats.query_count == 1);
  assert(!off_stats.query_profile_enabled);
  assert(off_stats.query_total_ms >= 0.0);
  assert(off_stats.router_lookup_ms == 0.0);
  assert(off_stats.anchor_distance_ms == 0.0);
  assert(off_stats.mbb_filter_ms == 0.0);
  assert(off_stats.child_bound_ms == 0.0);
  assert(off_stats.center_distance_ms == 0.0);
  assert(off_stats.leaf_collect_ms == 0.0);
  assert(off_stats.leaf_mbb_filter_ms == 0.0);
  assert(off_stats.leaf_verify_ms == 0.0);
  assert(off_stats.result_dedup_ms == 0.0);
  assert(off_stats.path_reuse_ms == 0.0);

  navigamer::SearchConfig profile_on = profile_off;
  profile_on.query_profile = true;
  profile_on.path_reuse_enabled = true;
  navigamer::BioGeometrySearchEngine on_engine(builder, profile_on);
  auto [on_results, on_stats] = on_engine.search_adaptive(query, 2);
  assert(on_results.size() == off_results.size());
  assert(on_stats.query_count == 1);
  assert(on_stats.query_profile_enabled);
  assert(on_stats.query_total_ms >= 0.0);
  assert(on_stats.anchor_distance_ms >= 0.0);
  assert(on_stats.mbb_filter_ms >= 0.0);
  assert(on_stats.center_distance_ms >= 0.0);
  assert(on_stats.leaf_verify_ms >= 0.0);
  assert(on_stats.result_dedup_ms >= 0.0);
  assert(on_stats.result_count == on_results.size());
  assert(on_stats.world_access_count > 0);
  assert(on_stats.anchor_distance_count > 0);
  assert(on_stats.center_distance_count > 0);
  assert(on_stats.raw_candidate_count >= on_stats.leaf_verify_count);
  auto [repeat_results, repeat_stats] = on_engine.search_adaptive(query, 2);
  assert(repeat_results.size() == on_results.size());
  assert(repeat_stats.path_reuse_attempt_count > 0);
  assert(repeat_stats.path_reuse_hit_count > 0);
  assert(repeat_stats.anchor_cache_hit_count > 0);

  navigamer::QueryBenchmarkConfig benchmark_config;
  benchmark_config.ref_input =
      "ACGTGCACTGATTGCATAGCTACGAAAAAAAAAAAACCCCGGGGCCCCCCCCCGGGCCCC";
  benchmark_config.window_length = 12;
  benchmark_config.stride = 12;
  benchmark_config.query_length = 12;
  benchmark_config.tolerance = 1;
  benchmark_config.queries_per_class = 1;
  benchmark_config.warmup_iterations = 0;
  benchmark_config.measured_iterations = 1;
  benchmark_config.cold_cache_bytes = 0;
  benchmark_config.detail_tsv_path = "/tmp/navigamer_query_profile_detail.tsv";
  benchmark_config.summary_tsv_path = "/tmp/navigamer_query_profile_summary.tsv";
  benchmark_config.json_path = "/tmp/navigamer_query_profile_summary.json";

  navigamer::BuildRangeConfig build_config;
  auto benchmark_result = navigamer::run_query_benchmark(
      benchmark_config, navigamer::HierarchyConfig({12, 6, 2}), build_config,
      profile_on);
  assert(benchmark_result.gate_passed);

  {
    std::ifstream detail(benchmark_config.detail_tsv_path);
    std::string header;
    std::getline(detail, header);
    assert(header.find("query_total_ms") != std::string::npos);
    assert(header.find("anchor_distance_ms") != std::string::npos);
    assert(header.find("local_router_invoked_count") != std::string::npos);
    assert(header.find("local_router_shortlist_child_count") !=
           std::string::npos);
    assert(header.find("best_first_invoked_count") != std::string::npos);
    assert(header.find("best_first_bound_candidate_count") !=
           std::string::npos);
    assert(header.find("path_reuse_attempt_count") != std::string::npos);
    assert(header.find("anchor_cache_hit_count") != std::string::npos);
    assert(header.find("center_distance_count") != std::string::npos);
    assert(header.find("raw_candidate_count") != std::string::npos);
  }
  {
    std::ifstream summary(benchmark_config.summary_tsv_path);
    std::string header;
    std::getline(summary, header);
    assert(header.find("warm_p50_query_ms") != std::string::npos);
    assert(header.find("avg_world_access_count") != std::string::npos);
    assert(header.find("avg_anchor_distance_count") != std::string::npos);
    assert(header.find("avg_local_router_invoked_count") != std::string::npos);
    assert(header.find("avg_best_first_invoked_count") != std::string::npos);
    assert(header.find("avg_path_reuse_hit_count") != std::string::npos);
    assert(header.find("avg_anchor_cache_hit_count") != std::string::npos);
    assert(header.find("avg_child_safe_bound_pruned_count") !=
           std::string::npos);
    assert(header.find("avg_center_distance_count") != std::string::npos);
    assert(header.find("avg_raw_candidate_count") != std::string::npos);
  }

  std::cout << "query profile smoke passed\n";
  return 0;
}
