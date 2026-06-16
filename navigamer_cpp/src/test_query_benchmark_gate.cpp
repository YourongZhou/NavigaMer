#include "query_benchmark.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

int main() {
  using navigamer::compare_result_ids;
  using navigamer::nearest_rank_percentile;

  assert(nearest_rank_percentile({1.0, 2.0, 3.0, 4.0}, 0.50) == 2.0);
  assert(nearest_rank_percentile({1.0, 2.0, 3.0, 4.0}, 0.95) == 4.0);

  auto equal = compare_result_ids({"a", "b"}, {"b", "a"}, {"a", "b"});
  assert(equal.baseline_equals_optimized);
  assert(equal.baseline_no_fn);
  assert(equal.optimized_no_fn);

  auto mismatch = compare_result_ids({"a"}, {"b"}, {"a", "b"});
  assert(!mismatch.baseline_equals_optimized);
  assert((mismatch.baseline_only == std::vector<std::string>{"a"}));
  assert((mismatch.optimized_only == std::vector<std::string>{"b"}));
  assert((mismatch.brute_force_missing_from_baseline ==
          std::vector<std::string>{"b"}));
  assert((mismatch.brute_force_missing_from_optimized ==
          std::vector<std::string>{"a"}));
  assert(!navigamer::comparison_passes_gate(mismatch));
  auto shared_false_positive =
      compare_result_ids({"extra"}, {"extra"}, {});
  assert(!navigamer::comparison_passes_gate(shared_false_positive));
  assert((shared_false_positive.baseline_extra_vs_brute_force ==
          std::vector<std::string>{"extra"}));
  assert((shared_false_positive.optimized_extra_vs_brute_force ==
          std::vector<std::string>{"extra"}));
  assert(!navigamer::profile_results_equal_brute_force(
      shared_false_positive, "baseline"));
  assert(!navigamer::profile_results_equal_brute_force(
      shared_false_positive, "optimized"));

  std::vector<std::shared_ptr<navigamer::BioSequence>> index_sequences = {
      std::make_shared<navigamer::BioSequence>("unique_a", "ACGTGCACTGAT"),
      std::make_shared<navigamer::BioSequence>("unique_b", "TGCATAGCTACG"),
      std::make_shared<navigamer::BioSequence>("low", "AAAAAAAAAAAA"),
      std::make_shared<navigamer::BioSequence>("repeat_a", "CCCCGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("repeat_b", "CCCCGGGGCCCC"),
  };
  auto first = navigamer::generate_benchmark_queries(
      index_sequences, index_sequences, 12, 1, 1234, 1);
  auto second = navigamer::generate_benchmark_queries(
      index_sequences, index_sequences, 12, 1, 1234, 1);
  assert(first.size() == 6);
  assert(first.size() == second.size());
  std::set<std::string> class_names;
  for (size_t i = 0; i < first.size(); ++i) {
    assert(first[i].query_class == second[i].query_class);
    assert(first[i].query.seq == second[i].query.seq);
    assert(first[i].brute_force_ids == second[i].brute_force_ids);
    class_names.insert(navigamer::query_class_name(first[i].query_class));
    if (first[i].query_class == navigamer::QueryClass::NoHit) {
      assert(first[i].brute_force_ids.empty());
    } else if (first[i].query_class == navigamer::QueryClass::SingleHit) {
      assert(first[i].brute_force_ids.size() == 1);
    } else if (first[i].query_class == navigamer::QueryClass::MultiHit) {
      assert(first[i].brute_force_ids.size() >= 2);
    }
  }
  assert(class_names.size() == 6);

  navigamer::QueryBenchmarkConfig config;
  config.ref_input =
      "ACGTGCACTGATTGCATAGCTACGAAAAAAAAAAAACCCCGGGGCCCCCCCCCGGGCCCC";
  config.window_length = 12;
  config.stride = 12;
  config.query_length = 12;
  config.tolerance = 1;
  config.queries_per_class = 1;
  config.warmup_iterations = 1;
  config.measured_iterations = 2;
  config.cold_cache_bytes = 0;
  config.detail_tsv_path = "/tmp/navigamer_query_benchmark_test_detail.tsv";
  config.summary_tsv_path = "/tmp/navigamer_query_benchmark_test_summary.tsv";
  config.json_path = "/tmp/navigamer_query_benchmark_test_summary.json";

  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::SearchConfig optimized_config;
  optimized_config.mbb_filter_mode = navigamer::MBBFilterMode::RectIndex;
  optimized_config.visited_mode = navigamer::VisitedMode::Epoch;
  optimized_config.graph_view_mode = navigamer::GraphViewMode::Flat;
  optimized_config.simd_mode = navigamer::SimdMode::Auto;
  optimized_config.search_qgram_prefilter = true;
  optimized_config.search_qgram_q = 3;
  auto result = navigamer::run_query_benchmark(
      config, navigamer::HierarchyConfig({12, 6, 2}), build_config,
      optimized_config);
  assert(result.gate_passed);
  assert(result.mismatch_count == 0);
  assert(result.detail_rows.size() == 6 * 2 * (1 + 2));
  assert(!result.summary_rows.empty());
  assert(result.json_summary.find("\"gate_passed\":true") != std::string::npos);
  assert(result.json_summary.find(
             "\"baseline\":{\"mbb_filter_mode\":\"scan\","
             "\"visited_mode\":\"string\","
             "\"graph_view\":\"original\","
             "\"simd_mode\":\"scalar\"") != std::string::npos);
  assert(result.json_summary.find(
             "\"optimized\":{\"mbb_filter_mode\":\"rect\","
             "\"visited_mode\":\"epoch\","
             "\"graph_view\":\"flat\","
             "\"simd_mode\":\"auto\"") != std::string::npos);
  assert(result.json_summary.find("\"candidate_set_comparison\":\"unavailable\"")
         != std::string::npos);
  {
    std::ifstream detail(config.detail_tsv_path);
    std::string header;
    std::getline(detail, header);
    assert(header.find("leaf_beacon_scalar_checks") != std::string::npos);
    assert(header.find("leaf_beacon_simd_batches") != std::string::npos);
    assert(header.find("leaf_beacon_simd_fallbacks") != std::string::npos);
  }
  {
    std::ifstream summary(config.summary_tsv_path);
    std::string header;
    std::getline(summary, header);
    assert(header.find("avg_leaf_beacon_scalar_checks") != std::string::npos);
    assert(header.find("avg_leaf_beacon_simd_batches") != std::string::npos);
    assert(header.find("avg_leaf_beacon_simd_fallbacks") != std::string::npos);
  }

  auto bad_output_config = config;
  bad_output_config.detail_tsv_path =
      "/tmp/navigamer_query_benchmark_missing_dir/detail.tsv";
  bad_output_config.summary_tsv_path =
      "/tmp/navigamer_query_benchmark_test_bad_summary.tsv";
  bad_output_config.json_path =
      "/tmp/navigamer_query_benchmark_test_bad_summary.json";
  bool saw_tsv_error = false;
  try {
    (void)navigamer::run_query_benchmark(
        bad_output_config, navigamer::HierarchyConfig({12, 6, 2}),
        build_config, optimized_config);
  } catch (const std::runtime_error&) {
    saw_tsv_error = true;
  }
  assert(saw_tsv_error);

  std::cout << "query benchmark gate tests passed\n";
  return 0;
}
