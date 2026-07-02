#include "query_benchmark.hpp"
#include "index_persistence.hpp"
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

  auto near_anchor = navigamer::compute_proximal_anchor_set_diagnostics(
      "ACGTACGT", {"ACGTACGT", "ACGTACGA"}, {"TTTTTTTT", "GGGGGGGG"},
      {1, 2, 4});
  assert(near_anchor.nearest_anchor_dist == 0.0);
  assert(near_anchor.oracle_envelope_by_k.size() == 3);
  assert(near_anchor.random_envelope_by_k.size() == 3);
  assert(near_anchor.oracle_envelope_by_k[0] == 0.0);
  assert(near_anchor.oracle_envelope_by_k[1] <
         near_anchor.random_envelope_by_k[1]);

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

  auto clustered = navigamer::generate_locality_benchmark_queries(
      "ACGTACGTACGTACGTACGTACGTACGTACGT", 8, 12, 1, 7);
  assert(clustered.same_template.size() == 8);
  assert(clustered.nearby_windows.size() == 8);
  assert(clustered.random_windows.size() == 8);
  assert(clustered.repeat.size() == 8);
  assert(clustered.real_dup_1x.size() == 8);
  assert(clustered.real_dup_4x.size() == 32);
  assert(clustered.real_dup_16x.size() == 128);
  assert(clustered.source_sorted_stride1.size() == 8);
  assert(clustered.source_sorted_mutated_tau5.size() == 8);
  assert(clustered.source_sorted_mutated_tau8.size() == 8);
  assert(clustered.same_template[0].source_pos ==
         clustered.same_template[7].source_pos);
  assert(clustered.nearby_windows[0].source_pos + 7 ==
         clustered.nearby_windows[7].source_pos);
  assert(clustered.repeat[0].query.seq == clustered.repeat[5].query.seq);
  assert(clustered.real_dup_4x[0].query.seq ==
         clustered.real_dup_4x[1].query.seq);
  assert(clustered.real_dup_4x[0].source_pos ==
         clustered.real_dup_4x[1].source_pos);
  assert(clustered.real_dup_4x[4].query.seq !=
         clustered.real_dup_4x[0].query.seq);
  assert(clustered.source_sorted_stride1[0].source_pos + 7 ==
         clustered.source_sorted_stride1[7].source_pos);
  assert(clustered.source_sorted_mutated_tau5[0].source_pos + 7 ==
         clustered.source_sorted_mutated_tau5[7].source_pos);
  assert(clustered.source_sorted_mutated_tau8[0].source_pos + 7 ==
         clustered.source_sorted_mutated_tau8[7].source_pos);
  assert(clustered.random_windows[0].query.seq.size() == 12);

  {
    const std::string locality_ref =
        "ACGTGCACTGATTGCATAGCTACGACGTGCACTGATTGCATAGCTACG";
    std::vector<std::shared_ptr<navigamer::BioSequence>> locality_windows;
    for (size_t start = 0; start + 12 <= locality_ref.size(); start += 6) {
      locality_windows.push_back(std::make_shared<navigamer::BioSequence>(
          "locality_" + std::to_string(start), locality_ref.substr(start, 12)));
    }
    navigamer::BuildRangeConfig locality_build_config;
    locality_build_config.min_rect_index_fanout = 1;
    navigamer::HierarchyConfig locality_hierarchy({12, 6, 2});
    navigamer::BioGeometryIndexBuilder locality_builder(
        locality_hierarchy, locality_build_config);
    locality_builder.build(locality_windows);
    const std::string locality_index_path = "/tmp/navigamer_locality_test.navidx";
    navigamer::save_index(
        locality_index_path, locality_builder,
        navigamer::make_index_manifest("locality_ref", "locality_windows",
                                       locality_hierarchy,
                                       locality_build_config));

    navigamer::LocalityBenchmarkConfig locality_config;
    locality_config.index_path = locality_index_path;
    locality_config.ref_input = locality_ref;
    locality_config.query_count = 4;
    locality_config.query_length = 12;
    locality_config.tolerance = 1;
    locality_config.edits = 1;
    locality_config.profiles = {"baseline", "path_reuse", "optimized"};
    locality_config.datasets = {"same_template", "nearby_windows"};
    locality_config.scenarios = {"all"};
    assert((locality_config.batch_schedules ==
            std::vector<std::string>{"original"}));
    locality_config.batch_schedules = {"original", "random", "qgram-signature",
                                       "source-oracle"};
    locality_config.out_tsv_path = "/tmp/navigamer_locality_test.tsv";
    locality_config.query_fastq_out_path =
        "/tmp/navigamer_locality_test_queries.fastq";
    auto locality_result =
        navigamer::run_persisted_locality_benchmark(locality_config);
    assert(locality_result.gate_passed);
    assert(locality_result.load_ms >= 0.0);
    assert(locality_result.rows.size() > 24);
    std::ifstream locality_tsv(locality_config.out_tsv_path);
    std::string locality_header;
    std::getline(locality_tsv, locality_header);
    assert(locality_header.find("batch_schedule_mode") != std::string::npos);
    assert(locality_header.find("load_ms") != std::string::npos);
    assert(locality_header.find("engine_init_ms") != std::string::npos);
    assert(locality_header.find("query_wall_ms") != std::string::npos);
    assert(locality_header.find("mean_fanout") != std::string::npos);
    assert(locality_header.find("p95_fanout") != std::string::npos);
    assert(locality_header.find("max_fanout") != std::string::npos);
    assert(locality_header.find("router_invoked_ratio") != std::string::npos);
    assert(locality_header.find("local_router_invoked_ratio") !=
           std::string::npos);
    assert(locality_header.find("safe_child_router_invoked_ratio") !=
           std::string::npos);
    assert(locality_header.find("path_reuse_hit_ratio") != std::string::npos);
    assert(locality_header.find("anchor_cache_hit_count") !=
           std::string::npos);
    assert(locality_header.find("child_shortlist_cache_hit_count") !=
           std::string::npos);
    assert(locality_header.find("safe_child_candidate_cache_hit_count") !=
           std::string::npos);
    assert(locality_header.find("productive_world_reuse_hit_count") !=
           std::string::npos);
    assert(locality_header.find("unique_query_count") != std::string::npos);
    assert(locality_header.find("duplicate_group_count") != std::string::npos);
    assert(locality_header.find("duplicate_ratio") != std::string::npos);
    assert(locality_header.find("verified_result_cache_hit_count") !=
           std::string::npos);
    assert(locality_header.find("near_query_reuse_hit_count") !=
           std::string::npos);
    assert(locality_header.find("near_query_triangle_pruned_count") !=
           std::string::npos);
    assert(locality_header.find("near_query_center_distance_reused_count") !=
           std::string::npos);
    assert(locality_header.find("near_query_bound_fallback_count") !=
           std::string::npos);
    assert(locality_header.find("near_query_direct_verify_count") !=
           std::string::npos);
    assert(locality_header.find("center_distance_reduction") !=
           std::string::npos);
    assert(locality_header.find("world_access_reduction") !=
           std::string::npos);
    assert(locality_header.find("p95_speedup") != std::string::npos);
    assert(locality_header.find("mean_neighbor_edit_distance") !=
           std::string::npos);
    assert(locality_header.find("p95_neighbor_edit_distance") !=
           std::string::npos);
    assert(locality_header.find("mean_neighbor_qgram_jaccard") !=
           std::string::npos);
    assert(locality_header.find("mean_child_count_before_router") !=
           std::string::npos);
    assert(locality_header.find("mean_post_mbb_survivor_count") !=
           std::string::npos);
    assert(locality_header.find("mean_safe_router_candidate_count") !=
           std::string::npos);
    assert(locality_header.find("mean_candidate_ratio_to_all_children") !=
           std::string::npos);
    assert(locality_header.find(
               "mean_candidate_ratio_to_post_mbb_survivors") !=
           std::string::npos);
    assert(locality_header.find("mean_children_actually_processed") !=
           std::string::npos);
    assert(locality_header.find("mean_center_checks_saved") !=
           std::string::npos);
    bool saw_source_oracle = false;
    bool saw_repeat = false;
    bool saw_batch_locality = false;
    bool saw_oracle = false;
    bool saw_real_dup_4x_random_cache = false;
    bool saw_source_sorted_near_reuse = false;
    for (const auto& row : locality_result.rows) {
      if (row.batch_schedule_mode == "source-oracle") saw_source_oracle = true;
      if (row.dataset == "repeat") saw_repeat = true;
      if (row.dataset == "batch_locality") saw_batch_locality = true;
      if (row.dataset == "oracle") saw_oracle = true;
      if (row.dataset == "real_dup_4x" && row.profile == "optimized" &&
          row.batch_schedule_mode == "random") {
        assert(row.query_count == 16);
        assert(row.unique_query_count == 4);
        assert(row.duplicate_group_count == 4);
        assert(row.duplicate_ratio == 0.75);
        assert(row.verified_result_cache_hit_count == 12);
        assert(row.mismatch_count == 0);
        saw_real_dup_4x_random_cache = true;
      }
      if (row.dataset == "source_sorted_stride1" &&
          row.profile == "optimized" &&
          row.batch_schedule_mode == "source-oracle") {
        assert(row.mismatch_count == 0);
        assert(row.mean_neighbor_edit_distance >= 0.0);
        assert(row.mean_neighbor_qgram_jaccard >= 0.0);
        saw_source_sorted_near_reuse = true;
      }
    }
    assert(saw_source_oracle);
    assert(saw_repeat);
    assert(saw_batch_locality);
    assert(saw_oracle);
    assert(saw_real_dup_4x_random_cache);
    assert(saw_source_sorted_near_reuse);
    {
      std::ifstream query_fastq(locality_config.query_fastq_out_path);
      std::string header;
      std::string sequence;
      std::string plus;
      std::string quality;
      std::getline(query_fastq, header);
      std::getline(query_fastq, sequence);
      std::getline(query_fastq, plus);
      std::getline(query_fastq, quality);
      assert(header.rfind("@low_fanout_0 source_pos=", 0) == 0);
      assert(sequence.size() == 12);
      assert(plus == "+");
      assert(quality.size() == sequence.size());
    }

    const std::string locality_json = "/tmp/navigamer_locality_report.json";
    const std::string locality_md = "/tmp/navigamer_locality_report.md";
    navigamer::write_locality_report_outputs(
        locality_result, locality_json, locality_md);
    {
      std::ifstream json(locality_json);
      std::string content((std::istreambuf_iterator<char>(json)),
                          std::istreambuf_iterator<char>());
      assert(content.find("\"gate_passed\":true") != std::string::npos);
      assert(content.find("\"batch_schedule_mode\":\"source-oracle\"") !=
             std::string::npos);
      assert(content.find("\"rows\"") != std::string::npos);
    }
    {
      std::ifstream markdown(locality_md);
      std::string content((std::istreambuf_iterator<char>(markdown)),
                          std::istreambuf_iterator<char>());
      assert(content.find("# NavigaMer Query Locality Report") !=
             std::string::npos);
      assert(content.find("source-oracle") != std::string::npos);
      assert(content.find("optimized") != std::string::npos);
    }
  }

  {
    std::vector<std::shared_ptr<navigamer::BioSequence>> high_fanout_sequences;
    std::string high_fanout_ref;
    const char alphabet[] = {'A', 'C', 'G', 'T'};
    for (size_t i = 0; i < 80; ++i) {
      uint32_t state = static_cast<uint32_t>(17 + i * 7919);
      std::string seq;
      seq.reserve(16);
      for (size_t j = 0; j < 16; ++j) {
        state = state * 1664525u + 1013904223u;
        seq.push_back(alphabet[(state >> 30) & 3u]);
      }
      high_fanout_sequences.push_back(
          std::make_shared<navigamer::BioSequence>(
              "high_fanout_" + std::to_string(i), seq));
      high_fanout_ref += seq;
    }

    navigamer::BuildRangeConfig high_fanout_build_config;
    high_fanout_build_config.min_rect_index_fanout = 1;
    navigamer::HierarchyConfig high_fanout_hierarchy({16, 1});
    navigamer::BioGeometryIndexBuilder high_fanout_builder(
        high_fanout_hierarchy, high_fanout_build_config);
    high_fanout_builder.build(high_fanout_sequences);
    const std::string high_fanout_index_path =
        "/tmp/navigamer_locality_high_fanout_test.navidx";
    navigamer::save_index(
        high_fanout_index_path, high_fanout_builder,
        navigamer::make_index_manifest("high_fanout_ref",
                                       "high_fanout_sequences",
                                       high_fanout_hierarchy,
                                       high_fanout_build_config));

    navigamer::LocalityBenchmarkConfig high_fanout_config;
    high_fanout_config.index_path = high_fanout_index_path;
    high_fanout_config.ref_input = high_fanout_ref;
    high_fanout_config.query_count = 8;
    high_fanout_config.query_length = 16;
    high_fanout_config.tolerance = 8;
    high_fanout_config.edits = 2;
    high_fanout_config.profiles = {"baseline", "optimized"};
    high_fanout_config.scenarios = {"same-template"};
    high_fanout_config.batch_schedules = {"source-oracle"};
    high_fanout_config.out_tsv_path =
        "/tmp/navigamer_locality_high_fanout_test.tsv";
    auto high_fanout_result =
        navigamer::run_persisted_locality_benchmark(high_fanout_config);
    assert(high_fanout_result.gate_passed);
    bool saw_high_fanout_optimized = false;
    for (const auto& row : high_fanout_result.rows) {
      if (row.dataset == "same_template" && row.profile == "optimized") {
        saw_high_fanout_optimized = true;
        assert(row.mismatch_count == 0);
        assert(row.max_fanout >= 64.0);
        assert(row.router_invoked_ratio == 0.0);
        assert(row.safe_child_router_invoked_ratio == 0.0);
      }
    }
    assert(saw_high_fanout_optimized);
  }

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
  config.enable_ablation_profiles = true;
  config.proximal_oracle_enabled = true;
  config.proximal_oracle_k_values = {1, 2, 4};
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
  optimized_config.path_reuse_enabled = true;
  optimized_config.router_hint_enabled = true;
  optimized_config.router_hint_qgram_q = 3;
  optimized_config.router_hint_minimizer_k = 3;
  optimized_config.router_hint_minimizer_w = 5;
  optimized_config.safe_child_router_enabled = true;
  optimized_config.safe_child_router_min_fanout = 1;
  optimized_config.safe_child_router_max_ratio = 1.0;
  optimized_config.safe_child_router_min_seed_len = 2;
  optimized_config.safe_child_router_mode = "qgram";
  optimized_config.query_planner_enabled = true;
  optimized_config.planner_router_min_fanout = 1;
  optimized_config.planner_safe_child_router_min_fanout = 1;
  auto result = navigamer::run_query_benchmark(
      config, navigamer::HierarchyConfig({12, 6, 2}), build_config,
      optimized_config);
  assert(result.gate_passed);
  assert(result.mismatch_count == 0);
  assert(result.detail_rows.size() == 6 * 7 * (1 + 2));
  assert(!result.summary_rows.empty());
  assert(result.json_summary.find("\"gate_passed\":true") != std::string::npos);
  assert(result.json_summary.find(
             "\"baseline\":{\"mbb_filter_mode\":\"scan\","
             "\"visited_mode\":\"string\","
             "\"graph_view\":\"original\","
             "\"simd_mode\":\"scalar\","
             "\"distance_mode\":\"dp\"") != std::string::npos);
  assert(result.json_summary.find(
             "\"optimized\":{\"mbb_filter_mode\":\"rect\","
             "\"visited_mode\":\"epoch\","
             "\"graph_view\":\"flat\","
             "\"simd_mode\":\"auto\","
             "\"distance_mode\":\"myers\"") != std::string::npos);
  assert(result.json_summary.find("\"router_hint_enabled\":true")
         != std::string::npos);
  assert(result.json_summary.find("\"path_reuse_enabled\":true")
         != std::string::npos);
  assert(result.json_summary.find("\"safe_child_router_enabled\":true")
         != std::string::npos);
  assert(result.json_summary.find("\"query_planner_enabled\":true")
         != std::string::npos);
  assert(result.json_summary.find("\"planner_router_min_fanout\":1")
         != std::string::npos);
  assert(result.json_summary.find("\"enable_ablation_profiles\":true")
         != std::string::npos);
  assert(result.json_summary.find("\"ablation_no_search_qgram\"")
         != std::string::npos);
  assert(result.json_summary.find("\"ablation_no_router_hints\"")
         != std::string::npos);
  assert(result.json_summary.find("\"ablation_no_path_reuse\"")
         != std::string::npos);
  assert(result.json_summary.find("\"ablation_no_safe_child_router\"")
         != std::string::npos);
  assert(result.json_summary.find("\"ablation_no_query_planner\"")
         != std::string::npos);
  assert(result.json_summary.find("\"candidate_set_comparison\":\"unavailable\"")
         != std::string::npos);
  {
    std::ifstream detail(config.detail_tsv_path);
    std::string header;
    std::getline(detail, header);
    assert(header.find("leaf_beacon_scalar_checks") != std::string::npos);
    assert(header.find("leaf_beacon_simd_batches") != std::string::npos);
    assert(header.find("leaf_beacon_simd_fallbacks") != std::string::npos);
    assert(header.find("router_hint_invoked_count") != std::string::npos);
    assert(header.find("router_qgram_ranked_count") != std::string::npos);
    assert(header.find("router_minimizer_ranked_count") != std::string::npos);
    assert(header.find("best_first_invoked_count") != std::string::npos);
    assert(header.find("path_reuse_hit_count") != std::string::npos);
    assert(header.find("near_query_triangle_pruned_count") != std::string::npos);
    assert(header.find("near_query_center_distance_reused_count") !=
           std::string::npos);
    assert(header.find("near_query_bound_fallback_count") != std::string::npos);
    assert(header.find("near_query_direct_verify_count") != std::string::npos);
    assert(header.find("anchor_cache_hit_count") != std::string::npos);
    assert(header.find("safe_child_router_invoked_count") != std::string::npos);
    assert(header.find("safe_child_router_pruned_by_not_candidate_count") !=
           std::string::npos);
    assert(header.find("safe_child_router_exact_pruned_count") !=
           std::string::npos);
    assert(header.find("safe_child_router_center_distance_reused_count") !=
           std::string::npos);
    assert(header.find("child_count_before_router") != std::string::npos);
    assert(header.find("post_mbb_survivor_count") != std::string::npos);
    assert(header.find("safe_router_candidate_count") != std::string::npos);
    assert(header.find("candidate_ratio_to_all_children") != std::string::npos);
    assert(header.find("candidate_ratio_to_post_mbb_survivors") !=
           std::string::npos);
    assert(header.find("children_actually_processed") != std::string::npos);
    assert(header.find("center_checks_saved") != std::string::npos);
    assert(header.find("planner_strategy_router_count") != std::string::npos);
    assert(header.find("planner_decision_ms") != std::string::npos);
    assert(header.find("actual_envelope_k1") != std::string::npos);
    assert(header.find("frontier_oracle_envelope_k2") != std::string::npos);
    assert(header.find("true_path_oracle_envelope_k4") != std::string::npos);
    assert(header.find("global_oracle_envelope_k1") != std::string::npos);
    assert(header.find("random_envelope_k1") != std::string::npos);
    assert(header.find("actual_nearest_anchor_dist") != std::string::npos);
    assert(header.find("global_oracle_gap_vs_actual_k1") != std::string::npos);
    assert(header.find("frontier_total_pushed") != std::string::npos);
    assert(header.find("profile_rank") != std::string::npos);
  }
  {
    std::ifstream summary(config.summary_tsv_path);
    std::string header;
    std::getline(summary, header);
    assert(header.find("avg_leaf_beacon_scalar_checks") != std::string::npos);
    assert(header.find("avg_leaf_beacon_simd_batches") != std::string::npos);
    assert(header.find("avg_leaf_beacon_simd_fallbacks") != std::string::npos);
    assert(header.find("avg_router_hint_invoked_count") != std::string::npos);
    assert(header.find("avg_router_qgram_ranked_count") != std::string::npos);
    assert(header.find("avg_router_minimizer_ranked_count") != std::string::npos);
    assert(header.find("avg_best_first_invoked_count") != std::string::npos);
    assert(header.find("avg_path_reuse_hit_count") != std::string::npos);
    assert(header.find("avg_near_query_triangle_pruned_count") !=
           std::string::npos);
    assert(header.find("avg_near_query_center_distance_reused_count") !=
           std::string::npos);
    assert(header.find("avg_near_query_bound_fallback_count") !=
           std::string::npos);
    assert(header.find("avg_near_query_direct_verify_count") !=
           std::string::npos);
    assert(header.find("center_distance_reduction") != std::string::npos);
    assert(header.find("world_access_reduction") != std::string::npos);
    assert(header.find("p95_speedup") != std::string::npos);
    assert(header.find("avg_anchor_cache_hit_count") != std::string::npos);
    assert(header.find("avg_safe_child_router_invoked_count") !=
           std::string::npos);
    assert(header.find("avg_safe_child_router_pruned_by_not_candidate_count") !=
           std::string::npos);
    assert(header.find("avg_safe_child_router_exact_pruned_count") !=
           std::string::npos);
    assert(header.find("avg_safe_child_router_center_distance_reused_count") !=
           std::string::npos);
    assert(header.find("avg_child_count_before_router") != std::string::npos);
    assert(header.find("avg_post_mbb_survivor_count") != std::string::npos);
    assert(header.find("avg_safe_router_candidate_count") != std::string::npos);
    assert(header.find("avg_candidate_ratio_to_all_children") !=
           std::string::npos);
    assert(header.find("avg_candidate_ratio_to_post_mbb_survivors") !=
           std::string::npos);
    assert(header.find("avg_children_actually_processed") !=
           std::string::npos);
    assert(header.find("avg_center_checks_saved") != std::string::npos);
    assert(header.find("avg_planner_strategy_router_count") !=
           std::string::npos);
    assert(header.find("avg_planner_decision_ms") != std::string::npos);
    assert(header.find("mean_actual_envelope_k1") != std::string::npos);
    assert(header.find("mean_frontier_oracle_envelope_k2") != std::string::npos);
    assert(header.find("mean_true_path_oracle_envelope_k4") != std::string::npos);
    assert(header.find("mean_global_oracle_envelope_k1") != std::string::npos);
    assert(header.find("mean_random_envelope_k1") != std::string::npos);
    assert(header.find("frac_global_oracle_much_better_than_actual_k1") !=
           std::string::npos);
    assert(header.find("avg_frontier_total_pushed") != std::string::npos);
    assert(header.find("cold_avg_speedup_vs_baseline") != std::string::npos);
    assert(header.find("warm_avg_speedup_vs_baseline") != std::string::npos);
    assert(header.find("avg_world_access_ratio_vs_baseline") != std::string::npos);
    assert(header.find("avg_center_distance_ratio_vs_baseline") != std::string::npos);
    assert(header.find("avg_raw_candidate_ratio_vs_baseline") != std::string::npos);
  }
  assert(result.json_summary.find("\"proximal_oracle_enabled\":true")
         != std::string::npos);
  assert(result.json_summary.find("\"proximal_oracle_k_values\":[1,2,4]")
         != std::string::npos);

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
