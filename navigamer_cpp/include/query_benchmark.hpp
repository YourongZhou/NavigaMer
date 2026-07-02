#ifndef NAVIGAMER_QUERY_BENCHMARK_HPP
#define NAVIGAMER_QUERY_BENCHMARK_HPP

#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"
#include <memory>
#include <string>
#include <vector>

namespace navigamer {

enum class QueryClass {
  RandomRegion,
  OrdinaryRegion,
  LowComplexityRegion,
  NoHit,
  SingleHit,
  MultiHit,
};

struct QueryBenchmarkConfig {
  std::string ref_input;
  size_t reference_subset_length = 0;
  int window_length = 200;
  int stride = 1;
  int query_length = 200;
  int tolerance = 2;
  unsigned seed = 42;
  int threads = 1;
  size_t queries_per_class = 1;
  size_t warmup_iterations = 2;
  size_t measured_iterations = 10;
  size_t cold_cache_bytes = 256ULL * 1024ULL * 1024ULL;
  bool enable_ablation_profiles = false;
  bool proximal_oracle_enabled = false;
  std::vector<size_t> proximal_oracle_k_values = {1, 2, 4};
  std::string detail_tsv_path;
  std::string summary_tsv_path;
  std::string json_path;
};

struct GeneratedBenchmarkQuery {
  QueryClass query_class;
  BioSequence query;
  std::vector<std::string> brute_force_ids;
};

struct LocalityBenchmarkQuery {
  BioSequence query;
  size_t source_pos = 0;
};

struct LocalityBenchmarkQuerySets {
  std::vector<LocalityBenchmarkQuery> same_template;
  std::vector<LocalityBenchmarkQuery> nearby_windows;
  std::vector<LocalityBenchmarkQuery> random_windows;
  std::vector<LocalityBenchmarkQuery> repeat;
  std::vector<LocalityBenchmarkQuery> batch_locality;
  std::vector<LocalityBenchmarkQuery> oracle;
  std::vector<LocalityBenchmarkQuery> real_dup_1x;
  std::vector<LocalityBenchmarkQuery> real_dup_4x;
  std::vector<LocalityBenchmarkQuery> real_dup_16x;
  std::vector<LocalityBenchmarkQuery> source_sorted_stride1;
  std::vector<LocalityBenchmarkQuery> source_sorted_mutated_tau5;
  std::vector<LocalityBenchmarkQuery> source_sorted_mutated_tau8;
};

struct LocalityBenchmarkConfig {
  std::string index_path;
  std::string ref_input;
  size_t query_count = 256;
  int query_length = 250;
  int tolerance = 5;
  int edits = 5;
  unsigned seed = 42;
  std::vector<std::string> profiles = {"baseline", "path_reuse", "optimized"};
  std::vector<std::string> datasets = {
      "same_template", "nearby_windows", "random_windows"};
  std::vector<std::string> scenarios;
  std::vector<std::string> batch_schedules = {"original"};
  std::string out_tsv_path;
  std::string query_fastq_out_path;
};

struct LocalityBenchmarkRow {
  std::string dataset;
  std::string profile;
  std::string batch_schedule_mode = "original";
  size_t query_count = 0;
  size_t result_total = 0;
  size_t fn_count = 0;
  size_t mismatch_count = 0;
  double load_ms = 0.0;
  double engine_init_ms = 0.0;
  double query_wall_ms = 0.0;
  double mean_query_ms = 0.0;
  double p50_query_ms = 0.0;
  double p95_query_ms = 0.0;
  double mean_world_access = 0.0;
  double mean_center_distance = 0.0;
  double mean_leaf_verify = 0.0;
  double mean_fanout = 0.0;
  double p50_fanout = 0.0;
  double p95_fanout = 0.0;
  double max_fanout = 0.0;
  double router_invoked_ratio = 0.0;
  double local_router_invoked_ratio = 0.0;
  double safe_child_router_invoked_ratio = 0.0;
  double path_reuse_hit_ratio = 0.0;
  size_t unique_query_count = 0;
  size_t duplicate_group_count = 0;
  double duplicate_ratio = 0.0;
  size_t verified_result_cache_hit_count = 0;
  size_t near_query_reuse_hit_count = 0;
  size_t near_query_triangle_pruned_count = 0;
	  size_t near_query_center_distance_reused_count = 0;
	  size_t near_query_bound_fallback_count = 0;
	  size_t near_query_direct_verify_count = 0;
	  size_t near_query_leaf_triangle_pruned_count = 0;
	  size_t near_query_leaf_distance_reused_count = 0;
	  size_t near_query_leaf_bound_fallback_count = 0;
	  double center_distance_reduction = 0.0;
  double world_access_reduction = 0.0;
  double p95_speedup = 1.0;
  double mean_neighbor_edit_distance = 0.0;
  double p95_neighbor_edit_distance = 0.0;
  double mean_neighbor_qgram_jaccard = 0.0;
  size_t anchor_cache_hit_count = 0;
  size_t child_shortlist_cache_hit_count = 0;
  size_t safe_child_candidate_cache_hit_count = 0;
  size_t productive_world_reuse_hit_count = 0;
  double mean_path_reuse_hits = 0.0;
  double mean_anchor_cache_hits = 0.0;
  double mean_child_shortlist_hits = 0.0;
  double mean_router_invoked = 0.0;
  double mean_router_hits = 0.0;
  double mean_local_router_invoked = 0.0;
  double mean_local_router_shortlisted = 0.0;
  double mean_best_first_invoked = 0.0;
  double mean_best_first_reordered = 0.0;
  double mean_child_count_before_router = 0.0;
  double mean_post_mbb_survivor_count = 0.0;
  double mean_safe_router_candidate_count = 0.0;
  double mean_candidate_ratio_to_all_children = 0.0;
  double mean_candidate_ratio_to_post_mbb_survivors = 0.0;
  double mean_children_actually_processed = 0.0;
  double mean_center_checks_saved = 0.0;
};

struct LocalityBenchmarkRunResult {
  bool gate_passed = false;
  double load_ms = 0.0;
  std::vector<LocalityBenchmarkRow> rows;
};

struct ResultComparison {
  bool baseline_equals_optimized = false;
  bool baseline_equals_brute_force = false;
  bool optimized_equals_brute_force = false;
  bool baseline_no_fn = false;
  bool optimized_no_fn = false;
  std::vector<std::string> baseline_only;
  std::vector<std::string> optimized_only;
  std::vector<std::string> baseline_extra_vs_brute_force;
  std::vector<std::string> optimized_extra_vs_brute_force;
  std::vector<std::string> brute_force_missing_from_baseline;
  std::vector<std::string> brute_force_missing_from_optimized;
};

struct QueryBenchmarkRunResult {
  bool gate_passed = false;
  size_t mismatch_count = 0;
  std::vector<std::vector<std::string>> detail_rows;
  std::vector<std::vector<std::string>> summary_rows;
  std::string json_summary;
};

struct ProximalAnchorSetDiagnostics {
  double nearest_anchor_dist = -1.0;
  double random_nearest_anchor_dist = -1.0;
  std::vector<double> oracle_envelope_by_k;
  std::vector<double> random_envelope_by_k;
};

const char* query_class_name(QueryClass value);
double nearest_rank_percentile(std::vector<double> values, double quantile);
ResultComparison compare_result_ids(std::vector<std::string> baseline,
                                    std::vector<std::string> optimized,
                                    std::vector<std::string> brute_force);
bool comparison_passes_gate(const ResultComparison& comparison);
bool profile_results_equal_brute_force(const ResultComparison& comparison,
                                       const std::string& profile);
ProximalAnchorSetDiagnostics compute_proximal_anchor_set_diagnostics(
    const std::string& query,
    const std::vector<std::string>& oracle_anchors,
    const std::vector<std::string>& random_anchors,
    const std::vector<size_t>& k_values);
std::vector<GeneratedBenchmarkQuery> generate_benchmark_queries(
    const std::vector<std::shared_ptr<BioSequence>>& index_sequences,
    const std::vector<std::shared_ptr<BioSequence>>& unique_sequences,
    int query_length,
    int tolerance,
    unsigned seed,
    size_t queries_per_class);
LocalityBenchmarkQuerySets generate_locality_benchmark_queries(
    const std::string& reference,
    size_t query_count,
    int query_length,
    int edits,
    unsigned seed);
LocalityBenchmarkRunResult run_persisted_locality_benchmark(
    const LocalityBenchmarkConfig& config);
void write_locality_report_outputs(
    const LocalityBenchmarkRunResult& result,
    const std::string& json_path,
    const std::string& markdown_path);
QueryBenchmarkRunResult run_query_benchmark(
    const QueryBenchmarkConfig& config,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& build_config,
    const SearchConfig& optimized_search_config);

}  // namespace navigamer

#endif  // NAVIGAMER_QUERY_BENCHMARK_HPP
