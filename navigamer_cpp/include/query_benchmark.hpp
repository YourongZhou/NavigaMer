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
  std::string detail_tsv_path;
  std::string summary_tsv_path;
  std::string json_path;
};

struct GeneratedBenchmarkQuery {
  QueryClass query_class;
  BioSequence query;
  std::vector<std::string> brute_force_ids;
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

const char* query_class_name(QueryClass value);
double nearest_rank_percentile(std::vector<double> values, double quantile);
ResultComparison compare_result_ids(std::vector<std::string> baseline,
                                    std::vector<std::string> optimized,
                                    std::vector<std::string> brute_force);
bool comparison_passes_gate(const ResultComparison& comparison);
bool profile_results_equal_brute_force(const ResultComparison& comparison,
                                       const std::string& profile);
std::vector<GeneratedBenchmarkQuery> generate_benchmark_queries(
    const std::vector<std::shared_ptr<BioSequence>>& index_sequences,
    const std::vector<std::shared_ptr<BioSequence>>& unique_sequences,
    int query_length,
    int tolerance,
    unsigned seed,
    size_t queries_per_class);
QueryBenchmarkRunResult run_query_benchmark(
    const QueryBenchmarkConfig& config,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& build_config,
    const SearchConfig& optimized_search_config);

}  // namespace navigamer

#endif  // NAVIGAMER_QUERY_BENCHMARK_HPP
