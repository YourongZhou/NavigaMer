#include "query_benchmark.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace navigamer {
namespace {

struct ExecutionRecord {
  std::string query_id;
  std::string profile;
  QueryClass query_class = QueryClass::RandomRegion;
  std::string sample_kind;
  size_t iteration = 0;
  double latency_ms = 0.0;
  size_t result_count = 0;
  size_t brute_force_result_count = 0;
  bool result_equal = false;
  bool no_fn = false;
  SearchStats stats;
};

struct AggregateRecord {
  std::string query_class;
  std::string profile;
  size_t query_count = 0;
  size_t sample_count = 0;
  size_t result_total = 0;
  size_t equality_failure_count = 0;
  size_t false_negative_count = 0;
  std::vector<double> cold_latencies;
  std::vector<double> warm_latencies;
};

std::vector<std::string> sorted_unique(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<std::string> difference(const std::vector<std::string>& left,
                                    const std::vector<std::string>& right) {
  std::vector<std::string> out;
  std::set_difference(left.begin(), left.end(), right.begin(), right.end(),
                      std::back_inserter(out));
  return out;
}

[[maybe_unused]] double average(const std::vector<double>& values) {
  return values.empty()
             ? 0.0
             : std::accumulate(values.begin(), values.end(), 0.0) /
                   static_cast<double>(values.size());
}

[[maybe_unused]] std::vector<AggregateRecord> aggregate_records(
    const std::vector<ExecutionRecord>& records) {
  std::map<std::pair<std::string, std::string>, AggregateRecord> grouped;
  for (const auto& record : records) {
    for (const std::string& class_name :
         {std::string(query_class_name(record.query_class)), std::string("all")}) {
      auto& aggregate = grouped[{class_name, record.profile}];
      aggregate.query_class = class_name;
      aggregate.profile = record.profile;
      aggregate.sample_count++;
      aggregate.result_total += record.result_count;
      aggregate.equality_failure_count += record.result_equal ? 0 : 1;
      aggregate.false_negative_count += record.no_fn ? 0 : 1;
      if (record.sample_kind == "cold") {
        aggregate.cold_latencies.push_back(record.latency_ms);
      } else if (record.sample_kind == "warm") {
        aggregate.warm_latencies.push_back(record.latency_ms);
      }
    }
  }
  std::vector<AggregateRecord> out;
  for (auto& entry : grouped) out.push_back(std::move(entry.second));
  return out;
}

}  // namespace

const char* query_class_name(QueryClass value) {
  switch (value) {
    case QueryClass::RandomRegion:
      return "random_region";
    case QueryClass::OrdinaryRegion:
      return "ordinary_region";
    case QueryClass::LowComplexityRegion:
      return "low_complexity_region";
    case QueryClass::NoHit:
      return "no_hit";
    case QueryClass::SingleHit:
      return "single_hit";
    case QueryClass::MultiHit:
      return "multi_hit";
  }
  throw std::invalid_argument("unknown query class");
}

double nearest_rank_percentile(std::vector<double> values, double quantile) {
  if (values.empty()) throw std::invalid_argument("percentile sample must not be empty");
  if (quantile < 0.0 || quantile > 1.0) {
    throw std::invalid_argument("percentile quantile must be in [0, 1]");
  }
  std::sort(values.begin(), values.end());
  const size_t rank = quantile == 0.0
                          ? 1
                          : static_cast<size_t>(
                                std::ceil(quantile * static_cast<double>(values.size())));
  return values[rank - 1];
}

ResultComparison compare_result_ids(std::vector<std::string> baseline,
                                    std::vector<std::string> optimized,
                                    std::vector<std::string> brute_force) {
  baseline = sorted_unique(std::move(baseline));
  optimized = sorted_unique(std::move(optimized));
  brute_force = sorted_unique(std::move(brute_force));

  ResultComparison comparison;
  comparison.baseline_equals_optimized = baseline == optimized;
  comparison.baseline_only = difference(baseline, optimized);
  comparison.optimized_only = difference(optimized, baseline);
  comparison.brute_force_missing_from_baseline = difference(brute_force, baseline);
  comparison.brute_force_missing_from_optimized = difference(brute_force, optimized);
  comparison.baseline_no_fn = comparison.brute_force_missing_from_baseline.empty();
  comparison.optimized_no_fn = comparison.brute_force_missing_from_optimized.empty();
  return comparison;
}

}  // namespace navigamer
