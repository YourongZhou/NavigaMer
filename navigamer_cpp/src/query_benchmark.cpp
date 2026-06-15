#include "query_benchmark.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
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

std::vector<std::string> exact_hit_ids(
    const std::string& query,
    const std::vector<std::shared_ptr<BioSequence>>& unique_sequences,
    int tolerance) {
  std::vector<std::string> ids;
  for (const auto& sequence : unique_sequences) {
    if (sequence && compute_distance(query, sequence->seq) <= tolerance) {
      ids.push_back(sequence->id);
    }
  }
  return sorted_unique(std::move(ids));
}

double shannon_entropy(const std::string& sequence) {
  std::array<size_t, 256> counts{};
  for (unsigned char c : sequence) counts[c]++;
  double entropy = 0.0;
  for (size_t count : counts) {
    if (count == 0) continue;
    const double p = static_cast<double>(count) /
                     static_cast<double>(sequence.size());
    entropy -= p * std::log2(p);
  }
  return entropy;
}

std::string random_dna(size_t length, std::mt19937& generator) {
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> pick(0, 3);
  std::string out(length, 'A');
  for (char& base : out) base = bases[pick(generator)];
  return out;
}

std::string mutate_substitutions(const std::string& input, size_t edit_count,
                                 std::mt19937& generator) {
  if (input.empty() || edit_count == 0) return input;
  std::string out = input;
  std::vector<size_t> positions(out.size());
  std::iota(positions.begin(), positions.end(), 0);
  std::shuffle(positions.begin(), positions.end(), generator);
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> pick(0, 3);
  for (size_t i = 0; i < std::min(edit_count, positions.size()); ++i) {
    char replacement = out[positions[i]];
    while (replacement == out[positions[i]]) replacement = bases[pick(generator)];
    out[positions[i]] = replacement;
  }
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

std::vector<GeneratedBenchmarkQuery> generate_benchmark_queries(
    const std::vector<std::shared_ptr<BioSequence>>& index_sequences,
    const std::vector<std::shared_ptr<BioSequence>>& unique_sequences,
    int query_length,
    int tolerance,
    unsigned seed,
    size_t queries_per_class) {
  if (query_length <= 0) throw std::invalid_argument("query length must be positive");
  if (tolerance < 0) throw std::invalid_argument("tolerance must be non-negative");
  if (queries_per_class == 0) {
    throw std::invalid_argument("queries per class must be positive");
  }
  if (index_sequences.empty() || unique_sequences.empty()) {
    throw std::invalid_argument("benchmark sequence sets must not be empty");
  }

  std::vector<std::string> windows;
  for (const auto& sequence : index_sequences) {
    if (!sequence || sequence->seq.size() < static_cast<size_t>(query_length)) {
      throw std::invalid_argument("index sequences must be at least query length");
    }
    for (size_t start = 0;
         start + static_cast<size_t>(query_length) <= sequence->seq.size();
         ++start) {
      windows.push_back(sequence->seq.substr(start, static_cast<size_t>(query_length)));
    }
  }

  struct ClassifiedWindow {
    std::string sequence;
    std::vector<std::string> hit_ids;
    double entropy = 0.0;
  };
  std::vector<ClassifiedWindow> classified;
  classified.reserve(windows.size());
  for (const auto& window : windows) {
    classified.push_back(
        {window, exact_hit_ids(window, unique_sequences, tolerance),
         shannon_entropy(window)});
  }

  std::mt19937 generator(seed);
  std::vector<size_t> shuffled_indices(classified.size());
  std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
  std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), generator);

  std::vector<size_t> entropy_desc = shuffled_indices;
  std::stable_sort(entropy_desc.begin(), entropy_desc.end(),
                   [&](size_t left, size_t right) {
                     return classified[left].entropy > classified[right].entropy;
                   });
  std::vector<size_t> entropy_asc = shuffled_indices;
  std::stable_sort(entropy_asc.begin(), entropy_asc.end(),
                   [&](size_t left, size_t right) {
                     return classified[left].entropy < classified[right].entropy;
                   });

  std::vector<GeneratedBenchmarkQuery> out;
  size_t query_id = 0;
  auto append = [&](QueryClass kind, const std::string& sequence,
                    std::vector<std::string> hit_ids) {
    out.push_back({kind,
                   BioSequence("query_benchmark_" + std::to_string(query_id++),
                               sequence),
                   sorted_unique(std::move(hit_ids))});
  };
  auto append_from_order = [&](QueryClass kind, const std::vector<size_t>& order,
                               bool reject_multi) {
    size_t added = 0;
    for (size_t index : order) {
      if (reject_multi && classified[index].hit_ids.size() >= 2) continue;
      append(kind, classified[index].sequence, classified[index].hit_ids);
      if (++added == queries_per_class) return;
    }
    throw std::runtime_error(
        "unable to generate query class " + std::string(query_class_name(kind)) +
        " after 4096 deterministic attempts");
  };

  append_from_order(QueryClass::RandomRegion, shuffled_indices, false);
  append_from_order(QueryClass::OrdinaryRegion, entropy_desc, true);
  append_from_order(QueryClass::LowComplexityRegion, entropy_asc, false);

  auto append_by_hit_count = [&](QueryClass kind, size_t minimum_hits,
                                 size_t maximum_hits) {
    for (size_t requested = 0; requested < queries_per_class; ++requested) {
      bool found = false;
      for (size_t attempt = 0; attempt < 4096; ++attempt) {
        std::string candidate;
        if (kind == QueryClass::NoHit) {
          candidate = random_dna(static_cast<size_t>(query_length), generator);
        } else {
          const auto& source =
              classified[shuffled_indices[attempt % shuffled_indices.size()]].sequence;
          const size_t edit_count =
              tolerance <= 0 ? 0 : attempt % (static_cast<size_t>(tolerance) + 1);
          candidate = mutate_substitutions(source, edit_count, generator);
        }
        auto hit_ids = exact_hit_ids(candidate, unique_sequences, tolerance);
        if (hit_ids.size() >= minimum_hits && hit_ids.size() <= maximum_hits) {
          append(kind, candidate, std::move(hit_ids));
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error(
            "unable to generate query class " +
            std::string(query_class_name(kind)) +
            " after 4096 deterministic attempts");
      }
    }
  };
  append_by_hit_count(QueryClass::NoHit, 0, 0);
  append_by_hit_count(QueryClass::SingleHit, 1, 1);
  append_by_hit_count(QueryClass::MultiHit, 2,
                      std::numeric_limits<size_t>::max());
  return out;
}

}  // namespace navigamer
