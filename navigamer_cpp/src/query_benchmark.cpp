#include "query_benchmark.hpp"
#include "io_utils.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <sys/resource.h>
#include <omp.h>

namespace navigamer {
namespace {

struct ExecutionRecord {
  std::string query_id;
  std::string profile;
  QueryClass query_class = QueryClass::RandomRegion;
  std::string sample_kind;
  std::string first_profile;
  size_t iteration = 0;
  double latency_ms = 0.0;
  size_t result_count = 0;
  size_t brute_force_result_count = 0;
  bool result_equal = false;
  bool no_fn = false;
  std::vector<std::string> result_ids;
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
  size_t world_access_count = 0;
  size_t node_access_count = 0;
  size_t edge_access_count = 0;
  size_t mbb_check_count = 0;
  size_t mbb_surviving_child_count = 0;
  size_t search_qgram_checks = 0;
  size_t center_exact_distance_call_count = 0;
  size_t leaf_beacon_check_count = 0;
  size_t leaf_exact_distance_call_count = 0;
  size_t visited_check_count = 0;
  size_t visited_hit_count = 0;
  size_t candidate_count = 0;
  size_t candidate_verify_count = 0;
  std::vector<double> cold_latencies;
  std::vector<double> warm_latencies;
};

struct MemorySnapshot {
  bool current_available = false;
  size_t current_rss_kb = 0;
  bool peak_available = false;
  size_t peak_rss_kb = 0;
};

struct MismatchDiagnostic {
  std::string query_id;
  bool repeated_results_equal = false;
  ResultComparison comparison;
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

double average(const std::vector<double>& values) {
  return values.empty()
             ? 0.0
             : std::accumulate(values.begin(), values.end(), 0.0) /
                   static_cast<double>(values.size());
}

std::vector<AggregateRecord> aggregate_records(
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
      aggregate.world_access_count += record.stats.world_access_count;
      aggregate.node_access_count += record.stats.node_access_count;
      aggregate.edge_access_count += record.stats.edge_access_count;
      aggregate.mbb_check_count += record.stats.mbb_check_count;
      aggregate.mbb_surviving_child_count += record.stats.mbb_surviving_child_count;
      aggregate.search_qgram_checks += record.stats.search_qgram_checks;
      aggregate.center_exact_distance_call_count +=
          record.stats.center_exact_distance_call_count;
      aggregate.leaf_beacon_check_count += record.stats.leaf_beacon_check_count;
      aggregate.leaf_exact_distance_call_count +=
          record.stats.leaf_exact_distance_call_count;
      aggregate.visited_check_count += record.stats.visited_check_count;
      aggregate.visited_hit_count += record.stats.visited_hit_count;
      aggregate.candidate_count += record.stats.candidate_count;
      aggregate.candidate_verify_count += record.stats.candidate_verify_count;
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

std::string format_double(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << value;
  return out.str();
}

std::string bool_string(bool value) { return value ? "true" : "false"; }

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

void append_json_string_array(std::ostringstream& out,
                              const std::vector<std::string>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ",";
    out << "\"" << json_escape(values[i]) << "\"";
  }
  out << "]";
}

std::vector<std::shared_ptr<BioSequence>> build_reference_windows(
    const std::string& ref_id, const std::string& reference, int window_length,
    int stride) {
  std::vector<std::shared_ptr<BioSequence>> out;
  for (int start = 0; start + window_length <= static_cast<int>(reference.size());
       start += stride) {
    auto sequence = std::make_shared<BioSequence>(
        "ref_" + std::to_string(start),
        reference.substr(static_cast<size_t>(start),
                         static_cast<size_t>(window_length)));
    sequence->add_occurrence(ref_id, start, start + window_length, "+");
    out.push_back(std::move(sequence));
  }
  return out;
}

std::vector<std::string> result_ids(
    const std::vector<std::shared_ptr<BioSequence>>& results) {
  std::vector<std::string> ids;
  ids.reserve(results.size());
  for (const auto& result : results) {
    if (result) ids.push_back(result->id);
  }
  return sorted_unique(std::move(ids));
}

MemorySnapshot memory_snapshot() {
  MemorySnapshot snapshot;
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      status >> snapshot.current_rss_kb;
      snapshot.current_available = true;
      break;
    }
    std::string rest;
    std::getline(status, rest);
  }
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    snapshot.peak_available = true;
    snapshot.peak_rss_kb = static_cast<size_t>(usage.ru_maxrss);
  }
  return snapshot;
}

void append_memory_json(std::ostringstream& out, const char* name,
                        const MemorySnapshot& snapshot) {
  out << "\"" << name << "\":{";
  if (snapshot.current_available) {
    out << "\"current_rss_kb\":" << snapshot.current_rss_kb;
  } else {
    out << "\"current_rss_kb\":\"unavailable\"";
  }
  out << ",";
  if (snapshot.peak_available) {
    out << "\"peak_rss_kb\":" << snapshot.peak_rss_kb;
  } else {
    out << "\"peak_rss_kb\":\"unavailable\"";
  }
  out << "}";
}

double percentile_or_zero(const std::vector<double>& values, double quantile) {
  return values.empty() ? 0.0 : nearest_rank_percentile(values, quantile);
}

double average_counter(size_t value, size_t samples) {
  return samples == 0 ? 0.0 : static_cast<double>(value) /
                                  static_cast<double>(samples);
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
  comparison.baseline_equals_brute_force = baseline == brute_force;
  comparison.optimized_equals_brute_force = optimized == brute_force;
  comparison.baseline_only = difference(baseline, optimized);
  comparison.optimized_only = difference(optimized, baseline);
  comparison.baseline_extra_vs_brute_force = difference(baseline, brute_force);
  comparison.optimized_extra_vs_brute_force = difference(optimized, brute_force);
  comparison.brute_force_missing_from_baseline = difference(brute_force, baseline);
  comparison.brute_force_missing_from_optimized = difference(brute_force, optimized);
  comparison.baseline_no_fn = comparison.brute_force_missing_from_baseline.empty();
  comparison.optimized_no_fn = comparison.brute_force_missing_from_optimized.empty();
  return comparison;
}

bool comparison_passes_gate(const ResultComparison& comparison) {
  return comparison.baseline_equals_optimized &&
         comparison.baseline_equals_brute_force &&
         comparison.optimized_equals_brute_force && comparison.baseline_no_fn &&
         comparison.optimized_no_fn;
}

bool profile_results_equal_brute_force(const ResultComparison& comparison,
                                       const std::string& profile) {
  if (profile == "baseline") return comparison.baseline_equals_brute_force;
  if (profile == "optimized") return comparison.optimized_equals_brute_force;
  throw std::invalid_argument("profile must be baseline or optimized");
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

QueryBenchmarkRunResult run_query_benchmark(
    const QueryBenchmarkConfig& config,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& build_config,
    const SearchConfig& optimized_search_config) {
  if (config.window_length <= 0 || config.query_length <= 0 ||
      config.stride <= 0 || config.threads <= 0 ||
      config.queries_per_class == 0 || config.measured_iterations == 0) {
    throw std::invalid_argument(
        "window/query length, stride, threads, queries per class, and "
        "measured iterations must be positive");
  }
  if (config.tolerance < 0) {
    throw std::invalid_argument("tolerance must be non-negative");
  }
  if (config.detail_tsv_path.empty() || config.summary_tsv_path.empty() ||
      config.json_path.empty()) {
    throw std::invalid_argument("all query benchmark output paths are required");
  }

  omp_set_num_threads(config.threads);
  auto [ref_id, loaded_reference] = load_reference(config.ref_input);
  std::string reference = loaded_reference;
  if (config.reference_subset_length > 0 &&
      reference.size() > config.reference_subset_length) {
    reference.resize(config.reference_subset_length);
  }
  if (reference.size() < static_cast<size_t>(config.window_length)) {
    throw std::invalid_argument("reference is shorter than benchmark window length");
  }
  auto index_sequences = build_reference_windows(
      ref_id, reference, config.window_length, config.stride);
  if (index_sequences.empty()) {
    throw std::runtime_error("query benchmark could not generate index windows");
  }

  const MemorySnapshot before_build = memory_snapshot();
  const auto build_start = std::chrono::steady_clock::now();
  BioGeometryIndexBuilder builder(hierarchy, build_config);
  builder.build(index_sequences);
  const auto build_end = std::chrono::steady_clock::now();
  const double build_duration_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start).count();
  const MemorySnapshot after_build = memory_snapshot();
  const auto build_stats = builder.get_statistics();

  std::vector<std::shared_ptr<BioSequence>> unique_sequences;
  unique_sequences.reserve(builder.unique_sequences.size());
  for (const auto& entry : builder.unique_sequences) {
    unique_sequences.push_back(entry.second);
  }
  auto queries = generate_benchmark_queries(
      index_sequences, unique_sequences, config.query_length, config.tolerance,
      config.seed, config.queries_per_class);
  std::map<std::string, size_t> generation_counts;
  for (const auto& query : queries) {
    generation_counts[query_class_name(query.query_class)]++;
  }

  SearchConfig baseline_config;
  baseline_config.mbb_filter_mode = MBBFilterMode::Scan;
  baseline_config.search_qgram_prefilter = false;
  BioGeometrySearchEngine baseline(builder, baseline_config);
  BioGeometrySearchEngine optimized(builder, optimized_search_config);

  std::vector<uint8_t> eviction_buffer(config.cold_cache_bytes, 0);
  volatile uint64_t eviction_checksum = 0;
  auto evict_best_effort = [&]() {
    for (size_t offset = 0; offset < eviction_buffer.size(); offset += 64) {
      eviction_buffer[offset] =
          static_cast<uint8_t>(eviction_buffer[offset] + 1);
      eviction_checksum += eviction_buffer[offset];
    }
  };

  std::vector<ExecutionRecord> records;
  std::vector<MismatchDiagnostic> mismatch_diagnostics;
  QueryBenchmarkRunResult result;
  auto execute = [&](BioGeometrySearchEngine& engine,
                     const GeneratedBenchmarkQuery& generated,
                     const std::string& profile, const std::string& sample_kind,
                     size_t iteration, const std::string& first_profile,
                     bool timed, bool cold) {
    if (cold) evict_best_effort();
    const auto start = std::chrono::steady_clock::now();
    auto [hits, stats] =
        engine.search_adaptive(generated.query, config.tolerance);
    const auto end = std::chrono::steady_clock::now();
    ExecutionRecord record;
    record.query_id = generated.query.id;
    record.profile = profile;
    record.query_class = generated.query_class;
    record.sample_kind = sample_kind;
    record.first_profile = first_profile;
    record.iteration = iteration;
    record.latency_ms =
        timed ? std::chrono::duration<double, std::milli>(end - start).count()
              : 0.0;
    record.result_ids = result_ids(hits);
    record.result_count = record.result_ids.size();
    record.brute_force_result_count = generated.brute_force_ids.size();
    record.stats = std::move(stats);
    return record;
  };

  for (size_t query_index = 0; query_index < queries.size(); ++query_index) {
    const auto& generated = queries[query_index];
    const std::string first_profile =
        query_index % 2 == 0 ? "baseline" : "optimized";
    const std::array<std::string, 2> order =
        query_index % 2 == 0
            ? std::array<std::string, 2>{"baseline", "optimized"}
            : std::array<std::string, 2>{"optimized", "baseline"};
    const size_t query_record_start = records.size();
    std::map<std::string, std::vector<std::string>> canonical_ids;
    bool repeated_results_equal = true;

    for (const auto& profile : order) {
      BioGeometrySearchEngine& engine =
          profile == "baseline" ? baseline : optimized;
      auto cold = execute(engine, generated, profile, "cold", 0, first_profile,
                          true, true);
      canonical_ids[profile] = cold.result_ids;
      records.push_back(std::move(cold));

      for (size_t iteration = 0; iteration < config.warmup_iterations;
           ++iteration) {
        auto warmup = execute(engine, generated, profile, "warmup", iteration,
                              first_profile, false, false);
        repeated_results_equal &=
            warmup.result_ids == canonical_ids[profile];
      }
      for (size_t iteration = 0; iteration < config.measured_iterations;
           ++iteration) {
        auto warm = execute(engine, generated, profile, "warm", iteration,
                            first_profile, true, false);
        repeated_results_equal &= warm.result_ids == canonical_ids[profile];
        records.push_back(std::move(warm));
      }
    }

    const ResultComparison comparison = compare_result_ids(
        canonical_ids["baseline"], canonical_ids["optimized"],
        generated.brute_force_ids);
    const bool query_gate_passed =
        repeated_results_equal && comparison_passes_gate(comparison);
    if (!query_gate_passed) {
      result.mismatch_count++;
      mismatch_diagnostics.push_back(
          {generated.query.id, repeated_results_equal, comparison});
    }
    for (size_t i = query_record_start; i < records.size(); ++i) {
      records[i].result_equal =
          repeated_results_equal && comparison.baseline_equals_optimized &&
          profile_results_equal_brute_force(comparison, records[i].profile);
      records[i].no_fn =
          records[i].profile == "baseline" ? comparison.baseline_no_fn
                                             : comparison.optimized_no_fn;
    }
  }
  static_cast<void>(eviction_checksum);
  result.gate_passed = result.mismatch_count == 0;
  const MemorySnapshot after_benchmark = memory_snapshot();

  const std::vector<std::string> detail_columns = {
      "query_id", "query_class", "profile", "sample_kind", "iteration",
      "first_profile", "latency_ms", "result_count",
      "brute_force_result_count", "result_equal", "no_fn",
      "world_access_count", "node_access_count", "edge_access_count",
      "mbb_checks", "mbb_survivors", "qgram_checks",
      "center_exact_distance_calls", "leaf_beacon_checks",
      "leaf_exact_distance_calls", "visited_checks", "visited_hits",
      "candidate_count", "verified_candidate_count"};
  for (const auto& record : records) {
    result.detail_rows.push_back({
        record.query_id,
        query_class_name(record.query_class),
        record.profile,
        record.sample_kind,
        std::to_string(record.iteration),
        record.first_profile,
        format_double(record.latency_ms),
        std::to_string(record.result_count),
        std::to_string(record.brute_force_result_count),
        bool_string(record.result_equal),
        bool_string(record.no_fn),
        std::to_string(record.stats.world_access_count),
        std::to_string(record.stats.node_access_count),
        std::to_string(record.stats.edge_access_count),
        std::to_string(record.stats.mbb_check_count),
        std::to_string(record.stats.mbb_surviving_child_count),
        std::to_string(record.stats.search_qgram_checks),
        std::to_string(record.stats.center_exact_distance_call_count),
        std::to_string(record.stats.leaf_beacon_check_count),
        std::to_string(record.stats.leaf_exact_distance_call_count),
        std::to_string(record.stats.visited_check_count),
        std::to_string(record.stats.visited_hit_count),
        std::to_string(record.stats.candidate_count),
        std::to_string(record.stats.candidate_verify_count),
    });
  }

  const std::vector<std::string> summary_columns = {
      "query_class", "profile", "query_count", "sample_count", "result_total",
      "equality_failure_count", "false_negative_count", "cold_avg_ms",
      "cold_p50_ms", "cold_p95_ms", "cold_p99_ms", "warm_avg_ms",
      "warm_p50_ms", "warm_p95_ms", "warm_p99_ms", "avg_world_access_count",
      "avg_node_access_count", "avg_edge_access_count", "avg_mbb_checks",
      "avg_mbb_survivors", "avg_qgram_checks",
      "avg_center_exact_distance_calls", "avg_leaf_beacon_checks",
      "avg_leaf_exact_distance_calls", "avg_visited_checks",
      "avg_visited_hits", "avg_candidate_count",
      "avg_verified_candidate_count"};
  auto aggregates = aggregate_records(records);
  for (const auto& aggregate : aggregates) {
    const size_t sample_count = aggregate.sample_count;
    const size_t samples_per_query = 1 + config.measured_iterations;
    result.summary_rows.push_back({
        aggregate.query_class,
        aggregate.profile,
        std::to_string(sample_count / samples_per_query),
        std::to_string(sample_count),
        std::to_string(aggregate.result_total),
        std::to_string(aggregate.equality_failure_count),
        std::to_string(aggregate.false_negative_count),
        format_double(average(aggregate.cold_latencies)),
        format_double(percentile_or_zero(aggregate.cold_latencies, 0.50)),
        format_double(percentile_or_zero(aggregate.cold_latencies, 0.95)),
        format_double(percentile_or_zero(aggregate.cold_latencies, 0.99)),
        format_double(average(aggregate.warm_latencies)),
        format_double(percentile_or_zero(aggregate.warm_latencies, 0.50)),
        format_double(percentile_or_zero(aggregate.warm_latencies, 0.95)),
        format_double(percentile_or_zero(aggregate.warm_latencies, 0.99)),
        format_double(average_counter(aggregate.world_access_count, sample_count)),
        format_double(average_counter(aggregate.node_access_count, sample_count)),
        format_double(average_counter(aggregate.edge_access_count, sample_count)),
        format_double(average_counter(aggregate.mbb_check_count, sample_count)),
        format_double(average_counter(aggregate.mbb_surviving_child_count, sample_count)),
        format_double(average_counter(aggregate.search_qgram_checks, sample_count)),
        format_double(average_counter(aggregate.center_exact_distance_call_count, sample_count)),
        format_double(average_counter(aggregate.leaf_beacon_check_count, sample_count)),
        format_double(average_counter(aggregate.leaf_exact_distance_call_count, sample_count)),
        format_double(average_counter(aggregate.visited_check_count, sample_count)),
        format_double(average_counter(aggregate.visited_hit_count, sample_count)),
        format_double(average_counter(aggregate.candidate_count, sample_count)),
        format_double(average_counter(aggregate.candidate_verify_count, sample_count)),
    });
  }

  std::ostringstream json;
  json << "{\"schema_version\":1,\"configuration\":{"
       << "\"ref_input\":\"" << json_escape(config.ref_input) << "\","
       << "\"reference_subset_length\":" << config.reference_subset_length << ","
       << "\"window_length\":" << config.window_length << ","
       << "\"stride\":" << config.stride << ","
       << "\"query_length\":" << config.query_length << ","
       << "\"tolerance\":" << config.tolerance << ","
       << "\"seed\":" << config.seed << ","
       << "\"threads\":" << config.threads << ","
       << "\"queries_per_class\":" << config.queries_per_class << ","
       << "\"warmup_iterations\":" << config.warmup_iterations << ","
       << "\"measured_iterations\":" << config.measured_iterations << ","
       << "\"cold_cache_bytes\":" << config.cold_cache_bytes << ","
       << "\"detail_tsv_path\":\"" << json_escape(config.detail_tsv_path) << "\","
       << "\"summary_tsv_path\":\"" << json_escape(config.summary_tsv_path) << "\","
       << "\"json_path\":\"" << json_escape(config.json_path) << "\"},"
       << "\"build\":{\"duration_ms\":" << format_double(build_duration_ms)
       << ",\"added_sequences\":" << build_stats.added_sequences
       << ",\"unique_sequences\":" << build_stats.unique_sequences
       << ",\"deduplicated\":" << build_stats.deduplicated
       << ",\"created_auxiliary_nodes\":" << build_stats.created_auxiliary_nodes
       << ",\"compression_ratio\":" << format_double(build_stats.compression_ratio)
       << ",\"dag_redundancy\":" << format_double(build_stats.dag_redundancy)
       << ",\"phase2_total_possible_pairs\":"
       << build_stats.phase2_total_possible_pairs
       << ",\"phase2_candidate_pairs\":" << build_stats.phase2_candidate_pairs
       << ",\"phase2_exact_distance_calls\":"
       << build_stats.phase2_exact_distance_calls
       << ",\"phase2_edges_added\":" << build_stats.phase2_edges_added
       << ",\"total_possible_leaf_pairs\":"
       << build_stats.total_possible_leaf_pairs
       << ",\"leaf_candidate_pairs\":" << build_stats.leaf_candidate_pairs
       << ",\"leaf_exact_distance_calls\":"
       << build_stats.leaf_exact_distance_calls
       << ",\"leaf_attachments_added\":" << build_stats.leaf_attachments_added
       << "},"
       << "\"profiles\":{\"baseline\":{\"mbb_filter_mode\":\"scan\","
       << "\"search_qgram_prefilter\":false},\"optimized\":{"
       << "\"mbb_filter_mode\":\""
       << mbb_filter_mode_name(optimized_search_config.mbb_filter_mode) << "\","
       << "\"search_qgram_prefilter\":"
       << bool_string(optimized_search_config.search_qgram_prefilter) << ","
       << "\"search_qgram_q\":" << optimized_search_config.search_qgram_q << "}},"
       << "\"generation\":{\"query_count\":" << queries.size()
       << ",\"counts\":{";
  size_t generation_index = 0;
  for (const auto& entry : generation_counts) {
    if (generation_index++) json << ",";
    json << "\"" << json_escape(entry.first) << "\":" << entry.second;
  }
  json << "}},\"aggregate_rows\":[";
  for (size_t row_index = 0; row_index < result.summary_rows.size(); ++row_index) {
    if (row_index) json << ",";
    json << "[";
    for (size_t column_index = 0;
         column_index < result.summary_rows[row_index].size(); ++column_index) {
      if (column_index) json << ",";
      json << "\"" << json_escape(result.summary_rows[row_index][column_index])
           << "\"";
    }
    json << "]";
  }
  json << "],"
       << "\"mismatch_count\":" << result.mismatch_count << ","
       << "\"mismatch_queries\":[";
  for (size_t i = 0; i < mismatch_diagnostics.size(); ++i) {
    if (i) json << ",";
    json << "\"" << json_escape(mismatch_diagnostics[i].query_id) << "\"";
  }
  json << "],\"mismatch_diagnostics\":[";
  for (size_t i = 0; i < mismatch_diagnostics.size(); ++i) {
    if (i) json << ",";
    const auto& diagnostic = mismatch_diagnostics[i];
    const auto& comparison = diagnostic.comparison;
    json << "{\"query_id\":\"" << json_escape(diagnostic.query_id) << "\","
         << "\"repeated_results_equal\":"
         << bool_string(diagnostic.repeated_results_equal) << ","
         << "\"baseline_equals_optimized\":"
         << bool_string(comparison.baseline_equals_optimized) << ","
         << "\"baseline_equals_brute_force\":"
         << bool_string(comparison.baseline_equals_brute_force) << ","
         << "\"optimized_equals_brute_force\":"
         << bool_string(comparison.optimized_equals_brute_force) << ","
         << "\"baseline_no_fn\":" << bool_string(comparison.baseline_no_fn)
         << ",\"optimized_no_fn\":"
         << bool_string(comparison.optimized_no_fn)
         << ",\"baseline_only\":";
    append_json_string_array(json, comparison.baseline_only);
    json << ",\"optimized_only\":";
    append_json_string_array(json, comparison.optimized_only);
    json << ",\"baseline_extra_vs_brute_force\":";
    append_json_string_array(json, comparison.baseline_extra_vs_brute_force);
    json << ",\"optimized_extra_vs_brute_force\":";
    append_json_string_array(json, comparison.optimized_extra_vs_brute_force);
    json << ",\"brute_force_missing_from_baseline\":";
    append_json_string_array(json, comparison.brute_force_missing_from_baseline);
    json << ",\"brute_force_missing_from_optimized\":";
    append_json_string_array(json, comparison.brute_force_missing_from_optimized);
    json << "}";
  }
  json << "],\"memory\":{";
  append_memory_json(json, "before_build", before_build);
  json << ",";
  append_memory_json(json, "after_build", after_build);
  json << ",";
  append_memory_json(json, "after_benchmark", after_benchmark);
  json << "},\"candidate_set_comparison\":\"unavailable\","
       << "\"allocation_counting\":\"unavailable\","
       << "\"gate_passed\":" << bool_string(result.gate_passed) << "}";
  result.json_summary = json.str();

  write_tsv(config.detail_tsv_path, detail_columns, result.detail_rows);
  write_tsv(config.summary_tsv_path, summary_columns, result.summary_rows);
  std::ofstream json_out(config.json_path);
  if (!json_out) throw std::runtime_error("unable to open query benchmark JSON output");
  json_out << result.json_summary << "\n";
  json_out.close();
  if (!json_out) {
    throw std::runtime_error("failed to write query benchmark JSON output");
  }
  return result;
}

}  // namespace navigamer
