#include "query_benchmark.hpp"
#include "index_persistence.hpp"
#include "io_utils.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <sys/resource.h>
#include <omp.h>

namespace navigamer {
namespace {

constexpr std::array<size_t, 3> kProximalOutputK = {1, 2, 4};
constexpr double kUnavailableMetric = -1.0;

struct ProximalOracleRecord {
  bool enabled = false;
  std::array<double, 3> actual_envelope = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  std::array<double, 3> frontier_oracle_envelope = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  std::array<double, 3> true_path_oracle_envelope = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  std::array<double, 3> global_oracle_envelope = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  std::array<double, 3> random_envelope = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  std::array<double, 3> global_gap_vs_actual = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  std::array<double, 3> global_gap_vs_frontier = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  std::array<bool, 3> global_much_better_than_actual = {false, false, false};
  std::array<bool, 3> global_much_better_than_frontier = {false, false, false};
  double actual_nearest_anchor_dist = kUnavailableMetric;
  double frontier_oracle_nearest_anchor_dist = kUnavailableMetric;
  double true_path_oracle_nearest_anchor_dist = kUnavailableMetric;
  double global_oracle_nearest_anchor_dist = kUnavailableMetric;
  double random_nearest_anchor_dist = kUnavailableMetric;
};

struct ExecutionRecord {
  std::string query_id;
  std::string profile;
  QueryClass query_class = QueryClass::RandomRegion;
  size_t profile_rank = 0;
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
  ProximalOracleRecord proximal;
};

struct BenchmarkProfileSpec {
  std::string name;
  SearchConfig config;
};

struct AggregateRecord {
  std::string query_class;
  std::string profile;
  size_t query_count = 0;
  size_t sample_count = 0;
  size_t result_total = 0;
  size_t equality_failure_count = 0;
  size_t false_negative_count = 0;
  size_t anchor_distance_count = 0;
  size_t center_distance_count = 0;
  size_t raw_candidate_count = 0;
  size_t world_access_count = 0;
  size_t node_access_count = 0;
  size_t edge_access_count = 0;
  size_t mbb_check_count = 0;
  size_t mbb_surviving_child_count = 0;
  size_t mbb_scalar_checks = 0;
  size_t mbb_simd_batches = 0;
  size_t mbb_simd_fallbacks = 0;
  size_t search_qgram_checks = 0;
  size_t center_exact_distance_call_count = 0;
  size_t leaf_beacon_check_count = 0;
  size_t leaf_beacon_scalar_checks = 0;
  size_t leaf_beacon_simd_batches = 0;
  size_t leaf_beacon_simd_fallbacks = 0;
  size_t leaf_exact_distance_call_count = 0;
  size_t visited_check_count = 0;
  size_t visited_hit_count = 0;
  size_t candidate_count = 0;
  size_t candidate_verify_count = 0;
  size_t local_router_invoked_count = 0;
  size_t local_router_empty_count = 0;
  size_t local_router_shortlist_child_count = 0;
  size_t local_router_remaining_child_count = 0;
  size_t local_router_fallback_count = 0;
  size_t router_hint_invoked_count = 0;
  size_t router_qgram_ranked_count = 0;
  size_t router_minimizer_ranked_count = 0;
  size_t router_pigeonhole_query_count = 0;
  size_t router_candidate_count = 0;
  size_t router_candidate_hit_count = 0;
  size_t router_fallback_count = 0;
  size_t best_first_invoked_count = 0;
  size_t best_first_reordered_count = 0;
  size_t best_first_bound_candidate_count = 0;
  size_t child_safe_bound_pruned_count = 0;
  size_t safe_child_router_invoked_count = 0;
  size_t safe_child_router_skipped_low_fanout_count = 0;
  size_t safe_child_router_fallback_count = 0;
  size_t safe_child_router_candidate_count = 0;
  size_t safe_child_router_pruned_by_not_candidate_count = 0;
  size_t safe_child_router_exact_verify_count = 0;
  size_t child_count_before_router = 0;
  size_t post_mbb_survivor_count = 0;
  size_t safe_router_candidate_count = 0;
  double candidate_ratio_to_all_children = 0.0;
  double candidate_ratio_to_post_mbb_survivors = 0.0;
  size_t children_actually_processed = 0;
  size_t center_checks_saved = 0;
  size_t planner_invoked_count = 0;
  size_t planner_strategy_baseline_count = 0;
  size_t planner_strategy_direct_qgram_count = 0;
  size_t planner_strategy_router_count = 0;
  size_t planner_strategy_safe_child_router_count = 0;
  size_t planner_strategy_path_reuse_count = 0;
  size_t planner_near_reuse_enabled_count = 0;
  size_t planner_near_reuse_disabled_count = 0;
  size_t planner_fallback_count = 0;
  double planner_decision_ms = 0.0;
  size_t frontier_max_size = 0;
  size_t frontier_total_pushed = 0;
  size_t path_reuse_attempt_count = 0;
  size_t path_reuse_hit_count = 0;
  size_t near_query_triangle_pruned_count = 0;
	  size_t near_query_center_distance_reused_count = 0;
	  size_t near_query_bound_fallback_count = 0;
	  size_t near_query_direct_verify_count = 0;
	  size_t near_query_leaf_triangle_pruned_count = 0;
	  size_t near_query_leaf_distance_reused_count = 0;
	  size_t near_query_leaf_bound_fallback_count = 0;
	  size_t anchor_cache_hit_count = 0;
  size_t child_shortlist_reuse_hit_count = 0;
  std::array<double, 3> proximal_actual_envelope_sum = {0.0, 0.0, 0.0};
  std::array<double, 3> proximal_frontier_envelope_sum = {0.0, 0.0, 0.0};
  std::array<double, 3> proximal_true_path_envelope_sum = {0.0, 0.0, 0.0};
  std::array<double, 3> proximal_global_envelope_sum = {0.0, 0.0, 0.0};
  std::array<double, 3> proximal_random_envelope_sum = {0.0, 0.0, 0.0};
  std::array<size_t, 3> proximal_actual_envelope_count = {0, 0, 0};
  std::array<size_t, 3> proximal_frontier_envelope_count = {0, 0, 0};
  std::array<size_t, 3> proximal_true_path_envelope_count = {0, 0, 0};
  std::array<size_t, 3> proximal_global_envelope_count = {0, 0, 0};
  std::array<size_t, 3> proximal_random_envelope_count = {0, 0, 0};
  std::array<size_t, 3> proximal_global_much_better_actual_count = {0, 0, 0};
  std::array<size_t, 3> proximal_global_much_better_frontier_count = {0, 0, 0};
  std::array<size_t, 3> proximal_global_much_better_actual_denominator = {0, 0, 0};
  std::array<size_t, 3> proximal_global_much_better_frontier_denominator = {0, 0, 0};
  std::vector<double> cold_latencies;
  std::vector<double> warm_latencies;
  std::vector<double> cold_profiled_query_ms;
  std::vector<double> warm_profiled_query_ms;
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
  std::vector<std::string> failing_profiles;
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

const BioSequence* sequence_pointer(
    const std::shared_ptr<BioSequence>& sequence) {
  return sequence.get();
}

const BioSequence* sequence_pointer(const BioSequence& sequence) {
  return &sequence;
}

template <typename SequenceContainer>
std::vector<std::string> exact_hit_ids(
    const std::string& query,
    const SequenceContainer& unique_sequences,
    int tolerance) {
  std::vector<std::string> ids;
  for (const auto& stored : unique_sequences) {
    const BioSequence* sequence = sequence_pointer(stored);
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

std::string build_query_schedule_key(const std::string& sequence) {
  constexpr size_t kScheduleQ = 4;
  if (sequence.size() < kScheduleQ) {
    return "seq:" + sequence;
  }
  std::vector<std::string> qgrams;
  qgrams.reserve(sequence.size() - kScheduleQ + 1);
  for (size_t i = 0; i + kScheduleQ <= sequence.size(); ++i) {
    qgrams.push_back(sequence.substr(i, kScheduleQ));
  }
  std::sort(qgrams.begin(), qgrams.end());
  qgrams.erase(std::unique(qgrams.begin(), qgrams.end()), qgrams.end());
  std::ostringstream out;
  out << "len:" << sequence.size();
  const size_t limit = std::min<size_t>(8, qgrams.size());
  for (size_t i = 0; i < limit; ++i) {
    out << "|" << qgrams[i];
  }
  return out.str();
}

SearchConfig build_query_benchmark_baseline_config() {
  SearchConfig config;
  config.mbb_filter_mode = MBBFilterMode::Scan;
  config.visited_mode = VisitedMode::StringSet;
  config.graph_view_mode = GraphViewMode::Original;
  config.simd_mode = SimdMode::Scalar;
  config.distance_mode = DistanceMode::DP;
  config.search_qgram_prefilter = false;
  return config;
}

void append_profile_if_distinct(std::vector<BenchmarkProfileSpec>& profiles,
                                const std::string& name,
                                const SearchConfig& candidate) {
  for (const auto& profile : profiles) {
    if (profile.name == name) return;
    const SearchConfig& existing = profile.config;
    if (existing.mbb_filter_mode == candidate.mbb_filter_mode &&
        existing.visited_mode == candidate.visited_mode &&
        existing.graph_view_mode == candidate.graph_view_mode &&
        existing.simd_mode == candidate.simd_mode &&
        existing.distance_mode == candidate.distance_mode &&
        existing.search_qgram_prefilter == candidate.search_qgram_prefilter &&
        existing.search_qgram_q == candidate.search_qgram_q &&
        existing.query_profile == candidate.query_profile &&
        existing.path_reuse_enabled == candidate.path_reuse_enabled &&
        existing.near_query_max_neighbor_edit_distance ==
            candidate.near_query_max_neighbor_edit_distance &&
        existing.near_query_min_qgram_jaccard ==
            candidate.near_query_min_qgram_jaccard &&
        existing.router_hint_enabled == candidate.router_hint_enabled &&
        existing.router_hint_qgram_q == candidate.router_hint_qgram_q &&
        existing.router_hint_minimizer_k == candidate.router_hint_minimizer_k &&
        existing.router_hint_minimizer_w == candidate.router_hint_minimizer_w &&
        existing.local_router_enabled == candidate.local_router_enabled &&
        existing.local_router_max_anchors == candidate.local_router_max_anchors &&
        existing.local_router_max_children == candidate.local_router_max_children &&
        existing.local_router_score_mode == candidate.local_router_score_mode &&
        existing.best_first_enabled == candidate.best_first_enabled &&
        existing.safe_child_router_enabled ==
            candidate.safe_child_router_enabled &&
        existing.safe_child_router_min_fanout ==
            candidate.safe_child_router_min_fanout &&
        existing.safe_child_router_max_candidates ==
            candidate.safe_child_router_max_candidates &&
        existing.safe_child_router_max_ratio ==
            candidate.safe_child_router_max_ratio &&
        existing.safe_child_router_min_seed_len ==
            candidate.safe_child_router_min_seed_len &&
        existing.safe_child_router_mode == candidate.safe_child_router_mode &&
        existing.safe_child_router_validate ==
            candidate.safe_child_router_validate &&
        existing.query_planner_enabled == candidate.query_planner_enabled &&
        existing.planner_direct_verify_max_candidates ==
            candidate.planner_direct_verify_max_candidates &&
        existing.planner_router_min_fanout ==
            candidate.planner_router_min_fanout &&
        existing.planner_safe_child_router_min_fanout ==
            candidate.planner_safe_child_router_min_fanout &&
        existing.planner_allow_direct_qgram_verify ==
            candidate.planner_allow_direct_qgram_verify) {
      return;
    }
  }
  profiles.push_back({name, candidate});
}

std::vector<BenchmarkProfileSpec> build_query_benchmark_profiles(
    const SearchConfig& optimized_search_config,
    bool enable_ablation_profiles) {
  std::vector<BenchmarkProfileSpec> profiles;
  profiles.push_back({"baseline", build_query_benchmark_baseline_config()});
  append_profile_if_distinct(profiles, "optimized", optimized_search_config);
  if (!enable_ablation_profiles) return profiles;

  if (optimized_search_config.search_qgram_prefilter) {
    SearchConfig ablation = optimized_search_config;
    ablation.search_qgram_prefilter = false;
    append_profile_if_distinct(profiles, "ablation_no_search_qgram", ablation);
  }
  if (optimized_search_config.router_hint_enabled) {
    SearchConfig ablation = optimized_search_config;
    ablation.router_hint_enabled = false;
    append_profile_if_distinct(profiles, "ablation_no_router_hints", ablation);
  }
  if (optimized_search_config.local_router_enabled) {
    SearchConfig ablation = optimized_search_config;
    ablation.local_router_enabled = false;
    append_profile_if_distinct(profiles, "ablation_no_local_router", ablation);
  }
  if (optimized_search_config.best_first_enabled) {
    SearchConfig ablation = optimized_search_config;
    ablation.best_first_enabled = false;
    append_profile_if_distinct(profiles, "ablation_no_best_first", ablation);
  }
  if (optimized_search_config.safe_child_router_enabled) {
    SearchConfig ablation = optimized_search_config;
    ablation.safe_child_router_enabled = false;
    append_profile_if_distinct(
        profiles, "ablation_no_safe_child_router", ablation);
  }
  if (optimized_search_config.path_reuse_enabled) {
    SearchConfig ablation = optimized_search_config;
    ablation.path_reuse_enabled = false;
    append_profile_if_distinct(profiles, "ablation_no_path_reuse", ablation);
  }
  if (optimized_search_config.query_planner_enabled) {
    SearchConfig ablation = optimized_search_config;
    ablation.query_planner_enabled = false;
    append_profile_if_distinct(profiles, "ablation_no_query_planner", ablation);
  }
  return profiles;
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

bool is_clean_dna_window(const std::string& sequence, size_t start,
                         size_t length) {
  if (start + length > sequence.size()) return false;
  for (size_t i = start; i < start + length; ++i) {
    const char c = sequence[i];
    if (c != 'A' && c != 'C' && c != 'G' && c != 'T' &&
        c != 'a' && c != 'c' && c != 'g' && c != 't') {
      return false;
    }
  }
  return true;
}

std::string uppercase_dna_window(const std::string& sequence, size_t start,
                                 size_t length) {
  std::string out = sequence.substr(start, length);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return out;
}

size_t find_clean_dna_window(const std::string& sequence, size_t preferred,
                             size_t length) {
  if (sequence.size() < length) {
    throw std::invalid_argument("reference must be at least query length");
  }
  preferred = std::min(preferred, sequence.size() - length);
  for (size_t delta = 0; preferred + delta + length <= sequence.size();
       ++delta) {
    const size_t right = preferred + delta;
    if (is_clean_dna_window(sequence, right, length)) return right;
    if (preferred >= delta &&
        is_clean_dna_window(sequence, preferred - delta, length)) {
      return preferred - delta;
    }
  }
  throw std::runtime_error("unable to find clean A/C/G/T query window");
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
      aggregate.anchor_distance_count += record.stats.anchor_distance_count;
      aggregate.center_distance_count += record.stats.center_distance_count;
      aggregate.raw_candidate_count += record.stats.raw_candidate_count;
      aggregate.world_access_count += record.stats.world_access_count;
      aggregate.node_access_count += record.stats.node_access_count;
      aggregate.edge_access_count += record.stats.edge_access_count;
      aggregate.mbb_check_count += record.stats.mbb_check_count;
      aggregate.mbb_surviving_child_count += record.stats.mbb_surviving_child_count;
      aggregate.mbb_scalar_checks += record.stats.mbb_scalar_checks;
      aggregate.mbb_simd_batches += record.stats.mbb_simd_batches;
      aggregate.mbb_simd_fallbacks += record.stats.mbb_simd_fallbacks;
      aggregate.search_qgram_checks += record.stats.search_qgram_checks;
      aggregate.center_exact_distance_call_count +=
          record.stats.center_exact_distance_call_count;
      aggregate.leaf_beacon_check_count += record.stats.leaf_beacon_check_count;
      aggregate.leaf_beacon_scalar_checks += record.stats.leaf_beacon_scalar_checks;
      aggregate.leaf_beacon_simd_batches += record.stats.leaf_beacon_simd_batches;
      aggregate.leaf_beacon_simd_fallbacks += record.stats.leaf_beacon_simd_fallbacks;
      aggregate.leaf_exact_distance_call_count +=
          record.stats.leaf_exact_distance_call_count;
      aggregate.visited_check_count += record.stats.visited_check_count;
      aggregate.visited_hit_count += record.stats.visited_hit_count;
      aggregate.candidate_count += record.stats.candidate_count;
      aggregate.candidate_verify_count += record.stats.candidate_verify_count;
      aggregate.local_router_invoked_count +=
          record.stats.local_router_invoked_count;
      aggregate.local_router_empty_count += record.stats.local_router_empty_count;
      aggregate.local_router_shortlist_child_count +=
          record.stats.local_router_shortlist_child_count;
      aggregate.local_router_remaining_child_count +=
          record.stats.local_router_remaining_child_count;
      aggregate.local_router_fallback_count +=
          record.stats.local_router_fallback_count;
      aggregate.router_hint_invoked_count +=
          record.stats.router_hint_invoked_count;
      aggregate.router_qgram_ranked_count +=
          record.stats.router_qgram_ranked_count;
      aggregate.router_minimizer_ranked_count +=
          record.stats.router_minimizer_ranked_count;
      aggregate.router_pigeonhole_query_count +=
          record.stats.router_pigeonhole_query_count;
      aggregate.router_candidate_count += record.stats.router_candidate_count;
      aggregate.router_candidate_hit_count +=
          record.stats.router_candidate_hit_count;
      aggregate.router_fallback_count += record.stats.router_fallback_count;
      aggregate.best_first_invoked_count += record.stats.best_first_invoked_count;
      aggregate.best_first_reordered_count +=
          record.stats.best_first_reordered_count;
      aggregate.best_first_bound_candidate_count +=
          record.stats.best_first_bound_candidate_count;
      aggregate.child_safe_bound_pruned_count +=
          record.stats.child_safe_bound_pruned_count;
      aggregate.safe_child_router_invoked_count +=
          record.stats.safe_child_router_invoked_count;
      aggregate.safe_child_router_skipped_low_fanout_count +=
          record.stats.safe_child_router_skipped_low_fanout_count;
      aggregate.safe_child_router_fallback_count +=
          record.stats.safe_child_router_fallback_count;
      aggregate.safe_child_router_candidate_count +=
          record.stats.safe_child_router_candidate_count;
      aggregate.safe_child_router_pruned_by_not_candidate_count +=
          record.stats.safe_child_router_pruned_by_not_candidate_count;
      aggregate.safe_child_router_exact_verify_count +=
          record.stats.safe_child_router_exact_verify_count;
      aggregate.child_count_before_router +=
          record.stats.child_count_before_router;
      aggregate.post_mbb_survivor_count += record.stats.post_mbb_survivor_count;
      aggregate.safe_router_candidate_count +=
          record.stats.safe_router_candidate_count;
      aggregate.candidate_ratio_to_all_children +=
          record.stats.candidate_ratio_to_all_children;
      aggregate.candidate_ratio_to_post_mbb_survivors +=
          record.stats.candidate_ratio_to_post_mbb_survivors;
      aggregate.children_actually_processed +=
          record.stats.children_actually_processed;
      aggregate.center_checks_saved += record.stats.center_checks_saved;
      aggregate.planner_invoked_count += record.stats.planner_invoked_count;
      aggregate.planner_strategy_baseline_count +=
          record.stats.planner_strategy_baseline_count;
      aggregate.planner_strategy_direct_qgram_count +=
          record.stats.planner_strategy_direct_qgram_count;
      aggregate.planner_strategy_router_count +=
          record.stats.planner_strategy_router_count;
      aggregate.planner_strategy_safe_child_router_count +=
          record.stats.planner_strategy_safe_child_router_count;
      aggregate.planner_strategy_path_reuse_count +=
          record.stats.planner_strategy_path_reuse_count;
      aggregate.planner_near_reuse_enabled_count +=
          record.stats.planner_near_reuse_enabled_count;
      aggregate.planner_near_reuse_disabled_count +=
          record.stats.planner_near_reuse_disabled_count;
      aggregate.planner_fallback_count += record.stats.planner_fallback_count;
      aggregate.planner_decision_ms += record.stats.planner_decision_ms;
      aggregate.frontier_max_size += record.stats.frontier_max_size;
      aggregate.frontier_total_pushed += record.stats.frontier_total_pushed;
      aggregate.path_reuse_attempt_count +=
          record.stats.path_reuse_attempt_count;
      aggregate.path_reuse_hit_count += record.stats.path_reuse_hit_count;
      aggregate.near_query_triangle_pruned_count +=
          record.stats.near_query_triangle_pruned_count;
      aggregate.near_query_center_distance_reused_count +=
          record.stats.near_query_center_distance_reused_count;
      aggregate.near_query_bound_fallback_count +=
          record.stats.near_query_bound_fallback_count;
	      aggregate.near_query_direct_verify_count +=
	          record.stats.near_query_direct_verify_count;
	      aggregate.near_query_leaf_triangle_pruned_count +=
	          record.stats.near_query_leaf_triangle_pruned_count;
	      aggregate.near_query_leaf_distance_reused_count +=
	          record.stats.near_query_leaf_distance_reused_count;
	      aggregate.near_query_leaf_bound_fallback_count +=
	          record.stats.near_query_leaf_bound_fallback_count;
	      aggregate.anchor_cache_hit_count += record.stats.anchor_cache_hit_count;
      aggregate.child_shortlist_reuse_hit_count +=
          record.stats.child_shortlist_reuse_hit_count;
      if (record.proximal.enabled) {
        for (size_t i = 0; i < kProximalOutputK.size(); ++i) {
          auto add_metric = [](double value, double& sum, size_t& count) {
            if (value >= 0.0) {
              sum += value;
              count++;
            }
          };
          add_metric(record.proximal.actual_envelope[i],
                     aggregate.proximal_actual_envelope_sum[i],
                     aggregate.proximal_actual_envelope_count[i]);
          add_metric(record.proximal.frontier_oracle_envelope[i],
                     aggregate.proximal_frontier_envelope_sum[i],
                     aggregate.proximal_frontier_envelope_count[i]);
          add_metric(record.proximal.true_path_oracle_envelope[i],
                     aggregate.proximal_true_path_envelope_sum[i],
                     aggregate.proximal_true_path_envelope_count[i]);
          add_metric(record.proximal.global_oracle_envelope[i],
                     aggregate.proximal_global_envelope_sum[i],
                     aggregate.proximal_global_envelope_count[i]);
          add_metric(record.proximal.random_envelope[i],
                     aggregate.proximal_random_envelope_sum[i],
                     aggregate.proximal_random_envelope_count[i]);
          if (record.proximal.global_oracle_envelope[i] >= 0.0 &&
              record.proximal.actual_envelope[i] >= 0.0) {
            aggregate.proximal_global_much_better_actual_denominator[i]++;
            if (record.proximal.global_much_better_than_actual[i]) {
              aggregate.proximal_global_much_better_actual_count[i]++;
            }
          }
          if (record.proximal.global_oracle_envelope[i] >= 0.0 &&
              record.proximal.frontier_oracle_envelope[i] >= 0.0) {
            aggregate.proximal_global_much_better_frontier_denominator[i]++;
            if (record.proximal.global_much_better_than_frontier[i]) {
              aggregate.proximal_global_much_better_frontier_count[i]++;
            }
          }
        }
      }
      const double profiled_query_ms =
          record.stats.query_total_ms > 0.0 ? record.stats.query_total_ms
                                            : record.latency_ms;
      if (record.sample_kind == "cold") {
        aggregate.cold_latencies.push_back(record.latency_ms);
        aggregate.cold_profiled_query_ms.push_back(profiled_query_ms);
      } else if (record.sample_kind == "warm") {
        aggregate.warm_latencies.push_back(record.latency_ms);
        aggregate.warm_profiled_query_ms.push_back(profiled_query_ms);
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

void append_json_size_array(std::ostringstream& out,
                            const std::vector<size_t>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ",";
    out << values[i];
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
    const SearchResult& results,
    const SequenceStore& sequence_store) {
  std::vector<std::string> ids;
  ids.reserve(results.size());
  for (LeafId result : results) {
    ids.push_back(sequence_store.reference_backed
                      ? sequence_store.reference_id + "_" +
                            std::to_string(
                                sequence_store.source_position(result))
                      : sequence_store.at(result).id);
  }
  return sorted_unique(std::move(ids));
}

std::vector<double> anchor_distances_sorted(
    const std::string& query,
    const std::vector<std::string>& anchors) {
  std::vector<double> distances;
  distances.reserve(anchors.size());
  for (const auto& anchor : anchors) {
    distances.push_back(static_cast<double>(compute_distance(query, anchor)));
  }
  std::sort(distances.begin(), distances.end());
  return distances;
}

double nearest_distance_from_sorted(const std::vector<double>& distances) {
  return distances.empty() ? kUnavailableMetric : distances.front();
}

double nearest_k_envelope_from_sorted(const std::vector<double>& distances,
                                      size_t k) {
  if (distances.empty() || k == 0) return kUnavailableMetric;
  const size_t limit = std::min(k, distances.size());
  return distances[limit - 1];
}

double envelope_from_random_order(const std::string& query,
                                  const std::vector<std::string>& anchors,
                                  size_t k) {
  if (anchors.empty() || k == 0) return kUnavailableMetric;
  const size_t limit = std::min(k, anchors.size());
  double envelope = 0.0;
  for (size_t i = 0; i < limit; ++i) {
    envelope = std::max(envelope,
                        static_cast<double>(compute_distance(query, anchors[i])));
  }
  return envelope;
}

uint64_t stable_hash64(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char c : value) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<std::string> unique_anchor_sequences(
    const std::vector<std::string>& anchors) {
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  out.reserve(anchors.size());
  for (const auto& anchor : anchors) {
    if (anchor.empty()) continue;
    if (seen.insert(anchor).second) out.push_back(anchor);
  }
  return out;
}

void append_node_anchors(const SearchGraphView& view, NodeId node_id,
                         std::vector<std::string>& anchors) {
  if (node_id >= view.node_records.size()) return;
  const auto& node = view.node_records[node_id];
  if (node.center_sequence_id < view.sequences.size()) {
    anchors.emplace_back(view.sequences.sequence(node.center_sequence_id));
  }
  for (uint32_t offset = 0; offset < node.beacon_count; ++offset) {
    const LeafId beacon_id = view.beacon_ids[node.beacon_begin + offset];
    if (beacon_id < view.sequences.size()) {
      anchors.emplace_back(view.sequences.sequence(beacon_id));
    }
  }
}

std::vector<size_t> normalized_proximal_k_values(
    const std::vector<size_t>& configured) {
  std::vector<size_t> values;
  for (size_t k : configured) {
    if (k > 0) values.push_back(k);
  }
  for (size_t k : kProximalOutputK) values.push_back(k);
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<std::string> deterministic_random_anchors(
    const std::string& query_id,
    const std::string& query,
    const std::vector<std::string>& anchors,
    size_t max_k) {
  std::vector<std::string> shuffled = anchors;
  if (shuffled.empty() || max_k == 0) return {};
  std::mt19937 generator(static_cast<uint32_t>(
      stable_hash64(query_id + "\t" + query) & 0xffffffffULL));
  std::shuffle(shuffled.begin(), shuffled.end(), generator);
  if (shuffled.size() > max_k) shuffled.resize(max_k);
  return shuffled;
}

std::array<double, 3> output_envelopes_by_k(
    const std::vector<double>& envelopes,
    const std::vector<size_t>& k_values) {
  std::array<double, 3> out = {
      kUnavailableMetric, kUnavailableMetric, kUnavailableMetric};
  for (size_t out_idx = 0; out_idx < kProximalOutputK.size(); ++out_idx) {
    auto it = std::find(k_values.begin(), k_values.end(),
                        kProximalOutputK[out_idx]);
    if (it == k_values.end()) continue;
    const size_t metric_idx = static_cast<size_t>(it - k_values.begin());
    if (metric_idx < envelopes.size()) out[out_idx] = envelopes[metric_idx];
  }
  return out;
}

bool much_better(double oracle, double observed) {
  return oracle >= 0.0 && observed > 0.0 && oracle <= observed * 0.75;
}

double gap_or_unavailable(double observed, double oracle) {
  if (observed < 0.0 || oracle < 0.0) return kUnavailableMetric;
  return observed - oracle;
}

struct ProximalOracleContext {
  bool enabled = false;
  std::vector<size_t> k_values;
  std::vector<std::string> global_anchors;
  const SearchGraphView* view = nullptr;
  std::unordered_map<std::string, NodeId> nodes_by_id;
  std::vector<NodeId> nodes;
};

ProximalOracleContext build_proximal_oracle_context(
    const BioGeometryIndexBuilder& builder,
    const QueryBenchmarkConfig& config) {
  ProximalOracleContext context;
  context.enabled = config.proximal_oracle_enabled;
  context.k_values = normalized_proximal_k_values(config.proximal_oracle_k_values);
  if (!context.enabled) return context;
  context.view = &builder.search_graph_view();
  for (NodeId node_id = 0; node_id < context.view->node_records.size();
       ++node_id) {
    const std::string id = std::to_string(node_id);
    context.nodes_by_id[id] = node_id;
    context.nodes.push_back(node_id);
    append_node_anchors(*context.view, node_id, context.global_anchors);
  }
  context.global_anchors = unique_anchor_sequences(context.global_anchors);
  return context;
}

std::vector<std::string> collect_anchors_for_node_ids(
    const ProximalOracleContext& context,
    const std::vector<std::string>& node_ids) {
  std::vector<std::string> anchors;
  for (const auto& node_id : node_ids) {
    auto it = context.nodes_by_id.find(node_id);
    if (it != context.nodes_by_id.end() && context.view) {
      append_node_anchors(*context.view, it->second, anchors);
    }
  }
  return unique_anchor_sequences(anchors);
}

bool node_subtree_has_hit(
    const SearchGraphView& view,
    NodeId node_id,
    const std::unordered_set<std::string>& hit_ids,
    std::unordered_map<NodeId, bool>& memo) {
  if (node_id >= view.node_records.size() || hit_ids.empty()) return false;
  auto memo_it = memo.find(node_id);
  if (memo_it != memo.end()) return memo_it->second;
  const auto& node = view.node_records[node_id];
  for (uint32_t offset = 0; offset < node.leaf_count; ++offset) {
    const LeafId leaf_id = view.leaf_ids[node.leaf_begin + offset];
    if (leaf_id < view.sequences.size() &&
        hit_ids.count(view.sequences[leaf_id].id)) {
      memo[node_id] = true;
      return true;
    }
  }
  for (uint32_t offset = 0; offset < node.child_count; ++offset) {
    if (node_subtree_has_hit(
            view, view.child_ids[node.child_begin + offset], hit_ids, memo)) {
      memo[node_id] = true;
      return true;
    }
  }
  memo[node_id] = false;
  return false;
}

std::vector<std::string> collect_true_path_anchors(
    const ProximalOracleContext& context,
    const std::vector<std::string>& brute_force_ids) {
  std::vector<std::string> anchors;
  std::unordered_set<std::string> hit_ids(brute_force_ids.begin(),
                                          brute_force_ids.end());
  std::unordered_map<NodeId, bool> memo;
  if (!context.view) return anchors;
  for (NodeId node_id : context.nodes) {
    if (node_subtree_has_hit(*context.view, node_id, hit_ids, memo)) {
      append_node_anchors(*context.view, node_id, anchors);
    }
  }
  return unique_anchor_sequences(anchors);
}

void assign_anchor_set_metrics(const std::string& query,
                               const std::vector<size_t>& k_values,
                               const std::vector<std::string>& oracle_anchors,
                               const std::vector<std::string>& random_anchors,
                               double* nearest,
                               std::array<double, 3>* envelopes,
                               double* random_nearest,
                               std::array<double, 3>* random_envelopes) {
  const auto metrics = compute_proximal_anchor_set_diagnostics(
      query, oracle_anchors, random_anchors, k_values);
  if (nearest) *nearest = metrics.nearest_anchor_dist;
  if (envelopes) {
    *envelopes =
        output_envelopes_by_k(metrics.oracle_envelope_by_k, k_values);
  }
  if (random_nearest) *random_nearest = metrics.random_nearest_anchor_dist;
  if (random_envelopes) {
    *random_envelopes =
        output_envelopes_by_k(metrics.random_envelope_by_k, k_values);
  }
}

ProximalOracleRecord compute_proximal_oracle_record(
    const ProximalOracleContext& context,
    const GeneratedBenchmarkQuery& generated,
    const SearchStats& stats) {
  ProximalOracleRecord record;
  if (!context.enabled) return record;
  record.enabled = true;
  const size_t max_k =
      context.k_values.empty()
          ? *std::max_element(kProximalOutputK.begin(), kProximalOutputK.end())
          : *std::max_element(context.k_values.begin(), context.k_values.end());
  const auto random_anchors = deterministic_random_anchors(
      generated.query.id, generated.query.seq, context.global_anchors, max_k);
  const auto actual_anchors = collect_anchors_for_node_ids(
      context, stats.proximal_actual_anchor_node_ids);
  const auto frontier_anchors = collect_anchors_for_node_ids(
      context, stats.proximal_frontier_node_ids);
  const auto true_path_anchors =
      collect_true_path_anchors(context, generated.brute_force_ids);

  assign_anchor_set_metrics(generated.query.seq, context.k_values, actual_anchors,
                            random_anchors, &record.actual_nearest_anchor_dist,
                            &record.actual_envelope, nullptr, nullptr);
  assign_anchor_set_metrics(generated.query.seq, context.k_values, frontier_anchors,
                            random_anchors,
                            &record.frontier_oracle_nearest_anchor_dist,
                            &record.frontier_oracle_envelope, nullptr, nullptr);
  assign_anchor_set_metrics(generated.query.seq, context.k_values, true_path_anchors,
                            random_anchors,
                            &record.true_path_oracle_nearest_anchor_dist,
                            &record.true_path_oracle_envelope, nullptr, nullptr);
  assign_anchor_set_metrics(generated.query.seq, context.k_values,
                            context.global_anchors, random_anchors,
                            &record.global_oracle_nearest_anchor_dist,
                            &record.global_oracle_envelope,
                            &record.random_nearest_anchor_dist,
                            &record.random_envelope);
  for (size_t i = 0; i < kProximalOutputK.size(); ++i) {
    record.global_gap_vs_actual[i] = gap_or_unavailable(
        record.actual_envelope[i], record.global_oracle_envelope[i]);
    record.global_gap_vs_frontier[i] = gap_or_unavailable(
        record.frontier_oracle_envelope[i], record.global_oracle_envelope[i]);
    record.global_much_better_than_actual[i] = much_better(
        record.global_oracle_envelope[i], record.actual_envelope[i]);
    record.global_much_better_than_frontier[i] = much_better(
        record.global_oracle_envelope[i], record.frontier_oracle_envelope[i]);
  }
  return record;
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

double average_counter(double value, size_t samples) {
  return samples == 0 ? 0.0 : value / static_cast<double>(samples);
}

double average_metric(double sum, size_t count) {
  return count == 0 ? kUnavailableMetric : sum / static_cast<double>(count);
}

double fraction_metric(size_t numerator, size_t denominator) {
  return denominator == 0
             ? 0.0
             : static_cast<double>(numerator) / static_cast<double>(denominator);
}

double safe_ratio(double numerator, double denominator) {
  if (denominator == 0.0) return numerator == 0.0 ? 1.0 : 0.0;
  return numerator / denominator;
}

double safe_speedup(double baseline_ms, double current_ms) {
  if (current_ms == 0.0) return baseline_ms == 0.0 ? 1.0 : 0.0;
  return baseline_ms / current_ms;
}

void locality_progress(const std::string& message) {
  std::cerr << "[locality-benchmark] " << message << std::endl;
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

ProximalAnchorSetDiagnostics compute_proximal_anchor_set_diagnostics(
    const std::string& query,
    const std::vector<std::string>& oracle_anchors,
    const std::vector<std::string>& random_anchors,
    const std::vector<size_t>& k_values) {
  ProximalAnchorSetDiagnostics out;
  const auto oracle_distances =
      anchor_distances_sorted(query, unique_anchor_sequences(oracle_anchors));
  out.nearest_anchor_dist = nearest_distance_from_sorted(oracle_distances);
  const auto unique_random = unique_anchor_sequences(random_anchors);
  const auto random_distances = anchor_distances_sorted(query, unique_random);
  out.random_nearest_anchor_dist =
      nearest_distance_from_sorted(random_distances);
  out.oracle_envelope_by_k.reserve(k_values.size());
  out.random_envelope_by_k.reserve(k_values.size());
  for (size_t k : k_values) {
    out.oracle_envelope_by_k.push_back(
        nearest_k_envelope_from_sorted(oracle_distances, k));
    out.random_envelope_by_k.push_back(
        envelope_from_random_order(query, unique_random, k));
  }
  return out;
}

template <typename SequenceContainer>
std::vector<GeneratedBenchmarkQuery> generate_benchmark_queries_impl(
    const std::vector<std::shared_ptr<BioSequence>>& index_sequences,
    const SequenceContainer& unique_sequences,
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

std::vector<GeneratedBenchmarkQuery> generate_benchmark_queries(
    const std::vector<std::shared_ptr<BioSequence>>& index_sequences,
    const std::vector<std::shared_ptr<BioSequence>>& unique_sequences,
    int query_length,
    int tolerance,
    unsigned seed,
    size_t queries_per_class) {
  return generate_benchmark_queries_impl(
      index_sequences, unique_sequences, query_length, tolerance, seed,
      queries_per_class);
}

std::vector<GeneratedBenchmarkQuery> generate_benchmark_queries(
    const std::vector<std::shared_ptr<BioSequence>>& index_sequences,
    const SequenceStore& unique_sequences,
    int query_length,
    int tolerance,
    unsigned seed,
    size_t queries_per_class) {
  return generate_benchmark_queries_impl(
      index_sequences, unique_sequences.records, query_length, tolerance, seed,
      queries_per_class);
}

LocalityBenchmarkQuerySets generate_locality_benchmark_queries(
    const std::string& reference,
    size_t query_count,
    int query_length,
    int edits,
    unsigned seed) {
  if (query_count == 0) {
    throw std::invalid_argument("query count must be positive");
  }
  if (query_length <= 0) {
    throw std::invalid_argument("query length must be positive");
  }
  if (edits < 0) {
    throw std::invalid_argument("edits must be non-negative");
  }
  const size_t length = static_cast<size_t>(query_length);
  if (reference.size() < length) {
    throw std::invalid_argument("reference must be at least query length");
  }

  std::mt19937 generator(seed);
  LocalityBenchmarkQuerySets out;
  out.same_template.reserve(query_count);
  out.nearby_windows.reserve(query_count);
  out.random_windows.reserve(query_count);
  out.repeat.reserve(query_count);
  out.batch_locality.reserve(query_count);
  out.oracle.reserve(query_count);
  out.real_dup_1x.reserve(query_count);
  out.real_dup_4x.reserve(query_count * 4);
  out.real_dup_16x.reserve(query_count * 16);
  out.source_sorted_stride1.reserve(query_count);
  out.source_sorted_mutated_tau5.reserve(query_count);
  out.source_sorted_mutated_tau8.reserve(query_count);

  const size_t same_pos =
      find_clean_dna_window(reference, reference.size() / 3, length);
  const std::string same_template =
      uppercase_dna_window(reference, same_pos, length);
  for (size_t i = 0; i < query_count; ++i) {
    out.same_template.push_back(
        {BioSequence("same_template_" + std::to_string(i),
                     mutate_substitutions(same_template,
                                          static_cast<size_t>(edits),
                                          generator)),
         same_pos});
  }

  const size_t nearby_span = length + query_count - 1;
  const size_t nearby_pos =
      find_clean_dna_window(reference, reference.size() / 3, nearby_span);
  for (size_t i = 0; i < query_count; ++i) {
    const size_t source_pos = nearby_pos + i;
    std::string window = uppercase_dna_window(reference, source_pos, length);
    out.nearby_windows.push_back(
        {BioSequence("nearby_windows_" + std::to_string(i),
                     mutate_substitutions(window, static_cast<size_t>(edits),
                                          generator)),
         source_pos});
  }

  std::uniform_int_distribution<size_t> pick(0, reference.size() - length);
  size_t attempts = 0;
  while (out.random_windows.size() < query_count && attempts < query_count * 4096) {
    ++attempts;
    const size_t source_pos = pick(generator);
    if (!is_clean_dna_window(reference, source_pos, length)) continue;
    std::string window = uppercase_dna_window(reference, source_pos, length);
    const size_t id = out.random_windows.size();
    out.random_windows.push_back(
        {BioSequence("random_windows_" + std::to_string(id),
                     mutate_substitutions(window, static_cast<size_t>(edits),
                                          generator)),
         source_pos});
  }
  if (out.random_windows.size() != query_count) {
    throw std::runtime_error("unable to generate enough random clean windows");
  }

  const size_t repeat_base =
      find_clean_dna_window(reference, reference.size() / 4, length);
  const size_t cluster_span =
      std::min(reference.size() - length, repeat_base + length);
  const size_t repeat_template_count = std::min<size_t>(query_count, 5);
  std::vector<std::string> repeated_templates;
  repeated_templates.reserve(repeat_template_count);
  for (size_t offset = 0; offset < repeat_template_count; ++offset) {
    const size_t source_pos = std::min(repeat_base + offset, cluster_span);
    std::string window = uppercase_dna_window(reference, source_pos, length);
    repeated_templates.push_back(
        mutate_substitutions(window, static_cast<size_t>(edits), generator));
  }
  for (size_t i = 0; i < query_count; ++i) {
    const size_t offset = repeat_template_count == 0 ? 0 : i % repeat_template_count;
    const size_t source_pos = std::min(repeat_base + offset, cluster_span);
    out.repeat.push_back(
        {BioSequence("repeat_" + std::to_string(i),
                     repeated_templates[offset]),
         source_pos});
  }

  const size_t batch_base =
      find_clean_dna_window(reference, reference.size() / 2, length);
  const size_t group_size = std::max<size_t>(1, std::min<size_t>(8, query_count));
  for (size_t i = 0; i < query_count; ++i) {
    const size_t group = i / group_size;
    const size_t source_pos =
        std::min(batch_base + group, reference.size() - length);
    std::string window = uppercase_dna_window(reference, source_pos, length);
    out.batch_locality.push_back(
        {BioSequence("batch_locality_" + std::to_string(i),
                     mutate_substitutions(window, static_cast<size_t>(edits),
                                          generator)),
         source_pos});
  }

  const size_t oracle_span = length + query_count - 1;
  const size_t oracle_base =
      find_clean_dna_window(reference, reference.size() / 5, oracle_span);
  for (size_t i = 0; i < query_count; ++i) {
    const size_t source_pos = oracle_base + i;
    std::string window = uppercase_dna_window(reference, source_pos, length);
    out.oracle.push_back(
        {BioSequence("oracle_" + std::to_string(i),
                     mutate_substitutions(window, static_cast<size_t>(edits),
                                          generator)),
         source_pos});
  }

  const size_t real_span = length + query_count - 1;
  const size_t real_base =
      find_clean_dna_window(reference, reference.size() / 6, real_span);
  std::vector<LocalityBenchmarkQuery> real_unique;
  real_unique.reserve(query_count);
  for (size_t i = 0; i < query_count; ++i) {
    const size_t source_pos = real_base + i;
    std::string window = uppercase_dna_window(reference, source_pos, length);
    real_unique.push_back(
        {BioSequence("real_dup_unique_" + std::to_string(i),
                     mutate_substitutions(window, static_cast<size_t>(edits),
                                          generator)),
         source_pos});
  }
  auto append_duplicates = [](std::vector<LocalityBenchmarkQuery>& target,
                              const std::vector<LocalityBenchmarkQuery>& unique,
                              size_t copies,
                              const std::string& prefix) {
    for (size_t i = 0; i < unique.size(); ++i) {
      for (size_t copy = 0; copy < copies; ++copy) {
        target.push_back(
            {BioSequence(prefix + "_" + std::to_string(i) + "_" +
                             std::to_string(copy),
                         unique[i].query.seq),
             unique[i].source_pos});
      }
    }
  };
  append_duplicates(out.real_dup_1x, real_unique, 1, "real_dup_1x");
  append_duplicates(out.real_dup_4x, real_unique, 4, "real_dup_4x");
  append_duplicates(out.real_dup_16x, real_unique, 16, "real_dup_16x");

  const size_t sorted_span = length + query_count - 1;
  const size_t sorted_base =
      find_clean_dna_window(reference, reference.size() / 7, sorted_span);
  for (size_t i = 0; i < query_count; ++i) {
    const size_t source_pos = sorted_base + i;
    std::string window = uppercase_dna_window(reference, source_pos, length);
    out.source_sorted_stride1.push_back(
        {BioSequence("source_sorted_stride1_" + std::to_string(i), window),
         source_pos});
    out.source_sorted_mutated_tau5.push_back(
        {BioSequence("source_sorted_mutated_tau5_" + std::to_string(i),
                     mutate_substitutions(window, std::min<size_t>(5, length),
                                          generator)),
         source_pos});
    out.source_sorted_mutated_tau8.push_back(
        {BioSequence("source_sorted_mutated_tau8_" + std::to_string(i),
                     mutate_substitutions(window, std::min<size_t>(8, length),
                                          generator)),
         source_pos});
  }

  return out;
}

std::vector<std::string> search_result_ids(
    const SearchResult& hits,
    const SequenceStore& sequence_store) {
  std::vector<std::string> ids;
  ids.reserve(hits.size());
  for (LeafId hit : hits) {
    ids.push_back(sequence_store.reference_backed
                      ? sequence_store.reference_id + "_" +
                            std::to_string(
                                sequence_store.source_position(hit))
                      : sequence_store.at(hit).id);
  }
  return sorted_unique(std::move(ids));
}

std::string exact_result_cache_key(const std::string& sequence, int tolerance) {
  return std::to_string(tolerance) + "\n" + sequence;
}

struct DuplicateSummary {
  size_t unique_query_count = 0;
  size_t duplicate_group_count = 0;
  double duplicate_ratio = 0.0;
};

DuplicateSummary summarize_duplicates(
    const std::vector<LocalityBenchmarkQuery>& queries) {
  std::unordered_map<std::string, size_t> counts;
  counts.reserve(queries.size());
  for (const auto& query : queries) {
    counts[query.query.seq]++;
  }
  DuplicateSummary summary;
  summary.unique_query_count = counts.size();
  for (const auto& entry : counts) {
    if (entry.second > 1) summary.duplicate_group_count++;
  }
  if (!queries.empty()) {
    summary.duplicate_ratio =
        static_cast<double>(queries.size() - summary.unique_query_count) /
        static_cast<double>(queries.size());
  }
  return summary;
}

std::unordered_set<std::string> qgram_set(const std::string& sequence,
                                          size_t q) {
  std::unordered_set<std::string> qgrams;
  if (q == 0 || sequence.size() < q) {
    qgrams.insert(sequence);
    return qgrams;
  }
  qgrams.reserve(sequence.size() - q + 1);
  for (size_t i = 0; i + q <= sequence.size(); ++i) {
    qgrams.insert(sequence.substr(i, q));
  }
  return qgrams;
}

double qgram_jaccard(const std::string& left, const std::string& right) {
  constexpr size_t kQgramJaccardQ = 5;
  const auto left_qgrams = qgram_set(left, kQgramJaccardQ);
  const auto right_qgrams = qgram_set(right, kQgramJaccardQ);
  size_t intersection = 0;
  for (const auto& qgram : left_qgrams) {
    if (right_qgrams.find(qgram) != right_qgrams.end()) intersection++;
  }
  const size_t union_size =
      left_qgrams.size() + right_qgrams.size() - intersection;
  if (union_size == 0) return 1.0;
  return static_cast<double>(intersection) / static_cast<double>(union_size);
}

struct NeighborSummary {
  double mean_edit_distance = 0.0;
  double p95_edit_distance = 0.0;
  double mean_qgram_jaccard = 0.0;
};

NeighborSummary summarize_neighbors(
    const std::vector<LocalityBenchmarkQuery>& queries) {
  NeighborSummary summary;
  if (queries.size() < 2) return summary;
  std::vector<double> edit_distances;
  std::vector<double> qgram_jaccards;
  edit_distances.reserve(queries.size() - 1);
  qgram_jaccards.reserve(queries.size() - 1);
  for (size_t i = 1; i < queries.size(); ++i) {
    edit_distances.push_back(static_cast<double>(
        compute_distance(queries[i - 1].query.seq, queries[i].query.seq)));
    qgram_jaccards.push_back(
        qgram_jaccard(queries[i - 1].query.seq, queries[i].query.seq));
  }
  summary.mean_edit_distance = average(edit_distances);
  summary.p95_edit_distance = percentile_or_zero(edit_distances, 0.95);
  summary.mean_qgram_jaccard = average(qgram_jaccards);
  return summary;
}

struct FanoutSummary {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double max = 0.0;
};

FanoutSummary compute_fanout_summary(const BioGeometryIndexBuilder& builder) {
  std::vector<double> fanouts;
  for (const auto& node : builder.search_graph_view().node_records) {
    if (node.child_count == 0) continue;
    fanouts.push_back(static_cast<double>(node.child_count));
  }
  if (fanouts.empty()) return {};
  FanoutSummary summary;
  summary.mean = average(fanouts);
  summary.p50 = percentile_or_zero(fanouts, 0.50);
  summary.p95 = percentile_or_zero(fanouts, 0.95);
  summary.max = *std::max_element(fanouts.begin(), fanouts.end());
  return summary;
}

SearchConfig locality_profile_config(const std::string& profile) {
  SearchConfig config;
  config.mbb_filter_mode = MBBFilterMode::RectIndex;
  config.visited_mode = VisitedMode::Epoch;
  config.graph_view_mode = GraphViewMode::Flat;
  config.simd_mode = SimdMode::Auto;
  config.distance_mode = DistanceMode::Myers;
  config.query_profile = false;
  if (profile == "baseline") return config;
  if (profile == "path_reuse" || profile == "optimized") {
    config.path_reuse_enabled = true;
  }
  if (profile == "optimized") {
    config.local_router_enabled = false;
    config.local_router_max_anchors = 4;
    config.local_router_max_children = 64;
    config.best_first_enabled = false;
    config.safe_child_router_enabled = false;
  }
  return config;
}

LocalityBenchmarkRow run_locality_profile(
    const std::string& dataset,
    const std::string& profile,
    const std::string& batch_schedule_mode,
    const BioGeometryIndexBuilder& builder,
    const std::vector<LocalityBenchmarkQuery>& queries,
    const std::vector<std::vector<std::string>>& baseline_ids,
    int tolerance,
    double load_ms,
    std::vector<std::vector<std::string>>* observed_ids) {
  locality_progress("profile start dataset=" + dataset + " profile=" + profile +
                    " schedule=" + batch_schedule_mode +
                    " queries=" + std::to_string(queries.size()));
  auto init_start = std::chrono::high_resolution_clock::now();
  BioGeometrySearchEngine engine(builder, locality_profile_config(profile));
  auto init_end = std::chrono::high_resolution_clock::now();

  LocalityBenchmarkRow row;
  row.dataset = dataset;
  row.profile = profile;
  row.batch_schedule_mode = batch_schedule_mode;
  row.query_count = queries.size();
  row.load_ms = load_ms;
  row.engine_init_ms =
      std::chrono::duration<double, std::milli>(init_end - init_start).count();
  const FanoutSummary fanout = compute_fanout_summary(builder);
  row.mean_fanout = fanout.mean;
  row.p50_fanout = fanout.p50;
  row.p95_fanout = fanout.p95;
  row.max_fanout = fanout.max;
  const DuplicateSummary duplicate_summary = summarize_duplicates(queries);
  row.unique_query_count = duplicate_summary.unique_query_count;
  row.duplicate_group_count = duplicate_summary.duplicate_group_count;
  row.duplicate_ratio = duplicate_summary.duplicate_ratio;
  const NeighborSummary neighbor_summary = summarize_neighbors(queries);
  row.mean_neighbor_edit_distance = neighbor_summary.mean_edit_distance;
  row.p95_neighbor_edit_distance = neighbor_summary.p95_edit_distance;
  row.mean_neighbor_qgram_jaccard = neighbor_summary.mean_qgram_jaccard;

  std::vector<double> latencies;
  std::vector<double> world_access;
  std::vector<double> center_distance;
  std::vector<double> leaf_verify;
  std::vector<double> path_reuse_hits;
  std::vector<double> anchor_cache_hits;
  std::vector<double> child_shortlist_hits;
  std::vector<double> router_invoked;
  std::vector<double> router_hits;
  std::vector<double> local_router_invoked;
  std::vector<double> local_router_shortlisted;
  std::vector<double> best_first_invoked;
  std::vector<double> best_first_reordered;
  std::vector<double> child_count_before_router;
  std::vector<double> post_mbb_survivor_count;
  std::vector<double> safe_router_candidate_count;
  std::vector<double> candidate_ratio_to_all_children;
  std::vector<double> candidate_ratio_to_post_mbb_survivors;
  std::vector<double> children_actually_processed;
  std::vector<double> center_checks_saved;
  double router_query_invoked_count = 0.0;
  double local_router_query_invoked_count = 0.0;
  double safe_child_router_query_invoked_count = 0.0;
  size_t path_reuse_attempt_total = 0;
  size_t path_reuse_hit_total = 0;
  size_t anchor_cache_hit_total = 0;
  size_t child_shortlist_cache_hit_total = 0;
  size_t safe_child_candidate_cache_hit_total = 0;
  size_t productive_world_reuse_hit_total = 0;
  size_t verified_result_cache_hit_total = 0;
  size_t near_query_reuse_hit_total = 0;
  size_t near_query_triangle_pruned_total = 0;
  size_t near_query_center_distance_reused_total = 0;
	  size_t near_query_bound_fallback_total = 0;
	  size_t near_query_direct_verify_total = 0;
	  size_t near_query_leaf_triangle_pruned_total = 0;
	  size_t near_query_leaf_distance_reused_total = 0;
	  size_t near_query_leaf_bound_fallback_total = 0;
  std::unordered_map<std::string, SearchResult> exact_result_cache;
  exact_result_cache.reserve(queries.size());
  const bool enable_exact_result_cache = profile != "baseline";

  auto query_start = std::chrono::high_resolution_clock::now();
  const size_t progress_interval =
      queries.size() <= 32 ? 1 : std::max<size_t>(1, queries.size() / 8);
  for (size_t i = 0; i < queries.size(); ++i) {
    const auto& query = queries[i].query;
    const bool report_query =
        i == 0 || queries.size() <= 32 ||
        ((i + 1) % progress_interval == 0) || (i + 1 == queries.size());
    if (report_query) {
      locality_progress("query start dataset=" + dataset + " profile=" +
                        profile + " schedule=" + batch_schedule_mode +
                        " query=" + std::to_string(i + 1) + "/" +
                        std::to_string(queries.size()));
    }
    auto one_start = std::chrono::high_resolution_clock::now();
    SearchResult hits;
    SearchStats stats(static_cast<size_t>(builder.num_primary_layers()));
    bool used_exact_result_cache = false;
    const std::string cache_key = exact_result_cache_key(query.seq, tolerance);
    if (enable_exact_result_cache) {
      auto cache_it = exact_result_cache.find(cache_key);
      if (cache_it != exact_result_cache.end()) {
        bool cache_valid = true;
        SearchResult cached_hits;
        cached_hits.reserve(cache_it->second.size());
        for (LeafId hit : cache_it->second) {
          if (hit >= builder.sequence_store().size()) {
            cache_valid = false;
            break;
          }
          const int dist = compute_distance(
              query.seq, builder.sequence_store().sequence(hit));
          stats.dist_calc_count++;
          stats.candidate_verify_count++;
          stats.leaf_verify_count++;
          stats.leaf_exact_distance_call_count++;
          if (dist <= tolerance) {
            cached_hits.push_back(hit);
          } else {
            cache_valid = false;
            break;
          }
        }
        if (cache_valid) {
          used_exact_result_cache = true;
          verified_result_cache_hit_total++;
          hits = std::move(cached_hits);
          stats.result_count = hits.size();
        }
      }
    }
    if (!used_exact_result_cache) {
      auto search_result = engine.search_adaptive(query, tolerance);
      hits = std::move(search_result.first);
      stats = std::move(search_result.second);
      if (enable_exact_result_cache) {
        exact_result_cache[cache_key] = hits;
      }
    }
    auto one_end = std::chrono::high_resolution_clock::now();
    const auto ids =
        search_result_ids(hits, builder.sequence_store());
    if (observed_ids) observed_ids->push_back(ids);
    if (ids.empty()) ++row.fn_count;
    if (!baseline_ids.empty() && ids != baseline_ids[i]) ++row.mismatch_count;
    row.result_total += ids.size();
    latencies.push_back(
        std::chrono::duration<double, std::milli>(one_end - one_start).count());
    world_access.push_back(static_cast<double>(stats.world_access_count));
    center_distance.push_back(static_cast<double>(stats.center_distance_count));
    leaf_verify.push_back(static_cast<double>(stats.candidate_verify_count));
    path_reuse_hits.push_back(static_cast<double>(stats.path_reuse_hit_count));
    anchor_cache_hits.push_back(static_cast<double>(stats.anchor_cache_hit_count));
    child_shortlist_hits.push_back(
        static_cast<double>(stats.child_shortlist_reuse_hit_count));
    router_invoked.push_back(static_cast<double>(stats.router_hint_invoked_count));
    if (stats.router_hint_invoked_count > 0) router_query_invoked_count += 1.0;
    router_hits.push_back(static_cast<double>(stats.router_candidate_hit_count));
    local_router_invoked.push_back(
        static_cast<double>(stats.local_router_invoked_count));
    if (stats.local_router_invoked_count > 0) {
      local_router_query_invoked_count += 1.0;
    }
    if (stats.safe_child_router_invoked_count > 0) {
      safe_child_router_query_invoked_count += 1.0;
    }
    path_reuse_attempt_total += stats.path_reuse_attempt_count;
    path_reuse_hit_total += stats.path_reuse_hit_count;
    anchor_cache_hit_total += stats.anchor_cache_hit_count;
    child_shortlist_cache_hit_total += stats.child_shortlist_cache_hit_count;
    safe_child_candidate_cache_hit_total +=
        stats.safe_child_candidate_cache_hit_count;
    productive_world_reuse_hit_total += stats.productive_world_reuse_hit_count;
    near_query_reuse_hit_total += stats.near_query_reuse_hit_count +
                                  stats.anchor_cache_hit_count +
                                  stats.child_shortlist_cache_hit_count +
                                  stats.safe_child_candidate_cache_hit_count;
    near_query_triangle_pruned_total +=
        stats.near_query_triangle_pruned_count;
    near_query_center_distance_reused_total +=
        stats.near_query_center_distance_reused_count;
	    near_query_bound_fallback_total += stats.near_query_bound_fallback_count;
	    near_query_direct_verify_total += stats.near_query_direct_verify_count;
	    near_query_leaf_triangle_pruned_total +=
	        stats.near_query_leaf_triangle_pruned_count;
	    near_query_leaf_distance_reused_total +=
	        stats.near_query_leaf_distance_reused_count;
	    near_query_leaf_bound_fallback_total +=
	        stats.near_query_leaf_bound_fallback_count;
    local_router_shortlisted.push_back(
        static_cast<double>(stats.local_router_shortlist_child_count));
    best_first_invoked.push_back(
        static_cast<double>(stats.best_first_invoked_count));
    best_first_reordered.push_back(
        static_cast<double>(stats.best_first_reordered_count));
    child_count_before_router.push_back(
        static_cast<double>(stats.child_count_before_router));
    post_mbb_survivor_count.push_back(
        static_cast<double>(stats.post_mbb_survivor_count));
    safe_router_candidate_count.push_back(
        static_cast<double>(stats.safe_router_candidate_count));
    candidate_ratio_to_all_children.push_back(
        stats.candidate_ratio_to_all_children);
    candidate_ratio_to_post_mbb_survivors.push_back(
        stats.candidate_ratio_to_post_mbb_survivors);
    children_actually_processed.push_back(
        static_cast<double>(stats.children_actually_processed));
    center_checks_saved.push_back(static_cast<double>(stats.center_checks_saved));
    if (report_query) {
      locality_progress("query finish dataset=" + dataset + " profile=" +
                        profile + " schedule=" + batch_schedule_mode +
                        " query=" + std::to_string(i + 1) + "/" +
                        std::to_string(queries.size()) +
                        " latency_ms=" + std::to_string(latencies.back()) +
                        " hits=" + std::to_string(ids.size()) +
                        " leaf_verify=" +
                        std::to_string(stats.leaf_verify_count));
    }
  }
  auto query_end = std::chrono::high_resolution_clock::now();
  row.query_wall_ms =
      std::chrono::duration<double, std::milli>(query_end - query_start).count();
  row.mean_query_ms = average(latencies);
  row.p50_query_ms = percentile_or_zero(latencies, 0.50);
  row.p95_query_ms = percentile_or_zero(latencies, 0.95);
  row.mean_world_access = average(world_access);
  row.mean_center_distance = average(center_distance);
  row.mean_leaf_verify = average(leaf_verify);
  const double query_count = queries.empty() ? 1.0 : static_cast<double>(queries.size());
  row.router_invoked_ratio = router_query_invoked_count / query_count;
  row.local_router_invoked_ratio = local_router_query_invoked_count / query_count;
  row.safe_child_router_invoked_ratio =
      safe_child_router_query_invoked_count / query_count;
  row.path_reuse_hit_ratio =
      path_reuse_attempt_total == 0
          ? 0.0
          : static_cast<double>(path_reuse_hit_total) /
                static_cast<double>(path_reuse_attempt_total);
  row.anchor_cache_hit_count = anchor_cache_hit_total;
  row.child_shortlist_cache_hit_count = child_shortlist_cache_hit_total;
  row.safe_child_candidate_cache_hit_count = safe_child_candidate_cache_hit_total;
  row.productive_world_reuse_hit_count = productive_world_reuse_hit_total;
  row.verified_result_cache_hit_count = verified_result_cache_hit_total;
  row.near_query_reuse_hit_count = near_query_reuse_hit_total;
  row.near_query_triangle_pruned_count = near_query_triangle_pruned_total;
  row.near_query_center_distance_reused_count =
      near_query_center_distance_reused_total;
	  row.near_query_bound_fallback_count = near_query_bound_fallback_total;
	  row.near_query_direct_verify_count = near_query_direct_verify_total;
	  row.near_query_leaf_triangle_pruned_count =
	      near_query_leaf_triangle_pruned_total;
	  row.near_query_leaf_distance_reused_count =
	      near_query_leaf_distance_reused_total;
	  row.near_query_leaf_bound_fallback_count =
	      near_query_leaf_bound_fallback_total;
  row.mean_path_reuse_hits = average(path_reuse_hits);
  row.mean_anchor_cache_hits = average(anchor_cache_hits);
  row.mean_child_shortlist_hits = average(child_shortlist_hits);
  row.mean_router_invoked = average(router_invoked);
  row.mean_router_hits = average(router_hits);
  row.mean_local_router_invoked = average(local_router_invoked);
  row.mean_local_router_shortlisted = average(local_router_shortlisted);
  row.mean_best_first_invoked = average(best_first_invoked);
  row.mean_best_first_reordered = average(best_first_reordered);
  row.mean_child_count_before_router = average(child_count_before_router);
  row.mean_post_mbb_survivor_count = average(post_mbb_survivor_count);
  row.mean_safe_router_candidate_count = average(safe_router_candidate_count);
  row.mean_candidate_ratio_to_all_children =
      average(candidate_ratio_to_all_children);
  row.mean_candidate_ratio_to_post_mbb_survivors =
      average(candidate_ratio_to_post_mbb_survivors);
  row.mean_children_actually_processed = average(children_actually_processed);
  row.mean_center_checks_saved = average(center_checks_saved);
  locality_progress("profile finish dataset=" + dataset + " profile=" +
                    profile + " schedule=" + batch_schedule_mode +
                    " mean_ms=" + std::to_string(row.mean_query_ms) +
                    " p95_ms=" + std::to_string(row.p95_query_ms) +
                    " fn=" + std::to_string(row.fn_count) +
                    " mismatch=" + std::to_string(row.mismatch_count));
  return row;
}

bool contains_profile(const std::vector<std::string>& profiles,
                      const std::string& profile) {
  return std::find(profiles.begin(), profiles.end(), profile) != profiles.end();
}

std::string minimizer_sort_key(const std::string& sequence) {
  constexpr size_t kMinimizerK = 5;
  if (sequence.size() <= kMinimizerK) return sequence;
  std::string best = sequence.substr(0, kMinimizerK);
  for (size_t i = 1; i + kMinimizerK <= sequence.size(); ++i) {
    best = std::min(best, sequence.substr(i, kMinimizerK));
  }
  return best;
}

std::vector<size_t> locality_schedule_order(
    const std::vector<LocalityBenchmarkQuery>& queries,
    const std::string& schedule,
    unsigned seed) {
  std::vector<size_t> order(queries.size());
  std::iota(order.begin(), order.end(), 0);
  if (schedule == "original") {
    return order;
  }
  if (schedule == "random") {
    std::mt19937 generator(seed);
    std::shuffle(order.begin(), order.end(), generator);
    return order;
  }
  if (schedule == "qgram-signature" || schedule == "router-signature") {
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
      const std::string left_key =
          build_query_schedule_key(queries[left].query.seq);
      const std::string right_key =
          build_query_schedule_key(queries[right].query.seq);
      if (left_key != right_key) return left_key < right_key;
      return queries[left].query.seq < queries[right].query.seq;
    });
    return order;
  }
  if (schedule == "minimizer") {
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
      const std::string left_key = minimizer_sort_key(queries[left].query.seq);
      const std::string right_key = minimizer_sort_key(queries[right].query.seq);
      if (left_key != right_key) return left_key < right_key;
      return queries[left].query.seq < queries[right].query.seq;
    });
    return order;
  }
  if (schedule == "source-oracle") {
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
      return queries[left].source_pos < queries[right].source_pos;
    });
    return order;
  }
  throw std::invalid_argument("unknown locality batch schedule: " + schedule);
}

std::vector<LocalityBenchmarkQuery> apply_locality_schedule(
    const std::vector<LocalityBenchmarkQuery>& queries,
    const std::vector<size_t>& order) {
  std::vector<LocalityBenchmarkQuery> out;
  out.reserve(order.size());
  for (size_t idx : order) out.push_back(queries[idx]);
  return out;
}

std::vector<std::vector<std::string>> apply_result_schedule(
    const std::vector<std::vector<std::string>>& ids,
    const std::vector<size_t>& order) {
  std::vector<std::vector<std::string>> out;
  out.reserve(order.size());
  for (size_t idx : order) out.push_back(ids[idx]);
  return out;
}

void write_locality_queries_fastq(
    const std::string& path,
    const std::vector<std::string>& dataset_names,
    const std::map<std::string, const std::vector<LocalityBenchmarkQuery>*>&
        all_datasets) {
  if (path.empty()) return;
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open locality query FASTQ: " + path);
  }
  for (const auto& dataset_name : dataset_names) {
    const auto dataset_it = all_datasets.find(dataset_name);
    if (dataset_it == all_datasets.end() || dataset_it->second == nullptr) {
      throw std::runtime_error("unknown locality dataset for FASTQ export: " +
                               dataset_name);
    }
    const auto& queries = *dataset_it->second;
    for (size_t i = 0; i < queries.size(); ++i) {
      const auto& query = queries[i];
      out << '@' << dataset_name << '_' << i
          << " source_pos=" << query.source_pos << '\n'
          << query.query.seq << "\n+\n"
          << std::string(query.query.seq.size(), 'I') << '\n';
    }
  }
}

void validate_locality_profiles(const std::vector<std::string>& profiles) {
  if (profiles.empty()) {
    throw std::invalid_argument("locality benchmark profiles must not be empty");
  }
  for (const auto& profile : profiles) {
    if (profile != "baseline" && profile != "path_reuse" &&
        profile != "optimized") {
      throw std::invalid_argument(
          "locality benchmark profile must be baseline, path_reuse, or optimized");
    }
  }
}

void validate_locality_datasets(const std::vector<std::string>& datasets) {
  if (datasets.empty()) {
    throw std::invalid_argument("locality benchmark datasets must not be empty");
  }
  for (const auto& dataset : datasets) {
    if (dataset != "same_template" && dataset != "nearby_windows" &&
        dataset != "random_windows" && dataset != "repeat" &&
        dataset != "batch_locality" && dataset != "oracle" &&
        dataset != "low_fanout" && dataset != "high_fanout" &&
        dataset != "real_dup_1x" && dataset != "real_dup_4x" &&
        dataset != "real_dup_16x" &&
        dataset != "source_sorted_stride1" &&
        dataset != "source_sorted_mutated_tau5" &&
        dataset != "source_sorted_mutated_tau8") {
      throw std::invalid_argument(
          "locality benchmark dataset must be same_template, nearby_windows, "
          "random_windows, repeat, batch_locality, oracle, low_fanout, "
          "high_fanout, real_dup_1x, real_dup_4x, real_dup_16x, "
          "source_sorted_stride1, source_sorted_mutated_tau5, or "
          "source_sorted_mutated_tau8");
    }
  }
}

std::string normalize_locality_name(std::string name) {
  std::replace(name.begin(), name.end(), '-', '_');
  return name;
}

std::vector<std::string> expand_locality_scenarios(
    const std::vector<std::string>& scenarios,
    const std::vector<std::string>& fallback_datasets) {
  if (scenarios.empty()) {
    std::vector<std::string> normalized;
    normalized.reserve(fallback_datasets.size());
    for (const auto& dataset : fallback_datasets) {
      normalized.push_back(normalize_locality_name(dataset));
    }
    return normalized;
  }

  std::vector<std::string> expanded;
  auto add_unique = [&](const std::string& value) {
    if (std::find(expanded.begin(), expanded.end(), value) == expanded.end()) {
      expanded.push_back(value);
    }
  };
  auto add_all = [&]() {
    add_unique("low_fanout");
    add_unique("high_fanout");
    add_unique("repeat");
    add_unique("batch_locality");
    add_unique("oracle");
    add_unique("real_dup_1x");
    add_unique("real_dup_4x");
    add_unique("real_dup_16x");
    add_unique("source_sorted_stride1");
    add_unique("source_sorted_mutated_tau5");
    add_unique("source_sorted_mutated_tau8");
  };

  for (const auto& raw : scenarios) {
    const std::string scenario = normalize_locality_name(raw);
    if (scenario == "all") {
      add_all();
    } else if (scenario == "low_fanout" || scenario == "high_fanout" ||
               scenario == "repeat" || scenario == "batch_locality" ||
               scenario == "oracle" || scenario == "same_template" ||
               scenario == "nearby_windows" || scenario == "random_windows" ||
               scenario == "real_dup_1x" || scenario == "real_dup_4x" ||
               scenario == "real_dup_16x" ||
               scenario == "source_sorted_stride1" ||
               scenario == "source_sorted_mutated_tau5" ||
               scenario == "source_sorted_mutated_tau8") {
      add_unique(scenario);
    } else {
      throw std::invalid_argument(
          "locality benchmark scenario must be low-fanout, high-fanout, "
          "repeat, batch-locality, oracle, real-dup-1x, real-dup-4x, "
          "real-dup-16x, source-sorted-stride1, source-sorted-mutated-tau5, "
          "source-sorted-mutated-tau8, all, or a locality dataset name");
    }
  }
  return expanded;
}

void validate_locality_batch_schedules(const std::vector<std::string>& schedules) {
  if (schedules.empty()) {
    throw std::invalid_argument("locality benchmark batch schedules must not be empty");
  }
  for (const auto& schedule : schedules) {
    if (schedule != "original" && schedule != "random" &&
        schedule != "minimizer" && schedule != "qgram-signature" &&
        schedule != "router-signature" && schedule != "source-oracle") {
      throw std::invalid_argument(
          "locality benchmark batch schedule must be original, random, "
          "minimizer, qgram-signature, router-signature, or source-oracle");
    }
  }
}

std::vector<std::vector<std::string>> baseline_locality_ids(
    const BioGeometryIndexBuilder& builder,
    const std::vector<LocalityBenchmarkQuery>& queries,
    int tolerance) {
  BioGeometrySearchEngine engine(builder, locality_profile_config("baseline"));
  std::vector<std::vector<std::string>> out;
  out.reserve(queries.size());
  std::unordered_map<std::string, std::vector<std::string>> exact_query_cache;
  exact_query_cache.reserve(queries.size());
  for (const auto& query : queries) {
    const auto cached = exact_query_cache.find(query.query.seq);
    if (cached != exact_query_cache.end()) {
      out.push_back(cached->second);
      continue;
    }
    auto [hits, stats] = engine.search_adaptive(query.query, tolerance);
    (void)stats;
    auto ids =
        search_result_ids(hits, builder.sequence_store());
    auto inserted = exact_query_cache.emplace(query.query.seq, ids);
    out.push_back(inserted.first->second);
  }
  return out;
}

std::vector<std::string> locality_columns() {
  return {
      "dataset", "profile", "batch_schedule_mode", "query_count", "result_total", "fn_count",
      "mismatch_count", "load_ms", "engine_init_ms", "query_wall_ms",
      "mean_query_ms", "p50_query_ms", "p95_query_ms",
      "mean_world_access", "mean_center_distance", "mean_leaf_verify",
      "mean_fanout", "p50_fanout", "p95_fanout", "max_fanout",
      "router_invoked_ratio", "local_router_invoked_ratio",
      "safe_child_router_invoked_ratio", "path_reuse_hit_ratio",
      "unique_query_count", "duplicate_group_count", "duplicate_ratio",
      "verified_result_cache_hit_count", "near_query_reuse_hit_count",
      "near_query_triangle_pruned_count",
      "near_query_center_distance_reused_count",
      "near_query_bound_fallback_count", "near_query_direct_verify_count",
      "center_distance_reduction", "world_access_reduction", "p95_speedup",
      "mean_neighbor_edit_distance", "p95_neighbor_edit_distance",
      "mean_neighbor_qgram_jaccard",
      "anchor_cache_hit_count", "child_shortlist_cache_hit_count",
      "safe_child_candidate_cache_hit_count", "productive_world_reuse_hit_count",
      "mean_path_reuse_hits", "mean_anchor_cache_hits",
      "mean_child_shortlist_hits", "mean_router_invoked", "mean_router_hits",
      "mean_local_router_invoked", "mean_local_router_shortlisted",
      "mean_best_first_invoked", "mean_best_first_reordered",
      "mean_child_count_before_router", "mean_post_mbb_survivor_count",
      "mean_safe_router_candidate_count",
      "mean_candidate_ratio_to_all_children",
	      "mean_candidate_ratio_to_post_mbb_survivors",
	      "mean_children_actually_processed", "mean_center_checks_saved",
	      "near_query_leaf_triangle_pruned_count",
	      "near_query_leaf_distance_reused_count",
	      "near_query_leaf_bound_fallback_count"};
}

std::vector<std::string> locality_row_values(const LocalityBenchmarkRow& row) {
  return {
      row.dataset,
      row.profile,
      row.batch_schedule_mode,
      std::to_string(row.query_count),
      std::to_string(row.result_total),
      std::to_string(row.fn_count),
      std::to_string(row.mismatch_count),
      format_double(row.load_ms),
      format_double(row.engine_init_ms),
      format_double(row.query_wall_ms),
      format_double(row.mean_query_ms),
      format_double(row.p50_query_ms),
      format_double(row.p95_query_ms),
      format_double(row.mean_world_access),
      format_double(row.mean_center_distance),
      format_double(row.mean_leaf_verify),
      format_double(row.mean_fanout),
      format_double(row.p50_fanout),
      format_double(row.p95_fanout),
      format_double(row.max_fanout),
      format_double(row.router_invoked_ratio),
      format_double(row.local_router_invoked_ratio),
      format_double(row.safe_child_router_invoked_ratio),
      format_double(row.path_reuse_hit_ratio),
      std::to_string(row.unique_query_count),
      std::to_string(row.duplicate_group_count),
      format_double(row.duplicate_ratio),
      std::to_string(row.verified_result_cache_hit_count),
      std::to_string(row.near_query_reuse_hit_count),
      std::to_string(row.near_query_triangle_pruned_count),
      std::to_string(row.near_query_center_distance_reused_count),
      std::to_string(row.near_query_bound_fallback_count),
      std::to_string(row.near_query_direct_verify_count),
      format_double(row.center_distance_reduction),
      format_double(row.world_access_reduction),
      format_double(row.p95_speedup),
      format_double(row.mean_neighbor_edit_distance),
      format_double(row.p95_neighbor_edit_distance),
      format_double(row.mean_neighbor_qgram_jaccard),
      std::to_string(row.anchor_cache_hit_count),
      std::to_string(row.child_shortlist_cache_hit_count),
      std::to_string(row.safe_child_candidate_cache_hit_count),
      std::to_string(row.productive_world_reuse_hit_count),
      format_double(row.mean_path_reuse_hits),
      format_double(row.mean_anchor_cache_hits),
      format_double(row.mean_child_shortlist_hits),
      format_double(row.mean_router_invoked),
      format_double(row.mean_router_hits),
      format_double(row.mean_local_router_invoked),
      format_double(row.mean_local_router_shortlisted),
      format_double(row.mean_best_first_invoked),
      format_double(row.mean_best_first_reordered),
      format_double(row.mean_child_count_before_router),
      format_double(row.mean_post_mbb_survivor_count),
      format_double(row.mean_safe_router_candidate_count),
      format_double(row.mean_candidate_ratio_to_all_children),
	      format_double(row.mean_candidate_ratio_to_post_mbb_survivors),
	      format_double(row.mean_children_actually_processed),
	      format_double(row.mean_center_checks_saved),
	      std::to_string(row.near_query_leaf_triangle_pruned_count),
	      std::to_string(row.near_query_leaf_distance_reused_count),
	      std::to_string(row.near_query_leaf_bound_fallback_count)};
}

LocalityBenchmarkRunResult run_persisted_locality_benchmark(
    const LocalityBenchmarkConfig& config) {
  if (config.index_path.empty()) {
    throw std::invalid_argument("locality benchmark requires index_path");
  }
  if (config.ref_input.empty()) {
    throw std::invalid_argument("locality benchmark requires ref_input");
  }
  if (config.out_tsv_path.empty()) {
    throw std::invalid_argument("locality benchmark requires out_tsv_path");
  }

  locality_progress("load index start path=" + config.index_path);
  auto load_start = std::chrono::high_resolution_clock::now();
  LoadedIndex loaded = load_index(config.index_path);
  auto load_end = std::chrono::high_resolution_clock::now();

  LocalityBenchmarkRunResult result;
  result.load_ms =
      std::chrono::duration<double, std::milli>(load_end - load_start).count();
  locality_progress("load index finish ms=" + std::to_string(result.load_ms));

  locality_progress("load reference start input=" + config.ref_input);
  auto [ref_id, ref_seq] = load_reference(config.ref_input);
  (void)ref_id;
  locality_progress("load reference finish length=" +
                    std::to_string(ref_seq.size()));
  locality_progress("generate query sets start query_count=" +
                    std::to_string(config.query_count) +
                    " query_length=" + std::to_string(config.query_length) +
                    " edits=" + std::to_string(config.edits));
  const auto query_sets = generate_locality_benchmark_queries(
      ref_seq, config.query_count, config.query_length, config.edits,
      config.seed);
  locality_progress("generate query sets finish");
  validate_locality_profiles(config.profiles);
  const auto dataset_names =
      expand_locality_scenarios(config.scenarios, config.datasets);
  validate_locality_datasets(dataset_names);
  validate_locality_batch_schedules(config.batch_schedules);
  locality_progress("validated profiles=" + std::to_string(config.profiles.size()) +
                    " datasets=" + std::to_string(dataset_names.size()) +
                    " schedules=" +
                    std::to_string(config.batch_schedules.size()));

  const std::map<std::string, const std::vector<LocalityBenchmarkQuery>*>
      all_datasets = {{"same_template", &query_sets.same_template},
                      {"nearby_windows", &query_sets.nearby_windows},
                      {"random_windows", &query_sets.random_windows},
                      {"low_fanout", &query_sets.random_windows},
                      {"high_fanout", &query_sets.nearby_windows},
                      {"repeat", &query_sets.repeat},
                      {"batch_locality", &query_sets.batch_locality},
                      {"oracle", &query_sets.oracle},
                      {"real_dup_1x", &query_sets.real_dup_1x},
                      {"real_dup_4x", &query_sets.real_dup_4x},
                      {"real_dup_16x", &query_sets.real_dup_16x},
                      {"source_sorted_stride1",
                       &query_sets.source_sorted_stride1},
                      {"source_sorted_mutated_tau5",
                       &query_sets.source_sorted_mutated_tau5},
                      {"source_sorted_mutated_tau8",
                       &query_sets.source_sorted_mutated_tau8}};
  write_locality_queries_fastq(config.query_fastq_out_path, dataset_names,
                               all_datasets);
  bool gate_passed = true;
  for (const auto& dataset_name : dataset_names) {
    const auto& queries = *all_datasets.at(dataset_name);
    locality_progress("dataset start name=" + dataset_name +
                      " queries=" + std::to_string(queries.size()));
    locality_progress("baseline ids start dataset=" + dataset_name);
    const auto original_baseline_ids =
        baseline_locality_ids(loaded.builder, queries, config.tolerance);
    locality_progress("baseline ids finish dataset=" + dataset_name);
    for (const auto& schedule : config.batch_schedules) {
      locality_progress("schedule start dataset=" + dataset_name +
                        " schedule=" + schedule);
      const auto order =
          locality_schedule_order(queries, schedule, config.seed + 7919U);
      const auto scheduled_queries = apply_locality_schedule(queries, order);
      const auto baseline_ids = apply_result_schedule(original_baseline_ids, order);
      LocalityBenchmarkRow baseline_reference;
      bool have_baseline_reference = false;
      if (contains_profile(config.profiles, "baseline")) {
        LocalityBenchmarkRow row = run_locality_profile(
            dataset_name, "baseline", schedule, loaded.builder, scheduled_queries,
            {}, config.tolerance, result.load_ms, nullptr);
        if (row.mismatch_count != 0) gate_passed = false;
        row.center_distance_reduction = 0.0;
        row.world_access_reduction = 0.0;
        row.p95_speedup = 1.0;
        baseline_reference = row;
        have_baseline_reference = true;
        result.rows.push_back(std::move(row));
      } else {
        locality_progress("hidden baseline reference skipped dataset=" +
                          dataset_name + " schedule=" + schedule +
                          " reason=baseline_profile_not_requested");
      }
      for (const auto& profile : config.profiles) {
        if (profile == "baseline") continue;
        LocalityBenchmarkRow row = run_locality_profile(
            dataset_name, profile, schedule, loaded.builder, scheduled_queries,
            baseline_ids, config.tolerance, result.load_ms, nullptr);
        if (row.mismatch_count != 0) gate_passed = false;
        if (have_baseline_reference) {
          row.center_distance_reduction =
              baseline_reference.mean_center_distance - row.mean_center_distance;
          row.world_access_reduction =
              baseline_reference.mean_world_access - row.mean_world_access;
          row.p95_speedup =
              safe_speedup(baseline_reference.p95_query_ms, row.p95_query_ms);
        }
        result.rows.push_back(std::move(row));
      }
      locality_progress("schedule finish dataset=" + dataset_name +
                        " schedule=" + schedule);
    }
    locality_progress("dataset finish name=" + dataset_name);
  }
  result.gate_passed = gate_passed;

  std::vector<std::vector<std::string>> rows;
  rows.reserve(result.rows.size());
  for (const auto& row : result.rows) {
    rows.push_back(locality_row_values(row));
  }
  locality_progress("write summary start path=" + config.out_tsv_path +
                    " rows=" + std::to_string(rows.size()));
  write_tsv(config.out_tsv_path, locality_columns(), rows);
  locality_progress("write summary finish path=" + config.out_tsv_path);
  return result;
}

void write_locality_report_outputs(
    const LocalityBenchmarkRunResult& result,
    const std::string& json_path,
    const std::string& markdown_path) {
  const auto columns = locality_columns();
  std::ofstream json(json_path);
  if (!json) {
    throw std::runtime_error("failed to open locality report JSON: " +
                             json_path);
  }
  json << "{\"gate_passed\":" << bool_string(result.gate_passed)
       << ",\"load_ms\":" << format_double(result.load_ms) << ",\"rows\":[";
  for (size_t row_index = 0; row_index < result.rows.size(); ++row_index) {
    if (row_index) json << ",";
    const auto values = locality_row_values(result.rows[row_index]);
    json << "{";
    for (size_t column_index = 0; column_index < columns.size(); ++column_index) {
      if (column_index) json << ",";
      json << "\"" << json_escape(columns[column_index]) << "\":\""
           << json_escape(values[column_index]) << "\"";
    }
    json << "}";
  }
  json << "]}";

  std::ofstream markdown(markdown_path);
  if (!markdown) {
    throw std::runtime_error("failed to open locality report Markdown: " +
                             markdown_path);
  }
  markdown << "# NavigaMer Query Locality Report\n\n";
  markdown << "- gate_passed: " << (result.gate_passed ? "true" : "false")
           << "\n";
  markdown << "- index_load_ms: " << format_double(result.load_ms) << "\n";
  markdown << "- rows: " << result.rows.size() << "\n\n";
  markdown << "| dataset | profile | batch_schedule_mode | query_count | "
              "mismatch_count | mean_query_ms | p95_query_ms | "
              "mean_world_access | mean_center_distance | router_invoked_ratio | "
              "safe_child_router_invoked_ratio | path_reuse_hit_ratio |\n";
  markdown << "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  for (const auto& row : result.rows) {
    markdown << "| " << row.dataset << " | " << row.profile << " | "
             << row.batch_schedule_mode << " | " << row.query_count << " | "
             << row.mismatch_count << " | " << format_double(row.mean_query_ms)
             << " | " << format_double(row.p95_query_ms) << " | "
             << format_double(row.mean_world_access) << " | "
             << format_double(row.mean_center_distance) << " | "
             << format_double(row.router_invoked_ratio) << " | "
             << format_double(row.safe_child_router_invoked_ratio) << " | "
             << format_double(row.path_reuse_hit_ratio) << " |\n";
  }
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

  auto queries = generate_benchmark_queries(
      index_sequences, builder.sequence_store(), config.query_length,
      config.tolerance,
      config.seed, config.queries_per_class);
  if (optimized_search_config.path_reuse_enabled) {
    std::stable_sort(queries.begin(), queries.end(),
                     [](const GeneratedBenchmarkQuery& left,
                        const GeneratedBenchmarkQuery& right) {
                       return build_query_schedule_key(left.query.seq) <
                              build_query_schedule_key(right.query.seq);
                     });
  }
  std::map<std::string, size_t> generation_counts;
  for (const auto& query : queries) {
    generation_counts[query_class_name(query.query_class)]++;
  }

  auto profile_specs = build_query_benchmark_profiles(
      optimized_search_config, config.enable_ablation_profiles);
  if (config.proximal_oracle_enabled) {
    for (auto& profile : profile_specs) {
      profile.config.proximal_oracle_enabled = true;
    }
  }
  std::map<std::string, SearchConfig> profile_configs;
  std::map<std::string, std::unique_ptr<BioGeometrySearchEngine>> engines;
  std::vector<std::string> profile_names;
  profile_names.reserve(profile_specs.size());
  for (const auto& profile : profile_specs) {
    profile_configs[profile.name] = profile.config;
    engines[profile.name] =
        std::make_unique<BioGeometrySearchEngine>(builder, profile.config);
    profile_names.push_back(profile.name);
  }

  std::vector<uint8_t> eviction_buffer(config.cold_cache_bytes, 0);
  const ProximalOracleContext proximal_context =
      build_proximal_oracle_context(builder, config);
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
                     const std::string& profile,
                     size_t profile_rank,
                     const std::string& sample_kind,
                     size_t iteration,
                     const std::string& first_profile,
                     bool timed,
                     bool cold) {
    if (cold) evict_best_effort();
    const auto start = std::chrono::steady_clock::now();
    auto [hits, stats] =
        engine.search_adaptive(generated.query, config.tolerance);
    const auto end = std::chrono::steady_clock::now();
    ExecutionRecord record;
    record.query_id = generated.query.id;
    record.profile = profile;
    record.query_class = generated.query_class;
    record.profile_rank = profile_rank;
    record.sample_kind = sample_kind;
    record.first_profile = first_profile;
    record.iteration = iteration;
    record.latency_ms =
        timed ? std::chrono::duration<double, std::milli>(end - start).count()
              : 0.0;
    record.result_ids =
        result_ids(hits, builder.sequence_store());
    record.result_count = record.result_ids.size();
    record.brute_force_result_count = generated.brute_force_ids.size();
    record.stats = std::move(stats);
    record.proximal =
        compute_proximal_oracle_record(proximal_context, generated, record.stats);
    return record;
  };

  for (size_t query_index = 0; query_index < queries.size(); ++query_index) {
    const auto& generated = queries[query_index];
    std::vector<std::string> order = profile_names;
    const size_t rotation = query_index % order.size();
    std::rotate(order.begin(), order.begin() + rotation, order.end());
    const std::string first_profile = order.front();
    const size_t query_record_start = records.size();
    std::map<std::string, std::vector<std::string>> canonical_ids;
    bool repeated_results_equal = true;

    for (size_t profile_rank = 0; profile_rank < order.size(); ++profile_rank) {
      const std::string& profile = order[profile_rank];
      BioGeometrySearchEngine& engine = *engines.at(profile);
      auto cold = execute(engine, generated, profile, profile_rank, "cold", 0,
                          first_profile, true, true);
      canonical_ids[profile] = cold.result_ids;
      records.push_back(std::move(cold));

      for (size_t iteration = 0; iteration < config.warmup_iterations;
           ++iteration) {
        auto warmup = execute(engine, generated, profile, profile_rank, "warmup",
                              iteration, first_profile, false, false);
        repeated_results_equal &= warmup.result_ids == canonical_ids[profile];
      }
      for (size_t iteration = 0; iteration < config.measured_iterations;
           ++iteration) {
        auto warm = execute(engine, generated, profile, profile_rank, "warm",
                            iteration, first_profile, true, false);
        repeated_results_equal &= warm.result_ids == canonical_ids[profile];
        records.push_back(std::move(warm));
      }
    }

    const ResultComparison comparison =
        canonical_ids.count("optimized")
            ? compare_result_ids(canonical_ids["baseline"],
                                 canonical_ids["optimized"],
                                 generated.brute_force_ids)
            : compare_result_ids(canonical_ids["baseline"],
                                 canonical_ids["baseline"],
                                 generated.brute_force_ids);
    std::vector<std::string> failing_profiles;
    for (const auto& profile : profile_names) {
      const auto& ids = canonical_ids[profile];
      if (ids != generated.brute_force_ids || ids != canonical_ids["baseline"]) {
        failing_profiles.push_back(profile);
      }
    }
    const bool query_gate_passed =
        repeated_results_equal && failing_profiles.empty() &&
        comparison_passes_gate(comparison);
    if (!query_gate_passed) {
      result.mismatch_count++;
      mismatch_diagnostics.push_back(
          {generated.query.id, repeated_results_equal, comparison,
           failing_profiles});
    }
    for (size_t i = query_record_start; i < records.size(); ++i) {
      records[i].result_equal =
          repeated_results_equal &&
          records[i].result_ids == canonical_ids["baseline"] &&
          records[i].result_ids == generated.brute_force_ids;
      records[i].no_fn =
          difference(generated.brute_force_ids, records[i].result_ids).empty();
    }
  }
  static_cast<void>(eviction_checksum);
  result.gate_passed = result.mismatch_count == 0;
  const MemorySnapshot after_benchmark = memory_snapshot();

  const std::vector<std::string> detail_columns = {
      "query_id", "query_class", "profile", "profile_rank", "sample_kind",
      "iteration", "first_profile", "latency_ms", "result_count",
      "brute_force_result_count", "result_equal", "no_fn",
      "query_profile_enabled", "query_count", "query_total_ms",
      "router_lookup_ms", "anchor_distance_ms", "mbb_filter_ms",
      "child_bound_ms", "center_distance_ms", "best_first_queue_ms",
      "leaf_collect_ms", "leaf_mbb_filter_ms", "leaf_verify_ms",
      "result_dedup_ms", "path_reuse_ms",
      "path_reuse_attempt_count", "path_reuse_hit_count",
      "near_query_triangle_pruned_count",
      "near_query_center_distance_reused_count",
	      "near_query_bound_fallback_count", "near_query_direct_verify_count",
	      "near_query_leaf_triangle_pruned_count",
	      "near_query_leaf_distance_reused_count",
	      "near_query_leaf_bound_fallback_count",
	      "anchor_cache_hit_count", "child_shortlist_reuse_hit_count",
      "router_hint_invoked_count", "router_qgram_ranked_count",
      "router_minimizer_ranked_count", "router_pigeonhole_query_count",
      "router_candidate_count", "router_candidate_hit_count",
      "router_fallback_count",
      "local_router_enabled_count", "local_router_invoked_count",
      "local_router_empty_count", "local_router_shortlist_child_count",
      "local_router_remaining_child_count", "local_router_fallback_count",
      "best_first_invoked_count", "best_first_reordered_count",
      "best_first_bound_candidate_count", "child_safe_bound_pruned_count",
      "safe_child_router_invoked_count",
      "safe_child_router_skipped_low_fanout_count",
      "safe_child_router_fallback_count",
      "safe_child_router_candidate_count",
      "safe_child_router_pruned_by_not_candidate_count",
      "safe_child_router_exact_verify_count",
      "child_count_before_router", "post_mbb_survivor_count",
      "safe_router_candidate_count", "candidate_ratio_to_all_children",
      "candidate_ratio_to_post_mbb_survivors",
      "children_actually_processed", "center_checks_saved",
      "planner_invoked_count", "planner_strategy_baseline_count",
      "planner_strategy_direct_qgram_count", "planner_strategy_router_count",
      "planner_strategy_safe_child_router_count",
      "planner_strategy_path_reuse_count",
      "planner_near_reuse_enabled_count",
      "planner_near_reuse_disabled_count", "planner_fallback_count",
      "planner_decision_ms",
      "actual_envelope_k1", "actual_envelope_k2", "actual_envelope_k4",
      "frontier_oracle_envelope_k1", "frontier_oracle_envelope_k2",
      "frontier_oracle_envelope_k4",
      "true_path_oracle_envelope_k1", "true_path_oracle_envelope_k2",
      "true_path_oracle_envelope_k4",
      "global_oracle_envelope_k1", "global_oracle_envelope_k2",
      "global_oracle_envelope_k4",
      "random_envelope_k1", "random_envelope_k2", "random_envelope_k4",
      "actual_nearest_anchor_dist", "frontier_oracle_nearest_anchor_dist",
      "true_path_oracle_nearest_anchor_dist",
      "global_oracle_nearest_anchor_dist", "random_nearest_anchor_dist",
      "global_oracle_gap_vs_actual_k1", "global_oracle_gap_vs_actual_k2",
      "global_oracle_gap_vs_actual_k4",
      "global_oracle_gap_vs_frontier_k1",
      "global_oracle_gap_vs_frontier_k2",
      "global_oracle_gap_vs_frontier_k4",
      "world_access_count", "node_access_count", "edge_access_count",
      "frontier_max_size", "frontier_total_pushed",
      "anchor_distance_count", "center_distance_count",
      "raw_candidate_count",
      "mbb_checks", "mbb_survivors", "mbb_scalar_checks",
      "mbb_simd_batches", "mbb_simd_fallbacks", "qgram_checks",
      "center_exact_distance_calls", "leaf_beacon_checks",
      "leaf_beacon_scalar_checks", "leaf_beacon_simd_batches",
      "leaf_beacon_simd_fallbacks",
      "leaf_exact_distance_calls", "visited_checks", "visited_hits",
      "candidate_count", "verified_candidate_count"};
  for (const auto& record : records) {
    result.detail_rows.push_back({
        record.query_id,
        query_class_name(record.query_class),
        record.profile,
        std::to_string(record.profile_rank),
        record.sample_kind,
        std::to_string(record.iteration),
        record.first_profile,
        format_double(record.latency_ms),
        std::to_string(record.result_count),
        std::to_string(record.brute_force_result_count),
        bool_string(record.result_equal),
        bool_string(record.no_fn),
        bool_string(record.stats.query_profile_enabled),
        std::to_string(record.stats.query_count),
        format_double(record.stats.query_total_ms),
        format_double(record.stats.router_lookup_ms),
        format_double(record.stats.anchor_distance_ms),
        format_double(record.stats.mbb_filter_ms),
        format_double(record.stats.child_bound_ms),
        format_double(record.stats.center_distance_ms),
        format_double(record.stats.best_first_queue_ms),
        format_double(record.stats.leaf_collect_ms),
        format_double(record.stats.leaf_mbb_filter_ms),
        format_double(record.stats.leaf_verify_ms),
        format_double(record.stats.result_dedup_ms),
        format_double(record.stats.path_reuse_ms),
        std::to_string(record.stats.path_reuse_attempt_count),
        std::to_string(record.stats.path_reuse_hit_count),
        std::to_string(record.stats.near_query_triangle_pruned_count),
        std::to_string(record.stats.near_query_center_distance_reused_count),
	        std::to_string(record.stats.near_query_bound_fallback_count),
	        std::to_string(record.stats.near_query_direct_verify_count),
	        std::to_string(record.stats.near_query_leaf_triangle_pruned_count),
	        std::to_string(record.stats.near_query_leaf_distance_reused_count),
	        std::to_string(record.stats.near_query_leaf_bound_fallback_count),
	        std::to_string(record.stats.anchor_cache_hit_count),
        std::to_string(record.stats.child_shortlist_reuse_hit_count),
        std::to_string(record.stats.router_hint_invoked_count),
        std::to_string(record.stats.router_qgram_ranked_count),
        std::to_string(record.stats.router_minimizer_ranked_count),
        std::to_string(record.stats.router_pigeonhole_query_count),
        std::to_string(record.stats.router_candidate_count),
        std::to_string(record.stats.router_candidate_hit_count),
        std::to_string(record.stats.router_fallback_count),
        std::to_string(record.stats.local_router_enabled_count),
        std::to_string(record.stats.local_router_invoked_count),
        std::to_string(record.stats.local_router_empty_count),
        std::to_string(record.stats.local_router_shortlist_child_count),
        std::to_string(record.stats.local_router_remaining_child_count),
        std::to_string(record.stats.local_router_fallback_count),
        std::to_string(record.stats.best_first_invoked_count),
        std::to_string(record.stats.best_first_reordered_count),
        std::to_string(record.stats.best_first_bound_candidate_count),
        std::to_string(record.stats.child_safe_bound_pruned_count),
        std::to_string(record.stats.safe_child_router_invoked_count),
        std::to_string(record.stats.safe_child_router_skipped_low_fanout_count),
        std::to_string(record.stats.safe_child_router_fallback_count),
        std::to_string(record.stats.safe_child_router_candidate_count),
        std::to_string(
            record.stats.safe_child_router_pruned_by_not_candidate_count),
        std::to_string(record.stats.safe_child_router_exact_verify_count),
        std::to_string(record.stats.child_count_before_router),
        std::to_string(record.stats.post_mbb_survivor_count),
        std::to_string(record.stats.safe_router_candidate_count),
        format_double(record.stats.candidate_ratio_to_all_children),
        format_double(record.stats.candidate_ratio_to_post_mbb_survivors),
        std::to_string(record.stats.children_actually_processed),
        std::to_string(record.stats.center_checks_saved),
        std::to_string(record.stats.planner_invoked_count),
        std::to_string(record.stats.planner_strategy_baseline_count),
        std::to_string(record.stats.planner_strategy_direct_qgram_count),
        std::to_string(record.stats.planner_strategy_router_count),
        std::to_string(record.stats.planner_strategy_safe_child_router_count),
        std::to_string(record.stats.planner_strategy_path_reuse_count),
        std::to_string(record.stats.planner_near_reuse_enabled_count),
        std::to_string(record.stats.planner_near_reuse_disabled_count),
        std::to_string(record.stats.planner_fallback_count),
        format_double(record.stats.planner_decision_ms),
        format_double(record.proximal.actual_envelope[0]),
        format_double(record.proximal.actual_envelope[1]),
        format_double(record.proximal.actual_envelope[2]),
        format_double(record.proximal.frontier_oracle_envelope[0]),
        format_double(record.proximal.frontier_oracle_envelope[1]),
        format_double(record.proximal.frontier_oracle_envelope[2]),
        format_double(record.proximal.true_path_oracle_envelope[0]),
        format_double(record.proximal.true_path_oracle_envelope[1]),
        format_double(record.proximal.true_path_oracle_envelope[2]),
        format_double(record.proximal.global_oracle_envelope[0]),
        format_double(record.proximal.global_oracle_envelope[1]),
        format_double(record.proximal.global_oracle_envelope[2]),
        format_double(record.proximal.random_envelope[0]),
        format_double(record.proximal.random_envelope[1]),
        format_double(record.proximal.random_envelope[2]),
        format_double(record.proximal.actual_nearest_anchor_dist),
        format_double(record.proximal.frontier_oracle_nearest_anchor_dist),
        format_double(record.proximal.true_path_oracle_nearest_anchor_dist),
        format_double(record.proximal.global_oracle_nearest_anchor_dist),
        format_double(record.proximal.random_nearest_anchor_dist),
        format_double(record.proximal.global_gap_vs_actual[0]),
        format_double(record.proximal.global_gap_vs_actual[1]),
        format_double(record.proximal.global_gap_vs_actual[2]),
        format_double(record.proximal.global_gap_vs_frontier[0]),
        format_double(record.proximal.global_gap_vs_frontier[1]),
        format_double(record.proximal.global_gap_vs_frontier[2]),
        std::to_string(record.stats.world_access_count),
        std::to_string(record.stats.node_access_count),
        std::to_string(record.stats.edge_access_count),
        std::to_string(record.stats.frontier_max_size),
        std::to_string(record.stats.frontier_total_pushed),
        std::to_string(record.stats.anchor_distance_count),
        std::to_string(record.stats.center_distance_count),
        std::to_string(record.stats.raw_candidate_count),
        std::to_string(record.stats.mbb_check_count),
        std::to_string(record.stats.mbb_surviving_child_count),
        std::to_string(record.stats.mbb_scalar_checks),
        std::to_string(record.stats.mbb_simd_batches),
        std::to_string(record.stats.mbb_simd_fallbacks),
        std::to_string(record.stats.search_qgram_checks),
        std::to_string(record.stats.center_exact_distance_call_count),
        std::to_string(record.stats.leaf_beacon_check_count),
        std::to_string(record.stats.leaf_beacon_scalar_checks),
        std::to_string(record.stats.leaf_beacon_simd_batches),
        std::to_string(record.stats.leaf_beacon_simd_fallbacks),
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
      "warm_p50_ms", "warm_p95_ms", "warm_p99_ms",
      "cold_avg_query_ms", "cold_p50_query_ms", "cold_p95_query_ms",
      "warm_avg_query_ms", "warm_p50_query_ms", "warm_p95_query_ms",
      "cold_avg_speedup_vs_baseline", "warm_avg_speedup_vs_baseline",
      "avg_world_access_ratio_vs_baseline",
      "avg_center_distance_ratio_vs_baseline",
      "avg_raw_candidate_ratio_vs_baseline",
      "avg_world_access_count", "avg_node_access_count", "avg_edge_access_count",
      "avg_anchor_distance_count", "avg_center_distance_count",
      "avg_raw_candidate_count", "avg_mbb_checks", "avg_mbb_survivors",
      "avg_mbb_scalar_checks",
      "avg_mbb_simd_batches", "avg_mbb_simd_fallbacks", "avg_qgram_checks",
      "avg_path_reuse_attempt_count", "avg_path_reuse_hit_count",
      "avg_near_query_triangle_pruned_count",
      "avg_near_query_center_distance_reused_count",
	      "avg_near_query_bound_fallback_count",
	      "avg_near_query_direct_verify_count",
	      "avg_near_query_leaf_triangle_pruned_count",
	      "avg_near_query_leaf_distance_reused_count",
	      "avg_near_query_leaf_bound_fallback_count",
	      "center_distance_reduction", "world_access_reduction", "p95_speedup",
      "avg_anchor_cache_hit_count", "avg_child_shortlist_reuse_hit_count",
      "avg_router_hint_invoked_count", "avg_router_qgram_ranked_count",
      "avg_router_minimizer_ranked_count",
      "avg_router_pigeonhole_query_count", "avg_router_candidate_count",
      "avg_router_candidate_hit_count", "avg_router_fallback_count",
      "avg_local_router_invoked_count", "avg_local_router_empty_count",
      "avg_local_router_shortlist_child_count",
      "avg_local_router_remaining_child_count",
      "avg_local_router_fallback_count",
      "avg_best_first_invoked_count", "avg_best_first_reordered_count",
      "avg_best_first_bound_candidate_count",
      "avg_child_safe_bound_pruned_count",
      "avg_safe_child_router_invoked_count",
      "avg_safe_child_router_skipped_low_fanout_count",
      "avg_safe_child_router_fallback_count",
      "avg_safe_child_router_candidate_count",
      "avg_safe_child_router_pruned_by_not_candidate_count",
      "avg_safe_child_router_exact_verify_count",
      "avg_child_count_before_router", "avg_post_mbb_survivor_count",
      "avg_safe_router_candidate_count",
      "avg_candidate_ratio_to_all_children",
      "avg_candidate_ratio_to_post_mbb_survivors",
      "avg_children_actually_processed", "avg_center_checks_saved",
      "avg_planner_invoked_count", "avg_planner_strategy_baseline_count",
      "avg_planner_strategy_direct_qgram_count",
      "avg_planner_strategy_router_count",
      "avg_planner_strategy_safe_child_router_count",
      "avg_planner_strategy_path_reuse_count",
      "avg_planner_near_reuse_enabled_count",
      "avg_planner_near_reuse_disabled_count", "avg_planner_fallback_count",
      "avg_planner_decision_ms",
      "mean_actual_envelope_k1", "mean_actual_envelope_k2",
      "mean_actual_envelope_k4",
      "mean_frontier_oracle_envelope_k1",
      "mean_frontier_oracle_envelope_k2",
      "mean_frontier_oracle_envelope_k4",
      "mean_true_path_oracle_envelope_k1",
      "mean_true_path_oracle_envelope_k2",
      "mean_true_path_oracle_envelope_k4",
      "mean_global_oracle_envelope_k1",
      "mean_global_oracle_envelope_k2",
      "mean_global_oracle_envelope_k4",
      "mean_random_envelope_k1", "mean_random_envelope_k2",
      "mean_random_envelope_k4",
      "frac_global_oracle_much_better_than_actual_k1",
      "frac_global_oracle_much_better_than_actual_k2",
      "frac_global_oracle_much_better_than_actual_k4",
      "frac_global_oracle_much_better_than_frontier_k1",
      "frac_global_oracle_much_better_than_frontier_k2",
      "frac_global_oracle_much_better_than_frontier_k4",
      "avg_center_exact_distance_calls", "avg_leaf_beacon_checks",
      "avg_leaf_beacon_scalar_checks", "avg_leaf_beacon_simd_batches",
      "avg_leaf_beacon_simd_fallbacks",
      "avg_leaf_exact_distance_calls", "avg_frontier_max_size",
      "avg_frontier_total_pushed", "avg_visited_checks",
      "avg_visited_hits", "avg_candidate_count",
      "avg_verified_candidate_count"};
  auto aggregates = aggregate_records(records);
  struct BaselineAggregateView {
    double cold_avg_ms = 0.0;
    double warm_avg_ms = 0.0;
    double warm_p95_ms = 0.0;
    double avg_world_access_count = 0.0;
    double avg_center_distance_count = 0.0;
    double avg_raw_candidate_count = 0.0;
  };
  std::map<std::string, BaselineAggregateView> baseline_by_class;
  for (const auto& aggregate : aggregates) {
    if (aggregate.profile != "baseline") continue;
    const size_t sample_count = aggregate.sample_count;
    baseline_by_class[aggregate.query_class] = {
        average(aggregate.cold_latencies),
        average(aggregate.warm_latencies),
        percentile_or_zero(aggregate.warm_latencies, 0.95),
        average_counter(aggregate.world_access_count, sample_count),
        average_counter(aggregate.center_distance_count, sample_count),
        average_counter(aggregate.raw_candidate_count, sample_count),
    };
  }
  for (const auto& aggregate : aggregates) {
    const size_t sample_count = aggregate.sample_count;
    const size_t samples_per_query = 1 + config.measured_iterations;
    const auto baseline_it = baseline_by_class.find(aggregate.query_class);
    const BaselineAggregateView baseline =
        baseline_it == baseline_by_class.end()
            ? BaselineAggregateView{}
            : baseline_it->second;
    const double cold_avg_ms = average(aggregate.cold_latencies);
    const double warm_avg_ms = average(aggregate.warm_latencies);
    const double avg_world_access_count =
        average_counter(aggregate.world_access_count, sample_count);
    const double avg_center_distance_count =
        average_counter(aggregate.center_distance_count, sample_count);
    const double avg_raw_candidate_count =
        average_counter(aggregate.raw_candidate_count, sample_count);
    result.summary_rows.push_back({
        aggregate.query_class,
        aggregate.profile,
        std::to_string(sample_count / samples_per_query),
        std::to_string(sample_count),
        std::to_string(aggregate.result_total),
        std::to_string(aggregate.equality_failure_count),
        std::to_string(aggregate.false_negative_count),
        format_double(cold_avg_ms),
        format_double(percentile_or_zero(aggregate.cold_latencies, 0.50)),
        format_double(percentile_or_zero(aggregate.cold_latencies, 0.95)),
        format_double(percentile_or_zero(aggregate.cold_latencies, 0.99)),
        format_double(warm_avg_ms),
        format_double(percentile_or_zero(aggregate.warm_latencies, 0.50)),
        format_double(percentile_or_zero(aggregate.warm_latencies, 0.95)),
        format_double(percentile_or_zero(aggregate.warm_latencies, 0.99)),
        format_double(average(aggregate.cold_profiled_query_ms)),
        format_double(percentile_or_zero(aggregate.cold_profiled_query_ms, 0.50)),
        format_double(percentile_or_zero(aggregate.cold_profiled_query_ms, 0.95)),
        format_double(average(aggregate.warm_profiled_query_ms)),
        format_double(percentile_or_zero(aggregate.warm_profiled_query_ms, 0.50)),
        format_double(percentile_or_zero(aggregate.warm_profiled_query_ms, 0.95)),
        format_double(safe_speedup(baseline.cold_avg_ms, cold_avg_ms)),
        format_double(safe_speedup(baseline.warm_avg_ms, warm_avg_ms)),
        format_double(
            safe_ratio(avg_world_access_count, baseline.avg_world_access_count)),
        format_double(safe_ratio(avg_center_distance_count,
                                 baseline.avg_center_distance_count)),
        format_double(
            safe_ratio(avg_raw_candidate_count, baseline.avg_raw_candidate_count)),
        format_double(avg_world_access_count),
        format_double(average_counter(aggregate.node_access_count, sample_count)),
        format_double(average_counter(aggregate.edge_access_count, sample_count)),
        format_double(average_counter(aggregate.anchor_distance_count, sample_count)),
        format_double(avg_center_distance_count),
        format_double(avg_raw_candidate_count),
        format_double(average_counter(aggregate.mbb_check_count, sample_count)),
        format_double(average_counter(aggregate.mbb_surviving_child_count, sample_count)),
        format_double(average_counter(aggregate.mbb_scalar_checks, sample_count)),
        format_double(average_counter(aggregate.mbb_simd_batches, sample_count)),
        format_double(average_counter(aggregate.mbb_simd_fallbacks, sample_count)),
        format_double(average_counter(aggregate.search_qgram_checks, sample_count)),
        format_double(average_counter(aggregate.path_reuse_attempt_count,
                                      sample_count)),
        format_double(average_counter(aggregate.path_reuse_hit_count,
                                      sample_count)),
        format_double(average_counter(
            aggregate.near_query_triangle_pruned_count, sample_count)),
        format_double(average_counter(
            aggregate.near_query_center_distance_reused_count, sample_count)),
        format_double(average_counter(
            aggregate.near_query_bound_fallback_count, sample_count)),
	        format_double(average_counter(
	            aggregate.near_query_direct_verify_count, sample_count)),
	        format_double(average_counter(
	            aggregate.near_query_leaf_triangle_pruned_count, sample_count)),
	        format_double(average_counter(
	            aggregate.near_query_leaf_distance_reused_count, sample_count)),
	        format_double(average_counter(
	            aggregate.near_query_leaf_bound_fallback_count, sample_count)),
	        format_double(baseline.avg_center_distance_count -
                      avg_center_distance_count),
        format_double(baseline.avg_world_access_count -
                      avg_world_access_count),
        format_double(safe_speedup(
            baseline.warm_p95_ms,
            percentile_or_zero(aggregate.warm_latencies, 0.95))),
        format_double(average_counter(aggregate.anchor_cache_hit_count,
                                      sample_count)),
        format_double(average_counter(
            aggregate.child_shortlist_reuse_hit_count, sample_count)),
        format_double(average_counter(aggregate.router_hint_invoked_count,
                                      sample_count)),
        format_double(average_counter(aggregate.router_qgram_ranked_count,
                                      sample_count)),
        format_double(average_counter(aggregate.router_minimizer_ranked_count,
                                      sample_count)),
        format_double(average_counter(aggregate.router_pigeonhole_query_count,
                                      sample_count)),
        format_double(average_counter(aggregate.router_candidate_count,
                                      sample_count)),
        format_double(average_counter(aggregate.router_candidate_hit_count,
                                      sample_count)),
        format_double(average_counter(aggregate.router_fallback_count,
                                      sample_count)),
        format_double(average_counter(aggregate.local_router_invoked_count, sample_count)),
        format_double(average_counter(aggregate.local_router_empty_count, sample_count)),
        format_double(
            average_counter(aggregate.local_router_shortlist_child_count,
                            sample_count)),
        format_double(
            average_counter(aggregate.local_router_remaining_child_count,
                            sample_count)),
        format_double(average_counter(aggregate.local_router_fallback_count,
                                      sample_count)),
        format_double(average_counter(aggregate.best_first_invoked_count,
                                      sample_count)),
        format_double(average_counter(aggregate.best_first_reordered_count,
                                      sample_count)),
        format_double(
            average_counter(aggregate.best_first_bound_candidate_count,
                            sample_count)),
        format_double(average_counter(aggregate.child_safe_bound_pruned_count,
                                      sample_count)),
        format_double(average_counter(aggregate.safe_child_router_invoked_count,
                                      sample_count)),
        format_double(average_counter(
            aggregate.safe_child_router_skipped_low_fanout_count, sample_count)),
        format_double(average_counter(aggregate.safe_child_router_fallback_count,
                                      sample_count)),
        format_double(average_counter(aggregate.safe_child_router_candidate_count,
                                      sample_count)),
        format_double(average_counter(
            aggregate.safe_child_router_pruned_by_not_candidate_count,
            sample_count)),
        format_double(average_counter(
            aggregate.safe_child_router_exact_verify_count, sample_count)),
        format_double(average_counter(aggregate.child_count_before_router,
                                      sample_count)),
        format_double(average_counter(aggregate.post_mbb_survivor_count,
                                      sample_count)),
        format_double(average_counter(aggregate.safe_router_candidate_count,
                                      sample_count)),
        format_double(average_counter(
            aggregate.candidate_ratio_to_all_children, sample_count)),
        format_double(average_counter(
            aggregate.candidate_ratio_to_post_mbb_survivors, sample_count)),
        format_double(average_counter(aggregate.children_actually_processed,
                                      sample_count)),
        format_double(average_counter(aggregate.center_checks_saved,
                                      sample_count)),
        format_double(average_counter(aggregate.planner_invoked_count,
                                      sample_count)),
        format_double(average_counter(aggregate.planner_strategy_baseline_count,
                                      sample_count)),
        format_double(average_counter(
            aggregate.planner_strategy_direct_qgram_count, sample_count)),
        format_double(average_counter(aggregate.planner_strategy_router_count,
                                      sample_count)),
        format_double(average_counter(
            aggregate.planner_strategy_safe_child_router_count, sample_count)),
        format_double(average_counter(
            aggregate.planner_strategy_path_reuse_count, sample_count)),
        format_double(average_counter(
            aggregate.planner_near_reuse_enabled_count, sample_count)),
        format_double(average_counter(
            aggregate.planner_near_reuse_disabled_count, sample_count)),
        format_double(average_counter(aggregate.planner_fallback_count,
                                      sample_count)),
        format_double(sample_count == 0
                          ? 0.0
                          : aggregate.planner_decision_ms /
                                static_cast<double>(sample_count)),
        format_double(average_metric(aggregate.proximal_actual_envelope_sum[0],
                                     aggregate.proximal_actual_envelope_count[0])),
        format_double(average_metric(aggregate.proximal_actual_envelope_sum[1],
                                     aggregate.proximal_actual_envelope_count[1])),
        format_double(average_metric(aggregate.proximal_actual_envelope_sum[2],
                                     aggregate.proximal_actual_envelope_count[2])),
        format_double(average_metric(aggregate.proximal_frontier_envelope_sum[0],
                                     aggregate.proximal_frontier_envelope_count[0])),
        format_double(average_metric(aggregate.proximal_frontier_envelope_sum[1],
                                     aggregate.proximal_frontier_envelope_count[1])),
        format_double(average_metric(aggregate.proximal_frontier_envelope_sum[2],
                                     aggregate.proximal_frontier_envelope_count[2])),
        format_double(average_metric(aggregate.proximal_true_path_envelope_sum[0],
                                     aggregate.proximal_true_path_envelope_count[0])),
        format_double(average_metric(aggregate.proximal_true_path_envelope_sum[1],
                                     aggregate.proximal_true_path_envelope_count[1])),
        format_double(average_metric(aggregate.proximal_true_path_envelope_sum[2],
                                     aggregate.proximal_true_path_envelope_count[2])),
        format_double(average_metric(aggregate.proximal_global_envelope_sum[0],
                                     aggregate.proximal_global_envelope_count[0])),
        format_double(average_metric(aggregate.proximal_global_envelope_sum[1],
                                     aggregate.proximal_global_envelope_count[1])),
        format_double(average_metric(aggregate.proximal_global_envelope_sum[2],
                                     aggregate.proximal_global_envelope_count[2])),
        format_double(average_metric(aggregate.proximal_random_envelope_sum[0],
                                     aggregate.proximal_random_envelope_count[0])),
        format_double(average_metric(aggregate.proximal_random_envelope_sum[1],
                                     aggregate.proximal_random_envelope_count[1])),
        format_double(average_metric(aggregate.proximal_random_envelope_sum[2],
                                     aggregate.proximal_random_envelope_count[2])),
        format_double(fraction_metric(
            aggregate.proximal_global_much_better_actual_count[0],
            aggregate.proximal_global_much_better_actual_denominator[0])),
        format_double(fraction_metric(
            aggregate.proximal_global_much_better_actual_count[1],
            aggregate.proximal_global_much_better_actual_denominator[1])),
        format_double(fraction_metric(
            aggregate.proximal_global_much_better_actual_count[2],
            aggregate.proximal_global_much_better_actual_denominator[2])),
        format_double(fraction_metric(
            aggregate.proximal_global_much_better_frontier_count[0],
            aggregate.proximal_global_much_better_frontier_denominator[0])),
        format_double(fraction_metric(
            aggregate.proximal_global_much_better_frontier_count[1],
            aggregate.proximal_global_much_better_frontier_denominator[1])),
        format_double(fraction_metric(
            aggregate.proximal_global_much_better_frontier_count[2],
            aggregate.proximal_global_much_better_frontier_denominator[2])),
        format_double(average_counter(aggregate.center_exact_distance_call_count, sample_count)),
        format_double(average_counter(aggregate.leaf_beacon_check_count, sample_count)),
        format_double(average_counter(aggregate.leaf_beacon_scalar_checks, sample_count)),
        format_double(average_counter(aggregate.leaf_beacon_simd_batches, sample_count)),
        format_double(average_counter(aggregate.leaf_beacon_simd_fallbacks, sample_count)),
        format_double(average_counter(aggregate.leaf_exact_distance_call_count, sample_count)),
        format_double(average_counter(aggregate.frontier_max_size, sample_count)),
        format_double(average_counter(aggregate.frontier_total_pushed,
                                      sample_count)),
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
       << "\"enable_ablation_profiles\":"
       << bool_string(config.enable_ablation_profiles) << ","
       << "\"proximal_oracle_enabled\":"
       << bool_string(config.proximal_oracle_enabled) << ","
       << "\"proximal_oracle_k_values\":";
  append_json_size_array(json, config.proximal_oracle_k_values);
  json << ","
       << "\"detail_tsv_path\":\"" << json_escape(config.detail_tsv_path) << "\","
       << "\"summary_tsv_path\":\"" << json_escape(config.summary_tsv_path) << "\","
       << "\"json_path\":\"" << json_escape(config.json_path) << "\"},"
       << "\"build\":{\"duration_ms\":" << format_double(build_duration_ms)
       << ",\"distance_mode\":\""
       << build_distance_mode_name(build_config.distance_mode) << "\""
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
       << "\"profiles\":{";
  for (size_t profile_index = 0; profile_index < profile_names.size();
       ++profile_index) {
    if (profile_index) json << ",";
    const std::string& profile_name = profile_names[profile_index];
    const SearchConfig& profile_config = profile_configs.at(profile_name);
    json << "\"" << json_escape(profile_name) << "\":{"
         << "\"mbb_filter_mode\":\""
         << mbb_filter_mode_name(profile_config.mbb_filter_mode) << "\","
         << "\"visited_mode\":\""
         << visited_mode_name(profile_config.visited_mode) << "\","
         << "\"graph_view\":\""
         << graph_view_mode_name(profile_config.graph_view_mode) << "\","
         << "\"simd_mode\":\""
         << simd_mode_name(profile_config.simd_mode) << "\","
         << "\"distance_mode\":\""
         << distance_mode_name(profile_config.distance_mode) << "\","
         << "\"search_qgram_prefilter\":"
         << bool_string(profile_config.search_qgram_prefilter) << ","
         << "\"path_reuse_enabled\":"
         << bool_string(profile_config.path_reuse_enabled) << ","
         << "\"near_query_max_neighbor_edit_distance\":"
         << profile_config.near_query_max_neighbor_edit_distance << ","
         << "\"near_query_min_qgram_jaccard\":"
         << format_double(profile_config.near_query_min_qgram_jaccard) << ","
         << "\"router_hint_enabled\":"
         << bool_string(profile_config.router_hint_enabled) << ","
         << "\"router_hint_qgram_q\":"
         << profile_config.router_hint_qgram_q << ","
         << "\"router_hint_minimizer_k\":"
         << profile_config.router_hint_minimizer_k << ","
         << "\"router_hint_minimizer_w\":"
         << profile_config.router_hint_minimizer_w << ","
         << "\"local_router_enabled\":"
         << bool_string(profile_config.local_router_enabled) << ","
         << "\"best_first_enabled\":"
         << bool_string(profile_config.best_first_enabled) << ","
         << "\"safe_child_router_enabled\":"
         << bool_string(profile_config.safe_child_router_enabled) << ","
         << "\"safe_child_router_min_fanout\":"
         << profile_config.safe_child_router_min_fanout << ","
         << "\"safe_child_router_max_candidates\":"
         << profile_config.safe_child_router_max_candidates << ","
         << "\"safe_child_router_max_ratio\":"
         << format_double(profile_config.safe_child_router_max_ratio) << ","
         << "\"safe_child_router_min_seed_len\":"
         << profile_config.safe_child_router_min_seed_len << ","
         << "\"safe_child_router_mode\":\""
         << json_escape(profile_config.safe_child_router_mode) << "\","
         << "\"safe_child_router_validate\":"
         << bool_string(profile_config.safe_child_router_validate) << ","
         << "\"query_planner_enabled\":"
         << bool_string(profile_config.query_planner_enabled) << ","
         << "\"planner_direct_verify_max_candidates\":"
         << profile_config.planner_direct_verify_max_candidates << ","
         << "\"planner_router_min_fanout\":"
         << profile_config.planner_router_min_fanout << ","
         << "\"planner_safe_child_router_min_fanout\":"
         << profile_config.planner_safe_child_router_min_fanout << ","
         << "\"planner_allow_direct_qgram_verify\":"
         << bool_string(profile_config.planner_allow_direct_qgram_verify) << ","
         << "\"proximal_oracle_enabled\":"
         << bool_string(profile_config.proximal_oracle_enabled) << ","
         << "\"query_profile\":"
         << bool_string(profile_config.query_profile) << ","
         << "\"search_qgram_q\":" << profile_config.search_qgram_q << "}";
  }
  json << "},\"profile_order\":[";
  for (size_t profile_index = 0; profile_index < profile_names.size();
       ++profile_index) {
    if (profile_index) json << ",";
    json << "\"" << json_escape(profile_names[profile_index]) << "\"";
  }
  json << "],\"ablation_profiles\":[";
  bool emitted_ablation = false;
  for (const auto& profile_name : profile_names) {
    if (profile_name.rfind("ablation_", 0) != 0) continue;
    if (emitted_ablation) json << ",";
    emitted_ablation = true;
    json << "\"" << json_escape(profile_name) << "\"";
  }
  json << "],"
       << "\"generation\":{\"query_count\":" << queries.size()
       << ",\"counts\":{";
  size_t generation_index = 0;
  for (const auto& entry : generation_counts) {
    if (generation_index++) json << ",";
    json << "\"" << json_escape(entry.first) << "\":" << entry.second;
  }
  json << "}},\"aggregate_columns\":[";
  for (size_t column_index = 0; column_index < summary_columns.size();
       ++column_index) {
    if (column_index) json << ",";
    json << "\"" << json_escape(summary_columns[column_index]) << "\"";
  }
  json << "],\"aggregate_rows\":[";
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
    json << ",\"failing_profiles\":";
    append_json_string_array(json, diagnostic.failing_profiles);
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
