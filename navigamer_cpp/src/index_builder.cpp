#include "index_builder.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>

namespace navigamer {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class ScopedTimer {
 public:
  explicit ScopedTimer(double* target_ms)
      : target_ms_(target_ms), start_(Clock::now()) {}

  ~ScopedTimer() {
    if (target_ms_) *target_ms_ += elapsed_ms_since(start_);
  }

 private:
  double* target_ms_;
  Clock::time_point start_;
};

int build_distance_bounded(const std::string& a, const std::string& b, int tau,
                           BuildDistanceMode mode);
int build_distance(const std::string& a, const std::string& b,
                   BuildDistanceMode mode);

constexpr size_t kPhase1ParallelScanMinFanout = 512;

struct Phase1CoverScanResult {
  std::shared_ptr<WorldNode> best;
  int best_dist = INT_MAX;
  size_t best_idx = std::numeric_limits<size_t>::max();
  size_t candidate_scans = 0;
  size_t length_pruned = 0;
  size_t exact_distance_calls = 0;
};

bool phase1_better_cover(size_t idx, int dist,
                         const Phase1CoverScanResult& current) {
  if (dist < current.best_dist) return true;
  return dist == current.best_dist && idx < current.best_idx;
}

Phase1CoverScanResult find_best_phase1_cover(
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    const std::shared_ptr<BioSequence>& sequence,
    int radius,
    BuildDistanceMode distance_mode) {
  Phase1CoverScanResult result;
  if (candidates.empty()) return result;

  auto scan_one = [&](size_t idx, Phase1CoverScanResult& local) {
    const auto& node = candidates[idx];
    if (!node || !node->center_ptr) return;
    local.candidate_scans++;
    if (std::llabs(static_cast<long long>(sequence->seq.size()) -
                   static_cast<long long>(node->center_ptr->seq.size())) >
        radius) {
      local.length_pruned++;
      return;
    }
    local.exact_distance_calls++;
    const int dist = build_distance_bounded(
        sequence->seq, node->center_ptr->seq, radius, distance_mode);
    if (dist <= radius && phase1_better_cover(idx, dist, local)) {
      local.best = node;
      local.best_dist = dist;
      local.best_idx = idx;
    }
  };

  if (candidates.size() < kPhase1ParallelScanMinFanout) {
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
      scan_one(idx, result);
    }
    return result;
  }

  const int thread_count = std::max(1, omp_get_max_threads());
  std::vector<Phase1CoverScanResult> local_results(
      static_cast<size_t>(thread_count));
  #pragma omp parallel for schedule(static)
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    const int tid = omp_get_thread_num();
    Phase1CoverScanResult& local =
        local_results[static_cast<size_t>(std::min(tid, thread_count - 1))];
    scan_one(idx, local);
  }

  for (const auto& local : local_results) {
    result.candidate_scans += local.candidate_scans;
    result.length_pruned += local.length_pruned;
    result.exact_distance_calls += local.exact_distance_calls;
    if (local.best &&
        phase1_better_cover(local.best_idx, local.best_dist, result)) {
      result.best = local.best;
      result.best_dist = local.best_dist;
      result.best_idx = local.best_idx;
    }
  }
  return result;
}

Phase1CoverScanResult find_best_phase1_cover_by_indices(
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    const std::vector<size_t>& candidate_indices,
    const std::shared_ptr<BioSequence>& sequence,
    int radius,
    BuildDistanceMode distance_mode) {
  Phase1CoverScanResult result;
  if (candidate_indices.empty()) return result;

  auto scan_one = [&](size_t pos, Phase1CoverScanResult& local) {
    const size_t idx = candidate_indices[pos];
    if (idx >= candidates.size()) return;
    const auto& node = candidates[idx];
    if (!node || !node->center_ptr) return;
    local.candidate_scans++;
    if (std::llabs(static_cast<long long>(sequence->seq.size()) -
                   static_cast<long long>(node->center_ptr->seq.size())) >
        radius) {
      local.length_pruned++;
      return;
    }
    local.exact_distance_calls++;
    const int dist = build_distance_bounded(
        sequence->seq, node->center_ptr->seq, radius, distance_mode);
    if (dist <= radius && phase1_better_cover(idx, dist, local)) {
      local.best = node;
      local.best_dist = dist;
      local.best_idx = idx;
    }
  };

  if (candidate_indices.size() < kPhase1ParallelScanMinFanout) {
    for (size_t pos = 0; pos < candidate_indices.size(); ++pos) {
      scan_one(pos, result);
    }
    return result;
  }

  const int thread_count = std::max(1, omp_get_max_threads());
  std::vector<Phase1CoverScanResult> local_results(
      static_cast<size_t>(thread_count));
  #pragma omp parallel for schedule(static)
  for (size_t pos = 0; pos < candidate_indices.size(); ++pos) {
    const int tid = omp_get_thread_num();
    Phase1CoverScanResult& local =
        local_results[static_cast<size_t>(std::min(tid, thread_count - 1))];
    scan_one(pos, local);
  }

  for (const auto& local : local_results) {
    result.candidate_scans += local.candidate_scans;
    result.length_pruned += local.length_pruned;
    result.exact_distance_calls += local.exact_distance_calls;
    if (local.best &&
        phase1_better_cover(local.best_idx, local.best_dist, result)) {
      result.best = local.best;
      result.best_dist = local.best_dist;
      result.best_idx = local.best_idx;
    }
  }
  return result;
}

enum class Phase1CoverSource {
  Scan,
  Metric,
  QGram,
  FallbackScan,
};

struct Phase1CandidateQueryResult {
  Phase1CoverSource source = Phase1CoverSource::Scan;
  bool fallback_scan = false;
  std::vector<size_t> candidate_indices;
  size_t total_possible = 0;
  size_t metric_distance_calls = 0;
  size_t metric_build_distance_calls = 0;
  size_t qgram_touched_candidates = 0;
  size_t qgram_pruned_candidates = 0;
};

bool phase1_length_compatible(size_t lhs_len, size_t rhs_len, int radius) {
  return std::llabs(static_cast<long long>(lhs_len) -
                    static_cast<long long>(rhs_len)) <= radius;
}

class Phase1CoverGroupIndex {
 public:
  Phase1CandidateQueryResult query(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      const std::shared_ptr<BioSequence>& sequence,
      int radius,
      const BuildRangeConfig& config) {
    Phase1CandidateQueryResult result;
    result.total_possible = candidates.size();
    if (candidates.empty() ||
        config.phase1_candidate_mode == Phase1CandidateMode::Scan ||
        candidates.size() < config.phase1_metric_min_fanout) {
      result.source = Phase1CoverSource::Scan;
      return result;
    }

    const size_t build_calls_before = metric_build_distance_calls_;
    sync_items(candidates, config);
    result.metric_build_distance_calls =
        metric_build_distance_calls_ - build_calls_before;

    if (candidates.size() >= config.phase1_qgram_min_fanout) {
      auto qgram = query_qgram(candidates, sequence, radius, config);
      qgram.metric_build_distance_calls = result.metric_build_distance_calls;
      return qgram;
    }

    auto metric = query_metric(candidates, sequence, radius, config);
    metric.metric_build_distance_calls = result.metric_build_distance_calls;
    return metric;
  }

 private:
  struct ItemInfo {
    size_t sequence_length = 0;
    size_t total_qgrams = 0;
    bool qgram_safe = false;
  };

  struct MetricTreeNode {
    size_t item_idx = 0;
    std::unordered_map<int, size_t> children;
  };

  struct QGramPosting {
    uint32_t item_idx = 0;
    uint32_t count = 0;
  };

  static const std::string& center_sequence(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      size_t idx) {
    static const std::string empty;
    if (idx >= candidates.size() || !candidates[idx] ||
        !candidates[idx]->center_ptr) {
      return empty;
    }
    return candidates[idx]->center_ptr->seq;
  }

  static Phase1CandidateQueryResult fallback_scan_result(size_t total_possible) {
    Phase1CandidateQueryResult result;
    result.source = Phase1CoverSource::FallbackScan;
    result.fallback_scan = true;
    result.total_possible = total_possible;
    return result;
  }

  void sync_items(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      const BuildRangeConfig& config) {
    while (items_.size() < candidates.size()) {
      const size_t idx = items_.size();
      ItemInfo info;
      info.sequence_length = center_sequence(candidates, idx).size();
      items_.push_back(info);
      if (metric_built_ && items_.size() < config.phase1_qgram_min_fanout) {
        insert_metric_item(candidates, idx, config.distance_mode);
      }
      if (qgram_built_) {
        add_qgram_item(candidates, idx, config.range_join.qgram_q);
      }
    }
  }

  void ensure_metric(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      BuildDistanceMode distance_mode) {
    if (metric_built_) return;
    metric_nodes_.clear();
    metric_nodes_.reserve(items_.size());
    metric_root_ = std::numeric_limits<size_t>::max();
    metric_built_ = true;
    for (size_t idx = 0; idx < items_.size(); ++idx) {
      insert_metric_item(candidates, idx, distance_mode);
    }
  }

  void insert_metric_item(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      size_t item_idx,
      BuildDistanceMode distance_mode) {
    if (item_idx >= candidates.size() || !candidates[item_idx] ||
        !candidates[item_idx]->center_ptr) {
      return;
    }
    if (metric_root_ == std::numeric_limits<size_t>::max()) {
      metric_root_ = metric_nodes_.size();
      metric_nodes_.push_back({item_idx, {}});
      return;
    }

    const std::string& sequence = candidates[item_idx]->center_ptr->seq;
    size_t current = metric_root_;
    while (true) {
      MetricTreeNode& node = metric_nodes_[current];
      const int dist = build_distance(
          sequence, center_sequence(candidates, node.item_idx), distance_mode);
      metric_build_distance_calls_++;
      auto child_it = node.children.find(dist);
      if (child_it == node.children.end()) {
        const size_t new_node_idx = metric_nodes_.size();
        node.children.emplace(dist, new_node_idx);
        metric_nodes_.push_back({item_idx, {}});
        return;
      }
      current = child_it->second;
    }
  }

  Phase1CandidateQueryResult query_metric(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      const std::shared_ptr<BioSequence>& sequence,
      int radius,
      const BuildRangeConfig& config) {
    Phase1CandidateQueryResult result;
    result.source = Phase1CoverSource::Metric;
    result.total_possible = candidates.size();
    ensure_metric(candidates, config.distance_mode);
    if (metric_root_ == std::numeric_limits<size_t>::max()) return result;

    std::vector<size_t> stack;
    stack.push_back(metric_root_);
    while (!stack.empty()) {
      const size_t node_idx = stack.back();
      stack.pop_back();
      const MetricTreeNode& node = metric_nodes_[node_idx];
      const int dist = build_distance(
          sequence->seq, center_sequence(candidates, node.item_idx),
          config.distance_mode);
      result.metric_distance_calls++;
      if (dist <= radius) result.candidate_indices.push_back(node.item_idx);

      const int min_edge = std::max(0, dist - radius);
      const int max_edge = dist + radius;
      for (const auto& child : node.children) {
        if (child.first >= min_edge && child.first <= max_edge) {
          stack.push_back(child.second);
        }
      }
    }

    std::sort(result.candidate_indices.begin(), result.candidate_indices.end());
    result.candidate_indices.erase(
        std::unique(result.candidate_indices.begin(),
                    result.candidate_indices.end()),
        result.candidate_indices.end());
    return result;
  }

  void ensure_qgram(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      int q) {
    if (qgram_built_ && qgram_q_ == q) return;
    qgram_postings_.clear();
    qgram_unsafe_indices_.clear();
    qgram_zero_total_indices_.clear();
    min_safe_total_qgrams_ = std::numeric_limits<size_t>::max();
    qgram_q_ = q;
    qgram_built_ = true;
    for (size_t idx = 0; idx < items_.size(); ++idx) {
      add_qgram_item(candidates, idx, q);
    }
  }

  void add_qgram_item(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      size_t idx,
      int q) {
    if (idx >= items_.size()) return;
    const std::string& sequence = center_sequence(candidates, idx);
    auto signature = compute_qgram_signature(sequence, q);
    items_[idx].sequence_length = sequence.size();
    items_[idx].qgram_safe = signature.safe_for_pruning;
    items_[idx].total_qgrams = signature.total_qgrams;
    if (!signature.safe_for_pruning) {
      qgram_unsafe_indices_.push_back(idx);
      return;
    }
    min_safe_total_qgrams_ =
        std::min(min_safe_total_qgrams_, signature.total_qgrams);
    if (signature.total_qgrams == 0) {
      qgram_zero_total_indices_.push_back(idx);
      return;
    }
    for (const auto& entry : signature.entries) {
      qgram_postings_[entry.code].push_back(
          {static_cast<uint32_t>(idx), entry.count});
    }
  }

  void reset_qgram_workspace(size_t item_count) {
    if (qgram_shared_.size() < item_count) qgram_shared_.resize(item_count, 0);
    if (qgram_seen_epoch_.size() < item_count) {
      qgram_seen_epoch_.resize(item_count, 0);
    }
    qgram_touched_.clear();
    if (qgram_epoch_ == std::numeric_limits<uint32_t>::max()) {
      std::fill(qgram_seen_epoch_.begin(), qgram_seen_epoch_.end(), 0);
      qgram_epoch_ = 1;
    } else {
      qgram_epoch_++;
    }
  }

  Phase1CandidateQueryResult query_qgram(
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      const std::shared_ptr<BioSequence>& sequence,
      int radius,
      const BuildRangeConfig& config) {
    Phase1CandidateQueryResult result;
    result.source = Phase1CoverSource::QGram;
    result.total_possible = candidates.size();
    ensure_qgram(candidates, config.range_join.qgram_q);

    const int q = config.range_join.qgram_q;
    const auto query_signature = compute_qgram_signature(sequence->seq, q);
    if (!query_signature.safe_for_pruning ||
        query_signature.total_qgrams == 0 ||
        min_safe_total_qgrams_ == std::numeric_limits<size_t>::max()) {
      return fallback_scan_result(candidates.size());
    }

    const size_t tau = static_cast<size_t>(radius);
    const size_t q_size = static_cast<size_t>(q);
    if (tau > std::numeric_limits<size_t>::max() / q_size / 2) {
      return fallback_scan_result(candidates.size());
    }
    const size_t max_l1 = 2 * q_size * tau;
    if (query_signature.total_qgrams >
        std::numeric_limits<size_t>::max() - min_safe_total_qgrams_) {
      return fallback_scan_result(candidates.size());
    }
    if (query_signature.total_qgrams + min_safe_total_qgrams_ <= max_l1) {
      return fallback_scan_result(candidates.size());
    }

    reset_qgram_workspace(items_.size());
    for (const auto& query_entry : query_signature.entries) {
      auto posting_it = qgram_postings_.find(query_entry.code);
      if (posting_it == qgram_postings_.end()) continue;
      for (const auto& posting : posting_it->second) {
        const size_t idx = posting.item_idx;
        if (idx >= items_.size()) continue;
        if (qgram_seen_epoch_[idx] != qgram_epoch_) {
          qgram_seen_epoch_[idx] = qgram_epoch_;
          qgram_shared_[idx] = 0;
          qgram_touched_.push_back(idx);
          if (qgram_touched_.size() > config.phase1_qgram_max_touched) {
            return fallback_scan_result(candidates.size());
          }
        }
        const uint32_t shared =
            std::min(query_entry.count, posting.count);
        if (qgram_shared_[idx] >
            std::numeric_limits<uint32_t>::max() - shared) {
          return fallback_scan_result(candidates.size());
        }
        qgram_shared_[idx] += shared;
      }
    }
    result.qgram_touched_candidates = qgram_touched_.size();

    for (size_t idx : qgram_touched_) {
      const auto& item = items_[idx];
      if (!phase1_length_compatible(sequence->seq.size(),
                                    item.sequence_length, radius)) {
        continue;
      }
      const size_t total_sum =
          query_signature.total_qgrams + item.total_qgrams;
      if (total_sum <= max_l1) {
        return fallback_scan_result(candidates.size());
      }
      const size_t required_shared = (total_sum - max_l1 + 1) / 2;
      if (qgram_shared_[idx] >= required_shared) {
        result.candidate_indices.push_back(idx);
        if (result.candidate_indices.size() >
            config.phase1_qgram_max_touched) {
          return fallback_scan_result(candidates.size());
        }
      }
    }

    auto append_unprunable = [&](const std::vector<size_t>& indices) {
      for (size_t idx : indices) {
        if (idx >= items_.size()) continue;
        if (phase1_length_compatible(sequence->seq.size(),
                                     items_[idx].sequence_length, radius)) {
          result.candidate_indices.push_back(idx);
        }
      }
    };
    append_unprunable(qgram_unsafe_indices_);
    append_unprunable(qgram_zero_total_indices_);
    if (result.candidate_indices.size() > config.phase1_qgram_max_touched) {
      return fallback_scan_result(candidates.size());
    }

    std::sort(result.candidate_indices.begin(), result.candidate_indices.end());
    result.candidate_indices.erase(
        std::unique(result.candidate_indices.begin(),
                    result.candidate_indices.end()),
        result.candidate_indices.end());
    result.qgram_pruned_candidates =
        result.total_possible > result.candidate_indices.size()
            ? result.total_possible - result.candidate_indices.size()
            : 0;
    return result;
  }

  std::vector<ItemInfo> items_;
  bool metric_built_ = false;
  size_t metric_root_ = std::numeric_limits<size_t>::max();
  std::vector<MetricTreeNode> metric_nodes_;
  size_t metric_build_distance_calls_ = 0;

  bool qgram_built_ = false;
  int qgram_q_ = 0;
  size_t min_safe_total_qgrams_ = std::numeric_limits<size_t>::max();
  std::unordered_map<uint64_t, std::vector<QGramPosting>> qgram_postings_;
  std::vector<size_t> qgram_unsafe_indices_;
  std::vector<size_t> qgram_zero_total_indices_;
  std::vector<uint32_t> qgram_shared_;
  std::vector<uint32_t> qgram_seen_epoch_;
  std::vector<size_t> qgram_touched_;
  uint32_t qgram_epoch_ = 1;
};

std::vector<int> make_auxiliary_radii(const std::vector<int>& primary_radii) {
  std::vector<int> out;
  if (primary_radii.size() < 2) return out;
  out.reserve(primary_radii.size() - 1);
  for (size_t i = 0; i + 1 < primary_radii.size(); ++i) {
    out.push_back(std::max(1, (primary_radii[i] + primary_radii[i + 1]) / 2));
  }
  return out;
}

std::vector<int> leaf_beacon_distances(
    const std::shared_ptr<BioSequence>& leaf,
    const std::vector<std::shared_ptr<BioSequence>>& beacons,
    int center_dist,
    BuildDistanceMode distance_mode) {
  std::vector<int> dists;
  dists.reserve(beacons.size());
  for (size_t i = 0; i < beacons.size(); ++i) {
    if (!beacons[i]) {
      dists.push_back(0);
    } else if (i == 0) {
      dists.push_back(center_dist);
    } else {
      dists.push_back(distance_mode == BuildDistanceMode::Edlib
                          ? compute_distance_edlib(leaf->seq, beacons[i]->seq)
                          : compute_distance(leaf->seq, beacons[i]->seq));
    }
  }
  return dists;
}

int build_distance(const std::string& a, const std::string& b,
                   BuildDistanceMode mode) {
  return mode == BuildDistanceMode::Edlib ? compute_distance_edlib(a, b)
                                          : compute_distance(a, b);
}

int build_distance_bounded(const std::string& a, const std::string& b, int tau,
                           BuildDistanceMode mode) {
  return mode == BuildDistanceMode::Edlib
             ? compute_distance_bounded_edlib(a, b, tau)
             : compute_distance_bounded_dp(a, b, tau);
}

std::vector<int> build_expanded_radii(const HierarchyConfig& config) {
  std::vector<int> expanded;
  expanded.reserve(static_cast<size_t>(config.num_expanded_layers()));
  for (int i = 0; i < config.num_primary_layers(); ++i) {
    expanded.push_back(config.primary_radii[static_cast<size_t>(i)]);
    if (i < config.num_auxiliary_layers()) {
      expanded.push_back(config.auxiliary_radii[static_cast<size_t>(i)]);
    }
  }
  return expanded;
}

bool expanded_layer_is_primary(int expanded_layer_idx) {
  return expanded_layer_idx % 2 == 0;
}

int expanded_to_primary_index(int expanded_layer_idx) {
  return expanded_layer_idx / 2;
}

void reset_node_metadata(const std::shared_ptr<WorldNode>& node,
                         int expanded_layer_idx,
                         bool is_primary,
                         int primary_layer_idx) {
  node->expanded_layer_index = expanded_layer_idx;
  node->is_primary = is_primary;
  node->primary_layer_index = primary_layer_idx;
}

double reduction_ratio(size_t before, size_t after) {
  if (before == 0) return 0.0;
  return 1.0 - static_cast<double>(after) / static_cast<double>(before);
}

std::string format_ms(double value) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(3) << value;
  return os.str();
}

void accumulate_range_timing(BioGeometryIndexBuilder::Statistics& stats,
                             const RangeJoinQueryResult& result) {
  stats.range_posting_lookup_ms += result.range_posting_lookup_ms;
  stats.range_seed_union_ms += result.range_seed_union_ms;
  stats.range_length_filter_ms += result.range_length_filter_ms;
  stats.range_qgram_query_ms += result.range_qgram_query_ms;
  stats.range_hybrid_intersection_ms += result.range_hybrid_intersection_ms;
  stats.range_full_scan_ms += result.range_full_scan_ms;
}

std::vector<int> leaf_beacon_distances_timed(
    const std::shared_ptr<BioSequence>& leaf,
    const std::vector<std::shared_ptr<BioSequence>>& beacons,
    int center_dist,
    BuildDistanceMode distance_mode,
    double* target_ms) {
  if (beacons.size() <= 1) {
    return leaf_beacon_distances(leaf, beacons, center_dist, distance_mode);
  }
  ScopedTimer timer(target_ms);
  return leaf_beacon_distances(leaf, beacons, center_dist, distance_mode);
}

void validate_range_config(const BuildRangeConfig& config) {
  if (config.range_join.min_seed_len <= 0) {
    throw std::invalid_argument("range-join min seed length must be positive");
  }
  if (config.range_join.max_seed_len < config.range_join.min_seed_len) {
    throw std::invalid_argument(
        "range-join max seed length must be at least min seed length");
  }
  if (config.range_join.qgram_q <= 0) {
    throw std::invalid_argument("range-join q-gram length must be positive");
  }
  if (!std::isfinite(config.range_join.auto_pigeonhole_max_ratio) ||
      config.range_join.auto_pigeonhole_max_ratio < 0.0 ||
      config.range_join.auto_pigeonhole_max_ratio > 1.0) {
    throw std::invalid_argument(
        "auto pigeonhole max ratio must be finite and in [0, 1]");
  }
  if (config.min_rect_index_fanout == 0) {
    throw std::invalid_argument("minimum rectangle-index fanout must be positive");
  }
  if (config.phase1_metric_min_fanout == 0) {
    throw std::invalid_argument("phase1 metric fanout threshold must be positive");
  }
  if (config.phase1_qgram_min_fanout < config.phase1_metric_min_fanout) {
    throw std::invalid_argument(
        "phase1 q-gram fanout threshold must be at least metric threshold");
  }
  if (config.phase1_qgram_max_touched == 0) {
    throw std::invalid_argument("phase1 q-gram touched limit must be positive");
  }
}

}  // namespace

HierarchyConfig::HierarchyConfig(std::vector<int> primary_radii_in)
    : primary_radii(std::move(primary_radii_in)),
      auxiliary_radii(make_auxiliary_radii(primary_radii)) {
  validate();
}

HierarchyConfig::HierarchyConfig(std::vector<int> primary_radii_in,
                                 std::vector<int> auxiliary_radii_in)
    : primary_radii(std::move(primary_radii_in)),
      auxiliary_radii(std::move(auxiliary_radii_in)) {
  validate();
}

int HierarchyConfig::num_primary_layers() const {
  return static_cast<int>(primary_radii.size());
}

int HierarchyConfig::num_auxiliary_layers() const {
  return static_cast<int>(auxiliary_radii.size());
}

int HierarchyConfig::num_expanded_layers() const {
  if (primary_radii.empty()) return 0;
  return static_cast<int>(primary_radii.size() * 2 - 1);
}

void HierarchyConfig::validate() const {
  if (primary_radii.size() < 2) {
    throw std::invalid_argument("HierarchyConfig requires at least two primary radii");
  }
  if (auxiliary_radii.size() != primary_radii.size() - 1) {
    throw std::invalid_argument("HierarchyConfig auxiliary_radii must have size K-1");
  }
  for (size_t i = 0; i + 1 < primary_radii.size(); ++i) {
    if (primary_radii[i] <= primary_radii[i + 1]) {
      throw std::invalid_argument("HierarchyConfig primary_radii must be strictly decreasing");
    }
    int aux = auxiliary_radii[i];
    if (aux <= 0) {
      throw std::invalid_argument("HierarchyConfig auxiliary radii must be positive");
    }
    if (aux > primary_radii[i] || aux < primary_radii[i + 1]) {
      throw std::invalid_argument("HierarchyConfig auxiliary radius must lie between adjacent primary radii");
    }
  }
}

const char* build_range_mode_name(BuildRangeMode mode) {
  return mode == BuildRangeMode::Full ? "full" : "indexed";
}

BuildRangeMode parse_build_range_mode(const std::string& value) {
  if (value == "full") return BuildRangeMode::Full;
  if (value == "indexed") return BuildRangeMode::Indexed;
  throw std::invalid_argument("build range mode must be full or indexed");
}

const char* leaf_attach_direction_name(LeafAttachDirection direction) {
  switch (direction) {
    case LeafAttachDirection::SeqToWorld:
      return "seq_to_world";
    case LeafAttachDirection::WorldToSeq:
      return "world_to_seq";
    case LeafAttachDirection::Auto:
      return "auto";
  }
  return "auto";
}

LeafAttachDirection parse_leaf_attach_direction(const std::string& value) {
  if (value == "seq-to-world" || value == "seq_to_world") {
    return LeafAttachDirection::SeqToWorld;
  }
  if (value == "world-to-seq" || value == "world_to_seq") {
    return LeafAttachDirection::WorldToSeq;
  }
  if (value == "auto") return LeafAttachDirection::Auto;
  throw std::invalid_argument(
      "leaf attach direction must be auto, seq-to-world, or world-to-seq");
}

const char* build_distance_mode_name(BuildDistanceMode mode) {
  switch (mode) {
    case BuildDistanceMode::DP:
      return "dp";
    case BuildDistanceMode::Edlib:
      return "edlib";
    case BuildDistanceMode::Auto:
      return "auto";
  }
  return "dp";
}

BuildDistanceMode parse_build_distance_mode(const std::string& value) {
  if (value == "dp") return BuildDistanceMode::DP;
  if (value == "edlib") return BuildDistanceMode::Edlib;
  if (value == "auto") return BuildDistanceMode::Auto;
  throw std::invalid_argument("build distance mode must be dp, edlib, or auto");
}

const char* phase1_candidate_mode_name(Phase1CandidateMode mode) {
  switch (mode) {
    case Phase1CandidateMode::Scan:
      return "scan";
    case Phase1CandidateMode::Hybrid:
      return "hybrid";
  }
  return "hybrid";
}

Phase1CandidateMode parse_phase1_candidate_mode(const std::string& value) {
  if (value == "scan") return Phase1CandidateMode::Scan;
  if (value == "hybrid") return Phase1CandidateMode::Hybrid;
  throw std::invalid_argument("phase1 candidate mode must be scan or hybrid");
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder()
    : stats_{},
      hierarchy_(HierarchyConfig({R_LW, R_MW, R_SW})),
      range_config_{},
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())) {
  stats_.created_primary_nodes.assign(static_cast<size_t>(hierarchy_.num_primary_layers()), 0);
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw)
    : BioGeometryIndexBuilder(HierarchyConfig({r_lw, r_mw, r_sw})) {}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(const HierarchyConfig& config)
    : BioGeometryIndexBuilder(config, BuildRangeConfig{}) {}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(
    const HierarchyConfig& config, const BuildRangeConfig& range_config)
    : stats_{},
      hierarchy_(config),
      range_config_(range_config),
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())) {
  validate_range_config(range_config_);
  stats_.created_primary_nodes.assign(static_cast<size_t>(hierarchy_.num_primary_layers()), 0);
}

const std::vector<std::shared_ptr<WorldNode>>& BioGeometryIndexBuilder::primary_layer(int idx) const {
  return primary_layers_.at(static_cast<size_t>(idx));
}

bool BioGeometryIndexBuilder::validate_integer_ids() const {
  if (world_node_count_ == 0 && !primary_layers_.empty()) return false;
  std::vector<bool> seen_nodes(world_node_count_, false);
  size_t visited_nodes = 0;
  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id >= world_node_count_) return false;
      if (seen_nodes[node->integer_id]) return false;
      seen_nodes[node->integer_id] = true;
      visited_nodes++;
      for (const auto& child : node->child_nodes) {
        if (!child || child->integer_id >= world_node_count_) return false;
      }
      for (const auto& leaf : node->child_leaves) {
        if (!leaf || leaf->sequence_id >= sequence_count_) return false;
      }
    }
  }
  if (visited_nodes != world_node_count_) return false;
  for (bool seen : seen_nodes) {
    if (!seen) return false;
  }

  if (unique_sequences.size() != sequence_count_) return false;
  std::vector<bool> seen_sequences(sequence_count_, false);
  for (const auto& entry : unique_sequences) {
    const auto& sequence = entry.second;
    if (!sequence || sequence->sequence_id >= sequence_count_) return false;
    if (seen_sequences[sequence->sequence_id]) return false;
    seen_sequences[sequence->sequence_id] = true;
  }
  for (bool seen : seen_sequences) {
    if (!seen) return false;
  }
  return true;
}

bool BioGeometryIndexBuilder::validate_search_graph_view() const {
  const auto& view = search_graph_view_;
  if (!validate_integer_ids()) return false;
  if (view.nodes.size() != world_node_count_ ||
      view.leaves.size() != sequence_count_ ||
      view.child_begin.size() != world_node_count_ ||
      view.child_end.size() != world_node_count_ ||
      view.leaf_begin.size() != world_node_count_ ||
      view.leaf_end.size() != world_node_count_ ||
      view.mbb_begin.size() != world_node_count_ ||
      view.mbb_dim.size() != world_node_count_ ||
      view.beacon_begin.size() != world_node_count_ ||
      view.beacon_end.size() != world_node_count_ ||
      view.leaf_beacon_begin.size() != world_node_count_ ||
      view.leaf_beacon_dim.size() != world_node_count_) {
    return false;
  }

  for (const auto& entry : unique_sequences) {
    const auto& sequence = entry.second;
    if (!sequence || sequence->sequence_id >= view.leaves.size()) return false;
    if (view.leaves[sequence->sequence_id] != sequence) return false;
  }

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id >= view.nodes.size()) return false;
      const NodeId node_id = node->integer_id;
      if (view.nodes[node_id] != node) return false;

      if (view.child_end[node_id] < view.child_begin[node_id] ||
          view.leaf_end[node_id] < view.leaf_begin[node_id] ||
          view.beacon_end[node_id] < view.beacon_begin[node_id]) {
        return false;
      }
      if (view.child_end[node_id] > view.child_ids.size() ||
          view.leaf_end[node_id] > view.leaf_ids.size() ||
          view.beacon_end[node_id] > view.beacon_ids.size()) {
        return false;
      }

      const uint32_t child_begin = view.child_begin[node_id];
      if (view.child_end[node_id] - child_begin != node->child_nodes.size()) {
        return false;
      }
      for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
        const auto& child = node->child_nodes[child_idx];
        if (!child || view.child_ids[child_begin + child_idx] != child->integer_id) {
          return false;
        }
      }

      const uint32_t leaf_begin = view.leaf_begin[node_id];
      if (view.leaf_end[node_id] - leaf_begin != node->child_leaves.size()) {
        return false;
      }
      for (size_t leaf_idx = 0; leaf_idx < node->child_leaves.size(); ++leaf_idx) {
        const auto& leaf = node->child_leaves[leaf_idx];
        if (!leaf || view.leaf_ids[leaf_begin + leaf_idx] != leaf->sequence_id) {
          return false;
        }
      }

      const uint32_t beacon_begin = view.beacon_begin[node_id];
      if (view.beacon_end[node_id] - beacon_begin != node->beacons.size()) {
        return false;
      }
      for (size_t beacon_idx = 0; beacon_idx < node->beacons.size(); ++beacon_idx) {
        const auto& beacon = node->beacons[beacon_idx];
        if (!beacon || view.beacon_ids[beacon_begin + beacon_idx] !=
                           beacon->sequence_id) {
          return false;
        }
      }

      const size_t child_count = node->child_nodes.size();
      const size_t mbb_dim = view.mbb_dim[node_id];
      const size_t mbb_begin = view.mbb_begin[node_id];
      if (mbb_dim != node->beacons.size()) return false;
      if (mbb_begin + mbb_dim * child_count > view.mbb_lo.size() ||
          mbb_begin + mbb_dim * child_count > view.mbb_hi.size()) {
        return false;
      }
      if (!node->child_beacon_mbbs.empty() &&
          node->child_beacon_mbbs.size() != child_count) {
        return false;
      }
      for (size_t child_idx = 0; child_idx < node->child_beacon_mbbs.size(); ++child_idx) {
        if (node->child_beacon_mbbs[child_idx].size() != mbb_dim) return false;
        for (size_t dim = 0; dim < mbb_dim; ++dim) {
          const size_t flat = mbb_begin + dim * child_count + child_idx;
          if (view.mbb_lo[flat] != node->child_beacon_mbbs[child_idx][dim].min_dist ||
              view.mbb_hi[flat] != node->child_beacon_mbbs[child_idx][dim].max_dist) {
            return false;
          }
        }
      }

      const size_t leaf_count = node->child_leaves.size();
      const size_t leaf_dim = view.leaf_beacon_dim[node_id];
      const size_t leaf_beacon_begin = view.leaf_beacon_begin[node_id];
      if (leaf_dim != node->beacons.size()) return false;
      if (leaf_beacon_begin + leaf_dim * leaf_count >
          view.leaf_beacon_dists.size()) {
        return false;
      }
      if (!node->leaf_beacon_dists.empty() &&
          node->leaf_beacon_dists.size() != leaf_count) {
        return false;
      }
      for (size_t leaf_idx = 0; leaf_idx < node->leaf_beacon_dists.size(); ++leaf_idx) {
        if (node->leaf_beacon_dists[leaf_idx].size() != leaf_dim) return false;
        for (size_t dim = 0; dim < leaf_dim; ++dim) {
          const size_t flat = leaf_beacon_begin + dim * leaf_count + leaf_idx;
          if (view.leaf_beacon_dists[flat] !=
              node->leaf_beacon_dists[leaf_idx][dim]) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

std::vector<std::shared_ptr<WorldNode>> BioGeometryIndexBuilder::find_neighbors(
    const BioSequence& query_seq,
    const std::vector<std::shared_ptr<WorldNode>>& candidates,
    int radius) const {
  std::vector<std::shared_ptr<WorldNode>> result;
  for (const auto& node : candidates) {
    if (!node->center_ptr) continue;
    int dist = build_distance(query_seq.seq, node->center_ptr->seq,
                              range_config_.distance_mode);
    if (dist <= radius) result.push_back(node);
  }
  return result;
}

std::vector<std::shared_ptr<BioSequence>> BioGeometryIndexBuilder::deduplicate(
    const std::vector<std::shared_ptr<BioSequence>>& raw) {
  std::unordered_map<std::string, std::shared_ptr<BioSequence>> seq_map;
  for (const auto& seq : raw) {
    stats_.added_sequences++;
    auto it = seq_map.find(seq->seq);
    if (it != seq_map.end()) {
      for (const auto& occ : seq->ref_positions) {
        it->second->add_occurrence(occ.ref_id, occ.start, occ.end, occ.strand);
      }
      if (seq->ref_positions.empty() && it->second->ref_positions.empty()) {
        it->second->add_occurrence(seq->id, 0, static_cast<int>(seq->seq.size()), "+");
      }
      if (!it->second->bwt_interval.valid() && seq->bwt_interval.valid()) {
        it->second->set_bwt_interval(seq->bwt_interval.start, seq->bwt_interval.end);
      }
      stats_.deduplicated++;
    } else {
      seq_map[seq->seq] = seq;
    }
  }

  unique_sequences.clear();
  for (const auto& entry : seq_map) unique_sequences[entry.second->id] = entry.second;
  stats_.unique_sequences = seq_map.size();

  std::vector<std::shared_ptr<BioSequence>> out;
  out.reserve(seq_map.size());
  for (const auto& entry : seq_map) out.push_back(entry.second);
  return out;
}

void BioGeometryIndexBuilder::phase1_build_extended_sketch(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  extended_layers_.assign(static_cast<size_t>(hierarchy_.num_expanded_layers()),
                          std::vector<std::shared_ptr<WorldNode>>());
  std::unordered_map<const void*, Phase1CoverGroupIndex> cover_group_indexes;

  auto add_phase1_scan_stats = [&](const Phase1CoverScanResult& scan) {
    stats_.phase1_cover_candidate_scans += scan.candidate_scans;
    stats_.phase1_length_pruned_candidates += scan.length_pruned;
    stats_.phase1_exact_distance_calls += scan.exact_distance_calls;
    stats_.phase1_candidate_pairs += scan.exact_distance_calls;
  };

  for (const auto& sequence : unique_seqs) {
    std::shared_ptr<WorldNode> parent;
    for (int layer_idx = 0; layer_idx < hierarchy_.num_expanded_layers(); ++layer_idx) {
      const int radius = expanded_radii_[static_cast<size_t>(layer_idx)];
      const std::vector<std::shared_ptr<WorldNode>>* candidates = nullptr;
      if (layer_idx == 0) {
        candidates = &extended_layers_[0];
      } else if (parent) {
        candidates = &parent->child_nodes;
      }

      Phase1CoverScanResult scan;
      if (candidates && !candidates->empty()) {
        stats_.phase1_total_possible_pairs += candidates->size();
        auto use_scan = [&]() {
          stats_.phase1_scan_queries++;
          scan = find_best_phase1_cover(*candidates, sequence, radius,
                                        range_config_.distance_mode);
          add_phase1_scan_stats(scan);
        };

        if (range_config_.phase1_candidate_mode == Phase1CandidateMode::Scan) {
          use_scan();
        } else {
          auto& group_index =
              cover_group_indexes[static_cast<const void*>(candidates)];
          auto candidate_query = group_index.query(
              *candidates, sequence, radius, range_config_);
          stats_.phase1_metric_build_distance_calls +=
              candidate_query.metric_build_distance_calls;
          switch (candidate_query.source) {
            case Phase1CoverSource::Scan:
              use_scan();
              break;
            case Phase1CoverSource::FallbackScan:
              stats_.phase1_fallback_scan_queries++;
              scan = find_best_phase1_cover(*candidates, sequence, radius,
                                            range_config_.distance_mode);
              add_phase1_scan_stats(scan);
              break;
            case Phase1CoverSource::Metric:
              stats_.phase1_metric_index_queries++;
              stats_.phase1_metric_distance_calls +=
                  candidate_query.metric_distance_calls;
              scan = find_best_phase1_cover_by_indices(
                  *candidates, candidate_query.candidate_indices, sequence,
                  radius, range_config_.distance_mode);
              add_phase1_scan_stats(scan);
              break;
            case Phase1CoverSource::QGram:
              stats_.phase1_qgram_index_queries++;
              stats_.phase1_qgram_touched_candidates +=
                  candidate_query.qgram_touched_candidates;
              stats_.phase1_qgram_pruned_candidates +=
                  candidate_query.qgram_pruned_candidates;
              scan = find_best_phase1_cover_by_indices(
                  *candidates, candidate_query.candidate_indices, sequence,
                  radius, range_config_.distance_mode);
              add_phase1_scan_stats(scan);
              break;
          }
        }
      }

      if (!scan.best) {
        auto new_node = std::make_shared<WorldNode>(
            sequence, radius, layer_idx);
        const bool is_primary = expanded_layer_is_primary(layer_idx);
        const int primary_idx = is_primary ? expanded_to_primary_index(layer_idx) : -1;
        reset_node_metadata(new_node, layer_idx, is_primary, primary_idx);
        extended_layers_[static_cast<size_t>(layer_idx)].push_back(new_node);
        if (parent) parent->child_nodes.push_back(new_node);
        if (is_primary) {
          stats_.created_primary_nodes[static_cast<size_t>(primary_idx)]++;
        } else {
          stats_.created_auxiliary_nodes++;
        }
        stats_.phase1_cover_misses++;
        parent = new_node;
      } else {
        stats_.phase1_best_cover_hits++;
        parent = scan.best;
      }
      if (!parent) break;
    }
  }
}

void BioGeometryIndexBuilder::phase2_inter_tier_rebinding() {
  for (int layer_idx = 0; layer_idx + 1 < hierarchy_.num_expanded_layers(); ++layer_idx) {
    auto& parents = extended_layers_[static_cast<size_t>(layer_idx)];
    auto& children = extended_layers_[static_cast<size_t>(layer_idx + 1)];
    stats_.phase2_total_possible_pairs += parents.size() * children.size();
    for (auto& parent : parents) parent->child_nodes.clear();

    if (range_config_.link_mode == BuildRangeMode::Full) {
      for (auto& parent : parents) {
        if (!parent->center_ptr) continue;
        {
          ScopedTimer verify_timer(&stats_.phase2_exact_verify_ms);
          for (auto& child : children) {
            if (!child->center_ptr) continue;
            stats_.phase2_candidate_pairs++;
            stats_.phase2_exact_distance_calls++;
            int dist = build_distance(parent->center_ptr->seq,
                                      child->center_ptr->seq,
                                      range_config_.distance_mode);
            if (dist <= parent->radius + child->radius) {
              {
                ScopedTimer timer(&stats_.phase2_edge_insert_ms);
                parent->child_nodes.push_back(child);
              }
              stats_.phase2_edges_added++;
            }
          }
        }
      }
      continue;
    }

    std::vector<RangeJoinItem> items;
    items.reserve(parents.size());
    int max_parent_radius = 0;
    for (size_t parent_idx = 0; parent_idx < parents.size(); ++parent_idx) {
      if (!parents[parent_idx]->center_ptr) continue;
      items.push_back({parent_idx, parents[parent_idx]->center_ptr->seq});
      max_parent_radius = std::max(max_parent_radius, parents[parent_idx]->radius);
    }
    ExactRangeJoinIndex parent_index(range_config_.range_join);
    {
      ScopedTimer timer(&stats_.phase2_index_build_ms);
      parent_index.build(items);
    }

    for (auto& child : children) {
      if (!child->center_ptr) continue;
      RangeJoinQueryResult candidates;
      {
        ScopedTimer timer(&stats_.phase2_candidate_query_ms);
        candidates =
            parent_index.query(child->center_ptr->seq, max_parent_radius + child->radius);
      }
      accumulate_range_timing(stats_, candidates);
      stats_.phase2_candidate_pairs += candidates.candidate_item_ids.size();
      if (candidates.used_full_scan) stats_.phase2_full_scan_fallback_count++;
      if (candidates.mode_used == RangeCandidateMode::PigeonholeOnly) {
        stats_.phase2_pigeonhole_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::QGramOnly) {
        stats_.phase2_qgram_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::Hybrid) {
        stats_.phase2_hybrid_queries++;
      }
      stats_.phase2_qgram_candidate_pairs += candidates.qgram_candidate_count;
      stats_.phase2_qgram_pruned_by_l1 += candidates.qgram_pruned_by_l1;
      stats_.phase2_length_pruned_pairs += candidates.length_filtered_items;
      stats_.phase2_seed_candidate_pairs_before_length_filter +=
          candidates.seed_candidate_pairs_before_length_filter;
      stats_.phase2_seed_length_pruned_candidates +=
          candidates.seed_length_pruned_candidates;
      stats_.phase2_pigeonhole_early_abort_count +=
          candidates.pigeonhole_early_abort_count;
      stats_.phase2_range_final_candidate_pairs +=
          candidates.final_candidate_pairs;
      stats_.phase2_required_shared_nonpositive_count +=
          candidates.required_shared_nonpositive;
      stats_.phase2_auto_pigeonhole_accepted +=
          candidates.auto_pigeonhole_accepted;
      stats_.phase2_auto_pigeonhole_rejected_large_candidates +=
          candidates.auto_pigeonhole_rejected_large_candidates;
      stats_.phase2_auto_qgram_invoked += candidates.auto_qgram_invoked;
      stats_.phase2_auto_hybrid_invoked += candidates.auto_hybrid_invoked;
      stats_.phase2_auto_final_candidate_pairs +=
          candidates.auto_final_candidate_pairs;
      stats_.phase2_auto_candidate_ratio_sum +=
          candidates.auto_candidate_ratio_sum;
      {
        ScopedTimer verify_timer(&stats_.phase2_exact_verify_ms);
        for (size_t parent_idx : candidates.candidate_item_ids) {
          auto& parent = parents[parent_idx];
          int tau = parent->radius + child->radius;
          if (std::llabs(
                  static_cast<long long>(parent->center_ptr->seq.size()) -
                  static_cast<long long>(child->center_ptr->seq.size())) > tau) {
            stats_.phase2_length_pruned_pairs++;
            continue;
          }
          stats_.phase2_exact_distance_calls++;
          int dist = build_distance_bounded(
              parent->center_ptr->seq, child->center_ptr->seq, tau,
              range_config_.distance_mode);
          if (dist <= tau) {
            {
              ScopedTimer timer(&stats_.phase2_edge_insert_ms);
              parent->child_nodes.push_back(child);
            }
            stats_.phase2_edges_added++;
          }
        }
      }
    }
  }
}

void BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb() {
  primary_layers_.assign(static_cast<size_t>(hierarchy_.num_primary_layers()),
                         std::vector<std::shared_ptr<WorldNode>>());

  for (int primary_idx = 0; primary_idx < hierarchy_.num_primary_layers(); ++primary_idx) {
    auto& target_layer = primary_layers_[static_cast<size_t>(primary_idx)];
    target_layer = extended_layers_[static_cast<size_t>(primary_idx * 2)];
    for (auto& node : target_layer) {
      reset_node_metadata(node, primary_idx * 2, true, primary_idx);
      node->beacons.clear();
      node->child_beacon_mbbs.clear();
      node->mbb_rect_index.reset();
      node->leaf_beacon_dists.clear();
    }
  }

  for (int primary_idx = 0; primary_idx < hierarchy_.num_primary_layers(); ++primary_idx) {
    auto& current_layer = primary_layers_[static_cast<size_t>(primary_idx)];
    const bool is_finest = (primary_idx == finest_primary_layer_index());
    for (auto& node : current_layer) {
      if (is_finest) {
        node->child_nodes.clear();
        continue;
      }

      std::vector<std::shared_ptr<WorldNode>> auxiliary_nodes = node->child_nodes;
      {
        ScopedTimer timer(&stats_.phase3_collect_beacons_ms);
        node->beacons.reserve(auxiliary_nodes.size());
        for (const auto& aux : auxiliary_nodes) {
          if (aux && aux->center_ptr) node->beacons.push_back(aux->center_ptr);
        }
      }

      std::vector<std::shared_ptr<WorldNode>> direct_children;
      {
        ScopedTimer timer(&stats_.phase3_collapse_children_ms);
        for (const auto& aux : auxiliary_nodes) {
          if (!aux) continue;
          for (const auto& child : aux->child_nodes) {
            if (std::find(direct_children.begin(), direct_children.end(), child) ==
                direct_children.end()) {
              direct_children.push_back(child);
            }
          }
        }
        node->child_nodes = std::move(direct_children);
      }

      {
        ScopedTimer timer(&stats_.phase3_child_mbb_distance_ms);
        node->child_beacon_mbbs.assign(node->child_nodes.size(), {});
        for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
          auto& child = node->child_nodes[child_idx];
          if (!child->center_ptr) continue;
          node->child_beacon_mbbs[child_idx].reserve(node->beacons.size());
          for (const auto& beacon : node->beacons) {
            if (!beacon) continue;
            int dist = build_distance(child->center_ptr->seq, beacon->seq,
                                      range_config_.distance_mode);
            MBB mbb;
            mbb.min_dist = std::max(0, dist - child->radius);
            mbb.max_dist = dist + child->radius;
            node->child_beacon_mbbs[child_idx].push_back(mbb);
          }
        }
      }

      if (node->child_nodes.size() >= range_config_.min_rect_index_fanout &&
          node->child_nodes.size() <= std::numeric_limits<uint32_t>::max() &&
          !node->beacons.empty() &&
          node->child_beacon_mbbs.size() == node->child_nodes.size()) {
        try {
          ScopedTimer timer(&stats_.phase3_rect_index_build_ms);
          std::vector<MBBRectIndex::Rect> rects;
          rects.reserve(node->child_nodes.size());
          bool valid = true;
          for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
            const auto& row = node->child_beacon_mbbs[child_idx];
            if (row.size() != node->beacons.size()) {
              valid = false;
              break;
            }
            MBBRectIndex::Rect rect;
            rect.child_id = static_cast<uint32_t>(child_idx);
            rect.lo.reserve(row.size());
            rect.hi.reserve(row.size());
            for (const auto& mbb : row) {
              rect.lo.push_back(mbb.min_dist);
              rect.hi.push_back(mbb.max_dist);
            }
            rects.push_back(std::move(rect));
          }
          if (valid) {
            auto rect_index = std::make_shared<MBBRectIndex>();
            rect_index->build(rects);
            if (rect_index->size() == node->child_nodes.size() &&
                rect_index->dim() == node->beacons.size()) {
              node->mbb_rect_index = std::move(rect_index);
            }
          }
        } catch (...) {
          node->mbb_rect_index.reset();
        }
      }
    }
  }
}

void BioGeometryIndexBuilder::attach_leaves(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs) {
  auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  stats_.total_possible_leaf_pairs = finest_layer.size() * unique_seqs.size();
  for (auto& node : finest_layer) {
    node->child_leaves.clear();
    node->beacons.clear();
    node->leaf_beacon_dists.clear();
    if (node->center_ptr) node->beacons.push_back(node->center_ptr);
  }

  LeafAttachDirection actual_direction = range_config_.leaf_attach_direction;
  if (range_config_.leaf_attach_mode == BuildRangeMode::Full) {
    actual_direction = LeafAttachDirection::WorldToSeq;
  } else if (actual_direction == LeafAttachDirection::Auto) {
    actual_direction =
        finest_layer.size() < unique_seqs.size()
            ? LeafAttachDirection::WorldToSeq
            : LeafAttachDirection::SeqToWorld;
  }
  stats_.leaf_attach_direction_used = actual_direction;

  if (range_config_.leaf_attach_mode == BuildRangeMode::Full) {
    const int thread_count = std::max(1, omp_get_max_threads());
    std::vector<double> thread_exact_ms(static_cast<size_t>(thread_count), 0.0);
    std::vector<double> thread_beacon_ms(static_cast<size_t>(thread_count), 0.0);
    std::vector<double> thread_tuple_ms(static_cast<size_t>(thread_count), 0.0);
    std::vector<double> thread_populate_ms(static_cast<size_t>(thread_count), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t layer_idx = 0; layer_idx < finest_layer.size(); ++layer_idx) {
      const int tid = omp_get_thread_num();
      const size_t timer_idx =
          static_cast<size_t>(std::min(tid, thread_count - 1));
      auto& node = finest_layer[layer_idx];
      std::string center = node->get_center_sequence();
      {
        ScopedTimer verify_timer(&thread_exact_ms[timer_idx]);
        for (const auto& seq : unique_seqs) {
          int dist = build_distance(center, seq->seq, range_config_.distance_mode);
          if (dist <= node->radius) {
            auto beacon_dists = leaf_beacon_distances_timed(
                seq, node->beacons, dist, range_config_.distance_mode,
                &thread_beacon_ms[timer_idx]);
            {
              ScopedTimer timer(&thread_tuple_ms[timer_idx]);
              node->child_leaves.push_back(seq);
              node->leaf_beacon_dists.push_back(std::move(beacon_dists));
            }
          }
        }
      }
      {
        ScopedTimer timer(&thread_populate_ms[timer_idx]);
        node->data_count = static_cast<int>(node->child_leaves.size());
      }
    }
    for (double value : thread_exact_ms) stats_.leaf_exact_verify_ms += value;
    for (double value : thread_beacon_ms) stats_.leaf_beacon_distance_ms += value;
    for (double value : thread_tuple_ms) stats_.leaf_tuple_emit_ms += value;
    for (double value : thread_populate_ms) stats_.leaf_populate_ms += value;
    stats_.leaf_candidate_pairs = stats_.total_possible_leaf_pairs;
    stats_.leaf_exact_distance_calls = stats_.total_possible_leaf_pairs;
  } else {
    auto record_leaf_candidates = [&](const RangeJoinQueryResult& candidates) {
      stats_.leaf_candidate_pairs += candidates.candidate_item_ids.size();
      if (candidates.used_full_scan) stats_.leaf_full_scan_fallback_count++;
      if (candidates.mode_used == RangeCandidateMode::PigeonholeOnly) {
        stats_.leaf_pigeonhole_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::QGramOnly) {
        stats_.leaf_qgram_queries++;
      } else if (candidates.mode_used == RangeCandidateMode::Hybrid) {
        stats_.leaf_hybrid_queries++;
      }
      stats_.leaf_qgram_candidate_pairs += candidates.qgram_candidate_count;
      stats_.leaf_qgram_pruned_by_l1 += candidates.qgram_pruned_by_l1;
      stats_.leaf_length_pruned_pairs += candidates.length_filtered_items;
      stats_.leaf_seed_candidate_pairs_before_length_filter +=
          candidates.seed_candidate_pairs_before_length_filter;
      stats_.leaf_seed_length_pruned_candidates +=
          candidates.seed_length_pruned_candidates;
      stats_.leaf_pigeonhole_early_abort_count +=
          candidates.pigeonhole_early_abort_count;
      stats_.leaf_range_final_candidate_pairs +=
          candidates.final_candidate_pairs;
      stats_.leaf_required_shared_nonpositive_count +=
          candidates.required_shared_nonpositive;
      stats_.leaf_auto_pigeonhole_accepted +=
          candidates.auto_pigeonhole_accepted;
      stats_.leaf_auto_pigeonhole_rejected_large_candidates +=
          candidates.auto_pigeonhole_rejected_large_candidates;
      stats_.leaf_auto_qgram_invoked += candidates.auto_qgram_invoked;
      stats_.leaf_auto_hybrid_invoked += candidates.auto_hybrid_invoked;
      stats_.leaf_auto_final_candidate_pairs +=
          candidates.auto_final_candidate_pairs;
      stats_.leaf_auto_candidate_ratio_sum +=
          candidates.auto_candidate_ratio_sum;
    };

    if (actual_direction == LeafAttachDirection::SeqToWorld) {
      std::vector<RangeJoinItem> items;
      items.reserve(finest_layer.size());
      int max_radius = 0;
      for (size_t world_idx = 0; world_idx < finest_layer.size(); ++world_idx) {
        const auto& world = finest_layer[world_idx];
        if (!world->center_ptr) continue;
        items.push_back({world_idx, world->center_ptr->seq});
        max_radius = std::max(max_radius, world->radius);
      }
      ExactRangeJoinIndex world_index(range_config_.range_join);
      {
        ScopedTimer timer(&stats_.leaf_index_build_ms);
        world_index.build(items);
      }
      for (const auto& seq : unique_seqs) {
        RangeJoinQueryResult candidates;
        {
          ScopedTimer timer(&stats_.leaf_candidate_query_ms);
          candidates = world_index.query(seq->seq, max_radius);
        }
        accumulate_range_timing(stats_, candidates);
        record_leaf_candidates(candidates);
        {
          ScopedTimer verify_timer(&stats_.leaf_exact_verify_ms);
          for (size_t world_idx : candidates.candidate_item_ids) {
            auto& world = finest_layer[world_idx];
            if (std::llabs(static_cast<long long>(seq->seq.size()) -
                           static_cast<long long>(world->center_ptr->seq.size())) >
                world->radius) {
              stats_.leaf_length_pruned_pairs++;
              continue;
            }
            stats_.leaf_exact_distance_calls++;
            int dist = build_distance_bounded(seq->seq, world->center_ptr->seq,
                                              world->radius,
                                              range_config_.distance_mode);
            if (dist <= world->radius) {
              auto beacon_dists = leaf_beacon_distances_timed(
                  seq, world->beacons, dist, range_config_.distance_mode,
                  &stats_.leaf_beacon_distance_ms);
              {
                ScopedTimer timer(&stats_.leaf_tuple_emit_ms);
                world->child_leaves.push_back(seq);
                world->leaf_beacon_dists.push_back(std::move(beacon_dists));
              }
            }
          }
        }
      }
      {
        ScopedTimer timer(&stats_.leaf_tuple_merge_sort_ms);
      }
      for (auto& node : finest_layer) {
        ScopedTimer timer(&stats_.leaf_populate_ms);
        node->data_count = static_cast<int>(node->child_leaves.size());
      }
    } else {
      struct LeafAttachTuple {
        size_t world_idx = 0;
        size_t seq_idx = 0;
        std::vector<int> beacon_dists;
      };

      std::vector<RangeJoinItem> items;
      items.reserve(unique_seqs.size());
      for (size_t seq_idx = 0; seq_idx < unique_seqs.size(); ++seq_idx) {
        items.push_back({seq_idx, unique_seqs[seq_idx]->seq});
      }

      ExactRangeJoinIndex sequence_index(range_config_.range_join);
      {
        ScopedTimer timer(&stats_.leaf_index_build_ms);
        sequence_index.build(items);
      }

      std::vector<LeafAttachTuple> tuples;
      for (size_t world_idx = 0; world_idx < finest_layer.size(); ++world_idx) {
        auto& world = finest_layer[world_idx];
        if (!world->center_ptr) continue;
        RangeJoinQueryResult candidates;
        {
          ScopedTimer timer(&stats_.leaf_candidate_query_ms);
          candidates =
              sequence_index.query(world->center_ptr->seq, world->radius);
        }
        accumulate_range_timing(stats_, candidates);
        record_leaf_candidates(candidates);
        {
          ScopedTimer verify_timer(&stats_.leaf_exact_verify_ms);
          for (size_t seq_idx : candidates.candidate_item_ids) {
            const auto& seq = unique_seqs[seq_idx];
            if (std::llabs(static_cast<long long>(seq->seq.size()) -
                           static_cast<long long>(world->center_ptr->seq.size())) >
                world->radius) {
              stats_.leaf_length_pruned_pairs++;
              continue;
            }
            stats_.leaf_exact_distance_calls++;
            int dist = build_distance_bounded(world->center_ptr->seq, seq->seq,
                                              world->radius,
                                              range_config_.distance_mode);
            if (dist <= world->radius) {
              auto beacon_dists = leaf_beacon_distances_timed(
                  seq, world->beacons, dist, range_config_.distance_mode,
                  &stats_.leaf_beacon_distance_ms);
              {
                ScopedTimer timer(&stats_.leaf_tuple_emit_ms);
                tuples.push_back(
                    {world_idx, seq_idx, std::move(beacon_dists)});
              }
            }
          }
        }
      }

      {
        ScopedTimer timer(&stats_.leaf_tuple_merge_sort_ms);
        std::sort(tuples.begin(), tuples.end(),
                  [](const LeafAttachTuple& lhs,
                     const LeafAttachTuple& rhs) {
                    if (lhs.world_idx != rhs.world_idx) {
                      return lhs.world_idx < rhs.world_idx;
                    }
                    return lhs.seq_idx < rhs.seq_idx;
                  });
      }
      {
        ScopedTimer timer(&stats_.leaf_populate_ms);
        for (auto& tuple : tuples) {
          auto& world = finest_layer[tuple.world_idx];
          world->child_leaves.push_back(unique_seqs[tuple.seq_idx]);
          world->leaf_beacon_dists.push_back(std::move(tuple.beacon_dists));
        }
        for (auto& node : finest_layer) {
          node->data_count = static_cast<int>(node->child_leaves.size());
        }
      }
    }
  }

  size_t total_links = 0;
  for (const auto& node : finest_layer) total_links += node->child_leaves.size();
  stats_.leaf_attachments_added = total_links;
  double avg_links =
      finest_layer.empty() ? 0.0 : static_cast<double>(total_links) / finest_layer.size();
  std::cerr << "    Attached " << total_links << " leaf links to finest primary layer"
            << " (avg " << avg_links << " per node)\n";
}

void BioGeometryIndexBuilder::assign_integer_ids() {
  world_node_count_ = 0;
  sequence_count_ = 0;

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (node) node->integer_id = INVALID_NODE_ID;
    }
  }

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id != INVALID_NODE_ID) continue;
      if (world_node_count_ > static_cast<size_t>(INVALID_NODE_ID - 1)) {
        throw std::runtime_error("too many world nodes for 32-bit NodeId");
      }
      node->integer_id = static_cast<NodeId>(world_node_count_++);
    }
  }

  std::vector<std::shared_ptr<BioSequence>> sequences;
  sequences.reserve(unique_sequences.size());
  for (const auto& entry : unique_sequences) {
    if (entry.second) {
      entry.second->sequence_id = INVALID_LEAF_ID;
      sequences.push_back(entry.second);
    }
  }
  std::sort(sequences.begin(), sequences.end(),
            [](const std::shared_ptr<BioSequence>& left,
               const std::shared_ptr<BioSequence>& right) {
              return left->id < right->id;
            });
  for (const auto& sequence : sequences) {
    if (sequence_count_ > static_cast<size_t>(INVALID_LEAF_ID - 1)) {
      throw std::runtime_error("too many sequences for 32-bit LeafId");
    }
    sequence->sequence_id = static_cast<LeafId>(sequence_count_++);
  }
}

void BioGeometryIndexBuilder::build_search_graph_view() {
  auto to_u32 = [](size_t value, const char* field) -> uint32_t {
    if (value > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(std::string(field) + " exceeds 32-bit view range");
    }
    return static_cast<uint32_t>(value);
  };

  SearchGraphView view;
  view.nodes.assign(world_node_count_, nullptr);
  view.leaves.assign(sequence_count_, nullptr);
  view.child_begin.assign(world_node_count_, 0);
  view.child_end.assign(world_node_count_, 0);
  view.leaf_begin.assign(world_node_count_, 0);
  view.leaf_end.assign(world_node_count_, 0);
  view.mbb_begin.assign(world_node_count_, 0);
  view.mbb_dim.assign(world_node_count_, 0);
  view.beacon_begin.assign(world_node_count_, 0);
  view.beacon_end.assign(world_node_count_, 0);
  view.leaf_beacon_begin.assign(world_node_count_, 0);
  view.leaf_beacon_dim.assign(world_node_count_, 0);

  for (const auto& entry : unique_sequences) {
    const auto& sequence = entry.second;
    if (!sequence || sequence->sequence_id >= sequence_count_) {
      throw std::runtime_error("cannot build search graph view with invalid leaf id");
    }
    view.leaves[sequence->sequence_id] = sequence;
  }

  for (const auto& layer : primary_layers_) {
    for (const auto& node : layer) {
      if (!node || node->integer_id >= world_node_count_) {
        throw std::runtime_error("cannot build search graph view with invalid node id");
      }
      const NodeId node_id = node->integer_id;
      view.nodes[node_id] = node;

      view.child_begin[node_id] = to_u32(view.child_ids.size(), "child_ids");
      for (const auto& child : node->child_nodes) {
        if (!child || child->integer_id >= world_node_count_) {
          throw std::runtime_error("cannot build search graph view with invalid child id");
        }
        view.child_ids.push_back(child->integer_id);
      }
      view.child_end[node_id] = to_u32(view.child_ids.size(), "child_ids");

      view.leaf_begin[node_id] = to_u32(view.leaf_ids.size(), "leaf_ids");
      for (const auto& leaf : node->child_leaves) {
        if (!leaf || leaf->sequence_id >= sequence_count_) {
          throw std::runtime_error("cannot build search graph view with invalid leaf id");
        }
        view.leaf_ids.push_back(leaf->sequence_id);
      }
      view.leaf_end[node_id] = to_u32(view.leaf_ids.size(), "leaf_ids");

      view.beacon_begin[node_id] = to_u32(view.beacon_ids.size(), "beacon_ids");
      for (const auto& beacon : node->beacons) {
        if (!beacon || beacon->sequence_id >= sequence_count_) {
          throw std::runtime_error("cannot build search graph view with invalid beacon id");
        }
        view.beacon_ids.push_back(beacon->sequence_id);
      }
      view.beacon_end[node_id] = to_u32(view.beacon_ids.size(), "beacon_ids");

      const size_t child_count = node->child_nodes.size();
      const size_t mbb_dim = node->beacons.size();
      view.mbb_begin[node_id] = to_u32(view.mbb_lo.size(), "mbb arrays");
      view.mbb_dim[node_id] = to_u32(mbb_dim, "mbb_dim");
      const size_t mbb_cells = child_count * mbb_dim;
      view.mbb_lo.resize(view.mbb_lo.size() + mbb_cells, 0);
      view.mbb_hi.resize(view.mbb_hi.size() + mbb_cells, 0);
      if (!node->child_beacon_mbbs.empty()) {
        if (node->child_beacon_mbbs.size() != child_count) {
          throw std::runtime_error("child MBB rows are not aligned with child nodes");
        }
        for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
          if (node->child_beacon_mbbs[child_idx].size() != mbb_dim) {
            throw std::runtime_error("child MBB row dimension mismatch");
          }
          for (size_t dim = 0; dim < mbb_dim; ++dim) {
            const size_t flat = view.mbb_begin[node_id] +
                                dim * child_count + child_idx;
            view.mbb_lo[flat] =
                static_cast<int32_t>(node->child_beacon_mbbs[child_idx][dim].min_dist);
            view.mbb_hi[flat] =
                static_cast<int32_t>(node->child_beacon_mbbs[child_idx][dim].max_dist);
          }
        }
      }

      const size_t leaf_count = node->child_leaves.size();
      const size_t leaf_dim = node->beacons.size();
      view.leaf_beacon_begin[node_id] =
          to_u32(view.leaf_beacon_dists.size(), "leaf_beacon_dists");
      view.leaf_beacon_dim[node_id] = to_u32(leaf_dim, "leaf_beacon_dim");
      const size_t leaf_cells = leaf_count * leaf_dim;
      view.leaf_beacon_dists.resize(view.leaf_beacon_dists.size() + leaf_cells, 0);
      if (!node->leaf_beacon_dists.empty()) {
        if (node->leaf_beacon_dists.size() != leaf_count) {
          throw std::runtime_error("leaf beacon rows are not aligned with leaves");
        }
        for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
          if (node->leaf_beacon_dists[leaf_idx].size() != leaf_dim) {
            throw std::runtime_error("leaf beacon row dimension mismatch");
          }
          for (size_t dim = 0; dim < leaf_dim; ++dim) {
            const size_t flat = view.leaf_beacon_begin[node_id] +
                                dim * leaf_count + leaf_idx;
            view.leaf_beacon_dists[flat] =
                static_cast<int32_t>(node->leaf_beacon_dists[leaf_idx][dim]);
          }
        }
      }
    }
  }

  search_graph_view_ = std::move(view);
}

void BioGeometryIndexBuilder::print_summary() const {
  Statistics stats = get_statistics();
  std::cerr << "  Build timing:\n"
            << "    total=" << format_ms(stats.total_build_ms) << " ms\n"
            << "    phase0_dedup=" << format_ms(stats.phase0_dedup_ms) << " ms\n"
            << "    phase1_sketch=" << format_ms(stats.phase1_sketch_ms) << " ms\n"
            << "      possible_pairs=" << stats.phase1_total_possible_pairs
            << " candidates=" << stats.phase1_candidate_pairs
            << " cover_scans=" << stats.phase1_cover_candidate_scans
            << " exact_calls=" << stats.phase1_exact_distance_calls
            << " length_pruned=" << stats.phase1_length_pruned_candidates
            << " hits=" << stats.phase1_best_cover_hits
            << " misses=" << stats.phase1_cover_misses
            << " scan_queries=" << stats.phase1_scan_queries
            << " metric_queries=" << stats.phase1_metric_index_queries
            << " qgram_queries=" << stats.phase1_qgram_index_queries
            << " fallback_scans=" << stats.phase1_fallback_scan_queries
            << " metric_dist_calls=" << stats.phase1_metric_distance_calls
            << " metric_build_dist_calls="
            << stats.phase1_metric_build_distance_calls
            << " qgram_touched=" << stats.phase1_qgram_touched_candidates
            << " qgram_pruned=" << stats.phase1_qgram_pruned_candidates
            << "\n"
            << "    phase2_rebinding=" << format_ms(stats.phase2_rebinding_ms) << " ms\n"
            << "      index_build=" << format_ms(stats.phase2_index_build_ms) << " ms\n"
            << "      candidate_query=" << format_ms(stats.phase2_candidate_query_ms) << " ms\n"
            << "      exact_verify=" << format_ms(stats.phase2_exact_verify_ms) << " ms\n"
            << "      edge_insert=" << format_ms(stats.phase2_edge_insert_ms) << " ms\n"
            << "    phase3_mbb=" << format_ms(stats.phase3_mbb_ms) << " ms\n"
            << "      collect_beacons=" << format_ms(stats.phase3_collect_beacons_ms) << " ms\n"
            << "      collapse_children=" << format_ms(stats.phase3_collapse_children_ms) << " ms\n"
            << "      child_mbb_distance=" << format_ms(stats.phase3_child_mbb_distance_ms) << " ms\n"
            << "      rect_index_build=" << format_ms(stats.phase3_rect_index_build_ms) << " ms\n"
            << "    phase4_attach=" << format_ms(stats.phase4_attach_ms) << " ms\n"
            << "      index_build=" << format_ms(stats.leaf_index_build_ms) << " ms\n"
            << "      candidate_query=" << format_ms(stats.leaf_candidate_query_ms) << " ms\n"
            << "      exact_verify=" << format_ms(stats.leaf_exact_verify_ms) << " ms\n"
            << "      tuple_emit=" << format_ms(stats.leaf_tuple_emit_ms) << " ms\n"
            << "      tuple_merge_sort=" << format_ms(stats.leaf_tuple_merge_sort_ms) << " ms\n"
            << "      populate=" << format_ms(stats.leaf_populate_ms) << " ms\n"
            << "      leaf_beacon_distance=" << format_ms(stats.leaf_beacon_distance_ms) << " ms\n"
            << "    assign_ids=" << format_ms(stats.assign_ids_ms) << " ms\n"
            << "    graph_view=" << format_ms(stats.graph_view_ms) << " ms\n"
            << "    range_join:\n"
            << "      posting_lookup=" << format_ms(stats.range_posting_lookup_ms) << " ms\n"
            << "      seed_union=" << format_ms(stats.range_seed_union_ms) << " ms\n"
            << "      length_filter=" << format_ms(stats.range_length_filter_ms) << " ms\n"
            << "      qgram_query=" << format_ms(stats.range_qgram_query_ms) << " ms\n"
            << "      hybrid_intersection="
            << format_ms(stats.range_hybrid_intersection_ms) << " ms\n"
            << "      full_scan=" << format_ms(stats.range_full_scan_ms) << " ms\n";
  std::cerr << "  Construction range modes: links="
            << build_range_mode_name(range_config_.link_mode)
            << " leaves=" << build_range_mode_name(range_config_.leaf_attach_mode)
            << " phase1=" << phase1_candidate_mode_name(
                   range_config_.phase1_candidate_mode)
            << " seeds=" << range_config_.range_join.min_seed_len
            << ".." << range_config_.range_join.max_seed_len
            << " candidates="
            << range_candidate_mode_name(range_config_.range_join.candidate_mode)
            << " leaf_direction="
            << leaf_attach_direction_name(stats.leaf_attach_direction_used)
            << " qgram_q=" << range_config_.range_join.qgram_q
            << " auto_max_candidates="
            << range_config_.range_join.auto_pigeonhole_max_candidates
            << " auto_max_ratio_ignored="
            << range_config_.range_join.auto_pigeonhole_max_ratio
            << " auto_hybrid="
            << (range_config_.range_join.auto_hybrid_on_large_candidates
                    ? "true"
                    : "false")
            << "\n";
  std::cerr << "  Phase2 range join: possible=" << stats.phase2_total_possible_pairs
            << " candidates=" << stats.phase2_candidate_pairs
            << " exact_calls=" << stats.phase2_exact_distance_calls
            << " edges=" << stats.phase2_edges_added
            << " fallbacks=" << stats.phase2_full_scan_fallback_count
            << " pigeonhole_queries=" << stats.phase2_pigeonhole_queries
            << " qgram_queries=" << stats.phase2_qgram_queries
            << " hybrid_queries=" << stats.phase2_hybrid_queries
            << " qgram_candidates=" << stats.phase2_qgram_candidate_pairs
            << " qgram_l1_pruned=" << stats.phase2_qgram_pruned_by_l1
            << " length_pruned=" << stats.phase2_length_pruned_pairs
            << " seed_candidates_before_length_filter="
            << stats.phase2_seed_candidate_pairs_before_length_filter
            << " seed_length_pruned="
            << stats.phase2_seed_length_pruned_candidates
            << " pigeonhole_early_abort="
            << stats.phase2_pigeonhole_early_abort_count
            << " range_final_candidates="
            << stats.phase2_range_final_candidate_pairs
            << " required_shared_nonpositive="
            << stats.phase2_required_shared_nonpositive_count
            << " auto_pigeonhole_accepted="
            << stats.phase2_auto_pigeonhole_accepted
            << " auto_pigeonhole_rejected_large="
            << stats.phase2_auto_pigeonhole_rejected_large_candidates
            << " auto_qgram_invoked=" << stats.phase2_auto_qgram_invoked
            << " auto_hybrid_invoked=" << stats.phase2_auto_hybrid_invoked
            << " auto_final_candidates="
            << stats.phase2_auto_final_candidate_pairs
            << " auto_avg_candidate_ratio_ignored="
            << stats.phase2_auto_candidate_ratio_avg
            << " candidate_reduction=" << (stats.phase2_candidate_reduction_ratio * 100.0)
            << "% exact_reduction=" << (stats.phase2_exact_distance_reduction_ratio * 100.0)
            << "%\n";
  std::cerr << "  Leaf range join: possible=" << stats.total_possible_leaf_pairs
            << " candidates=" << stats.leaf_candidate_pairs
            << " exact_calls=" << stats.leaf_exact_distance_calls
            << " attachments=" << stats.leaf_attachments_added
            << " fallbacks=" << stats.leaf_full_scan_fallback_count
            << " pigeonhole_queries=" << stats.leaf_pigeonhole_queries
            << " qgram_queries=" << stats.leaf_qgram_queries
            << " hybrid_queries=" << stats.leaf_hybrid_queries
            << " qgram_candidates=" << stats.leaf_qgram_candidate_pairs
            << " qgram_l1_pruned=" << stats.leaf_qgram_pruned_by_l1
            << " length_pruned=" << stats.leaf_length_pruned_pairs
            << " seed_candidates_before_length_filter="
            << stats.leaf_seed_candidate_pairs_before_length_filter
            << " seed_length_pruned="
            << stats.leaf_seed_length_pruned_candidates
            << " pigeonhole_early_abort="
            << stats.leaf_pigeonhole_early_abort_count
            << " range_final_candidates="
            << stats.leaf_range_final_candidate_pairs
            << " required_shared_nonpositive="
            << stats.leaf_required_shared_nonpositive_count
            << " auto_pigeonhole_accepted="
            << stats.leaf_auto_pigeonhole_accepted
            << " auto_pigeonhole_rejected_large="
            << stats.leaf_auto_pigeonhole_rejected_large_candidates
            << " auto_qgram_invoked=" << stats.leaf_auto_qgram_invoked
            << " auto_hybrid_invoked=" << stats.leaf_auto_hybrid_invoked
            << " auto_final_candidates="
            << stats.leaf_auto_final_candidate_pairs
            << " auto_avg_candidate_ratio_ignored="
            << stats.leaf_auto_candidate_ratio_avg
            << " candidate_reduction=" << (stats.leaf_candidate_reduction_ratio * 100.0)
            << "% exact_reduction=" << (stats.leaf_exact_distance_reduction_ratio * 100.0)
            << "%\n";
  std::cerr << "  Primary layers: " << num_primary_layers() << "\n";
  for (int layer_idx = 0; layer_idx < num_primary_layers(); ++layer_idx) {
    std::cerr << "    W" << layer_idx
              << " radius=" << hierarchy_.primary_radii[static_cast<size_t>(layer_idx)]
              << " nodes=" << primary_layers_[static_cast<size_t>(layer_idx)].size() << "\n";
  }

  const auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  if (stats_.unique_sequences > 0 && !finest_layer.empty()) {
    double compression =
        1.0 - static_cast<double>(finest_layer.size()) / stats_.unique_sequences;
    std::cerr << "  Finest-layer compression: " << (compression * 100.0) << "% ("
              << stats_.unique_sequences << " unique -> " << finest_layer.size() << " nodes)\n";
  }

  for (int layer_idx = 0; layer_idx + 1 < num_primary_layers(); ++layer_idx) {
    const auto& layer = primary_layers_[static_cast<size_t>(layer_idx)];
    size_t total_edges = 0;
    for (const auto& node : layer) total_edges += node->child_nodes.size();
    double avg_edges = layer.empty() ? 0.0 : static_cast<double>(total_edges) / layer.size();
    std::cerr << "  Avg W" << layer_idx << " -> W" << (layer_idx + 1)
              << " edges: " << avg_edges << "\n";
  }
}

void BioGeometryIndexBuilder::build(
    const std::vector<std::shared_ptr<BioSequence>>& raw_sequences) {
  const auto build_start = Clock::now();
  stats_ = Statistics{};
  stats_.created_primary_nodes.assign(static_cast<size_t>(num_primary_layers()), 0);
  unique_sequences.clear();
  world_node_count_ = 0;
  sequence_count_ = 0;
  search_graph_view_ = SearchGraphView{};
  primary_layers_.assign(static_cast<size_t>(num_primary_layers()),
                         std::vector<std::shared_ptr<WorldNode>>());
  extended_layers_.clear();

  std::cerr << "[Build generalized hierarchy] Starting for " << raw_sequences.size()
            << " sequences...\n";
  std::cerr << "  Phase 0: Deduplicating sequences...\n";
  std::vector<std::shared_ptr<BioSequence>> unique_seqs;
  {
    ScopedTimer timer(&stats_.phase0_dedup_ms);
    unique_seqs = deduplicate(raw_sequences);
  }
  std::cerr << "    " << raw_sequences.size() << " -> " << unique_seqs.size() << " unique ("
            << stats_.deduplicated << " merged)\n";

  std::cerr << "  Phase 1: Extended hierarchy sketch (top-down)...\n";
  std::cerr << "    Primary radii: ";
  for (size_t i = 0; i < hierarchy_.primary_radii.size(); ++i) {
    if (i) std::cerr << ",";
    std::cerr << hierarchy_.primary_radii[i];
  }
  std::cerr << "\n";
  std::cerr << "    Auxiliary radii: ";
  for (size_t i = 0; i < hierarchy_.auxiliary_radii.size(); ++i) {
    if (i) std::cerr << ",";
    std::cerr << hierarchy_.auxiliary_radii[i];
  }
  std::cerr << "\n";
  {
    ScopedTimer timer(&stats_.phase1_sketch_ms);
    phase1_build_extended_sketch(unique_seqs);
  }

  std::cerr << "    Expanded layers:";
  for (size_t i = 0; i < extended_layers_.size(); ++i) {
    std::cerr << " L" << i << "=" << extended_layers_[i].size();
  }
  std::cerr << "\n";

  std::cerr << "  Phase 2: Inter-tier rebinding (DAG)...\n";
  {
    ScopedTimer timer(&stats_.phase2_rebinding_ms);
    phase2_inter_tier_rebinding();
  }

  std::cerr << "  Phase 3: Collapse auxiliary tiers + MBB...\n";
  {
    ScopedTimer timer(&stats_.phase3_mbb_ms);
    phase3_collapse_and_compute_mbb();
  }

  std::cerr << "  Phase 4: Leaf attachment...\n";
  {
    ScopedTimer timer(&stats_.phase4_attach_ms);
    attach_leaves(unique_seqs);
  }
  {
    ScopedTimer timer(&stats_.assign_ids_ms);
    assign_integer_ids();
  }
  {
    ScopedTimer timer(&stats_.graph_view_ms);
    build_search_graph_view();
  }

  std::cerr << "[Build generalized hierarchy] Completed.\n";
  stats_.total_build_ms = elapsed_ms_since(build_start);
  {
    ScopedTimer timer(&stats_.print_summary_ms);
    print_summary();
  }
  stats_.total_build_ms = elapsed_ms_since(build_start);
}

BioGeometryIndexBuilder::Statistics BioGeometryIndexBuilder::get_statistics() const {
  Statistics stats = stats_;
  stats.phase2_candidate_reduction_ratio =
      reduction_ratio(stats.phase2_total_possible_pairs, stats.phase2_candidate_pairs);
  stats.phase2_exact_distance_reduction_ratio =
      reduction_ratio(stats.phase2_total_possible_pairs, stats.phase2_exact_distance_calls);
  stats.leaf_candidate_reduction_ratio =
      reduction_ratio(stats.total_possible_leaf_pairs, stats.leaf_candidate_pairs);
  stats.leaf_exact_distance_reduction_ratio =
      reduction_ratio(stats.total_possible_leaf_pairs, stats.leaf_exact_distance_calls);
  const size_t phase2_auto_ratio_count =
      stats.phase2_auto_pigeonhole_accepted +
      stats.phase2_auto_pigeonhole_rejected_large_candidates;
  if (phase2_auto_ratio_count > 0) {
    stats.phase2_auto_candidate_ratio_avg =
        stats.phase2_auto_candidate_ratio_sum /
        static_cast<double>(phase2_auto_ratio_count);
  }
  const size_t leaf_auto_ratio_count =
      stats.leaf_auto_pigeonhole_accepted +
      stats.leaf_auto_pigeonhole_rejected_large_candidates;
  if (leaf_auto_ratio_count > 0) {
    stats.leaf_auto_candidate_ratio_avg =
        stats.leaf_auto_candidate_ratio_sum /
        static_cast<double>(leaf_auto_ratio_count);
  }
  const auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  if (stats.unique_sequences > 0 && !finest_layer.empty()) {
    stats.compression_ratio =
        1.0 - static_cast<double>(finest_layer.size()) / stats.unique_sequences;
  }

  if (num_primary_layers() >= 2) {
    size_t total_edges = 0;
    size_t total_nodes = 0;
    for (int layer_idx = 0; layer_idx + 1 < num_primary_layers(); ++layer_idx) {
      const auto& layer = primary_layers_[static_cast<size_t>(layer_idx)];
      total_nodes += layer.size();
      for (const auto& node : layer) total_edges += node->child_nodes.size();
    }
    if (total_nodes > 0) {
      stats.dag_redundancy =
          (static_cast<double>(total_edges) / static_cast<double>(total_nodes) - 1.0) * 100.0;
    }
  }
  return stats;
}

}  // namespace navigamer
