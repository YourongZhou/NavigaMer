#include "range_join.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unordered_set>

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

void merge_range_timing(RangeJoinQueryResult& target,
                        const RangeJoinQueryResult& source) {
  target.range_posting_lookup_ms += source.range_posting_lookup_ms;
  target.range_seed_union_ms += source.range_seed_union_ms;
  target.range_length_filter_ms += source.range_length_filter_ms;
  target.range_qgram_query_ms += source.range_qgram_query_ms;
  target.range_hybrid_intersection_ms += source.range_hybrid_intersection_ms;
  target.range_full_scan_ms += source.range_full_scan_ms;
}

void copy_seed_counters(RangeJoinQueryResult& target,
                        const RangeJoinQueryResult& source) {
  target.seed_candidate_pairs_before_length_filter =
      source.seed_candidate_pairs_before_length_filter;
  target.seed_length_pruned_candidates = source.seed_length_pruned_candidates;
  target.pigeonhole_early_abort_count = source.pigeonhole_early_abort_count;
  target.pigeonhole_candidate_count = source.pigeonhole_candidate_count;
}

}  // namespace

const char* range_candidate_mode_name(RangeCandidateMode mode) {
  switch (mode) {
    case RangeCandidateMode::Auto: return "auto";
    case RangeCandidateMode::PigeonholeOnly: return "pigeonhole";
    case RangeCandidateMode::QGramOnly: return "qgram";
    case RangeCandidateMode::Hybrid: return "hybrid";
    case RangeCandidateMode::FullScan: return "full";
  }
  return "unknown";
}

RangeCandidateMode parse_range_candidate_mode(const std::string& value) {
  if (value == "auto") return RangeCandidateMode::Auto;
  if (value == "pigeonhole") return RangeCandidateMode::PigeonholeOnly;
  if (value == "qgram") return RangeCandidateMode::QGramOnly;
  if (value == "hybrid") return RangeCandidateMode::Hybrid;
  if (value == "full") return RangeCandidateMode::FullScan;
  throw std::invalid_argument(
      "range candidate mode must be auto, pigeonhole, qgram, hybrid, or full");
}

ExactRangeJoinIndex::ExactRangeJoinIndex(
    RangeJoinConfig config, bool defer_qgram_build)
    : config_(config),
      qgram_index_(config.qgram_q),
      defer_qgram_build_(defer_qgram_build),
      deferred_qgram_mutex_(defer_qgram_build
                                ? std::make_shared<std::mutex>()
                                : nullptr) {
  if (config_.min_seed_len <= 0) {
    throw std::invalid_argument("range-join min seed length must be positive");
  }
  if (config_.max_seed_len < config_.min_seed_len) {
    throw std::invalid_argument(
        "range-join max seed length must be at least min seed length");
  }
  if (!std::isfinite(config_.auto_pigeonhole_max_ratio) ||
      config_.auto_pigeonhole_max_ratio < 0.0 ||
      config_.auto_pigeonhole_max_ratio > 1.0) {
    throw std::invalid_argument(
        "auto pigeonhole max ratio must be finite and in [0, 1]");
  }
}

void ExactRangeJoinIndex::build(const std::vector<RangeJoinItem>& items) {
  items_ = items;
  item_lengths_by_id_.clear();
  item_lengths_by_id_.reserve(items.size());
  postings_by_seed_len_.clear();
  qgram_ready_ = false;
  std::vector<QGramCountIndex::Item> qgram_items;
  if (!defer_qgram_build_) qgram_items.reserve(items.size());
  for (const auto& item : items) {
    item_lengths_by_id_[item.item_id] = item.sequence.size();
    if (!defer_qgram_build_) {
      qgram_items.push_back({item.item_id, item.sequence});
    }
  }
  if (!defer_qgram_build_) {
    qgram_index_.build(qgram_items);
    qgram_ready_ = true;
  }
}

const QGramCountIndex& ExactRangeJoinIndex::ensure_qgram_index() const {
  const auto build_qgram = [this]() {
    std::vector<QGramCountIndex::Item> qgram_items;
    qgram_items.reserve(items_.size());
    for (const auto& item : items_) {
      qgram_items.push_back({item.item_id, item.sequence});
    }
    qgram_index_.build(qgram_items);
    qgram_ready_ = true;
  };

  if (defer_qgram_build_) {
    std::lock_guard<std::mutex> lock(*deferred_qgram_mutex_);
    if (!qgram_ready_) build_qgram();
  } else if (!qgram_ready_) {
    build_qgram();
  }
  return qgram_index_;
}

void ExactRangeJoinIndex::prepare_qgram() {
  (void)ensure_qgram_index();
}

const ExactRangeJoinIndex::PostingLists&
ExactRangeJoinIndex::postings_for_seed_len(int seed_len) {
  auto existing = postings_by_seed_len_.find(seed_len);
  if (existing != postings_by_seed_len_.end()) return existing->second;

  PostingLists postings;
  for (const auto& item : items_) {
    if (item.sequence.size() < static_cast<size_t>(seed_len)) continue;
    std::unordered_set<std::string> seen;
    const size_t last = item.sequence.size() - static_cast<size_t>(seed_len);
    for (size_t pos = 0; pos <= last; ++pos) {
      std::string seed = item.sequence.substr(pos, static_cast<size_t>(seed_len));
      if (seen.insert(seed).second) postings[seed].push_back(item.item_id);
    }
  }
  return postings_by_seed_len_.emplace(seed_len, std::move(postings)).first->second;
}

const ExactRangeJoinIndex::PostingLists*
ExactRangeJoinIndex::find_postings_for_seed_len(int seed_len) const {
  auto existing = postings_by_seed_len_.find(seed_len);
  if (existing == postings_by_seed_len_.end()) return nullptr;
  return &existing->second;
}

void ExactRangeJoinIndex::prepare_seed_lengths(
    const std::vector<int>& seed_lengths) {
  for (int seed_len : seed_lengths) {
    if (seed_len >= config_.min_seed_len) {
      (void)postings_for_seed_len(seed_len);
    }
  }
}

bool ExactRangeJoinIndex::query_needs_seed_postings(int seed_len) const {
  if (seed_len < config_.min_seed_len) return false;
  switch (config_.candidate_mode) {
    case RangeCandidateMode::Auto:
    case RangeCandidateMode::PigeonholeOnly:
    case RangeCandidateMode::Hybrid:
      return true;
    case RangeCandidateMode::QGramOnly:
    case RangeCandidateMode::FullScan:
      return false;
  }
  return false;
}

RangeJoinQueryResult ExactRangeJoinIndex::query(
    const std::string& query_sequence, int tau) {
  if (tau < 0) throw std::invalid_argument("range-join threshold must be non-negative");

  const int block_count = tau + 1;
  const int block_len =
      static_cast<int>(query_sequence.size() / static_cast<size_t>(block_count));
  const int seed_len = std::min(config_.max_seed_len, block_len);
  if (query_needs_seed_postings(seed_len)) {
    prepare_seed_lengths({seed_len});
  }

  RangeJoinQueryWorkspace workspace;
  return static_cast<const ExactRangeJoinIndex&>(*this).query(
      query_sequence, tau, &workspace);
}

RangeJoinQueryResult ExactRangeJoinIndex::query(
    const std::string& query_sequence, int tau,
    RangeJoinQueryWorkspace* workspace) const {
  if (tau < 0) throw std::invalid_argument("range-join threshold must be non-negative");
  RangeJoinQueryWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;

  const int block_count = tau + 1;
  const int block_len =
      static_cast<int>(query_sequence.size() / static_cast<size_t>(block_count));
  const int seed_len = std::min(config_.max_seed_len, block_len);

  if (config_.candidate_mode == RangeCandidateMode::FullScan) {
    auto result = full_scan(query_sequence, tau, false);
    result.block_len = block_len;
    result.seed_len = seed_len;
    return result;
  }

  if (config_.candidate_mode == RangeCandidateMode::QGramOnly) {
    auto result = qgram_query(query_sequence, tau, workspace);
    result.block_len = block_len;
    result.seed_len = seed_len;
    return result;
  }

  if (config_.candidate_mode == RangeCandidateMode::PigeonholeOnly) {
    return pigeonhole_query(query_sequence, tau, block_len, seed_len,
                            std::numeric_limits<size_t>::max());
  }

  if (config_.candidate_mode == RangeCandidateMode::Hybrid) {
    auto pigeonhole =
        pigeonhole_query(query_sequence, tau, block_len, seed_len,
                         std::numeric_limits<size_t>::max());
    auto qgram = qgram_query(query_sequence, tau, workspace);
    return hybrid_result(pigeonhole, qgram);
  }

  if (seed_len < config_.min_seed_len) {
    auto qgram = qgram_query(query_sequence, tau, workspace);
    qgram.block_len = block_len;
    qgram.seed_len = seed_len;
    qgram.auto_qgram_invoked = 1;
    qgram.auto_final_candidate_pairs = qgram.candidate_item_ids.size();
    return qgram;
  }

  auto pigeonhole =
      pigeonhole_query(query_sequence, tau, block_len, seed_len,
                       config_.auto_pigeonhole_max_candidates);
  if (pigeonhole.pigeonhole_early_abort_count > 0) {
    auto qgram = qgram_query(query_sequence, tau, workspace);
    merge_range_timing(qgram, pigeonhole);
    qgram.block_len = block_len;
    qgram.seed_len = seed_len;
    copy_seed_counters(qgram, pigeonhole);
    qgram.auto_pigeonhole_rejected_large_candidates = 1;
    qgram.auto_qgram_invoked = 1;
    qgram.auto_final_candidate_pairs = qgram.candidate_item_ids.size();
    qgram.final_candidate_pairs = qgram.candidate_item_ids.size();
    return qgram;
  }
  pigeonhole.pigeonhole_candidate_count =
      pigeonhole.candidate_item_ids.size();
  if (pigeonhole.pigeonhole_candidate_count <=
          config_.auto_pigeonhole_max_candidates) {
    pigeonhole.auto_pigeonhole_accepted = 1;
    pigeonhole.auto_final_candidate_pairs =
        pigeonhole.candidate_item_ids.size();
    return pigeonhole;
  }

  auto qgram = qgram_query(query_sequence, tau, workspace);
  if (!config_.auto_hybrid_on_large_candidates) {
    merge_range_timing(qgram, pigeonhole);
    qgram.block_len = block_len;
    qgram.seed_len = seed_len;
    copy_seed_counters(qgram, pigeonhole);
    qgram.auto_pigeonhole_rejected_large_candidates = 1;
    qgram.auto_qgram_invoked = 1;
    qgram.auto_final_candidate_pairs = qgram.candidate_item_ids.size();
    qgram.final_candidate_pairs = qgram.candidate_item_ids.size();
    return qgram;
  }

  auto result = hybrid_result(pigeonhole, qgram);
  copy_seed_counters(result, pigeonhole);
  result.auto_pigeonhole_rejected_large_candidates = 1;
  result.auto_qgram_invoked = 1;
  result.auto_hybrid_invoked = 1;
  result.auto_final_candidate_pairs = result.candidate_item_ids.size();
  result.final_candidate_pairs = result.candidate_item_ids.size();
  return result;
}

RangeJoinQueryResult ExactRangeJoinIndex::full_scan(
    const std::string& query_sequence, int tau, bool fallback) const {
  RangeJoinQueryResult result;
  ScopedTimer full_timer(&result.range_full_scan_ms);
  result.mode_used = RangeCandidateMode::FullScan;
  result.used_full_scan = fallback;
  {
    ScopedTimer length_timer(&result.range_length_filter_ms);
    for (const auto& item : items_) {
      if (std::abs(static_cast<long long>(query_sequence.size()) -
                   static_cast<long long>(item.sequence.size())) <= tau) {
        result.candidate_item_ids.push_back(item.item_id);
      } else {
        result.length_filtered_items++;
      }
    }
  }
  std::sort(result.candidate_item_ids.begin(), result.candidate_item_ids.end());
  result.candidate_item_ids.erase(
      std::unique(result.candidate_item_ids.begin(),
                  result.candidate_item_ids.end()),
      result.candidate_item_ids.end());
  result.compatible_item_count = result.candidate_item_ids.size();
  result.final_candidate_pairs = result.candidate_item_ids.size();
  return result;
}

RangeJoinQueryResult ExactRangeJoinIndex::pigeonhole_query(
    const std::string& query_sequence, int tau, int block_len, int seed_len,
    size_t early_abort_candidate_limit) const {
  if (seed_len < config_.min_seed_len) {
    auto result = full_scan(query_sequence, tau, true);
    result.block_len = block_len;
    result.seed_len = seed_len;
    return result;
  }

  RangeJoinQueryResult result;
  result.mode_used = RangeCandidateMode::PigeonholeOnly;
  result.block_len = block_len;
  result.seed_len = seed_len;
  const PostingLists* postings_ptr = nullptr;
  {
    ScopedTimer timer(&result.range_posting_lookup_ms);
    postings_ptr = find_postings_for_seed_len(result.seed_len);
  }
  if (!postings_ptr) {
    auto fallback = full_scan(query_sequence, tau, true);
    merge_range_timing(fallback, result);
    fallback.block_len = block_len;
    fallback.seed_len = seed_len;
    return fallback;
  }
  const auto& postings = *postings_ptr;
  std::unordered_set<size_t> candidates;
  const int block_count = tau + 1;
  for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    const size_t block_start =
        static_cast<size_t>(block_idx) * static_cast<size_t>(result.block_len);
    std::string seed =
        query_sequence.substr(block_start, static_cast<size_t>(result.seed_len));
    PostingLists::const_iterator posting;
    {
      ScopedTimer timer(&result.range_posting_lookup_ms);
      posting = postings.find(seed);
    }
    if (posting == postings.end()) continue;
    {
      ScopedTimer timer(&result.range_seed_union_ms);
      for (size_t item_id : posting->second) {
        candidates.insert(item_id);
        if (candidates.size() > early_abort_candidate_limit) {
          result.seed_candidate_pairs_before_length_filter = candidates.size();
          result.pigeonhole_candidate_count = candidates.size();
          result.pigeonhole_early_abort_count = 1;
          result.final_candidate_pairs = 0;
          return result;
        }
      }
    }
  }

  result.seed_candidate_pairs_before_length_filter = candidates.size();
  {
    ScopedTimer timer(&result.range_length_filter_ms);
    for (size_t item_id : candidates) {
      auto length_it = item_lengths_by_id_.find(item_id);
      if (length_it == item_lengths_by_id_.end()) {
        auto fallback = full_scan(query_sequence, tau, true);
        merge_range_timing(fallback, result);
        fallback.block_len = block_len;
        fallback.seed_len = seed_len;
        return fallback;
      }
      if (std::abs(static_cast<long long>(query_sequence.size()) -
                   static_cast<long long>(length_it->second)) <= tau) {
        result.candidate_item_ids.push_back(item_id);
      } else {
        result.length_filtered_items++;
        result.seed_length_pruned_candidates++;
      }
    }
  }

  std::sort(result.candidate_item_ids.begin(), result.candidate_item_ids.end());
  result.pigeonhole_candidate_count = result.candidate_item_ids.size();
  result.final_candidate_pairs = result.candidate_item_ids.size();
  return result;
}

RangeJoinQueryResult ExactRangeJoinIndex::hybrid_result(
    const RangeJoinQueryResult& pigeonhole,
    const RangeJoinQueryResult& qgram) const {
  RangeJoinQueryResult result;
  merge_range_timing(result, pigeonhole);
  merge_range_timing(result, qgram);
  result.mode_used = RangeCandidateMode::Hybrid;
  result.used_full_scan = pigeonhole.used_full_scan || qgram.used_full_scan;
  result.block_len = pigeonhole.block_len;
  result.seed_len = pigeonhole.seed_len;
  result.length_filtered_items =
      std::max(pigeonhole.length_filtered_items, qgram.length_filtered_items);
  result.compatible_item_count =
      std::max(pigeonhole.compatible_item_count, qgram.compatible_item_count);
  copy_seed_counters(result, pigeonhole);
  result.qgram_candidate_count = qgram.candidate_item_ids.size();
  result.qgram_pruned_by_l1 = qgram.qgram_pruned_by_l1;
  result.required_shared_nonpositive = qgram.required_shared_nonpositive;
  {
    ScopedTimer timer(&result.range_hybrid_intersection_ms);
    std::set_intersection(
        pigeonhole.candidate_item_ids.begin(), pigeonhole.candidate_item_ids.end(),
        qgram.candidate_item_ids.begin(), qgram.candidate_item_ids.end(),
        std::back_inserter(result.candidate_item_ids));
  }
  result.final_candidate_pairs = result.candidate_item_ids.size();
  return result;
}

RangeJoinQueryResult ExactRangeJoinIndex::qgram_query(
    const std::string& query_sequence, int tau,
    RangeJoinQueryWorkspace* workspace) const {
  QGramCountIndex::QueryStats stats;
  RangeJoinQueryResult result;
  result.mode_used = RangeCandidateMode::QGramOnly;
  {
    ScopedTimer timer(&result.range_qgram_query_ms);
    const QGramCountIndex& qgram_index =
        defer_qgram_build_ ? ensure_qgram_index() : qgram_index_;
    result.candidate_item_ids =
        qgram_index.query(query_sequence, tau, &stats,
                          workspace ? &workspace->qgram : nullptr);
  }
  result.used_full_scan = stats.full_scan_fallbacks > 0;
  result.length_filtered_items = stats.length_filtered_items;
  result.qgram_candidate_count = stats.qgram_candidates;
  result.qgram_pruned_by_l1 = stats.pruned_by_l1;
  result.required_shared_nonpositive = stats.required_shared_nonpositive;
  result.compatible_item_count = stats.total_items - stats.length_filtered_items;
  result.final_candidate_pairs = result.candidate_item_ids.size();
  return result;
}

}  // namespace navigamer
