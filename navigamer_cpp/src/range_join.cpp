#include "range_join.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

namespace navigamer {

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

ExactRangeJoinIndex::ExactRangeJoinIndex(RangeJoinConfig config)
    : config_(config), qgram_index_(config.qgram_q) {
  if (config_.min_seed_len <= 0) {
    throw std::invalid_argument("range-join min seed length must be positive");
  }
  if (config_.max_seed_len < config_.min_seed_len) {
    throw std::invalid_argument(
        "range-join max seed length must be at least min seed length");
  }
}

void ExactRangeJoinIndex::build(const std::vector<RangeJoinItem>& items) {
  items_ = items;
  postings_by_seed_len_.clear();
  std::vector<QGramCountIndex::Item> qgram_items;
  qgram_items.reserve(items.size());
  for (const auto& item : items) {
    qgram_items.push_back({item.item_id, item.sequence});
  }
  qgram_index_.build(qgram_items);
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

RangeJoinQueryResult ExactRangeJoinIndex::query(
    const std::string& query_sequence, int tau) {
  if (tau < 0) throw std::invalid_argument("range-join threshold must be non-negative");

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

  if (config_.candidate_mode == RangeCandidateMode::QGramOnly ||
      (config_.candidate_mode == RangeCandidateMode::Auto &&
       seed_len < config_.min_seed_len)) {
    auto result = qgram_query(query_sequence, tau);
    result.block_len = block_len;
    result.seed_len = seed_len;
    return result;
  }

  if (config_.candidate_mode == RangeCandidateMode::PigeonholeOnly ||
      config_.candidate_mode == RangeCandidateMode::Auto) {
    return pigeonhole_query(query_sequence, tau, block_len, seed_len);
  }

  auto pigeonhole = pigeonhole_query(query_sequence, tau, block_len, seed_len);
  auto qgram = qgram_query(query_sequence, tau);
  RangeJoinQueryResult result;
  result.mode_used = RangeCandidateMode::Hybrid;
  result.used_full_scan = pigeonhole.used_full_scan || qgram.used_full_scan;
  result.block_len = block_len;
  result.seed_len = seed_len;
  result.length_filtered_items =
      std::max(pigeonhole.length_filtered_items, qgram.length_filtered_items);
  result.qgram_candidate_count = qgram.candidate_item_ids.size();
  result.qgram_pruned_by_l1 = qgram.qgram_pruned_by_l1;
  result.required_shared_nonpositive = qgram.required_shared_nonpositive;
  std::set_intersection(
      pigeonhole.candidate_item_ids.begin(), pigeonhole.candidate_item_ids.end(),
      qgram.candidate_item_ids.begin(), qgram.candidate_item_ids.end(),
      std::back_inserter(result.candidate_item_ids));
  return result;
}

RangeJoinQueryResult ExactRangeJoinIndex::full_scan(
    const std::string& query_sequence, int tau, bool fallback) const {
  RangeJoinQueryResult result;
  result.mode_used = RangeCandidateMode::FullScan;
  result.used_full_scan = fallback;
  for (const auto& item : items_) {
    if (std::abs(static_cast<long long>(query_sequence.size()) -
                 static_cast<long long>(item.sequence.size())) <= tau) {
      result.candidate_item_ids.push_back(item.item_id);
    } else {
      result.length_filtered_items++;
    }
  }
  std::sort(result.candidate_item_ids.begin(), result.candidate_item_ids.end());
  result.candidate_item_ids.erase(
      std::unique(result.candidate_item_ids.begin(),
                  result.candidate_item_ids.end()),
      result.candidate_item_ids.end());
  return result;
}

RangeJoinQueryResult ExactRangeJoinIndex::pigeonhole_query(
    const std::string& query_sequence, int tau, int block_len, int seed_len) {
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
  const auto compatible = full_scan(query_sequence, tau, false);
  result.length_filtered_items = compatible.length_filtered_items;
  std::unordered_set<size_t> compatible_ids(
      compatible.candidate_item_ids.begin(), compatible.candidate_item_ids.end());
  const auto& postings = postings_for_seed_len(result.seed_len);
  std::unordered_set<size_t> candidates;
  const int block_count = tau + 1;
  for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    const size_t block_start =
        static_cast<size_t>(block_idx) * static_cast<size_t>(result.block_len);
    std::string seed =
        query_sequence.substr(block_start, static_cast<size_t>(result.seed_len));
    auto posting = postings.find(seed);
    if (posting == postings.end()) continue;
    for (size_t item_id : posting->second) {
      if (compatible_ids.count(item_id) != 0) candidates.insert(item_id);
    }
  }

  result.candidate_item_ids.assign(candidates.begin(), candidates.end());
  std::sort(result.candidate_item_ids.begin(), result.candidate_item_ids.end());
  return result;
}

RangeJoinQueryResult ExactRangeJoinIndex::qgram_query(
    const std::string& query_sequence, int tau) const {
  QGramCountIndex::QueryStats stats;
  RangeJoinQueryResult result;
  result.mode_used = RangeCandidateMode::QGramOnly;
  result.candidate_item_ids = qgram_index_.query(query_sequence, tau, &stats);
  result.used_full_scan = stats.full_scan_fallbacks > 0;
  result.length_filtered_items = stats.length_filtered_items;
  result.qgram_candidate_count = stats.qgram_candidates;
  result.qgram_pruned_by_l1 = stats.pruned_by_l1;
  result.required_shared_nonpositive = stats.required_shared_nonpositive;
  return result;
}

}  // namespace navigamer
