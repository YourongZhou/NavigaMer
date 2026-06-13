#include "range_join.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

namespace navigamer {

ExactRangeJoinIndex::ExactRangeJoinIndex(RangeJoinConfig config)
    : config_(config) {
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

  RangeJoinQueryResult result;
  const int block_count = tau + 1;
  result.block_len =
      static_cast<int>(query_sequence.size() / static_cast<size_t>(block_count));
  result.seed_len = std::min(config_.max_seed_len, result.block_len);

  if (result.seed_len < config_.min_seed_len) {
    result.used_full_scan = true;
    for (const auto& item : items_) {
      if (std::abs(static_cast<long long>(query_sequence.size()) -
                   static_cast<long long>(item.sequence.size())) <= tau) {
        result.candidate_item_ids.push_back(item.item_id);
      }
    }
    std::sort(result.candidate_item_ids.begin(), result.candidate_item_ids.end());
    return result;
  }

  const auto& postings = postings_for_seed_len(result.seed_len);
  std::unordered_set<size_t> candidates;
  for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    const size_t block_start =
        static_cast<size_t>(block_idx) * static_cast<size_t>(result.block_len);
    std::string seed =
        query_sequence.substr(block_start, static_cast<size_t>(result.seed_len));
    auto posting = postings.find(seed);
    if (posting == postings.end()) continue;
    for (size_t item_id : posting->second) candidates.insert(item_id);
  }

  result.candidate_item_ids.assign(candidates.begin(), candidates.end());
  std::sort(result.candidate_item_ids.begin(), result.candidate_item_ids.end());
  return result;
}

}  // namespace navigamer
