#include "qgram_filter.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

namespace navigamer {

size_t qgram_total(const std::string& sequence, int q) {
  if (q <= 0) throw std::invalid_argument("q-gram length must be positive");
  const size_t q_size = static_cast<size_t>(q);
  return sequence.size() < q_size ? 0 : sequence.size() - q_size + 1;
}

QGramCounts compute_qgram_counts(const std::string& sequence, int q) {
  QGramCounts counts;
  const size_t total = qgram_total(sequence, q);
  const size_t q_size = static_cast<size_t>(q);
  for (size_t pos = 0; pos < total; ++pos) {
    counts[sequence.substr(pos, q_size)]++;
  }
  return counts;
}

size_t compute_qgram_l1(
    const std::string& lhs, const std::string& rhs, int q) {
  const auto lhs_counts = compute_qgram_counts(lhs, q);
  const auto rhs_counts = compute_qgram_counts(rhs, q);
  std::unordered_set<std::string> grams;
  for (const auto& entry : lhs_counts) grams.insert(entry.first);
  for (const auto& entry : rhs_counts) grams.insert(entry.first);

  size_t l1 = 0;
  for (const auto& gram : grams) {
    const auto lhs_it = lhs_counts.find(gram);
    const auto rhs_it = rhs_counts.find(gram);
    const size_t lhs_count = lhs_it == lhs_counts.end() ? 0 : lhs_it->second;
    const size_t rhs_count = rhs_it == rhs_counts.end() ? 0 : rhs_it->second;
    l1 += static_cast<size_t>(
        std::llabs(static_cast<long long>(lhs_count) -
                   static_cast<long long>(rhs_count)));
  }
  return l1;
}

QGramCountIndex::QGramCountIndex(int q) : q_(q) {
  if (q_ <= 0) throw std::invalid_argument("q-gram length must be positive");
}

void QGramCountIndex::build(const std::vector<Item>& items) {
  items_.clear();
  postings_.clear();
  items_.reserve(items.size());

  for (const auto& item : items) {
    const size_t internal_idx = items_.size();
    items_.push_back({item.item_id, item.sequence.size(),
                      qgram_total(item.sequence, q_)});
    for (const auto& entry : compute_qgram_counts(item.sequence, q_)) {
      postings_[entry.first].push_back({internal_idx, entry.second});
    }
  }
}

std::vector<size_t> QGramCountIndex::query(
    const std::string& query_sequence, int tau, QueryStats* stats) const {
  if (tau < 0) throw std::invalid_argument("q-gram threshold must be non-negative");

  QueryStats local_stats;
  local_stats.total_items = items_.size();
  const size_t query_total = qgram_total(query_sequence, q_);
  const auto query_counts = compute_qgram_counts(query_sequence, q_);
  std::vector<size_t> shared(items_.size(), 0);

  for (const auto& query_entry : query_counts) {
    auto posting_it = postings_.find(query_entry.first);
    if (posting_it == postings_.end()) continue;
    for (const auto& posting : posting_it->second) {
      shared[posting.internal_idx] +=
          std::min(query_entry.second, posting.count);
    }
  }

  std::vector<size_t> candidates;
  candidates.reserve(items_.size());
  const long long query_length = static_cast<long long>(query_sequence.size());
  const long long max_l1 = 2LL * static_cast<long long>(q_) * tau;
  if (query_total == 0) local_stats.full_scan_fallbacks = 1;

  for (size_t internal_idx = 0; internal_idx < items_.size(); ++internal_idx) {
    const auto& item = items_[internal_idx];
    if (std::llabs(query_length -
                   static_cast<long long>(item.sequence_length)) > tau) {
      local_stats.length_filtered_items++;
      continue;
    }

    if (query_total == 0 || item.total_qgrams == 0) {
      candidates.push_back(item.item_id);
      continue;
    }

    const long long numerator =
        static_cast<long long>(query_total) +
        static_cast<long long>(item.total_qgrams) - max_l1;
    if (numerator <= 0) {
      local_stats.required_shared_nonpositive++;
      candidates.push_back(item.item_id);
      continue;
    }

    const size_t required_shared =
        static_cast<size_t>((numerator + 1) / 2);
    if (shared[internal_idx] >= required_shared) {
      candidates.push_back(item.item_id);
    } else {
      local_stats.pruned_by_l1++;
    }
  }

  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  local_stats.qgram_candidates = candidates.size();
  if (stats) *stats = local_stats;
  return candidates;
}

}  // namespace navigamer
