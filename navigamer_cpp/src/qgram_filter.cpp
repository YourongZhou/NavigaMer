#include "qgram_filter.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
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

QGramSignature compute_qgram_signature(const std::string& sequence, int q) {
  QGramSignature signature;
  signature.q = q;
  signature.sequence_length = sequence.size();
  if (q <= 0 || q > 32) return signature;

  std::vector<uint8_t> bases;
  bases.reserve(sequence.size());
  for (char c : sequence) {
    switch (c) {
      case 'A': bases.push_back(0); break;
      case 'C': bases.push_back(1); break;
      case 'G': bases.push_back(2); break;
      case 'T': bases.push_back(3); break;
      default: return signature;
    }
  }

  const size_t q_size = static_cast<size_t>(q);
  signature.total_qgrams =
      sequence.size() < q_size ? 0 : sequence.size() - q_size + 1;
  signature.safe_for_pruning = true;
  if (signature.total_qgrams == 0) return signature;

  const uint64_t mask =
      q == 32 ? std::numeric_limits<uint64_t>::max()
              : (uint64_t{1} << (2 * q)) - 1;
  std::vector<uint64_t> codes;
  codes.reserve(signature.total_qgrams);
  uint64_t code = 0;
  for (size_t i = 0; i < bases.size(); ++i) {
    code = ((code << 2) | bases[i]) & mask;
    if (i + 1 >= q_size) codes.push_back(code);
  }
  std::sort(codes.begin(), codes.end());

  for (uint64_t current : codes) {
    if (signature.entries.empty() ||
        signature.entries.back().code != current) {
      signature.entries.push_back({current, 1});
    } else if (signature.entries.back().count ==
               std::numeric_limits<uint32_t>::max()) {
      signature.safe_for_pruning = false;
      signature.entries.clear();
      return signature;
    } else {
      signature.entries.back().count++;
    }
  }
  return signature;
}

size_t qgram_l1_distance(
    const QGramSignature& lhs, const QGramSignature& rhs) {
  if (!lhs.safe_for_pruning || !rhs.safe_for_pruning || lhs.q != rhs.q) {
    return std::numeric_limits<size_t>::max();
  }

  size_t l1 = 0;
  size_t lhs_idx = 0;
  size_t rhs_idx = 0;
  while (lhs_idx < lhs.entries.size() || rhs_idx < rhs.entries.size()) {
    size_t delta = 0;
    if (rhs_idx == rhs.entries.size() ||
        (lhs_idx < lhs.entries.size() &&
         lhs.entries[lhs_idx].code < rhs.entries[rhs_idx].code)) {
      delta = lhs.entries[lhs_idx++].count;
    } else if (lhs_idx == lhs.entries.size() ||
               rhs.entries[rhs_idx].code < lhs.entries[lhs_idx].code) {
      delta = rhs.entries[rhs_idx++].count;
    } else {
      const auto lhs_count = lhs.entries[lhs_idx++].count;
      const auto rhs_count = rhs.entries[rhs_idx++].count;
      delta = lhs_count > rhs_count ? lhs_count - rhs_count
                                    : rhs_count - lhs_count;
    }
    if (l1 > std::numeric_limits<size_t>::max() - delta) {
      return std::numeric_limits<size_t>::max();
    }
    l1 += delta;
  }
  return l1;
}

bool qgram_can_prune_edit_distance(
    const QGramSignature& lhs, const QGramSignature& rhs, int tau) {
  if (!lhs.safe_for_pruning || !rhs.safe_for_pruning ||
      lhs.q <= 0 || lhs.q != rhs.q || tau < 0) {
    return false;
  }
  const size_t q = static_cast<size_t>(lhs.q);
  const size_t threshold_tau = static_cast<size_t>(tau);
  if (threshold_tau > std::numeric_limits<size_t>::max() / q / 2) {
    return false;
  }
  return qgram_l1_distance(lhs, rhs) > 2 * q * threshold_tau;
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
