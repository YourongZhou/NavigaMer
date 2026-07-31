#include "qgram_filter.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace navigamer {

size_t qgram_total(std::string_view sequence, int q) {
  if (q <= 0) throw std::invalid_argument("q-gram length must be positive");
  const size_t q_size = static_cast<size_t>(q);
  return sequence.size() < q_size ? 0 : sequence.size() - q_size + 1;
}

QGramCounts compute_qgram_counts(std::string_view sequence, int q) {
  QGramCounts counts;
  const size_t total = qgram_total(sequence, q);
  const size_t q_size = static_cast<size_t>(q);
  for (size_t pos = 0; pos < total; ++pos) {
    counts[std::string(sequence.substr(pos, q_size))]++;
  }
  return counts;
}

size_t compute_qgram_l1(
    std::string_view lhs, std::string_view rhs, int q) {
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

QGramSignature compute_qgram_signature(std::string_view sequence, int q) {
  QGramSignature signature;
  signature.q = q;
  signature.sequence_length = sequence.size();
  if (q <= 0 || q > 32) return signature;

  for (char c : sequence) {
    switch (c) {
      case 'A':
      case 'C':
      case 'G':
      case 'T':
        break;
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
  if (q <= 5) {
    std::array<uint32_t, 1024> counts = {};
    uint64_t code = 0;
    for (size_t i = 0; i < sequence.size(); ++i) {
      uint64_t base = 0;
      switch (sequence[i]) {
        case 'C': base = 1; break;
        case 'G': base = 2; break;
        case 'T': base = 3; break;
        default: break;
      }
      code = ((code << 2) | base) & mask;
      if (i + 1 >= q_size) {
        auto& count = counts[static_cast<size_t>(code)];
        if (count == std::numeric_limits<uint32_t>::max()) {
          signature.safe_for_pruning = false;
          signature.entries.clear();
          return signature;
        }
        count++;
      }
    }
    const size_t code_count = size_t{1} << (2 * q);
    signature.entries.reserve(
        std::min(signature.total_qgrams, code_count));
    for (size_t current = 0; current < code_count; ++current) {
      if (counts[current] != 0) {
        signature.entries.push_back(
            {static_cast<uint64_t>(current), counts[current]});
      }
    }
    return signature;
  }

  std::vector<uint64_t> codes;
  codes.reserve(signature.total_qgrams);
  uint64_t code = 0;
  for (size_t i = 0; i < sequence.size(); ++i) {
    uint64_t base = 0;
    switch (sequence[i]) {
      case 'C': base = 1; break;
      case 'G': base = 2; break;
      case 'T': base = 3; break;
      default: break;
    }
    code = ((code << 2) | base) & mask;
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

void QGramQueryWorkspace::reset_seen(size_t item_count) {
  const bool compact =
      item_count <=
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;
  if (compact) {
    if (seen_epoch16.size() != item_count) {
      seen_epoch16.assign(item_count, 0);
    }
    if (!seen_epoch.empty()) std::vector<uint32_t>().swap(seen_epoch);
    if (epoch16 == std::numeric_limits<uint16_t>::max()) {
      std::fill(seen_epoch16.begin(), seen_epoch16.end(), 0);
      epoch16 = 1;
    } else {
      epoch16++;
    }
  } else {
    if (seen_epoch.size() != item_count) seen_epoch.assign(item_count, 0);
    if (!seen_epoch16.empty()) std::vector<uint16_t>().swap(seen_epoch16);
    if (epoch == std::numeric_limits<uint32_t>::max()) {
      std::fill(seen_epoch.begin(), seen_epoch.end(), 0);
      epoch = 1;
    } else {
      epoch++;
    }
  }
}

void QGramQueryWorkspace::reset(size_t item_count, bool compact_shared) {
  if (compact_shared) {
    if (shared16.size() != item_count) shared16.assign(item_count, 0);
    if (!shared.empty()) std::vector<size_t>().swap(shared);
  } else {
    if (shared.size() != item_count) shared.assign(item_count, 0);
    if (!shared16.empty()) std::vector<uint16_t>().swap(shared16);
  }
  reset_seen(item_count);
}

QGramCountIndex::QGramCountIndex(int q) : q_(q) {
  if (q_ <= 0) throw std::invalid_argument("q-gram length must be positive");
}

void QGramCountIndex::build(const std::vector<Item>& items) {
  std::vector<ItemView> views;
  views.reserve(items.size());
  for (const auto& item : items) {
    views.push_back({item.item_id, item.sequence});
  }
  build_views(views);
}

void QGramCountIndex::build_views(const std::vector<ItemView>& items) {
  items_.clear();
  postings_.clear();
  dense_postings_.clear();
  dense_byte_packed_postings_.clear();
  dense_packed_postings_.clear();
  if (q_ <= 5) {
    const bool compact_item_ids = items.size() <=
        static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;
    const bool byte_packable_item_ids =
        items.size() <= (size_t{1} << 24);
    bool byte_counts = true;
    bool compact_counts = true;
    const size_t q_size = static_cast<size_t>(q_);
    for (const auto& item : items) {
      const size_t total = item.sequence.size() < q_size
                               ? 0
                               : item.sequence.size() - q_size + 1;
      byte_counts =
          byte_counts && total <= std::numeric_limits<uint8_t>::max();
      compact_counts =
          compact_counts &&
          total <= std::numeric_limits<uint16_t>::max();
    }
    const size_t code_count = size_t{1} << (2 * q_);
    if (byte_counts && !compact_item_ids && byte_packable_item_ids) {
      dense_byte_packed_postings_.resize(code_count);
    } else if (compact_counts && compact_item_ids) {
      dense_packed_postings_.resize(code_count);
    } else {
      dense_postings_.resize(code_count);
    }
  }
  item_ids_strictly_increasing_ = true;
  items_.reserve(items.size());
  const bool posting_index_capacity =
      items.size() <= static_cast<size_t>(
                          std::numeric_limits<uint32_t>::max());

  for (const auto& item : items) {
    const size_t internal_idx = items_.size();
    if (internal_idx > 0 &&
        item.item_id <= items_[internal_idx - 1].item_id) {
      item_ids_strictly_increasing_ = false;
    }
    const std::string_view sequence = item.sequence;
    QGramSignature signature = compute_qgram_signature(sequence, q_);
    const bool qgram_indexable =
        posting_index_capacity && signature.safe_for_pruning;
    items_.push_back(
        {item.item_id, sequence.size(), signature.total_qgrams,
         qgram_indexable});
    if (!qgram_indexable) continue;
    for (const auto& entry : signature.entries) {
      if (!dense_byte_packed_postings_.empty()) {
        dense_byte_packed_postings_[static_cast<size_t>(entry.code)]
            .push_back(
                (static_cast<uint32_t>(internal_idx) << 8) |
                static_cast<uint8_t>(entry.count));
      } else if (!dense_packed_postings_.empty()) {
        dense_packed_postings_[static_cast<size_t>(entry.code)].push_back(
            (static_cast<uint32_t>(internal_idx) << 16) | entry.count);
      } else if (!dense_postings_.empty()) {
        dense_postings_[static_cast<size_t>(entry.code)].push_back(
            {static_cast<uint32_t>(internal_idx), entry.count});
      } else {
        postings_[entry.code].push_back(
            {static_cast<uint32_t>(internal_idx), entry.count});
      }
    }
  }
}

std::vector<size_t> QGramCountIndex::query(
    std::string_view query_sequence, int tau, QueryStats* stats,
    QGramQueryWorkspace* workspace) const {
  if (tau < 0) throw std::invalid_argument("q-gram threshold must be non-negative");

  QueryStats local_stats;
  local_stats.total_items = items_.size();
  const QGramSignature query_signature =
      compute_qgram_signature(query_sequence, q_);
  const size_t query_total = query_signature.total_qgrams;
  QGramQueryWorkspace local_workspace;
  QGramQueryWorkspace* ws = workspace ? workspace : &local_workspace;
  // Compact postings are selected only when every indexed item has at most
  // UINT16_MAX q-grams. Shared counts cannot exceed the item's total.
  const bool compact_shared =
      !dense_byte_packed_postings_.empty() ||
      !dense_packed_postings_.empty();
  ws->reset(items_.size(), compact_shared);
  const bool compact_seen =
      items_.size() <=
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;
  uint16_t* seen16 =
      compact_seen ? ws->seen_epoch16.data() : nullptr;
  uint32_t* seen32 =
      compact_seen ? nullptr : ws->seen_epoch.data();
  const uint32_t current_epoch =
      compact_seen ? ws->epoch16 : ws->epoch;

  if (query_signature.safe_for_pruning) {
    for (const auto& query_entry : query_signature.entries) {
      const auto consume_posting = [&](uint32_t internal_idx,
                                       uint32_t count) {
        const bool newly_seen =
            compact_seen
                ? seen16[internal_idx] != current_epoch
                : seen32[internal_idx] != current_epoch;
        if (newly_seen) {
          if (compact_seen) {
            seen16[internal_idx] =
                static_cast<uint16_t>(current_epoch);
          } else {
            seen32[internal_idx] = current_epoch;
          }
          if (compact_shared) {
            ws->shared16[internal_idx] = 0;
          } else {
            ws->shared[internal_idx] = 0;
          }
        }
        const uint32_t contribution =
            std::min(query_entry.count, count);
        if (compact_shared) {
          ws->shared16[internal_idx] = static_cast<uint16_t>(
              ws->shared16[internal_idx] + contribution);
        } else {
          ws->shared[internal_idx] += contribution;
        }
      };
      if (!dense_packed_postings_.empty()) {
        if (query_entry.code >= dense_packed_postings_.size()) continue;
        for (uint32_t packed :
             dense_packed_postings_[static_cast<size_t>(query_entry.code)]) {
          consume_posting(
              packed >> 16,
              packed & std::numeric_limits<uint16_t>::max());
        }
      } else if (!dense_byte_packed_postings_.empty()) {
        if (query_entry.code >= dense_byte_packed_postings_.size()) continue;
        for (uint32_t packed :
             dense_byte_packed_postings_[
                 static_cast<size_t>(query_entry.code)]) {
          consume_posting(packed >> 8, packed & 0xffU);
        }
      } else if (!dense_postings_.empty()) {
        if (query_entry.code >= dense_postings_.size()) continue;
        for (const auto& posting :
             dense_postings_[static_cast<size_t>(query_entry.code)]) {
          consume_posting(posting.internal_idx, posting.count);
        }
      } else {
        const auto posting_it = postings_.find(query_entry.code);
        if (posting_it == postings_.end()) continue;
        for (const auto& posting : posting_it->second) {
          consume_posting(posting.internal_idx, posting.count);
        }
      }
    }
  }

  std::vector<size_t> candidates;
  candidates.reserve(items_.size());
  const long long query_length = static_cast<long long>(query_sequence.size());
  const long long max_l1 = 2LL * static_cast<long long>(q_) * tau;
  if (!query_signature.safe_for_pruning || query_total == 0) {
    local_stats.full_scan_fallbacks = 1;
  }

  for (size_t internal_idx = 0; internal_idx < items_.size(); ++internal_idx) {
    const auto& item = items_[internal_idx];
    if (std::llabs(query_length -
                   static_cast<long long>(item.sequence_length)) > tau) {
      local_stats.length_filtered_items++;
      continue;
    }

    if (!query_signature.safe_for_pruning || !item.qgram_indexable ||
        query_total == 0 || item.total_qgrams == 0) {
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
    const size_t shared_count =
        (compact_seen
             ? seen16[internal_idx] == current_epoch
             : seen32[internal_idx] == current_epoch)
            ? (compact_shared
                   ? static_cast<size_t>(ws->shared16[internal_idx])
                   : ws->shared[internal_idx])
            : 0;
    if (shared_count >= required_shared) {
      candidates.push_back(item.item_id);
    } else {
      local_stats.pruned_by_l1++;
    }
  }

  if (!item_ids_strictly_increasing_) {
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
  }
  local_stats.qgram_candidates = candidates.size();
  if (stats) *stats = local_stats;
  return candidates;
}

}  // namespace navigamer
