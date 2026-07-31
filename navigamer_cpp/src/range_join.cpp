#include "range_join.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace navigamer {

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kMinShiftRunForRollingPostings = 8;

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

bool encode_dna_seed(std::string_view sequence, size_t start, int seed_len,
                     uint64_t* code) {
  if (!code || seed_len <= 0 || seed_len > 32 ||
      start + static_cast<size_t>(seed_len) > sequence.size()) {
    return false;
  }
  uint64_t value = 0;
  for (int offset = 0; offset < seed_len; ++offset) {
    value <<= 2;
    switch (sequence[start + static_cast<size_t>(offset)]) {
      case 'A':
        break;
      case 'C':
        value |= 1;
        break;
      case 'G':
        value |= 2;
        break;
      case 'T':
        value |= 3;
        break;
      default:
        return false;
    }
  }
  *code = value;
  return true;
}

bool is_acgt_sequence(std::string_view sequence) {
  for (char base : sequence) {
    if (base != 'A' && base != 'C' && base != 'G' && base != 'T') {
      return false;
    }
  }
  return true;
}

uint64_t dna_base_bits(char base) {
  switch (base) {
    case 'C': return 1;
    case 'G': return 2;
    case 'T': return 3;
    default: return 0;
  }
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

void RangeJoinQueryWorkspace::reset_seed(size_t item_count) {
  qgram.reset_seen(item_count);
  seed_touched.clear();
}

ExactRangeJoinIndex::ExactRangeJoinIndex(
    RangeJoinConfig config, bool defer_qgram_build,
    bool enable_shifted_window_postings,
    bool enable_positional_postings)
    : config_(config),
      qgram_index_(config.qgram_q),
      defer_qgram_build_(defer_qgram_build),
      enable_shifted_window_postings_(enable_shifted_window_postings),
      enable_positional_postings_(enable_positional_postings),
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

void ExactRangeJoinIndex::build(std::vector<RangeJoinItem> items) {
  owned_items_ = std::move(items);
  items_.clear();
  items_.reserve(owned_items_.size());
  for (size_t item_idx = 0; item_idx < owned_items_.size(); ++item_idx) {
    items_.push_back(
        {owned_items_[item_idx].item_id, item_idx, {}});
  }
  reset_after_items_changed();
}

void ExactRangeJoinIndex::build_views(std::vector<RangeJoinItemView> items) {
  owned_items_.clear();
  owned_items_.shrink_to_fit();
  items_.clear();
  items_.reserve(items.size());
  for (const auto& item : items) {
    items_.push_back(
        {item.item_id, std::numeric_limits<size_t>::max(), item.sequence});
  }
  reset_after_items_changed();
}

std::string_view ExactRangeJoinIndex::item_sequence(
    const StoredItem& item) const {
  if (item.owned_item_idx < owned_items_.size()) {
    return owned_items_[item.owned_item_idx].sequence;
  }
  return item.external_sequence;
}

void ExactRangeJoinIndex::reset_after_items_changed() {
  postings16_by_seed_len_.clear();
  postings_by_seed_len_.clear();
  positional_postings_by_seed_len_.clear();
  wide_positional_postings_by_seed_len_.clear();
  unindexable_items16_by_seed_len_.clear();
  unindexable_items_by_seed_len_.clear();
  seed_index_capacity_ =
      items_.size() <= static_cast<size_t>(
                           std::numeric_limits<uint32_t>::max());
  if (enable_positional_postings_ && seed_index_capacity_) {
    for (const auto& item : items_) {
      if (item_sequence(item).size() >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1) {
        seed_index_capacity_ = false;
        break;
      }
    }
  }
  seed_index_uses_16bit_ =
      items_.size() <= static_cast<size_t>(
                           std::numeric_limits<uint16_t>::max());
  positional_postings_use_32bit_ = seed_index_uses_16bit_;
  if (positional_postings_use_32bit_) {
    for (const auto& item : items_) {
      if (item_sequence(item).size() >
          static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1) {
        positional_postings_use_32bit_ = false;
        break;
      }
    }
  }
  qgram_ready_ = false;
  std::vector<QGramCountIndex::ItemView> qgram_items;
  if (!defer_qgram_build_) qgram_items.reserve(items_.size());
  for (const auto& item : items_) {
    if (!defer_qgram_build_) {
      qgram_items.push_back({item.item_id, item_sequence(item)});
    }
  }
  if (!defer_qgram_build_) {
    qgram_index_.build_views(qgram_items);
    qgram_ready_ = true;
  }
}

const QGramCountIndex& ExactRangeJoinIndex::ensure_qgram_index() const {
  const auto build_qgram = [this]() {
    std::vector<QGramCountIndex::ItemView> qgram_items;
    qgram_items.reserve(items_.size());
    for (const auto& item : items_) {
      qgram_items.push_back({item.item_id, item_sequence(item)});
    }
    qgram_index_.build_views(qgram_items);
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

void ExactRangeJoinIndex::prepare_postings_for_seed_len(int seed_len) {
  if (enable_positional_postings_) {
    if (positional_postings_use_32bit_) {
      if (positional_postings_by_seed_len_.count(seed_len)) return;
    } else if (wide_positional_postings_by_seed_len_.count(seed_len)) {
      return;
    }
  } else if (seed_index_uses_16bit_) {
    if (postings16_by_seed_len_.count(seed_len)) return;
  } else if (postings_by_seed_len_.count(seed_len)) {
    return;
  }
  if (!seed_index_capacity_) {
    if (enable_positional_postings_) {
      wide_positional_postings_by_seed_len_.emplace(
          seed_len, WidePositionalPostingLists{});
    } else {
      postings_by_seed_len_.emplace(seed_len, PostingLists{});
    }
    return;
  }

  if (enable_positional_postings_) {
    const auto populate_positional = [&](auto& postings,
                                         auto& unindexable_items) {
      using Posting =
          typename std::decay_t<decltype(postings)>::mapped_type::value_type;
      using CompactIndex =
          typename std::decay_t<decltype(unindexable_items)>::value_type;
      const uint64_t mask =
          seed_len >= 32
              ? std::numeric_limits<uint64_t>::max()
              : (uint64_t{1} << (2 * seed_len)) - 1;
      for (size_t item_idx = 0; item_idx < items_.size(); ++item_idx) {
        const auto& sequence = item_sequence(items_[item_idx]);
        const CompactIndex compact_idx =
            static_cast<CompactIndex>(item_idx);
        if (sequence.size() < static_cast<size_t>(seed_len)) {
          continue;
        }
        if (seed_len > 32 || !is_acgt_sequence(sequence)) {
          unindexable_items.push_back(compact_idx);
          continue;
        }
        const size_t last =
            sequence.size() - static_cast<size_t>(seed_len);
        uint64_t code = 0;
        for (int offset = 0; offset < seed_len; ++offset) {
          code = (code << 2) |
                 dna_base_bits(sequence[static_cast<size_t>(offset)]);
        }
        for (size_t pos = 0; pos <= last; ++pos) {
          if (pos > 0) {
            const size_t next =
                pos + static_cast<size_t>(seed_len) - 1;
            code = ((code << 2) | dna_base_bits(sequence[next])) & mask;
          }
          if constexpr (std::is_same_v<Posting, uint32_t>) {
            postings[code].push_back(
                (static_cast<uint32_t>(item_idx) << 16) |
                static_cast<uint32_t>(pos));
          } else {
            postings[code].push_back(
                (static_cast<uint64_t>(item_idx) << 32) |
                static_cast<uint64_t>(pos));
          }
        }
      }
    };

    if (positional_postings_use_32bit_) {
      PositionalPostingLists postings;
      postings.reserve(items_.size());
      std::vector<uint16_t> unindexable_items;
      populate_positional(postings, unindexable_items);
      positional_postings_by_seed_len_.emplace(
          seed_len, std::move(postings));
      unindexable_items16_by_seed_len_.emplace(
          seed_len, std::move(unindexable_items));
    } else {
      WidePositionalPostingLists postings;
      postings.reserve(items_.size());
      std::vector<uint32_t> unindexable_items;
      populate_positional(postings, unindexable_items);
      wide_positional_postings_by_seed_len_.emplace(
          seed_len, std::move(postings));
      unindexable_items_by_seed_len_.emplace(
          seed_len, std::move(unindexable_items));
    }
    return;
  }

  const auto populate = [&](auto& postings, auto& unindexable_items) {
    using CompactIndex =
        typename std::decay_t<decltype(unindexable_items)>::value_type;
    std::vector<uint64_t> item_codes;
    std::vector<uint64_t> rolling_codes;
    std::unordered_map<uint64_t, uint32_t> rolling_counts;
    std::string_view previous_sequence;
    bool has_previous_sequence = false;
    size_t rolling_head = 0;
    bool rolling_active = false;
    size_t consecutive_one_base_shifts = 0;
    const uint64_t mask =
        seed_len >= 32
            ? std::numeric_limits<uint64_t>::max()
            : (uint64_t{1} << (2 * seed_len)) - 1;

    const auto initialize_rolling = [&](std::string_view sequence) {
      const size_t code_count =
          sequence.size() - static_cast<size_t>(seed_len) + 1;
      rolling_codes.resize(code_count);
      rolling_counts.clear();
      rolling_counts.reserve(code_count);
      uint64_t code = 0;
      for (int offset = 0; offset < seed_len; ++offset) {
        code = (code << 2) |
               dna_base_bits(sequence[static_cast<size_t>(offset)]);
      }
      rolling_codes[0] = code;
      rolling_counts[code]++;
      for (size_t pos = 1; pos < code_count; ++pos) {
        const size_t next = pos + static_cast<size_t>(seed_len) - 1;
        code = ((code << 2) | dna_base_bits(sequence[next])) & mask;
        rolling_codes[pos] = code;
        rolling_counts[code]++;
      }
      rolling_head = 0;
      rolling_active = true;
    };

    for (size_t item_idx = 0; item_idx < items_.size(); ++item_idx) {
      const auto& item = items_[item_idx];
      const auto& sequence = item_sequence(item);
      const CompactIndex compact_idx =
          static_cast<CompactIndex>(item_idx);
      if (sequence.size() < static_cast<size_t>(seed_len)) {
        has_previous_sequence = false;
        rolling_active = false;
        consecutive_one_base_shifts = 0;
        continue;
      }
      if (seed_len > 32 || !is_acgt_sequence(sequence)) {
        unindexable_items.push_back(compact_idx);
        has_previous_sequence = false;
        rolling_active = false;
        consecutive_one_base_shifts = 0;
        continue;
      }
      const bool shifted_one_base =
          enable_shifted_window_postings_ && has_previous_sequence &&
          previous_sequence.size() == sequence.size() &&
          std::equal(previous_sequence.begin() + 1,
                     previous_sequence.end(), sequence.begin());
      consecutive_one_base_shifts =
          shifted_one_base ? consecutive_one_base_shifts + 1 : 0;
      if (shifted_one_base &&
          (rolling_active ||
           consecutive_one_base_shifts >=
               kMinShiftRunForRollingPostings)) {
        if (!rolling_active) initialize_rolling(previous_sequence);
        const size_t code_count = rolling_codes.size();
        const uint64_t outgoing = rolling_codes[rolling_head];
        auto outgoing_count = rolling_counts.find(outgoing);
        if (--outgoing_count->second == 0) {
          rolling_counts.erase(outgoing_count);
        }
        const size_t previous_tail =
            (rolling_head + code_count - 1) % code_count;
        const uint64_t incoming =
            ((rolling_codes[previous_tail] << 2) |
             dna_base_bits(sequence.back())) &
            mask;
        rolling_head = (rolling_head + 1) % code_count;
        const size_t new_tail =
            (rolling_head + code_count - 1) % code_count;
        rolling_codes[new_tail] = incoming;
        rolling_counts[incoming]++;
        for (const auto& entry : rolling_counts) {
          postings[entry.first].push_back(compact_idx);
        }
        previous_sequence = sequence;
        has_previous_sequence = true;
        continue;
      }

      item_codes.clear();
      item_codes.reserve(
          sequence.size() - static_cast<size_t>(seed_len) + 1);
      const size_t last = sequence.size() - static_cast<size_t>(seed_len);
      uint64_t code = 0;
      for (int offset = 0; offset < seed_len; ++offset) {
        code = (code << 2) |
               dna_base_bits(sequence[static_cast<size_t>(offset)]);
      }
      item_codes.push_back(code);
      for (size_t pos = 1; pos <= last; ++pos) {
        const size_t next = pos + static_cast<size_t>(seed_len) - 1;
        code = ((code << 2) | dna_base_bits(sequence[next])) & mask;
        item_codes.push_back(code);
      }
      std::sort(item_codes.begin(), item_codes.end());
      item_codes.erase(std::unique(item_codes.begin(), item_codes.end()),
                       item_codes.end());
      for (uint64_t item_code : item_codes) {
        postings[item_code].push_back(compact_idx);
      }
      previous_sequence = sequence;
      has_previous_sequence = true;
      rolling_active = false;
    }
  };

  if (seed_index_uses_16bit_) {
    PostingLists16 postings;
    postings.reserve(items_.size());
    std::vector<uint16_t> unindexable_items;
    populate(postings, unindexable_items);
    postings16_by_seed_len_.emplace(seed_len, std::move(postings));
    unindexable_items16_by_seed_len_.emplace(
        seed_len, std::move(unindexable_items));
  } else {
    PostingLists postings;
    postings.reserve(items_.size());
    std::vector<uint32_t> unindexable_items;
    populate(postings, unindexable_items);
    postings_by_seed_len_.emplace(seed_len, std::move(postings));
    unindexable_items_by_seed_len_.emplace(
        seed_len, std::move(unindexable_items));
  }
}

void ExactRangeJoinIndex::prepare_seed_lengths(
    const std::vector<int>& seed_lengths) {
  for (int seed_len : seed_lengths) {
    if (seed_len >= config_.min_seed_len) {
      prepare_postings_for_seed_len(seed_len);
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
    std::string_view query_sequence, int tau) {
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
    std::string_view query_sequence, int tau,
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
                            std::numeric_limits<size_t>::max(), workspace);
  }

  if (config_.candidate_mode == RangeCandidateMode::Hybrid) {
    auto pigeonhole =
        pigeonhole_query(query_sequence, tau, block_len, seed_len,
                         std::numeric_limits<size_t>::max(), workspace);
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
  if (!seed_index_capacity_) {
    auto result = full_scan(query_sequence, tau, true);
    result.block_len = block_len;
    result.seed_len = seed_len;
    return result;
  }

  auto pigeonhole =
      pigeonhole_query(query_sequence, tau, block_len, seed_len,
                       config_.auto_pigeonhole_max_candidates, workspace);
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
    std::string_view query_sequence, int tau, bool fallback) const {
  RangeJoinQueryResult result;
  ScopedTimer full_timer(&result.range_full_scan_ms);
  result.mode_used = RangeCandidateMode::FullScan;
  result.used_full_scan = fallback;
  {
    ScopedTimer length_timer(&result.range_length_filter_ms);
    for (const auto& item : items_) {
      const auto& sequence = item_sequence(item);
      if (std::abs(static_cast<long long>(query_sequence.size()) -
                   static_cast<long long>(sequence.size())) <= tau) {
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
    std::string_view query_sequence, int tau, int block_len, int seed_len,
    size_t early_abort_candidate_limit,
    RangeJoinQueryWorkspace* workspace) const {
  if (seed_len < config_.min_seed_len) {
    auto result = full_scan(query_sequence, tau, true);
    result.block_len = block_len;
    result.seed_len = seed_len;
    return result;
  }
  if (!seed_index_capacity_) {
    auto result = full_scan(query_sequence, tau, true);
    result.block_len = block_len;
    result.seed_len = seed_len;
    return result;
  }

  RangeJoinQueryWorkspace local_workspace;
  if (!workspace) workspace = &local_workspace;
  workspace->reset_seed(items_.size());
  auto add_candidate = [&](uint32_t item_idx) {
    if (item_idx >= workspace->qgram.seen_epoch.size() ||
        workspace->qgram.seen_epoch[item_idx] == workspace->qgram.epoch) {
      return false;
    }
    workspace->qgram.seen_epoch[item_idx] = workspace->qgram.epoch;
    workspace->seed_touched.push_back(item_idx);
    return true;
  };

  RangeJoinQueryResult result;
  result.mode_used = RangeCandidateMode::PigeonholeOnly;
  result.block_len = block_len;
  result.seed_len = seed_len;
  const PostingLists16* postings16_ptr = nullptr;
  const PostingLists* postings_ptr = nullptr;
  const PositionalPostingLists* positional_postings_ptr = nullptr;
  const WidePositionalPostingLists* wide_positional_postings_ptr = nullptr;
  {
    ScopedTimer timer(&result.range_posting_lookup_ms);
    if (enable_positional_postings_ &&
        positional_postings_use_32bit_) {
      const auto existing =
          positional_postings_by_seed_len_.find(result.seed_len);
      if (existing != positional_postings_by_seed_len_.end()) {
        positional_postings_ptr = &existing->second;
      }
    } else if (enable_positional_postings_) {
      const auto existing =
          wide_positional_postings_by_seed_len_.find(result.seed_len);
      if (existing != wide_positional_postings_by_seed_len_.end()) {
        wide_positional_postings_ptr = &existing->second;
      }
    } else if (seed_index_uses_16bit_) {
      const auto existing =
          postings16_by_seed_len_.find(result.seed_len);
      if (existing != postings16_by_seed_len_.end()) {
        postings16_ptr = &existing->second;
      }
    } else {
      const auto existing = postings_by_seed_len_.find(result.seed_len);
      if (existing != postings_by_seed_len_.end()) {
        postings_ptr = &existing->second;
      }
    }
  }
  if (!postings16_ptr && !postings_ptr &&
      !positional_postings_ptr && !wide_positional_postings_ptr) {
    auto fallback = full_scan(query_sequence, tau, true);
    merge_range_timing(fallback, result);
    fallback.block_len = block_len;
    fallback.seed_len = seed_len;
    return fallback;
  }
  const auto consume_indices = [&](const auto& indices) {
    for (auto compact_idx : indices) {
      add_candidate(static_cast<uint32_t>(compact_idx));
      if (workspace->seed_touched.size() >
          early_abort_candidate_limit) {
        result.seed_candidate_pairs_before_length_filter =
            workspace->seed_touched.size();
        result.pigeonhole_candidate_count =
            workspace->seed_touched.size();
        result.pigeonhole_early_abort_count = 1;
        result.final_candidate_pairs = 0;
        return false;
      }
    }
    return true;
  };
  const auto consume_positional_indices =
      [&](const auto& indices, size_t block_start) {
        using Posting =
            typename std::decay_t<decltype(indices)>::value_type;
        for (Posting packed : indices) {
          uint32_t item_idx = 0;
          uint32_t position = 0;
          if constexpr (std::is_same_v<Posting, uint32_t>) {
            item_idx = packed >> 16;
            position =
                packed & std::numeric_limits<uint16_t>::max();
          } else {
            item_idx = static_cast<uint32_t>(packed >> 32);
            position = static_cast<uint32_t>(
                packed & std::numeric_limits<uint32_t>::max());
          }
          if (std::llabs(static_cast<long long>(position) -
                         static_cast<long long>(block_start)) <= tau) {
            add_candidate(item_idx);
            if (workspace->seed_touched.size() >
                early_abort_candidate_limit) {
              result.seed_candidate_pairs_before_length_filter =
                  workspace->seed_touched.size();
              result.pigeonhole_candidate_count =
                  workspace->seed_touched.size();
              result.pigeonhole_early_abort_count = 1;
              result.final_candidate_pairs = 0;
              return false;
            }
          }
        }
        return true;
      };

  const int block_count = tau + 1;
  for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    const size_t block_start =
        static_cast<size_t>(block_idx) * static_cast<size_t>(result.block_len);
    uint64_t seed = 0;
    if (!encode_dna_seed(
            query_sequence, block_start, result.seed_len, &seed)) {
      auto fallback = full_scan(query_sequence, tau, true);
      merge_range_timing(fallback, result);
      fallback.block_len = block_len;
      fallback.seed_len = seed_len;
      return fallback;
    }
    if (positional_postings_ptr) {
      PositionalPostingLists::const_iterator posting;
      {
        ScopedTimer timer(&result.range_posting_lookup_ms);
        posting = positional_postings_ptr->find(seed);
      }
      if (posting != positional_postings_ptr->end()) {
        ScopedTimer timer(&result.range_seed_union_ms);
        if (!consume_positional_indices(
                posting->second, block_start)) {
          return result;
        }
      }
    } else if (wide_positional_postings_ptr) {
      WidePositionalPostingLists::const_iterator posting;
      {
        ScopedTimer timer(&result.range_posting_lookup_ms);
        posting = wide_positional_postings_ptr->find(seed);
      }
      if (posting != wide_positional_postings_ptr->end()) {
        ScopedTimer timer(&result.range_seed_union_ms);
        if (!consume_positional_indices(
                posting->second, block_start)) {
          return result;
        }
      }
    } else if (postings16_ptr) {
      PostingLists16::const_iterator posting;
      {
        ScopedTimer timer(&result.range_posting_lookup_ms);
        posting = postings16_ptr->find(seed);
      }
      if (posting != postings16_ptr->end()) {
        ScopedTimer timer(&result.range_seed_union_ms);
        if (!consume_indices(posting->second)) return result;
      }
    } else {
      PostingLists::const_iterator posting;
      {
        ScopedTimer timer(&result.range_posting_lookup_ms);
        posting = postings_ptr->find(seed);
      }
      if (posting != postings_ptr->end()) {
        ScopedTimer timer(&result.range_seed_union_ms);
        if (!consume_indices(posting->second)) return result;
      }
    }
  }
  const bool compact_unindexable =
      enable_positional_postings_
          ? positional_postings_use_32bit_
          : seed_index_uses_16bit_;
  if (compact_unindexable) {
    const auto unindexable_it =
        unindexable_items16_by_seed_len_.find(result.seed_len);
    if (unindexable_it != unindexable_items16_by_seed_len_.end()) {
      ScopedTimer timer(&result.range_seed_union_ms);
      if (!consume_indices(unindexable_it->second)) return result;
    }
  } else {
    const auto unindexable_it =
        unindexable_items_by_seed_len_.find(result.seed_len);
    if (unindexable_it != unindexable_items_by_seed_len_.end()) {
      ScopedTimer timer(&result.range_seed_union_ms);
      if (!consume_indices(unindexable_it->second)) return result;
    }
  }

  result.seed_candidate_pairs_before_length_filter =
      workspace->seed_touched.size();
  result.candidate_item_ids.reserve(workspace->seed_touched.size());
  {
    ScopedTimer timer(&result.range_length_filter_ms);
    for (uint32_t item_idx : workspace->seed_touched) {
      if (item_idx >= items_.size()) {
        auto fallback = full_scan(query_sequence, tau, true);
        merge_range_timing(fallback, result);
        fallback.block_len = block_len;
        fallback.seed_len = seed_len;
        return fallback;
      }
      const auto& item = items_[item_idx];
      const auto& sequence = item_sequence(item);
      if (std::abs(static_cast<long long>(query_sequence.size()) -
                   static_cast<long long>(sequence.size())) <= tau) {
        result.candidate_item_ids.push_back(item.item_id);
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
    std::string_view query_sequence, int tau,
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
