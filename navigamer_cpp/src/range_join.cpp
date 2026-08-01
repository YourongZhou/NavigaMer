#include "range_join.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
  if (item_count <=
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1) {
    seed_touched16.clear();
    if (!seed_touched.empty()) std::vector<uint32_t>().swap(seed_touched);
  } else {
    seed_touched.clear();
    if (!seed_touched16.empty()) {
      std::vector<uint16_t>().swap(seed_touched16);
    }
  }
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
  if (items.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::length_error("range-join index exceeds 32-bit item capacity");
  }
  for (const auto& item : items) {
    if (item.item_id >
        static_cast<size_t>(std::numeric_limits<RangeJoinItemId>::max())) {
      throw std::length_error("range-join item ID exceeds 32-bit capacity");
    }
  }
  owned_items_ = std::move(items);
  item_storage_ = std::monostate{};
  external_item_ids_.clear();
  external_item_ids_.shrink_to_fit();
  reset_after_items_changed();
}

void ExactRangeJoinIndex::build_views(std::vector<RangeJoinItemView> items) {
  if (items.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::length_error("range-join index exceeds 32-bit item capacity");
  }
  owned_items_.clear();
  owned_items_.shrink_to_fit();
  external_item_ids_.clear();
  bool identity_item_ids = true;
  for (size_t item_idx = 0; item_idx < items.size(); ++item_idx) {
    if (items[item_idx].item_id >
        static_cast<size_t>(std::numeric_limits<RangeJoinItemId>::max())) {
      throw std::length_error("range-join item ID exceeds 32-bit capacity");
    }
    if (items[item_idx].item_id != item_idx) {
      identity_item_ids = false;
    }
  }
  if (!identity_item_ids) {
    external_item_ids_.reserve(items.size());
    for (const auto& item : items) {
      external_item_ids_.push_back(
          static_cast<RangeJoinItemId>(item.item_id));
    }
  } else {
    external_item_ids_.shrink_to_fit();
  }
  std::vector<std::string_view> sequences;
  sequences.reserve(items.size());
  for (const auto& item : items) {
    sequences.push_back(item.sequence);
  }
  item_storage_ = std::move(sequences);
  reset_after_items_changed();
}

size_t ExactRangeJoinIndex::item_count() const {
  switch (item_storage_.index()) {
    case 0:
      return std::get<0>(item_storage_).size();
    case 1:
      return std::get<1>(item_storage_).size();
    case 2:
      return owned_items_.size();
  }
  return 0;
}

RangeJoinItemId ExactRangeJoinIndex::item_id(size_t item_idx) const {
  if (item_storage_.index() == 2) {
    return static_cast<RangeJoinItemId>(owned_items_[item_idx].item_id);
  }
  return external_item_ids_.empty()
             ? static_cast<RangeJoinItemId>(item_idx)
             : external_item_ids_[item_idx];
}

std::string_view ExactRangeJoinIndex::item_sequence(
    size_t item_idx) const {
  if (item_storage_.index() == 0) {
    if (max_item_sequence_length_ == 0) return {};
    const auto* reference_data =
        reinterpret_cast<const char*>(item_storage_aux_);
    return std::string_view(
        reference_data + std::get<0>(item_storage_)[item_idx],
        max_item_sequence_length_);
  }
  if (item_storage_.index() == 1) {
    return std::get<1>(item_storage_)[item_idx];
  }
  return owned_items_[item_idx].sequence;
}

size_t ExactRangeJoinIndex::min_item_sequence_length() const {
  return item_storage_.index() == 0
             ? max_item_sequence_length_
             : static_cast<size_t>(item_storage_aux_);
}

void ExactRangeJoinIndex::reset_after_items_changed() {
  const size_t count = item_count();
  const size_t uniform_sequence_length =
      item_storage_.index() == 0 ? max_item_sequence_length_ : 0;
  postings16_by_seed_len_.clear();
  postings_by_seed_len_.clear();
  positional_postings_by_seed_len_.clear();
  wide_positional_postings_by_seed_len_.clear();
  unindexable_items16_by_seed_len_.clear();
  unindexable_items_by_seed_len_.clear();
  seed_index_capacity_ =
      count <= static_cast<size_t>(
                           std::numeric_limits<uint32_t>::max());
  if (enable_positional_postings_ && seed_index_capacity_ &&
      uniform_sequence_length != 0) {
    seed_index_capacity_ =
        uniform_sequence_length <=
        static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  } else if (enable_positional_postings_ && seed_index_capacity_) {
    for (size_t item_idx = 0; item_idx < count; ++item_idx) {
      if (item_sequence(item_idx).size() >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1) {
        seed_index_capacity_ = false;
        break;
      }
    }
  }
  seed_index_uses_16bit_ =
      count <= static_cast<size_t>(
                           std::numeric_limits<uint16_t>::max());
  positional_postings_use_32bit_ = seed_index_uses_16bit_;
  if (positional_postings_use_32bit_ && uniform_sequence_length != 0) {
    positional_postings_use_32bit_ =
        uniform_sequence_length <=
        static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;
  } else if (positional_postings_use_32bit_) {
    for (size_t item_idx = 0; item_idx < count; ++item_idx) {
      if (item_sequence(item_idx).size() >
          static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1) {
        positional_postings_use_32bit_ = false;
        break;
      }
    }
  }
  size_t min_item_sequence_length =
      count == 0 ? 0 : std::numeric_limits<size_t>::max();
  max_item_sequence_length_ =
      count == 0 ? 0 : uniform_sequence_length;
  if (item_storage_.index() == 0 && count != 0) {
    min_item_sequence_length = uniform_sequence_length;
  }
  item_ids_strictly_increasing_ = true;
  qgram_ready_ = false;
  if (item_storage_.index() == 0 && defer_qgram_build_) return;
  std::vector<QGramCountIndex::ItemView> qgram_items;
  if (!defer_qgram_build_) qgram_items.reserve(count);
  for (size_t item_idx = 0; item_idx < count; ++item_idx) {
    const auto sequence = item_sequence(item_idx);
    min_item_sequence_length =
        std::min(min_item_sequence_length, sequence.size());
    max_item_sequence_length_ =
        std::max(max_item_sequence_length_, sequence.size());
    if (item_idx != 0 &&
        item_id(item_idx) <= item_id(item_idx - 1)) {
      item_ids_strictly_increasing_ = false;
    }
    if (!defer_qgram_build_) {
      qgram_items.push_back({item_id(item_idx), sequence});
    }
  }
  if (item_storage_.index() != 0) {
    item_storage_aux_ = min_item_sequence_length;
  }
  if (!defer_qgram_build_) {
    qgram_index_.build_views(qgram_items);
    qgram_ready_ = true;
  }
}

bool ExactRangeJoinIndex::qgram_bound_is_vacuous(
    std::string_view query_sequence, int tau,
    bool* full_scan_fallback, size_t* query_total) const {
  *full_scan_fallback = false;
  *query_total = 0;

  const int q = config_.qgram_q;
  if (q > 32) {
    *full_scan_fallback = true;
    return true;
  }
  for (char base : query_sequence) {
    if (base != 'A' && base != 'C' && base != 'G' && base != 'T') {
      *full_scan_fallback = true;
      return true;
    }
  }

  const size_t q_size = static_cast<size_t>(q);
  *query_total = query_sequence.size() < q_size
                     ? 0
                     : query_sequence.size() - q_size + 1;
  if (*query_total == 0) {
    *full_scan_fallback = true;
    return true;
  }
  if (item_count() == 0) return true;

  const size_t threshold = static_cast<size_t>(tau);
  const size_t min_compatible_length =
      query_sequence.size() > threshold
          ? query_sequence.size() - threshold
          : 0;
  const size_t max_compatible_length =
      query_sequence.size() >
              std::numeric_limits<size_t>::max() - threshold
          ? std::numeric_limits<size_t>::max()
          : query_sequence.size() + threshold;
  if (max_item_sequence_length_ < min_compatible_length ||
      min_item_sequence_length() > max_compatible_length) {
    return true;
  }

  // q-gram L1 pruning requires
  //   query_total + item_total - 2*q*tau > 0.
  // item_total grows monotonically with sequence length, so using the largest
  // possibly compatible length gives a safe O(1) upper bound. If even that
  // numerator is non-positive, no item can be removed by the posting index.
  const size_t largest_possible_length =
      std::min(max_item_sequence_length_, max_compatible_length);
  const size_t largest_item_total =
      largest_possible_length < q_size
          ? 0
          : largest_possible_length - q_size + 1;
  const uint64_t max_l1 =
      uint64_t{2} * static_cast<uint64_t>(q) *
      static_cast<uint64_t>(threshold);
  if (*query_total > max_l1) return false;
  return largest_item_total <= max_l1 - *query_total;
}

const QGramCountIndex& ExactRangeJoinIndex::ensure_qgram_index() const {
  const auto build_qgram = [this]() {
    std::vector<QGramCountIndex::ItemView> qgram_items;
    qgram_items.reserve(item_count());
    for (size_t item_idx = 0; item_idx < item_count(); ++item_idx) {
      qgram_items.push_back(
          {item_id(item_idx), item_sequence(item_idx)});
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
      for (size_t item_idx = 0; item_idx < item_count(); ++item_idx) {
        const auto sequence = item_sequence(item_idx);
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
      postings.reserve(item_count());
      std::vector<uint16_t> unindexable_items;
      populate_positional(postings, unindexable_items);
      positional_postings_by_seed_len_.emplace(
          seed_len, std::move(postings));
      unindexable_items16_by_seed_len_.emplace(
          seed_len, std::move(unindexable_items));
    } else {
      WidePositionalPostingLists postings;
      postings.reserve(item_count());
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
    const auto generate = [&](size_t item_limit,
                              bool collect_unindexable,
                              auto&& emit) {
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
          const size_t next =
              pos + static_cast<size_t>(seed_len) - 1;
          code = ((code << 2) | dna_base_bits(sequence[next])) & mask;
          rolling_codes[pos] = code;
          rolling_counts[code]++;
        }
        rolling_head = 0;
        rolling_active = true;
      };

      for (size_t item_idx = 0; item_idx < item_limit; ++item_idx) {
        const auto sequence = item_sequence(item_idx);
        const CompactIndex compact_idx =
            static_cast<CompactIndex>(item_idx);
        if (sequence.size() < static_cast<size_t>(seed_len)) {
          has_previous_sequence = false;
          rolling_active = false;
          consecutive_one_base_shifts = 0;
          continue;
        }
        if (seed_len > 32 || !is_acgt_sequence(sequence)) {
          if (collect_unindexable) {
            unindexable_items.push_back(compact_idx);
          }
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
            emit(entry.first, compact_idx);
          }
          previous_sequence = sequence;
          has_previous_sequence = true;
          continue;
        }

        item_codes.clear();
        item_codes.reserve(
            sequence.size() - static_cast<size_t>(seed_len) + 1);
        const size_t last =
            sequence.size() - static_cast<size_t>(seed_len);
        uint64_t code = 0;
        for (int offset = 0; offset < seed_len; ++offset) {
          code = (code << 2) |
                 dna_base_bits(sequence[static_cast<size_t>(offset)]);
        }
        item_codes.push_back(code);
        for (size_t pos = 1; pos <= last; ++pos) {
          const size_t next =
              pos + static_cast<size_t>(seed_len) - 1;
          code = ((code << 2) | dna_base_bits(sequence[next])) & mask;
          item_codes.push_back(code);
        }
        std::sort(item_codes.begin(), item_codes.end());
        item_codes.erase(
            std::unique(item_codes.begin(), item_codes.end()),
            item_codes.end());
        for (uint64_t item_code : item_codes) {
          emit(item_code, compact_idx);
        }
        previous_sequence = sequence;
        has_previous_sequence = true;
        rolling_active = false;
      }
    };

    const size_t sample_item_count = std::min<size_t>(item_count(), 256);
    const size_t sample_prefix_item_count = sample_item_count / 2;
    std::vector<uint64_t> sample_codes;
    std::vector<uint64_t> sample_prefix_codes;
    long double sample_prefix_possible_postings = 0.0L;
    long double sample_possible_postings = 0.0L;
    long double total_possible_postings = 0.0L;
    for (size_t item_idx = 0; item_idx < item_count(); ++item_idx) {
      const size_t length = item_sequence(item_idx).size();
      if (length < static_cast<size_t>(seed_len)) continue;
      const size_t possible =
          length - static_cast<size_t>(seed_len) + 1;
      total_possible_postings += static_cast<long double>(possible);
      if (item_idx < sample_item_count) {
        sample_possible_postings += static_cast<long double>(possible);
        if (item_idx < sample_prefix_item_count) {
          sample_prefix_possible_postings +=
              static_cast<long double>(possible);
        }
      }
    }
    if (sample_possible_postings != 0.0L) {
      generate(sample_item_count, false,
               [&](uint64_t code, CompactIndex item_idx) {
                 sample_codes.push_back(code);
                 if (item_idx < sample_prefix_item_count) {
                   sample_prefix_codes.push_back(code);
                 }
               });
      std::sort(sample_codes.begin(), sample_codes.end());
      sample_codes.erase(
          std::unique(sample_codes.begin(), sample_codes.end()),
          sample_codes.end());
      std::sort(sample_prefix_codes.begin(), sample_prefix_codes.end());
      sample_prefix_codes.erase(
          std::unique(sample_prefix_codes.begin(),
                      sample_prefix_codes.end()),
          sample_prefix_codes.end());
      const long double code_space =
          seed_len >= 32
              ? static_cast<long double>(
                    std::numeric_limits<uint64_t>::max())
              : static_cast<long double>(
                    uint64_t{1} << (2 * seed_len));
      const long double sample_tail_possible_postings =
          sample_possible_postings - sample_prefix_possible_postings;
      const long double remaining_possible_postings =
          total_possible_postings - sample_possible_postings;
      const long double sampled_new_code_count =
          sample_codes.size() - sample_prefix_codes.size();
      const long double extrapolated_code_count =
          sample_tail_possible_postings == 0.0L
              ? static_cast<long double>(sample_codes.size())
              : static_cast<long double>(sample_codes.size()) +
                    sampled_new_code_count *
                        remaining_possible_postings /
                        sample_tail_possible_postings;
      const long double estimate = std::min(
          {total_possible_postings, code_space,
           extrapolated_code_count,
           static_cast<long double>(
               std::numeric_limits<size_t>::max())});
      postings.reserve_unique_codes(
          std::max(sample_codes.size(),
                   static_cast<size_t>(std::ceil(estimate))));
      std::vector<uint64_t>().swap(sample_codes);
      std::vector<uint64_t>().swap(sample_prefix_codes);
    }

    generate(item_count(), true, [&](uint64_t code, CompactIndex) {
      postings.count(code);
    });
    postings.finish_counting(
        item_count() == 0
            ? 0
            : static_cast<uint32_t>(item_count() - 1));
    generate(item_count(), false,
             [&](uint64_t code, CompactIndex compact_idx) {
      postings.append(code, compact_idx);
    });
  };

  if (seed_index_uses_16bit_) {
    PostingLists16 postings;
    std::vector<uint16_t> unindexable_items;
    populate(postings, unindexable_items);
    postings16_by_seed_len_.emplace(seed_len, std::move(postings));
    unindexable_items16_by_seed_len_.emplace(
        seed_len, std::move(unindexable_items));
  } else {
    PostingLists postings;
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
  const auto query_length = static_cast<long long>(query_sequence.size());
  if (std::llabs(query_length -
                 static_cast<long long>(min_item_sequence_length())) <= tau &&
      std::llabs(query_length -
                 static_cast<long long>(max_item_sequence_length_)) <= tau) {
    result.candidate_item_ids.reserve(item_count());
  }
  {
    ScopedTimer length_timer(&result.range_length_filter_ms);
    const size_t uniform_sequence_length =
        item_storage_.index() == 0 ? max_item_sequence_length_ : 0;
    for (size_t item_idx = 0; item_idx < item_count(); ++item_idx) {
      const size_t item_length =
          uniform_sequence_length != 0
              ? uniform_sequence_length
              : item_sequence(item_idx).size();
      if (std::abs(static_cast<long long>(query_sequence.size()) -
                   static_cast<long long>(item_length)) <= tau) {
        result.candidate_item_ids.push_back(item_id(item_idx));
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
  workspace->reset_seed(item_count());
  const bool compact_seen =
      item_count() <=
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;
  uint16_t* seen16 =
      compact_seen ? workspace->qgram.seen_epoch16.data() : nullptr;
  uint32_t* seen32 =
      compact_seen ? nullptr : workspace->qgram.seen_epoch.data();
  const uint32_t current_epoch =
      compact_seen ? workspace->qgram.epoch16 : workspace->qgram.epoch;
  const auto touched_size = [&]() {
    return compact_seen ? workspace->seed_touched16.size()
                        : workspace->seed_touched.size();
  };
  auto add_candidate = [&](uint32_t item_idx) {
    if (item_idx >= item_count() ||
        (compact_seen
             ? seen16[item_idx] == current_epoch
             : seen32[item_idx] == current_epoch)) {
      return false;
    }
    if (compact_seen) {
      seen16[item_idx] = static_cast<uint16_t>(current_epoch);
    } else {
      seen32[item_idx] = current_epoch;
    }
    if (compact_seen) {
      workspace->seed_touched16.push_back(
          static_cast<uint16_t>(item_idx));
    } else {
      workspace->seed_touched.push_back(item_idx);
    }
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
      if (touched_size() > early_abort_candidate_limit) {
        result.seed_candidate_pairs_before_length_filter =
            touched_size();
        result.pigeonhole_candidate_count =
            touched_size();
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
            if (touched_size() > early_abort_candidate_limit) {
              result.seed_candidate_pairs_before_length_filter =
                  touched_size();
              result.pigeonhole_candidate_count =
                  touched_size();
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
      PostingLists16::Range posting;
      {
        ScopedTimer timer(&result.range_posting_lookup_ms);
        posting = postings16_ptr->find(seed);
      }
      if (posting) {
        ScopedTimer timer(&result.range_seed_union_ms);
        if (!consume_indices(posting)) return result;
      }
    } else {
      PostingLists::Range posting;
      {
        ScopedTimer timer(&result.range_posting_lookup_ms);
        posting = postings_ptr->find(seed);
      }
      if (posting) {
        ScopedTimer timer(&result.range_seed_union_ms);
        if (!consume_indices(posting)) return result;
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
      touched_size();
  result.candidate_item_ids.reserve(touched_size());
  bool valid_touched_ids = true;
  {
    ScopedTimer timer(&result.range_length_filter_ms);
    const auto append_compatible = [&](const auto& touched) {
      for (auto compact_idx : touched) {
        const uint32_t item_idx = static_cast<uint32_t>(compact_idx);
        if (item_idx >= item_count()) {
          valid_touched_ids = false;
          return;
        }
        const auto sequence = item_sequence(item_idx);
        if (std::abs(static_cast<long long>(query_sequence.size()) -
                     static_cast<long long>(sequence.size())) <= tau) {
          result.candidate_item_ids.push_back(item_id(item_idx));
        } else {
          result.length_filtered_items++;
          result.seed_length_pruned_candidates++;
        }
      }
    };
    if (compact_seen) {
      append_compatible(workspace->seed_touched16);
    } else {
      append_compatible(workspace->seed_touched);
    }
  }
  if (!valid_touched_ids) {
    auto fallback = full_scan(query_sequence, tau, true);
    merge_range_timing(fallback, result);
    fallback.block_len = block_len;
    fallback.seed_len = seed_len;
    return fallback;
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
    bool full_scan_fallback = false;
    size_t query_total = 0;
    if (qgram_bound_is_vacuous(
            query_sequence, tau, &full_scan_fallback, &query_total)) {
      stats.total_items = item_count();
      stats.full_scan_fallbacks = full_scan_fallback ? 1 : 0;
      const size_t q_size = static_cast<size_t>(config_.qgram_q);
      const bool posting_index_capacity =
          item_count() <= static_cast<size_t>(
                               std::numeric_limits<uint32_t>::max());
      const auto query_length =
          static_cast<long long>(query_sequence.size());
      if (std::llabs(
              query_length -
              static_cast<long long>(min_item_sequence_length())) <= tau &&
          std::llabs(
              query_length -
              static_cast<long long>(max_item_sequence_length_)) <= tau) {
        result.candidate_item_ids.reserve(item_count());
      }
      const size_t uniform_sequence_length =
          item_storage_.index() == 0 ? max_item_sequence_length_ : 0;
      for (size_t item_idx = 0; item_idx < item_count(); ++item_idx) {
        const size_t item_length =
            uniform_sequence_length != 0
                ? uniform_sequence_length
                : item_sequence(item_idx).size();
        if (std::llabs(
                static_cast<long long>(query_sequence.size()) -
                static_cast<long long>(item_length)) > tau) {
          stats.length_filtered_items++;
          continue;
        }
        result.candidate_item_ids.push_back(item_id(item_idx));
        const size_t item_total =
            item_length < q_size ? 0 : item_length - q_size + 1;
        if (!full_scan_fallback && posting_index_capacity &&
            query_total != 0 && item_total != 0) {
          stats.required_shared_nonpositive++;
        }
      }
      if (!item_ids_strictly_increasing_) {
        std::sort(result.candidate_item_ids.begin(),
                  result.candidate_item_ids.end());
        result.candidate_item_ids.erase(
            std::unique(result.candidate_item_ids.begin(),
                        result.candidate_item_ids.end()),
            result.candidate_item_ids.end());
      }
      stats.qgram_candidates = result.candidate_item_ids.size();
    } else {
      const QGramCountIndex& qgram_index =
          defer_qgram_build_ ? ensure_qgram_index() : qgram_index_;
      result.candidate_item_ids =
          qgram_index.query(query_sequence, tau, &stats,
                            workspace ? &workspace->qgram : nullptr);
    }
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

void ExactRangeJoinIndex::build_uniform_identity_views(
    std::vector<const char*> sequence_data,
    size_t sequence_length) {
  if (sequence_data.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::length_error("range-join index exceeds 32-bit item capacity");
  }
  uintptr_t reference_base = 0;
  uintptr_t reference_end = 0;
  if (!sequence_data.empty()) {
    if (!sequence_data.front()) {
      throw std::invalid_argument("range-join sequence pointer is null");
    }
    reference_base = reinterpret_cast<uintptr_t>(sequence_data.front());
    reference_end = reference_base;
    for (const char* sequence : sequence_data) {
      if (!sequence) {
        throw std::invalid_argument("range-join sequence pointer is null");
      }
      const uintptr_t address = reinterpret_cast<uintptr_t>(sequence);
      reference_base = std::min(reference_base, address);
      reference_end = std::max(reference_end, address);
    }
    if (reference_end - reference_base >
        std::numeric_limits<uint32_t>::max()) {
      throw std::length_error(
          "range-join sequence pointers exceed 32-bit reference span");
    }
  }
  std::vector<uint32_t> sequence_offsets;
  sequence_offsets.reserve(sequence_data.size());
  for (const char* sequence : sequence_data) {
    sequence_offsets.push_back(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(sequence) - reference_base));
  }
  owned_items_.clear();
  owned_items_.shrink_to_fit();
  item_storage_ = std::move(sequence_offsets);
  item_storage_aux_ = reference_base;
  max_item_sequence_length_ = sequence_length;
  external_item_ids_.clear();
  external_item_ids_.shrink_to_fit();
  reset_after_items_changed();
}

}  // namespace navigamer
