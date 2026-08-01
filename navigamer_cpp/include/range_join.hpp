#ifndef NAVIGAMER_RANGE_JOIN_HPP
#define NAVIGAMER_RANGE_JOIN_HPP

#include "qgram_filter.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace navigamer {

using RangeJoinItemId = uint32_t;

enum class RangeCandidateMode {
  Auto,
  PigeonholeOnly,
  QGramOnly,
  Hybrid,
  FullScan,
};

const char* range_candidate_mode_name(RangeCandidateMode mode);
RangeCandidateMode parse_range_candidate_mode(const std::string& value);

struct RangeJoinConfig {
  int min_seed_len = 6;
  int max_seed_len = 20;
  int qgram_q = 5;
  RangeCandidateMode candidate_mode = RangeCandidateMode::Auto;
  size_t auto_pigeonhole_max_candidates = 4096;
  double auto_pigeonhole_max_ratio = 0.25;
  bool auto_hybrid_on_large_candidates = true;
};

struct RangeJoinItem {
  size_t item_id = 0;
  std::string sequence;
};

struct RangeJoinItemView {
  size_t item_id = 0;
  std::string_view sequence;
};

struct RangeJoinQueryResult {
  std::vector<RangeJoinItemId> candidate_item_ids;
  bool used_full_scan = false;
  RangeCandidateMode mode_used = RangeCandidateMode::FullScan;
  int block_len = 0;
  int seed_len = 0;
  size_t length_filtered_items = 0;
  size_t qgram_candidate_count = 0;
  size_t qgram_pruned_by_l1 = 0;
  size_t required_shared_nonpositive = 0;
  size_t compatible_item_count = 0;
  size_t pigeonhole_candidate_count = 0;
  double pigeonhole_candidate_ratio = 0.0;
  size_t seed_candidate_pairs_before_length_filter = 0;
  size_t seed_length_pruned_candidates = 0;
  size_t pigeonhole_early_abort_count = 0;
  size_t final_candidate_pairs = 0;
  size_t auto_pigeonhole_accepted = 0;
  size_t auto_pigeonhole_rejected_large_candidates = 0;
  size_t auto_qgram_invoked = 0;
  size_t auto_hybrid_invoked = 0;
  size_t auto_final_candidate_pairs = 0;
  double auto_candidate_ratio_sum = 0.0;
  double range_posting_lookup_ms = 0.0;
  double range_seed_union_ms = 0.0;
  double range_length_filter_ms = 0.0;
  double range_qgram_query_ms = 0.0;
  double range_hybrid_intersection_ms = 0.0;
  double range_full_scan_ms = 0.0;
};

struct RangeJoinQueryWorkspace {
  QGramQueryWorkspace qgram;
  std::vector<uint32_t> seed_touched;
  std::vector<uint16_t> seed_touched16;

  void reset_seed(size_t item_count);
};

class ExactRangeJoinIndex {
 public:
  explicit ExactRangeJoinIndex(
      RangeJoinConfig config = {}, bool defer_qgram_build = false,
      bool enable_shifted_window_postings = false,
      bool enable_positional_postings = false);

  void build(std::vector<RangeJoinItem> items);
  void build_views(std::vector<RangeJoinItemView> items);
  // Every pointer must refer into the same stable backing allocation.
  void build_uniform_identity_views(
      std::vector<const char*> sequence_data, size_t sequence_length);
  void prepare_qgram();
  void prepare_seed_lengths(const std::vector<int>& seed_lengths);
  RangeJoinQueryResult query(std::string_view query_sequence, int tau);
  RangeJoinQueryResult query(
      std::string_view query_sequence, int tau,
      RangeJoinQueryWorkspace* workspace) const;

 private:
  // Build-time seed postings use contiguous open-addressed slots, exact-width
  // values, and adaptive intervals for consecutive reference-window IDs.
  template <typename Posting>
  class CompactPostingLists {
   public:
    class Iterator {
     public:
      Iterator() = default;
      Iterator(const uint8_t* data, uint64_t begin, uint64_t end,
               uint8_t bit_width, bool interval)
          : data_(data), index_(begin), end_(end),
            bit_width_(bit_width), interval_(interval),
            at_end_(begin == end) {
        if (interval_ && !at_end_) {
          current_ = decode(index_++);
          run_end_ = decode(index_++);
        }
      }

      Posting operator*() const {
        return interval_ ? current_ : decode(index_);
      }
      Iterator& operator++() {
        if (interval_) {
          if (current_ < run_end_) {
            ++current_;
          } else if (index_ < end_) {
            current_ = decode(index_++);
            run_end_ = decode(index_++);
          } else {
            at_end_ = true;
          }
        } else if (++index_ == end_) {
          at_end_ = true;
        }
        return *this;
      }
      bool operator!=(const Iterator& other) const {
        return at_end_ != other.at_end_;
      }

     private:
      Posting decode(uint64_t index) const {
        const uint64_t bit_offset = index * bit_width_;
        const size_t byte_offset =
            static_cast<size_t>(bit_offset >> 3);
        const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
        uint64_t word = 0;
        std::memcpy(&word, data_ + byte_offset, sizeof(word));
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        word = __builtin_bswap64(word);
#endif
        const uint64_t mask =
            bit_width_ == 32
                ? std::numeric_limits<uint32_t>::max()
                : (uint64_t{1} << bit_width_) - 1;
        return static_cast<Posting>((word >> shift) & mask);
      }

      const uint8_t* data_ = nullptr;
      uint64_t index_ = 0;
      uint64_t end_ = 0;
      uint8_t bit_width_ = 0;
      bool interval_ = false;
      bool at_end_ = true;
      Posting current_ = 0;
      Posting run_end_ = 0;
    };

    struct Range {
      using value_type = Posting;

      Iterator first;
      Iterator last;
      bool found = false;

      Iterator begin() const { return first; }
      Iterator end() const { return last; }
      explicit operator bool() const { return found; }
    };

    void reserve_unique_codes(size_t expected_unique_codes) {
      if (expected_unique_codes == 0 || !slots64_.empty() ||
          !slots32_.empty()) {
        return;
      }
      size_t capacity = 16;
      while (expected_unique_codes > capacity - capacity / 8) {
        if (capacity > std::numeric_limits<size_t>::max() / 2) {
          throw std::length_error("range-join seed table is too large");
        }
        capacity *= 2;
      }
      rehash(slots32_, capacity);
    }

    void count(uint64_t code, Posting posting) {
      if (!slots64_.empty()) {
        count_in(slots64_, code, posting);
        ++posting_count_;
        return;
      }
      Slot32* slot = find_or_insert(slots32_, code);
      if (slot->count == std::numeric_limits<uint32_t>::max() ||
          posting_count_ == std::numeric_limits<uint32_t>::max()) {
        promote_to_64bit();
        count_in(slots64_, code, posting);
      } else {
        if (slot->count == 0 ||
            slot->begin == std::numeric_limits<uint32_t>::max() ||
            posting != static_cast<Posting>(slot->begin + 1)) {
          ++slot->encoded_count;
        }
        slot->begin = posting;
        ++slot->count;
      }
      ++posting_count_;
    }

    void finish_counting(uint32_t maximum_posting) {
      if (!slots64_.empty()) {
        finish_slots(slots64_);
      } else {
        finish_slots(slots32_);
      }
      posting_bit_width_ = 1;
      while ((maximum_posting >>= 1) != 0) ++posting_bit_width_;
      if (encoded_posting_count_ >
          (std::numeric_limits<size_t>::max() - 7) /
              posting_bit_width_) {
        throw std::length_error("range-join posting storage is too large");
      }
      const size_t byte_count = static_cast<size_t>(
          (encoded_posting_count_ * posting_bit_width_ + 7) / 8);
      packed_postings_.assign(
          byte_count + sizeof(uint64_t) - 1, uint8_t{0});
    }

    void append(uint64_t code, Posting posting) {
      if (!slots64_.empty()) {
        auto* slot = find_existing(slots64_, code);
        append_to_slot(*slot, posting);
      } else {
        auto* slot = find_existing(slots32_, code);
        append_to_slot(*slot, posting);
      }
    }

    void finish_filling() {
      if (!slots64_.empty()) {
        finish_filling_slots(slots64_);
      } else {
        finish_filling_slots(slots32_);
      }
    }

    Range find(uint64_t code) const {
      if (!slots64_.empty()) return find_range(slots64_, code);
      return find_range(slots32_, code);
    }

   private:
    struct Slot32 {
      uint64_t code = 0;
      uint32_t begin = 0;
      uint32_t count = 0;
      uint32_t encoded_count = 0;
      bool interval = false;
    };
    struct Slot64 {
      uint64_t code = 0;
      uint64_t begin = 0;
      uint64_t count = 0;
      uint64_t encoded_count = 0;
      bool interval = false;
    };
    static_assert(sizeof(Slot32) == 24,
                  "run-aware posting slot must remain 24 bytes");
    static_assert(sizeof(Slot64) == 40,
                  "wide run-aware posting slot must remain 40 bytes");

    static uint64_t hash_code(uint64_t value) {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31);
    }

    template <typename Slot>
    static size_t slot_index(const std::vector<Slot>& slots,
                             uint64_t code) {
      return static_cast<size_t>(hash_code(code)) & (slots.size() - 1);
    }

    template <typename Slot>
    static Slot* find_existing(std::vector<Slot>& slots,
                               uint64_t code) {
      size_t idx = slot_index(slots, code);
      while (slots[idx].count != 0 && slots[idx].code != code) {
        idx = (idx + 1) & (slots.size() - 1);
      }
      return &slots[idx];
    }

    template <typename Slot>
    static const Slot* find_existing(const std::vector<Slot>& slots,
                                     uint64_t code) {
      if (slots.empty()) return nullptr;
      size_t idx = slot_index(slots, code);
      while (slots[idx].count != 0 && slots[idx].code != code) {
        idx = (idx + 1) & (slots.size() - 1);
      }
      return slots[idx].count == 0 ? nullptr : &slots[idx];
    }

    template <typename Slot>
    void rehash(std::vector<Slot>& slots, size_t capacity) {
      std::vector<Slot> replacement(capacity);
      for (const auto& slot : slots) {
        if (slot.count == 0) continue;
        auto* destination = find_existing(replacement, slot.code);
        *destination = slot;
      }
      slots.swap(replacement);
    }

    template <typename Slot>
    Slot* find_or_insert(std::vector<Slot>& slots, uint64_t code) {
      if (slots.empty()) rehash(slots, 16);
      Slot* slot = find_existing(slots, code);
      if (slot->count != 0) return slot;
      if (unique_code_count_ + 1 >
          slots.size() - slots.size() / 8) {
        rehash(slots, slots.size() * 2);
        slot = find_existing(slots, code);
      }
      slot->code = code;
      ++unique_code_count_;
      return slot;
    }

    template <typename Slot>
    void count_in(std::vector<Slot>& slots, uint64_t code,
                  Posting posting) {
      auto* slot = find_or_insert(slots, code);
      if (slot->count == std::numeric_limits<decltype(slot->count)>::max()) {
        throw std::length_error("range-join posting list is too large");
      }
      if (slot->count == 0 ||
          slot->begin ==
              std::numeric_limits<decltype(slot->begin)>::max() ||
          posting != static_cast<Posting>(slot->begin + 1)) {
        ++slot->encoded_count;
      }
      slot->begin = posting;
      ++slot->count;
    }

    void promote_to_64bit() {
      slots64_.resize(slots32_.size());
      for (const auto& slot : slots32_) {
        if (slot.count == 0) continue;
        auto* destination = find_existing(slots64_, slot.code);
        destination->code = slot.code;
        destination->begin = slot.begin;
        destination->count = slot.count;
        destination->encoded_count = slot.encoded_count;
      }
      std::vector<Slot32>().swap(slots32_);
    }

    template <typename Slot>
    void finish_slots(std::vector<Slot>& slots) {
      uint64_t posting_end = 0;
      for (auto& slot : slots) {
        if (slot.count == 0) continue;
        slot.interval =
            slot.encoded_count < slot.count - slot.encoded_count;
        const uint64_t record_count =
            slot.interval ? slot.encoded_count * 2 : slot.count;
        slot.begin = static_cast<decltype(slot.begin)>(posting_end);
        slot.encoded_count = 0;
        posting_end += record_count;
      }
      if (posting_end > std::numeric_limits<size_t>::max()) {
        throw std::length_error("range-join posting storage is too large");
      }
      encoded_posting_count_ = posting_end;
    }

    template <typename Slot>
    void append_to_slot(Slot& slot, Posting posting) {
      if (slot.interval && slot.encoded_count != 0) {
        const Posting previous = load_posting(slot.begin - 1);
        if (previous != std::numeric_limits<Posting>::max() &&
            posting == static_cast<Posting>(previous + 1)) {
          store_posting(slot.begin - 1, posting);
          return;
        }
      }
      store_posting(slot.begin++, posting);
      ++slot.encoded_count;
      if (slot.interval) {
        store_posting(slot.begin++, posting);
        ++slot.encoded_count;
      }
    }

    template <typename Slot>
    static void finish_filling_slots(std::vector<Slot>& slots) {
      for (auto& slot : slots) {
        if (slot.count != 0) slot.begin -= slot.encoded_count;
      }
    }

    template <typename Slot>
    Range find_range(const std::vector<Slot>& slots,
                     uint64_t code) const {
      const auto* slot = find_existing(slots, code);
      if (!slot) return {};
      const uint64_t begin = slot->begin;
      const uint64_t end = begin + slot->encoded_count;
      return {
          Iterator(packed_postings_.data(), begin, end,
                   posting_bit_width_, slot->interval),
          Iterator(),
          true};
    }

    void store_posting(uint64_t index, Posting posting) {
      const uint64_t bit_offset = index * posting_bit_width_;
      const size_t byte_offset =
          static_cast<size_t>(bit_offset >> 3);
      const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
      uint64_t word = 0;
      std::memcpy(&word, packed_postings_.data() + byte_offset,
                  sizeof(word));
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      word = __builtin_bswap64(word);
#endif
      const uint64_t value_mask =
          posting_bit_width_ == 32
              ? std::numeric_limits<uint32_t>::max()
              : (uint64_t{1} << posting_bit_width_) - 1;
      const uint64_t shifted_mask = value_mask << shift;
      word = (word & ~shifted_mask) |
             ((static_cast<uint64_t>(posting) & value_mask) << shift);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      word = __builtin_bswap64(word);
#endif
      std::memcpy(packed_postings_.data() + byte_offset, &word,
                  sizeof(word));
    }

    Posting load_posting(uint64_t index) const {
      return *Iterator(
          packed_postings_.data(), index, index + 1,
          posting_bit_width_, false);
    }

    std::vector<Slot32> slots32_;
    std::vector<Slot64> slots64_;
    std::vector<uint8_t> packed_postings_;
    uint64_t posting_count_ = 0;
    uint64_t encoded_posting_count_ = 0;
    size_t unique_code_count_ = 0;
    uint8_t posting_bit_width_ = 1;
  };

  using PostingLists16 = CompactPostingLists<uint16_t>;
  using PostingLists = CompactPostingLists<uint32_t>;
  using PositionalPostingLists = CompactPostingLists<uint32_t>;
  using WidePositionalPostingLists =
      std::unordered_map<uint64_t, std::vector<uint64_t>>;

  RangeJoinConfig config_;
  std::vector<RangeJoinItem> owned_items_;
  using ItemStorage = std::variant<
      std::vector<uint32_t>,
      std::vector<std::string_view>,
      std::monostate>;
  // Fixed-length reference sequences are the production/query-side common
  // case, so keep one base pointer plus compact 32-bit offsets in the first
  // variant alternative.
  ItemStorage item_storage_ = std::vector<uint32_t>{};
  // Item IDs are 32-bit throughout the production graph. External view IDs
  // are implicit when they are exactly 0..N-1; owned IDs already live in
  // owned_items_ and need no second per-item table.
  std::vector<RangeJoinItemId> external_item_ids_;
  std::unordered_map<int, PostingLists16> postings16_by_seed_len_;
  std::unordered_map<int, PostingLists> postings_by_seed_len_;
  std::unordered_map<int, PositionalPostingLists>
      positional_postings_by_seed_len_;
  std::unordered_map<int, uint8_t>
      positional_position_bits_by_seed_len_;
  std::unordered_map<int, WidePositionalPostingLists>
      wide_positional_postings_by_seed_len_;
  std::unordered_map<int, std::vector<uint16_t>>
      unindexable_items16_by_seed_len_;
  std::unordered_map<int, std::vector<uint32_t>>
      unindexable_items_by_seed_len_;
  bool seed_index_capacity_ = true;
  bool seed_index_uses_16bit_ = true;
  bool positional_postings_use_32bit_ = true;
  // Uniform mode uses this word for the shared reference base; other modes
  // use it for the minimum item length. This keeps the compact offset mode
  // from increasing the range-index object size.
  uintptr_t item_storage_aux_ = 0;
  size_t max_item_sequence_length_ = 0;
  bool item_ids_strictly_increasing_ = true;
  mutable QGramCountIndex qgram_index_;
  mutable bool qgram_ready_ = false;
  bool defer_qgram_build_ = false;
  bool enable_shifted_window_postings_ = false;
  bool enable_positional_postings_ = false;
  mutable std::shared_ptr<std::mutex> deferred_qgram_mutex_;

  size_t item_count() const;
  RangeJoinItemId item_id(size_t item_idx) const;
  std::string_view item_sequence(size_t item_idx) const;
  size_t min_item_sequence_length() const;
  void reset_after_items_changed();
  bool qgram_bound_is_vacuous(
      std::string_view query_sequence, int tau,
      bool* full_scan_fallback, size_t* query_total) const;
  const QGramCountIndex& ensure_qgram_index() const;
  void prepare_postings_for_seed_len(int seed_len);
  bool query_needs_seed_postings(int seed_len) const;
  RangeJoinQueryResult full_scan(
      std::string_view query_sequence, int tau, bool fallback) const;
  RangeJoinQueryResult pigeonhole_query(
      std::string_view query_sequence, int tau, int block_len, int seed_len,
      size_t early_abort_candidate_limit,
      RangeJoinQueryWorkspace* workspace) const;
  RangeJoinQueryResult qgram_query(
      std::string_view query_sequence, int tau,
      RangeJoinQueryWorkspace* workspace) const;
  RangeJoinQueryResult hybrid_result(
      const RangeJoinQueryResult& pigeonhole,
      const RangeJoinQueryResult& qgram) const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_RANGE_JOIN_HPP
