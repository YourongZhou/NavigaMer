#ifndef NAVIGAMER_PHASE1_SEED_INDEX_HPP
#define NAVIGAMER_PHASE1_SEED_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace navigamer {

struct Phase1SeedIndexConfig {
  int min_seed_len = 8;
  int max_seed_len = 20;
};

struct Phase1SeedQueryResult {
  bool safe = false;
  int block_len = 0;
  int seed_len = 0;
  size_t posting_entries_visited = 0;
  std::vector<size_t> candidate_indices;
};

class IncrementalPigeonholeIndex {
 public:
  explicit IncrementalPigeonholeIndex(Phase1SeedIndexConfig config = {});

  void append(size_t item_id, std::string_view sequence);
  Phase1SeedQueryResult query(std::string_view sequence, int tau);
  size_t size() const { return items_.size(); }

 private:
  template <typename Head>
  class PostingHeadMap {
   public:
    static constexpr Head invalid_head() {
      return std::numeric_limits<Head>::max();
    }

    size_t size() const { return size_; }

    void reserve(size_t expected_size) {
      size_t capacity = 16;
      while (capacity * 7 / 10 < expected_size) {
        if (capacity > std::numeric_limits<size_t>::max() / 2) {
          throw std::length_error("phase1 posting head table is too large");
        }
        capacity *= 2;
      }
      if (capacity > heads_.size()) rehash(capacity);
    }

    Head find(uint64_t key) const {
      if (heads_.empty()) return invalid_head();
      const size_t mask = heads_.size() - 1;
      size_t slot = hash_key(key) & mask;
      while (heads_[slot] != invalid_head()) {
        if (keys_[slot] == key) return heads_[slot];
        slot = (slot + 1) & mask;
      }
      return invalid_head();
    }

    Head& get_or_insert(uint64_t key) {
      if (heads_.empty() || (size_ + 1) * 10 > heads_.size() * 7) {
        rehash(heads_.empty() ? 16 : heads_.size() * 2);
      }
      const size_t mask = heads_.size() - 1;
      size_t slot = hash_key(key) & mask;
      while (heads_[slot] != invalid_head()) {
        if (keys_[slot] == key) return heads_[slot];
        slot = (slot + 1) & mask;
      }
      keys_[slot] = key;
      heads_[slot] = invalid_head();
      size_++;
      return heads_[slot];
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
      for (size_t slot = 0; slot < heads_.size(); ++slot) {
        if (heads_[slot] != invalid_head()) {
          fn(keys_[slot], heads_[slot]);
        }
      }
    }

    void clear_and_release() {
      keys_.clear();
      keys_.shrink_to_fit();
      heads_.clear();
      heads_.shrink_to_fit();
      size_ = 0;
    }

   private:
    static size_t hash_key(uint64_t key) {
      key ^= key >> 30;
      key *= UINT64_C(0xbf58476d1ce4e5b9);
      key ^= key >> 27;
      key *= UINT64_C(0x94d049bb133111eb);
      key ^= key >> 31;
      return static_cast<size_t>(key);
    }

    void rehash(size_t capacity) {
      if (capacity < 16) capacity = 16;
      std::vector<uint64_t> old_keys = std::move(keys_);
      std::vector<Head> old_heads = std::move(heads_);
      keys_.assign(capacity, 0);
      heads_.assign(capacity, invalid_head());
      size_ = 0;
      for (size_t slot = 0; slot < old_heads.size(); ++slot) {
        if (old_heads[slot] == invalid_head()) continue;
        Head& head = get_or_insert(old_keys[slot]);
        head = old_heads[slot];
      }
    }

    std::vector<uint64_t> keys_;
    std::vector<Head> heads_;
    size_t size_ = 0;
  };

  // The normal human-genome path uses seed lengths up to 20, so the 40-bit
  // DNA code and Compact24's 24-bit posting head fit in one exact 64-bit slot.
  // A wider key promotes the whole table to the generic representation.
  class Compact24PostingHeadMap {
   public:
    static constexpr uint32_t invalid_head() { return 0x00ffffffU; }

    size_t size() const { return wide_ ? wide_heads_.size() : size_; }

    void reserve(size_t expected_size) {
      if (wide_) {
        wide_heads_.reserve(expected_size);
        return;
      }
      size_t capacity = 16;
      while (capacity * 7 / 10 < expected_size) {
        if (capacity > std::numeric_limits<size_t>::max() / 2) {
          throw std::length_error("phase1 compact24 head table is too large");
        }
        capacity *= 2;
      }
      if (capacity > slots_.size()) rehash(capacity);
    }

    uint32_t find(uint64_t key) const {
      if (wide_) return wide_heads_.find(key);
      if (key > max_packed_key() || slots_.empty()) return invalid_head();
      const size_t mask = slots_.size() - 1;
      size_t slot_idx = hash_key(key) & mask;
      while (slot_head(slots_[slot_idx]) != invalid_head()) {
        if (slot_key(slots_[slot_idx]) == key) {
          return slot_head(slots_[slot_idx]);
        }
        slot_idx = (slot_idx + 1) & mask;
      }
      return invalid_head();
    }

    uint32_t exchange(uint64_t key, uint32_t new_head) {
      if (new_head >= invalid_head()) {
        throw std::overflow_error("phase1 compact24 posting head overflow");
      }
      if (!wide_ && key > max_packed_key()) promote_to_wide();
      if (wide_) {
        uint32_t& head = wide_heads_.get_or_insert(key);
        const uint32_t previous = head;
        head = new_head;
        return previous == std::numeric_limits<uint32_t>::max()
                   ? invalid_head()
                   : previous;
      }
      if (slots_.empty() || (size_ + 1) * 10 > slots_.size() * 7) {
        if (!slots_.empty() &&
            slots_.size() > std::numeric_limits<size_t>::max() / 2) {
          throw std::length_error("phase1 compact24 head table is too large");
        }
        rehash(slots_.empty() ? 16 : slots_.size() * 2);
      }
      const size_t mask = slots_.size() - 1;
      size_t slot_idx = hash_key(key) & mask;
      while (slot_head(slots_[slot_idx]) != invalid_head()) {
        if (slot_key(slots_[slot_idx]) == key) {
          const uint32_t previous = slot_head(slots_[slot_idx]);
          slots_[slot_idx] = pack_slot(key, new_head);
          return previous;
        }
        slot_idx = (slot_idx + 1) & mask;
      }
      slots_[slot_idx] = pack_slot(key, new_head);
      size_++;
      return invalid_head();
    }

    void set(uint64_t key, uint32_t head) {
      (void)exchange(key, head);
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
      if (wide_) {
        wide_heads_.for_each(std::forward<Fn>(fn));
        return;
      }
      for (uint64_t slot : slots_) {
        const uint32_t head = slot_head(slot);
        if (head != invalid_head()) fn(slot_key(slot), head);
      }
    }

    void clear_and_release() {
      slots_.clear();
      slots_.shrink_to_fit();
      wide_heads_.clear_and_release();
      size_ = 0;
      wide_ = false;
    }

   private:
    static constexpr uint64_t max_packed_key() {
      return (UINT64_C(1) << 40) - 1;
    }
    static constexpr uint64_t head_mask() { return invalid_head(); }
    static uint64_t pack_slot(uint64_t key, uint32_t head) {
      return (key << 24) | head;
    }
    static uint64_t slot_key(uint64_t slot) { return slot >> 24; }
    static uint32_t slot_head(uint64_t slot) {
      return static_cast<uint32_t>(slot & head_mask());
    }
    static size_t hash_key(uint64_t key) {
      key ^= key >> 30;
      key *= UINT64_C(0xbf58476d1ce4e5b9);
      key ^= key >> 27;
      key *= UINT64_C(0x94d049bb133111eb);
      key ^= key >> 31;
      return static_cast<size_t>(key);
    }

    void rehash(size_t capacity) {
      if (capacity < 16) capacity = 16;
      std::vector<uint64_t> old_slots = std::move(slots_);
      slots_.assign(capacity, invalid_head());
      size_ = 0;
      for (uint64_t slot : old_slots) {
        const uint32_t head = slot_head(slot);
        if (head != invalid_head()) set(slot_key(slot), head);
      }
    }

    void promote_to_wide() {
      wide_heads_.reserve(size_);
      for (uint64_t slot : slots_) {
        const uint32_t head = slot_head(slot);
        if (head != invalid_head()) {
          wide_heads_.get_or_insert(slot_key(slot)) = head;
        }
      }
      slots_.clear();
      slots_.shrink_to_fit();
      size_ = 0;
      wide_ = true;
    }

    std::vector<uint64_t> slots_;
    PostingHeadMap<uint32_t> wide_heads_;
    size_t size_ = 0;
    bool wide_ = false;
  };

  struct Item {
    size_t item_id = 0;
    std::string_view sequence;
  };

  struct SeedState {
    enum class PostingStorage : uint8_t {
      Compact16,
      Compact24,
      Packed32,
      Wide,
    };

    struct Compact24PostingEntry {
      uint16_t item_idx = 0;
      uint8_t position = 0;
      uint8_t next_low = 0xff;
      uint8_t next_mid = 0xff;
      uint8_t next_high = 0xff;

      uint32_t next() const {
        return static_cast<uint32_t>(next_low) |
               (static_cast<uint32_t>(next_mid) << 8) |
               (static_cast<uint32_t>(next_high) << 16);
      }
      void set_next(uint32_t value) {
        next_low = static_cast<uint8_t>(value);
        next_mid = static_cast<uint8_t>(value >> 8);
        next_high = static_cast<uint8_t>(value >> 16);
      }
    };
    static_assert(sizeof(Compact24PostingEntry) == 6,
                  "compact phase1 posting entry must remain 6 bytes");

    struct WidePostingEntry {
      uint32_t item_idx = 0;
      uint32_t position = 0;
      uint32_t next = std::numeric_limits<uint32_t>::max();
    };
    static_assert(sizeof(WidePostingEntry) == 12,
                  "wide phase1 posting entry must remain compact");

    size_t indexed_count = 0;
    PostingStorage posting_storage = PostingStorage::Compact16;
    PostingHeadMap<uint16_t> compact_heads;
    std::vector<uint32_t> compact_entries;
    Compact24PostingHeadMap compact24_heads;
    std::vector<Compact24PostingEntry> compact24_entries;
    PostingHeadMap<uint32_t> packed_heads;
    std::vector<uint64_t> packed_entries;
    PostingHeadMap<uint32_t> wide_heads;
    std::vector<WidePostingEntry> wide_entries;
    std::vector<uint32_t> unindexable_items;
  };

  Phase1SeedIndexConfig config_;
  std::vector<Item> items_;
  std::unordered_map<int, SeedState> states_;
  std::vector<uint32_t> seen_epoch_;
  uint32_t epoch_ = 0;

  SeedState& ensure_state(int seed_len);
  static void promote_to_compact24_postings(SeedState& state);
  static void promote_to_packed_postings(SeedState& state);
  static void promote_to_wide_postings(SeedState& state);
  void index_item(SeedState& state, int seed_len, uint32_t item_idx);
  void begin_query();
};

}  // namespace navigamer

#endif  // NAVIGAMER_PHASE1_SEED_INDEX_HPP
