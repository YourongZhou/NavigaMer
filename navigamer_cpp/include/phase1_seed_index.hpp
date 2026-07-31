#ifndef NAVIGAMER_PHASE1_SEED_INDEX_HPP
#define NAVIGAMER_PHASE1_SEED_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
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
  struct Item {
    size_t item_id = 0;
    std::string_view sequence;
  };

  struct SeedState {
    enum class PostingStorage : uint8_t {
      Compact16,
      Packed32,
      Wide,
    };

    struct WidePostingEntry {
      uint32_t item_idx = 0;
      uint32_t position = 0;
      uint32_t next = std::numeric_limits<uint32_t>::max();
    };
    static_assert(sizeof(WidePostingEntry) == 12,
                  "wide phase1 posting entry must remain compact");

    size_t indexed_count = 0;
    PostingStorage posting_storage = PostingStorage::Compact16;
    std::unordered_map<uint64_t, uint16_t> compact_heads;
    std::vector<uint32_t> compact_entries;
    std::unordered_map<uint64_t, uint32_t> packed_heads;
    std::vector<uint64_t> packed_entries;
    std::unordered_map<uint64_t, uint32_t> wide_heads;
    std::vector<WidePostingEntry> wide_entries;
    std::vector<uint32_t> unindexable_items;
  };

  Phase1SeedIndexConfig config_;
  std::vector<Item> items_;
  std::unordered_map<int, SeedState> states_;
  std::vector<uint32_t> seen_epoch_;
  uint32_t epoch_ = 0;

  SeedState& ensure_state(int seed_len);
  static void promote_to_packed_postings(SeedState& state);
  static void promote_to_wide_postings(SeedState& state);
  void index_item(SeedState& state, int seed_len, uint32_t item_idx);
  void begin_query();
};

}  // namespace navigamer

#endif  // NAVIGAMER_PHASE1_SEED_INDEX_HPP
