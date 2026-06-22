#ifndef NAVIGAMER_PHASE1_SEED_INDEX_HPP
#define NAVIGAMER_PHASE1_SEED_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <string>
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

  void append(size_t item_id, const std::string& sequence);
  Phase1SeedQueryResult query(const std::string& sequence, int tau);
  size_t size() const { return items_.size(); }

 private:
  struct Item {
    size_t item_id = 0;
    std::string sequence;
  };

  struct SeedState {
    struct Posting {
      uint32_t item_idx = 0;
      uint32_t position = 0;
    };

    size_t indexed_count = 0;
    std::unordered_map<uint64_t, std::vector<Posting>> postings;
    std::vector<uint32_t> unindexable_items;
  };

  Phase1SeedIndexConfig config_;
  std::vector<Item> items_;
  std::unordered_map<int, SeedState> states_;
  std::vector<uint32_t> seen_epoch_;
  uint32_t epoch_ = 0;

  SeedState& ensure_state(int seed_len);
  void index_item(SeedState& state, int seed_len, uint32_t item_idx);
  void begin_query();
};

}  // namespace navigamer

#endif  // NAVIGAMER_PHASE1_SEED_INDEX_HPP
