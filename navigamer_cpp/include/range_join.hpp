#ifndef NAVIGAMER_RANGE_JOIN_HPP
#define NAVIGAMER_RANGE_JOIN_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace navigamer {

struct RangeJoinConfig {
  int min_seed_len = 8;
  int max_seed_len = 20;
};

struct RangeJoinItem {
  size_t item_id = 0;
  std::string sequence;
};

struct RangeJoinQueryResult {
  std::vector<size_t> candidate_item_ids;
  bool used_full_scan = false;
  int block_len = 0;
  int seed_len = 0;
};

class ExactRangeJoinIndex {
 public:
  explicit ExactRangeJoinIndex(RangeJoinConfig config = {});

  void build(const std::vector<RangeJoinItem>& items);
  RangeJoinQueryResult query(const std::string& query_sequence, int tau);

 private:
  using PostingLists = std::unordered_map<std::string, std::vector<size_t>>;

  RangeJoinConfig config_;
  std::vector<RangeJoinItem> items_;
  std::unordered_map<int, PostingLists> postings_by_seed_len_;

  const PostingLists& postings_for_seed_len(int seed_len);
};

}  // namespace navigamer

#endif  // NAVIGAMER_RANGE_JOIN_HPP
