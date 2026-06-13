#ifndef NAVIGAMER_RANGE_JOIN_HPP
#define NAVIGAMER_RANGE_JOIN_HPP

#include "qgram_filter.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace navigamer {

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
  int min_seed_len = 8;
  int max_seed_len = 20;
  int qgram_q = 5;
  RangeCandidateMode candidate_mode = RangeCandidateMode::Auto;
};

struct RangeJoinItem {
  size_t item_id = 0;
  std::string sequence;
};

struct RangeJoinQueryResult {
  std::vector<size_t> candidate_item_ids;
  bool used_full_scan = false;
  RangeCandidateMode mode_used = RangeCandidateMode::FullScan;
  int block_len = 0;
  int seed_len = 0;
  size_t length_filtered_items = 0;
  size_t qgram_candidate_count = 0;
  size_t qgram_pruned_by_l1 = 0;
  size_t required_shared_nonpositive = 0;
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
  QGramCountIndex qgram_index_;

  const PostingLists& postings_for_seed_len(int seed_len);
  RangeJoinQueryResult full_scan(
      const std::string& query_sequence, int tau, bool fallback) const;
  RangeJoinQueryResult pigeonhole_query(
      const std::string& query_sequence, int tau, int block_len, int seed_len);
  RangeJoinQueryResult qgram_query(
      const std::string& query_sequence, int tau) const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_RANGE_JOIN_HPP
