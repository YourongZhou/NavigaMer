#ifndef NAVIGAMER_RANGE_JOIN_HPP
#define NAVIGAMER_RANGE_JOIN_HPP

#include "qgram_filter.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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
  size_t auto_pigeonhole_max_candidates = 4096;
  double auto_pigeonhole_max_ratio = 0.25;
  bool auto_hybrid_on_large_candidates = true;
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
};

class ExactRangeJoinIndex {
 public:
  explicit ExactRangeJoinIndex(
      RangeJoinConfig config = {}, bool defer_qgram_build = false);

  void build(const std::vector<RangeJoinItem>& items);
  void prepare_qgram();
  void prepare_seed_lengths(const std::vector<int>& seed_lengths);
  RangeJoinQueryResult query(const std::string& query_sequence, int tau);
  RangeJoinQueryResult query(
      const std::string& query_sequence, int tau,
      RangeJoinQueryWorkspace* workspace) const;

 private:
  using PostingLists = std::unordered_map<uint64_t, std::vector<size_t>>;

  RangeJoinConfig config_;
  std::vector<RangeJoinItem> items_;
  std::unordered_map<size_t, size_t> item_lengths_by_id_;
  std::unordered_map<int, PostingLists> postings_by_seed_len_;
  std::unordered_map<int, std::vector<size_t>>
      unindexable_items_by_seed_len_;
  mutable QGramCountIndex qgram_index_;
  mutable bool qgram_ready_ = false;
  bool defer_qgram_build_ = false;
  mutable std::shared_ptr<std::mutex> deferred_qgram_mutex_;

  const QGramCountIndex& ensure_qgram_index() const;
  const PostingLists& postings_for_seed_len(int seed_len);
  const PostingLists* find_postings_for_seed_len(int seed_len) const;
  bool query_needs_seed_postings(int seed_len) const;
  RangeJoinQueryResult full_scan(
      const std::string& query_sequence, int tau, bool fallback) const;
  RangeJoinQueryResult pigeonhole_query(
      const std::string& query_sequence, int tau, int block_len, int seed_len,
      size_t early_abort_candidate_limit) const;
  RangeJoinQueryResult qgram_query(
      const std::string& query_sequence, int tau,
      RangeJoinQueryWorkspace* workspace) const;
  RangeJoinQueryResult hybrid_result(
      const RangeJoinQueryResult& pigeonhole,
      const RangeJoinQueryResult& qgram) const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_RANGE_JOIN_HPP
