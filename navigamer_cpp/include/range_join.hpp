#ifndef NAVIGAMER_RANGE_JOIN_HPP
#define NAVIGAMER_RANGE_JOIN_HPP

#include "qgram_filter.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
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
  using PostingLists16 =
      std::unordered_map<uint64_t, std::vector<uint16_t>>;
  using PostingLists = std::unordered_map<uint64_t, std::vector<uint32_t>>;
  using PositionalPostingLists =
      std::unordered_map<uint64_t, std::vector<uint32_t>>;
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
