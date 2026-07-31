#ifndef NAVIGAMER_QGRAM_FILTER_HPP
#define NAVIGAMER_QGRAM_FILTER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace navigamer {

using QGramCounts = std::unordered_map<std::string, size_t>;

QGramCounts compute_qgram_counts(std::string_view sequence, int q);
size_t qgram_total(std::string_view sequence, int q);
size_t compute_qgram_l1(
    std::string_view lhs, std::string_view rhs, int q);

struct QGramEntry {
  uint64_t code = 0;
  uint32_t count = 0;
};

struct QGramSignature {
  int q = 0;
  size_t sequence_length = 0;
  size_t total_qgrams = 0;
  bool safe_for_pruning = false;
  std::vector<QGramEntry> entries;
};

QGramSignature compute_qgram_signature(std::string_view sequence, int q);
size_t qgram_l1_distance(
    const QGramSignature& lhs, const QGramSignature& rhs);
bool qgram_can_prune_edit_distance(
    const QGramSignature& lhs, const QGramSignature& rhs, int tau);

struct QGramQueryWorkspace {
  std::vector<size_t> shared;
  std::vector<uint32_t> seen_epoch;
  std::vector<size_t> touched;
  uint32_t epoch = 1;

  void reset(size_t item_count);
};

class QGramCountIndex {
 public:
  struct Item {
    size_t item_id = 0;
    std::string sequence;
  };

  struct ItemView {
    size_t item_id = 0;
    std::string_view sequence;
  };

  struct QueryStats {
    size_t total_items = 0;
    size_t length_filtered_items = 0;
    size_t qgram_candidates = 0;
    size_t full_scan_fallbacks = 0;
    size_t required_shared_nonpositive = 0;
    size_t pruned_by_l1 = 0;
  };

  explicit QGramCountIndex(int q = 5);

  void build(const std::vector<Item>& items);
  void build_views(const std::vector<ItemView>& items);
  std::vector<size_t> query(
      std::string_view query_sequence, int tau,
      QueryStats* stats = nullptr,
      QGramQueryWorkspace* workspace = nullptr) const;

  int q() const { return q_; }
  size_t size() const { return items_.size(); }

 private:
  struct StoredItem {
    size_t item_id = 0;
    size_t sequence_length = 0;
    size_t total_qgrams = 0;
    bool qgram_indexable = false;
  };

  struct Posting {
    uint32_t internal_idx = 0;
    uint32_t count = 0;
  };

  int q_;
  std::vector<StoredItem> items_;
  std::unordered_map<uint64_t, std::vector<Posting>> postings_;
  std::vector<std::vector<Posting>> dense_postings_;
  std::vector<std::vector<uint32_t>> dense_packed_postings_;
  bool item_ids_strictly_increasing_ = true;
};

}  // namespace navigamer

#endif  // NAVIGAMER_QGRAM_FILTER_HPP
