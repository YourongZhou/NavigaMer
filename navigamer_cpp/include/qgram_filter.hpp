#ifndef NAVIGAMER_QGRAM_FILTER_HPP
#define NAVIGAMER_QGRAM_FILTER_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace navigamer {

using QGramCounts = std::unordered_map<std::string, size_t>;

QGramCounts compute_qgram_counts(const std::string& sequence, int q);
size_t qgram_total(const std::string& sequence, int q);
size_t compute_qgram_l1(
    const std::string& lhs, const std::string& rhs, int q);

class QGramCountIndex {
 public:
  struct Item {
    size_t item_id = 0;
    std::string sequence;
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
  std::vector<size_t> query(
      const std::string& query_sequence, int tau,
      QueryStats* stats = nullptr) const;

  int q() const { return q_; }
  size_t size() const { return items_.size(); }

 private:
  struct StoredItem {
    size_t item_id = 0;
    size_t sequence_length = 0;
    size_t total_qgrams = 0;
  };

  struct Posting {
    size_t internal_idx = 0;
    size_t count = 0;
  };

  int q_;
  std::vector<StoredItem> items_;
  std::unordered_map<std::string, std::vector<Posting>> postings_;
};

}  // namespace navigamer

#endif  // NAVIGAMER_QGRAM_FILTER_HPP
