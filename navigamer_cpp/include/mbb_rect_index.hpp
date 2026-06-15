#ifndef NAVIGAMER_MBB_RECT_INDEX_HPP
#define NAVIGAMER_MBB_RECT_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace navigamer {

class MBBRectIndex {
 public:
  struct Rect {
    uint32_t child_id = 0;
    std::vector<int> lo;
    std::vector<int> hi;
  };

  void build(const std::vector<Rect>& rects);

  std::vector<uint32_t> query_intersect(
      const std::vector<int>& q_lo,
      const std::vector<int>& q_hi) const;

  size_t size() const { return child_ids_.size(); }
  size_t dim() const { return lo_by_dim_.size(); }

 private:
  void clear();

  std::vector<uint32_t> child_ids_;
  std::vector<std::vector<int>> lo_by_dim_;
  std::vector<std::vector<int>> hi_by_dim_;
};

}  // namespace navigamer

#endif  // NAVIGAMER_MBB_RECT_INDEX_HPP
