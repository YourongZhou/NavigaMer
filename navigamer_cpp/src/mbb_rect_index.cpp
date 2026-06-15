#include "mbb_rect_index.hpp"

#include <numeric>

namespace navigamer {

void MBBRectIndex::clear() {
  child_ids_.clear();
  lo_by_dim_.clear();
  hi_by_dim_.clear();
}

void MBBRectIndex::build(const std::vector<Rect>& rects) {
  clear();
  if (rects.empty()) return;

  const size_t dimensions = rects.front().lo.size();
  if (rects.front().hi.size() != dimensions) return;

  for (const auto& rect : rects) {
    if (rect.lo.size() != dimensions || rect.hi.size() != dimensions) return;
    for (size_t dim = 0; dim < dimensions; ++dim) {
      if (rect.lo[dim] > rect.hi[dim]) return;
    }
  }

  child_ids_.reserve(rects.size());
  lo_by_dim_.assign(dimensions, {});
  hi_by_dim_.assign(dimensions, {});
  for (size_t dim = 0; dim < dimensions; ++dim) {
    lo_by_dim_[dim].reserve(rects.size());
    hi_by_dim_[dim].reserve(rects.size());
  }

  for (const auto& rect : rects) {
    child_ids_.push_back(rect.child_id);
    for (size_t dim = 0; dim < dimensions; ++dim) {
      lo_by_dim_[dim].push_back(rect.lo[dim]);
      hi_by_dim_[dim].push_back(rect.hi[dim]);
    }
  }
}

std::vector<uint32_t> MBBRectIndex::query_intersect(
    const std::vector<int>& q_lo,
    const std::vector<int>& q_hi) const {
  std::vector<uint32_t> out;
  if (size() == 0 || q_lo.size() != dim() || q_hi.size() != dim()) {
    return out;
  }
  if (dim() == 0) return child_ids_;
  for (size_t dim_idx = 0; dim_idx < dim(); ++dim_idx) {
    if (q_lo[dim_idx] > q_hi[dim_idx]) return out;
  }

  std::vector<size_t> candidates(size());
  std::iota(candidates.begin(), candidates.end(), size_t{0});
  for (size_t dim_idx = 0; dim_idx < dim(); ++dim_idx) {
    size_t write_idx = 0;
    for (size_t rect_idx : candidates) {
      if (hi_by_dim_[dim_idx][rect_idx] >= q_lo[dim_idx] &&
          lo_by_dim_[dim_idx][rect_idx] <= q_hi[dim_idx]) {
        candidates[write_idx++] = rect_idx;
      }
    }
    candidates.resize(write_idx);
    if (candidates.empty()) break;
  }

  out.reserve(candidates.size());
  for (size_t rect_idx : candidates) out.push_back(child_ids_[rect_idx]);
  return out;
}

}  // namespace navigamer
