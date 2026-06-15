#include "mbb_rect_index.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using navigamer::MBBRectIndex;

std::vector<uint32_t> sorted(std::vector<uint32_t> values) {
  std::sort(values.begin(), values.end());
  return values;
}

std::vector<uint32_t> naive_intersect(
    const std::vector<MBBRectIndex::Rect>& rects,
    const std::vector<int>& q_lo,
    const std::vector<int>& q_hi) {
  std::vector<uint32_t> out;
  for (const auto& rect : rects) {
    bool intersects = rect.lo.size() == q_lo.size() &&
                      rect.hi.size() == q_hi.size();
    for (size_t dim = 0; intersects && dim < q_lo.size(); ++dim) {
      intersects = rect.hi[dim] >= q_lo[dim] && rect.lo[dim] <= q_hi[dim];
    }
    if (intersects) out.push_back(rect.child_id);
  }
  return sorted(std::move(out));
}

void test_basic() {
  MBBRectIndex index;
  index.build({
      {10, {0, 0}, {3, 3}},
      {20, {4, 4}, {8, 8}},
      {30, {2, 7}, {5, 10}},
  });
  assert(index.size() == 3);
  assert(index.dim() == 2);
  assert(sorted(index.query_intersect({2, 2}, {4, 4})) ==
         std::vector<uint32_t>({10, 20}));
  assert(sorted(index.query_intersect({3, 7}, {3, 7})) ==
         std::vector<uint32_t>({30}));

  MBBRectIndex index3d;
  index3d.build({
      {1, {0, 0, 0}, {2, 2, 2}},
      {2, {2, 2, 3}, {4, 4, 5}},
  });
  assert(index3d.query_intersect({2, 2, 2}, {2, 2, 2}) ==
         std::vector<uint32_t>({1}));
}

void test_random_equivalence() {
  std::mt19937 gen(12345);
  std::uniform_int_distribution<int> coord(-100, 100);
  std::uniform_int_distribution<int> width(0, 30);

  for (size_t dim : {size_t(1), size_t(2), size_t(4), size_t(8), size_t(16)}) {
    std::vector<MBBRectIndex::Rect> rects;
    for (uint32_t id = 0; id < 250; ++id) {
      MBBRectIndex::Rect rect;
      rect.child_id = id;
      for (size_t d = 0; d < dim; ++d) {
        int lo = coord(gen);
        rect.lo.push_back(lo);
        rect.hi.push_back(lo + width(gen));
      }
      rects.push_back(std::move(rect));
    }

    MBBRectIndex index;
    index.build(rects);
    assert(index.size() == rects.size());
    assert(index.dim() == dim);
    for (int query = 0; query < 200; ++query) {
      std::vector<int> q_lo;
      std::vector<int> q_hi;
      for (size_t d = 0; d < dim; ++d) {
        int lo = coord(gen);
        q_lo.push_back(lo);
        q_hi.push_back(lo + width(gen));
      }
      assert(sorted(index.query_intersect(q_lo, q_hi)) ==
             naive_intersect(rects, q_lo, q_hi));
    }
  }
}

void test_invalid_input() {
  MBBRectIndex index;
  index.build({});
  assert(index.size() == 0);
  assert(index.dim() == 0);

  index.build({{1, {}, {}}});
  assert(index.size() == 1);
  assert(index.dim() == 0);
  assert(index.query_intersect({}, {}) == std::vector<uint32_t>({1}));

  index.build({{1, {0, 0}, {1}}});
  assert(index.size() == 0);

  index.build({{1, {2}, {1}}});
  assert(index.size() == 0);

  index.build({{1, {0}, {1}}, {2, {0, 0}, {1, 1}}});
  assert(index.size() == 0);

  index.build({{1, {0, 0}, {1, 1}}});
  assert(index.query_intersect({0}, {1}).empty());
  assert(index.query_intersect({2, 0}, {1, 1}).empty());
}

}  // namespace

int main() {
  test_basic();
  test_random_equivalence();
  test_invalid_input();
  std::cout << "MBB rectangle index tests passed\n";
  return 0;
}
