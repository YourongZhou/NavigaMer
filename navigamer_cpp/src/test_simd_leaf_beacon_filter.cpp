#include "simd_mbb_filter.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<uint32_t> reference_survivors(
    const std::vector<int32_t>& dist_by_dim,
    size_t leaf_count,
    size_t dim,
    const std::vector<int32_t>& query,
    int32_t tolerance) {
  std::vector<uint32_t> out;
  for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * leaf_count + leaf_idx;
      if (std::abs(query[dim_idx] - dist_by_dim[flat]) > tolerance) {
        ok = false;
        break;
      }
    }
    if (ok) out.push_back(static_cast<uint32_t>(leaf_idx));
  }
  return out;
}

void assert_random_equivalence() {
  std::mt19937 rng(20260616);
  std::uniform_int_distribution<int32_t> pick_dist(0, 200);
  const std::vector<size_t> dims = {1, 2, 4, 8, 16, 32};
  const std::vector<size_t> leaf_counts = {1, 7, 8, 9, 31, 64, 1000};

  for (size_t dim : dims) {
    for (size_t leaf_count : leaf_counts) {
      std::vector<int32_t> dist_by_dim(dim * leaf_count);
      std::vector<int32_t> query(dim);
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        query[dim_idx] = pick_dist(rng);
        for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
          dist_by_dim[dim_idx * leaf_count + leaf_idx] = pick_dist(rng);
        }
      }

      for (int32_t tolerance : {0, 1, 3, 7, 15}) {
        auto expected =
            reference_survivors(dist_by_dim, leaf_count, dim, query, tolerance);

        navigamer::LeafBeaconFilterSimdStats scalar_stats;
        auto scalar = navigamer::filter_leaf_beacon_survivors(
            dist_by_dim.data(), leaf_count, dim, query.data(), tolerance,
            navigamer::SimdMode::Scalar, &scalar_stats);
        assert(scalar == expected);
        assert(scalar_stats.scalar_checks == leaf_count);

        navigamer::LeafBeaconFilterSimdStats auto_stats;
        auto automatic = navigamer::filter_leaf_beacon_survivors(
            dist_by_dim.data(), leaf_count, dim, query.data(), tolerance,
            navigamer::SimdMode::Auto, &auto_stats);
        assert(automatic == expected);

        navigamer::LeafBeaconFilterSimdStats avx2_stats;
        auto avx2 = navigamer::filter_leaf_beacon_survivors(
            dist_by_dim.data(), leaf_count, dim, query.data(), tolerance,
            navigamer::SimdMode::AVX2, &avx2_stats);
        assert(avx2 == expected);
      }
    }
  }
}

}  // namespace

int main() {
  assert_random_equivalence();
  std::cout << "SIMD leaf beacon filter tests passed\n";
  return 0;
}
