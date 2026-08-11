#include "simd_mbb_filter.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<uint32_t> reference_survivors(
    const std::vector<uint8_t>& dist_by_dim,
    size_t leaf_count,
    size_t dim,
    const std::vector<int>& query,
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

std::vector<uint8_t> pack_distances(
    const std::vector<uint8_t>& distances, uint32_t bits) {
  std::vector<uint8_t> packed(
      (distances.size() * bits + 7) / 8, 0);
  for (size_t idx = 0; idx < distances.size(); ++idx) {
    const size_t bit_offset = idx * bits;
    const size_t byte_offset = bit_offset >> 3;
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    packed[byte_offset] |=
        static_cast<uint8_t>(distances[idx] << shift);
    if (shift + bits > 8) {
      packed[byte_offset + 1] |=
          static_cast<uint8_t>(distances[idx] >> (8 - shift));
    }
  }
  return packed;
}

void assert_random_equivalence() {
  std::mt19937 rng(20260616);
  std::uniform_int_distribution<int32_t> pick_dist(0, 255);
  std::uniform_int_distribution<int32_t> pick_query(0, 300);
  const std::vector<size_t> dims = {1, 2, 4, 8, 16, 32};
  const std::vector<size_t> leaf_counts = {
      1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 1000};

  for (size_t dim : dims) {
    for (size_t leaf_count : leaf_counts) {
      std::vector<uint8_t> dist_by_dim(dim * leaf_count);
      std::vector<int> query(dim);
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        query[dim_idx] = pick_query(rng);
        for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
          dist_by_dim[dim_idx * leaf_count + leaf_idx] = pick_dist(rng);
        }
      }

      for (int32_t tolerance : {0, 1, 3, 15, 5000}) {
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

void assert_packed_random_equivalence() {
  std::mt19937 rng(20260801);
  std::uniform_int_distribution<int32_t> pick_query(0, 300);
  const std::vector<size_t> dims = {1, 2, 4, 9};
  const std::vector<size_t> leaf_counts = {
      1, 2, 3, 7, 8, 9, 17, 33, 100};

  for (uint32_t bits = 1; bits < 8; ++bits) {
    const uint32_t mask = (uint32_t{1} << bits) - 1;
    for (size_t dim : dims) {
      for (size_t leaf_count : leaf_counts) {
        std::vector<uint8_t> distances(dim * leaf_count);
        std::vector<int> query(dim);
        for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
          query[dim_idx] = pick_query(rng);
          for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
            distances[dim_idx * leaf_count + leaf_idx] =
                static_cast<uint8_t>(rng() & mask);
          }
        }
        const auto packed = pack_distances(distances, bits);
        for (int32_t tolerance : {0, 1, 3, 15, 5000}) {
          const auto expected = reference_survivors(
              distances, leaf_count, dim, query, tolerance);
          for (navigamer::SimdMode mode : {
                   navigamer::SimdMode::Scalar,
                   navigamer::SimdMode::Auto,
                   navigamer::SimdMode::AVX2}) {
            navigamer::LeafBeaconFilterSimdStats stats;
            const auto actual =
                navigamer::filter_leaf_beacon_survivors(
                    packed.data(), leaf_count, dim, query.data(),
                    tolerance, mode, &stats, bits);
            assert(actual == expected);
            assert(stats.scalar_checks == leaf_count);
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  assert_random_equivalence();
  assert_packed_random_equivalence();
  std::cout << "SIMD leaf beacon filter tests passed\n";
  return 0;
}
