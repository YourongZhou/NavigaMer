#include "simd_mbb_filter.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> reference_survivors(
    const std::vector<int32_t>& lo,
    const std::vector<int32_t>& hi,
    size_t child_count,
    size_t dim,
    const std::vector<int32_t>& query,
    int32_t tolerance) {
  std::vector<uint32_t> out;
  for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * child_count + child_idx;
      if (hi[flat] < query[dim_idx] - tolerance ||
          lo[flat] > query[dim_idx] + tolerance) {
        ok = false;
        break;
      }
    }
    if (ok) out.push_back(static_cast<uint32_t>(child_idx));
  }
  return out;
}

void assert_random_equivalence() {
  std::mt19937 rng(20260616);
  std::uniform_int_distribution<int32_t> pick_center(0, 200);
  std::uniform_int_distribution<int32_t> pick_width(0, 20);
  std::uniform_int_distribution<int32_t> pick_query(0, 200);
  const std::vector<size_t> dims = {1, 2, 4, 8, 16, 32};
  const std::vector<size_t> child_counts = {1, 7, 8, 9, 31, 64, 1000};

  for (size_t dim : dims) {
    for (size_t child_count : child_counts) {
      std::vector<int32_t> lo(dim * child_count);
      std::vector<int32_t> hi(dim * child_count);
      std::vector<int32_t> query(dim);
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        query[dim_idx] = pick_query(rng);
        for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
          int32_t center = pick_center(rng);
          int32_t width = pick_width(rng);
          const size_t flat = dim_idx * child_count + child_idx;
          lo[flat] = center - width;
          hi[flat] = center + width;
        }
      }
      for (int32_t tolerance : {0, 1, 3, 7, 15}) {
        auto expected =
            reference_survivors(lo, hi, child_count, dim, query, tolerance);

        navigamer::MBBFilterSimdStats scalar_stats;
        auto scalar = navigamer::filter_mbb_survivors(
            lo.data(), hi.data(), child_count, dim, query.data(), tolerance,
            navigamer::SimdMode::Scalar, &scalar_stats);
        assert(scalar == expected);
        assert(scalar_stats.scalar_checks == child_count);

        navigamer::MBBFilterSimdStats auto_stats;
        auto automatic = navigamer::filter_mbb_survivors(
            lo.data(), hi.data(), child_count, dim, query.data(), tolerance,
            navigamer::SimdMode::Auto, &auto_stats);
        assert(automatic == expected);

        navigamer::MBBFilterSimdStats avx2_stats;
        auto avx2 = navigamer::filter_mbb_survivors(
            lo.data(), hi.data(), child_count, dim, query.data(), tolerance,
            navigamer::SimdMode::AVX2, &avx2_stats);
        assert(avx2 == expected);
      }
    }
  }
}

void assert_mode_parsing() {
  assert(navigamer::parse_simd_mode("auto") == navigamer::SimdMode::Auto);
  assert(navigamer::parse_simd_mode("scalar") == navigamer::SimdMode::Scalar);
  assert(navigamer::parse_simd_mode("avx2") == navigamer::SimdMode::AVX2);
  assert(navigamer::parse_simd_mode("avx512") == navigamer::SimdMode::AVX512);
  assert(std::string(navigamer::simd_mode_name(navigamer::SimdMode::Auto)) ==
         "auto");
  assert(std::string(navigamer::simd_mode_name(navigamer::SimdMode::Scalar)) ==
         "scalar");
}

}  // namespace

int main() {
  assert_mode_parsing();
  assert_random_equivalence();
  std::cout << "SIMD MBB filter tests passed\n";
  return 0;
}

