#include "simd_mbb_filter.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> reference_survivors(
    const std::vector<uint8_t>& center_dist,
    size_t child_count,
    size_t dim,
    const std::vector<int>& query,
    int32_t child_radius,
    int32_t tolerance) {
  std::vector<uint32_t> out;
  for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * child_count + child_idx;
      if (std::abs(
              static_cast<int>(center_dist[flat]) -
              query[dim_idx]) >
          child_radius + tolerance) {
        ok = false;
        break;
      }
    }
    if (ok) out.push_back(static_cast<uint32_t>(child_idx));
  }
  return out;
}

std::vector<uint8_t> pack_distances(
    const std::vector<uint8_t>& values, uint32_t bits) {
  std::vector<uint8_t> packed(
      (values.size() * bits + 7) / 8, 0);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    const size_t bit_offset = idx * bits;
    const size_t byte_offset = bit_offset >> 3;
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    packed[byte_offset] |=
        static_cast<uint8_t>(values[idx] << shift);
    if (shift + bits > 8) {
      packed[byte_offset + 1] |=
          static_cast<uint8_t>(values[idx] >> (8 - shift));
    }
  }
  return packed;
}

std::vector<uint8_t> pack_paired_base11(
    const std::vector<uint8_t>& values,
    size_t child_count,
    size_t dim) {
  const size_t pairs_per_child = (dim + 1) / 2;
  std::vector<uint8_t> codes(child_count * pairs_per_child, 0);
  for (size_t child = 0; child < child_count; ++child) {
    for (size_t pair_dim = 0; pair_dim < pairs_per_child; ++pair_dim) {
      const size_t first_dim = pair_dim * 2;
      const uint8_t first = values[first_dim * child_count + child];
      const uint8_t second =
          first_dim + 1 < dim
              ? values[(first_dim + 1) * child_count + child]
              : 0;
      codes[child * pairs_per_child + pair_dim] =
          static_cast<uint8_t>(first + 11 * second);
    }
  }
  return pack_distances(codes, 7);
}

void assert_random_equivalence() {
  std::mt19937 rng(20260616);
  std::uniform_int_distribution<int32_t> pick_center(20, 235);
  std::uniform_int_distribution<int32_t> pick_radius(0, 20);
  std::uniform_int_distribution<int32_t> pick_query(0, 300);
  const std::vector<size_t> dims = {1, 2, 4, 8, 16, 32};
  const std::vector<size_t> child_counts = {
      1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 1000};

  for (size_t dim : dims) {
    for (size_t child_count : child_counts) {
      std::vector<uint8_t> center_dist(dim * child_count);
      std::vector<int> query(dim);
      const int32_t child_radius = pick_radius(rng);
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        query[dim_idx] = pick_query(rng);
        for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
          const size_t flat = dim_idx * child_count + child_idx;
          center_dist[flat] =
              static_cast<uint8_t>(pick_center(rng));
        }
      }
      for (int32_t tolerance : {0, 1, 3, 15, 5000}) {
        auto expected =
            reference_survivors(
                center_dist, child_count, dim, query,
                child_radius, tolerance);

        navigamer::MBBFilterSimdStats scalar_stats;
        auto scalar = navigamer::filter_mbb_survivors(
            center_dist.data(), child_count, dim, query.data(),
            child_radius, tolerance,
            navigamer::SimdMode::Scalar, &scalar_stats);
        assert(scalar == expected);
        assert(scalar_stats.scalar_checks == child_count);

        navigamer::MBBFilterSimdStats auto_stats;
        auto automatic = navigamer::filter_mbb_survivors(
            center_dist.data(), child_count, dim, query.data(),
            child_radius, tolerance,
            navigamer::SimdMode::Auto, &auto_stats);
        assert(automatic == expected);

        navigamer::MBBFilterSimdStats avx2_stats;
        auto avx2 = navigamer::filter_mbb_survivors(
            center_dist.data(), child_count, dim, query.data(),
            child_radius, tolerance,
            navigamer::SimdMode::AVX2, &avx2_stats);
        assert(avx2 == expected);
      }
    }
  }
}

void assert_packed_equivalence() {
  std::mt19937 rng(20260801);
  for (uint32_t bits = 1; bits < 8; ++bits) {
    std::uniform_int_distribution<int> pick_distance(
        0, (1 << bits) - 1);
    for (size_t dim : {size_t{1}, size_t{3}, size_t{12}}) {
      for (size_t child_count :
           {size_t{1}, size_t{7}, size_t{32}, size_t{257}}) {
        std::vector<uint8_t> distances(dim * child_count);
        for (auto& distance : distances) {
          distance = static_cast<uint8_t>(pick_distance(rng));
        }
        const auto packed = pack_distances(distances, bits);
        std::vector<int> query(dim);
        for (auto& distance : query) distance = pick_distance(rng);
        for (int32_t tolerance : {0, 2, 9}) {
          const auto expected = reference_survivors(
              distances, child_count, dim, query, 3, tolerance);
          navigamer::MBBFilterSimdStats stats;
          const auto actual = navigamer::filter_mbb_survivors(
              packed.data(), child_count, dim, query.data(), 3,
              tolerance, navigamer::SimdMode::Auto, &stats, bits);
          assert(actual == expected);
          assert(stats.scalar_checks == child_count);
        }
      }
    }
  }
}

void assert_quantized_filter_has_no_false_negatives() {
  std::mt19937 rng(20260803);
  std::uniform_int_distribution<int> pick_distance(0, 255);
  for (uint32_t bin_width :
       {uint32_t{4}, uint32_t{6}, uint32_t{8}}) {
    uint32_t maximum_encoded = 255 / bin_width;
    uint32_t bits = 1;
    while ((maximum_encoded >>= 1) != 0) ++bits;
    for (size_t dim : {size_t{1}, size_t{4}, size_t{10}}) {
      for (size_t child_count : {size_t{1}, size_t{31}, size_t{257}}) {
        std::vector<uint8_t> exact(dim * child_count);
        std::vector<uint8_t> encoded(dim * child_count);
        for (size_t idx = 0; idx < exact.size(); ++idx) {
          exact[idx] = static_cast<uint8_t>(pick_distance(rng));
          encoded[idx] =
              static_cast<uint8_t>(exact[idx] / bin_width);
        }
        const auto packed = pack_distances(encoded, bits);
        std::vector<int> query(dim);
        for (auto& distance : query) distance = pick_distance(rng);
        for (int32_t tolerance : {0, 1, 5, 20}) {
          const auto expected = reference_survivors(
              exact, child_count, dim, query, 7, tolerance);
          const auto actual = navigamer::filter_mbb_survivors(
              packed.data(), child_count, dim, query.data(), 7,
              tolerance, navigamer::SimdMode::Auto, nullptr, bits,
              bin_width);
          assert(std::includes(
              actual.begin(), actual.end(), expected.begin(), expected.end()));
        }
      }
    }

    std::vector<uint8_t> exact(256);
    std::vector<uint8_t> encoded(256);
    for (size_t distance = 0; distance < exact.size(); ++distance) {
      exact[distance] = static_cast<uint8_t>(distance);
      encoded[distance] =
          static_cast<uint8_t>(distance / bin_width);
    }
    const auto packed = pack_distances(encoded, bits);
    for (int query_distance = 0; query_distance <= 255; ++query_distance) {
      const std::vector<int> query = {query_distance};
      for (int32_t tolerance : {0, 1, 5, 20, 255}) {
        const auto expected = reference_survivors(
            exact, exact.size(), 1, query, 0, tolerance);
        const auto actual = navigamer::filter_mbb_survivors(
            packed.data(), exact.size(), 1, query.data(), 0,
            tolerance, navigamer::SimdMode::Auto, nullptr, bits,
            bin_width);
        assert(std::includes(
            actual.begin(), actual.end(), expected.begin(), expected.end()));
      }
    }
  }
}

void assert_paired_base11_equivalence() {
  constexpr uint32_t bin_width = 6;
  constexpr int32_t quantization_error = bin_width / 2;
  std::mt19937 rng(20260802);
  std::uniform_int_distribution<int> pick_encoded(0, 10);
  std::uniform_int_distribution<int> pick_query(0, 80);
  for (size_t dim : {size_t{2}, size_t{3}, size_t{10}}) {
    for (size_t child_count : {size_t{1}, size_t{7}, size_t{44},
                               size_t{257}}) {
      std::vector<uint8_t> encoded(dim * child_count);
      std::vector<uint8_t> midpoints(dim * child_count);
      for (size_t idx = 0; idx < encoded.size(); ++idx) {
        encoded[idx] = static_cast<uint8_t>(pick_encoded(rng));
        midpoints[idx] = static_cast<uint8_t>(
            encoded[idx] * bin_width + quantization_error);
      }
      const auto packed =
          pack_paired_base11(encoded, child_count, dim);
      std::vector<int> query(dim);
      for (int& value : query) value = pick_query(rng);
      for (int32_t tolerance : {0, 2, 9}) {
        constexpr int32_t child_radius = 7;
        const auto expected = reference_survivors(
            midpoints, child_count, dim, query, child_radius,
            tolerance + quantization_error);
        navigamer::MBBFilterSimdStats stats;
        const auto actual = navigamer::filter_mbb_survivors(
            packed.data(), child_count, dim, query.data(), child_radius,
            tolerance, navigamer::SimdMode::Auto, &stats, 7, bin_width);
        assert(actual == expected);
        assert(stats.scalar_checks == child_count);
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
  assert_packed_equivalence();
  assert_quantized_filter_has_no_false_negatives();
  assert_paired_base11_equivalence();
  std::cout << "SIMD MBB filter tests passed\n";
  return 0;
}
