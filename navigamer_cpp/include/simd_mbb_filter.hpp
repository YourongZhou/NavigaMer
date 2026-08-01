#ifndef NAVIGAMER_SIMD_MBB_FILTER_HPP
#define NAVIGAMER_SIMD_MBB_FILTER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace navigamer {

enum class SimdMode {
  Auto,
  Scalar,
  AVX2,
  AVX512,
};

struct MBBFilterSimdStats {
  size_t scalar_checks = 0;
  size_t simd_batches = 0;
  size_t simd_fallbacks = 0;
};

struct LeafBeaconFilterSimdStats {
  size_t scalar_checks = 0;
  size_t simd_batches = 0;
  size_t simd_fallbacks = 0;
};

const char* simd_mode_name(SimdMode mode);
SimdMode parse_simd_mode(const std::string& value);
bool simd_avx2_runtime_supported();

// Rank/unrank one pair of quantized beacon distances among only the states
// permitted by the triangle inequality for the exact beacon-pair distance.
// Width must be 6 or 12 and both quantized values must be in [0, 10].
uint32_t metric_pair_rank_bits(
    uint8_t beacon_pair_distance, uint32_t quantization_bin_width);
uint8_t metric_pair_rank(
    uint8_t first, uint8_t second,
    uint8_t beacon_pair_distance, uint32_t quantization_bin_width);
uint8_t metric_pair_code(
    uint8_t rank, uint8_t beacon_pair_distance,
    uint32_t quantization_bin_width);

std::vector<uint32_t> filter_mbb_survivors(
    const uint8_t* center_dist_by_dim,
    size_t child_count,
    size_t dim,
    const int* query_beacon_dists,
    int32_t child_radius,
    int32_t tolerance,
    SimdMode mode,
    MBBFilterSimdStats* stats = nullptr,
    uint32_t packed_bits = 8,
    uint32_t quantization_bin_width = 1);

std::vector<uint32_t> filter_leaf_beacon_survivors(
    const uint8_t* dist_by_dim,
    size_t leaf_count,
    size_t dim,
    const int* query_beacon_dists,
    int32_t tolerance,
    SimdMode mode,
    LeafBeaconFilterSimdStats* stats = nullptr,
    uint32_t packed_bits = 8);

}  // namespace navigamer

#endif  // NAVIGAMER_SIMD_MBB_FILTER_HPP
