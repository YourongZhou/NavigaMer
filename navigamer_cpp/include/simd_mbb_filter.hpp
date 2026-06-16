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

const char* simd_mode_name(SimdMode mode);
SimdMode parse_simd_mode(const std::string& value);
bool simd_avx2_runtime_supported();

std::vector<uint32_t> filter_mbb_survivors(
    const int32_t* lo_by_dim,
    const int32_t* hi_by_dim,
    size_t child_count,
    size_t dim,
    const int32_t* query_beacon_dists,
    int32_t tolerance,
    SimdMode mode,
    MBBFilterSimdStats* stats = nullptr);

}  // namespace navigamer

#endif  // NAVIGAMER_SIMD_MBB_FILTER_HPP
