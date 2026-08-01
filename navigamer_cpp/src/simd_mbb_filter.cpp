#include "simd_mbb_filter.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define NAVIGAMER_HAS_AVX2_TARGET 1
#else
#define NAVIGAMER_HAS_AVX2_TARGET 0
#endif

namespace navigamer {
namespace {

void validate_inputs(const uint8_t* center_dist_by_dim,
                     size_t child_count,
                     size_t dim,
                     const int* query_beacon_dists) {
  if (child_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::invalid_argument("MBB child_count exceeds uint32_t output range");
  }
  if (child_count == 0 || dim == 0) return;
  if (!center_dist_by_dim || !query_beacon_dists) {
    throw std::invalid_argument("MBB filter received null input");
  }
}

void validate_leaf_inputs(const uint8_t* dist_by_dim,
                          size_t leaf_count,
                          size_t dim,
                          const int* query_beacon_dists) {
  if (leaf_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::invalid_argument("leaf_count exceeds uint32_t output range");
  }
  if (leaf_count == 0 || dim == 0) return;
  if (!dist_by_dim || !query_beacon_dists) {
    throw std::invalid_argument("leaf beacon filter received null input");
  }
}

std::vector<uint32_t> filter_scalar(const uint8_t* center_dist_by_dim,
                                    size_t child_count,
                                    size_t dim,
                                    const int* query_beacon_dists,
                                    int32_t tolerance,
                                    MBBFilterSimdStats* stats) {
  if (stats) stats->scalar_checks += child_count;
  std::vector<uint32_t> survivors;
  survivors.reserve(child_count);
  for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * child_count + child_idx;
      const int64_t query_lo =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_hi =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (center_dist_by_dim[flat] < query_lo ||
          center_dist_by_dim[flat] > query_hi) {
        ok = false;
        break;
      }
    }
    if (ok) survivors.push_back(static_cast<uint32_t>(child_idx));
  }
  return survivors;
}

uint8_t packed_distance(const uint8_t* data, size_t cell,
                        uint32_t bits) {
  const size_t bit_offset = cell * bits;
  const size_t byte_offset = bit_offset >> 3;
  const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
  uint16_t word = data[byte_offset];
  if (shift + bits > 8) {
    word |= static_cast<uint16_t>(data[byte_offset + 1]) << 8;
  }
  return static_cast<uint8_t>(
      (word >> shift) & ((uint32_t{1} << bits) - 1));
}

std::vector<uint32_t> filter_packed_scalar(
    const uint8_t* center_dist_by_dim,
    size_t child_count,
    size_t dim,
    const int* query_beacon_dists,
    int32_t tolerance,
    uint32_t bits,
    MBBFilterSimdStats* stats) {
  if (stats) stats->scalar_checks += child_count;
  std::vector<uint32_t> survivors;
  survivors.reserve(child_count);
  for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t cell = dim_idx * child_count + child_idx;
      const int64_t query_lo =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_hi =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      const uint8_t center_dist =
          packed_distance(center_dist_by_dim, cell, bits);
      if (center_dist < query_lo || center_dist > query_hi) {
        ok = false;
        break;
      }
    }
    if (ok) survivors.push_back(static_cast<uint32_t>(child_idx));
  }
  return survivors;
}

std::vector<uint32_t> filter_leaf_scalar(const uint8_t* dist_by_dim,
                                         size_t leaf_count,
                                         size_t dim,
                                         const int* query_beacon_dists,
                                         int32_t tolerance,
                                         LeafBeaconFilterSimdStats* stats) {
  if (stats) stats->scalar_checks += leaf_count;
  std::vector<uint32_t> survivors;
  survivors.reserve(leaf_count);
  for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * leaf_count + leaf_idx;
      const int64_t query_lower =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_upper =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (dist_by_dim[flat] < query_lower ||
          dist_by_dim[flat] > query_upper) {
        ok = false;
        break;
      }
    }
    if (ok) survivors.push_back(static_cast<uint32_t>(leaf_idx));
  }
  return survivors;
}

#if NAVIGAMER_HAS_AVX2_TARGET
__attribute__((target("avx2")))
std::vector<uint32_t> filter_avx2(
    const uint8_t* center_dist_by_dim,
    size_t child_count,
    size_t dim,
    const int* query_beacon_dists,
    int32_t tolerance,
    MBBFilterSimdStats* stats) {
  constexpr size_t kWidth = 32;
  std::vector<uint32_t> survivors;
  survivors.reserve(child_count);

  const __m256i zero = _mm256_setzero_si256();
  const __m256i all_true = _mm256_set1_epi8(-1);
  const __m256i unsigned_bias =
      _mm256_set1_epi8(static_cast<char>(0x80));
  size_t child_idx = 0;
  for (; child_idx + kWidth <= child_count; child_idx += kWidth) {
    if (stats) ++stats->simd_batches;
    __m256i alive = all_true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const int64_t query_lower =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_upper =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (query_upper < 0 ||
          query_lower >
              std::numeric_limits<uint8_t>::max()) {
        alive = zero;
        break;
      }
      const uint8_t bounded_lower = static_cast<uint8_t>(
          std::max<int64_t>(0, query_lower));
      const uint8_t bounded_upper = static_cast<uint8_t>(
          std::min<int64_t>(
              std::numeric_limits<uint8_t>::max(), query_upper));
      const size_t flat = dim_idx * child_count + child_idx;
      const __m256i center_dist = _mm256_xor_si256(
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(
                  center_dist_by_dim + flat)),
          unsigned_bias);
      const __m256i query_lo = _mm256_xor_si256(
          _mm256_set1_epi8(static_cast<char>(bounded_lower)),
          unsigned_bias);
      const __m256i query_hi = _mm256_xor_si256(
          _mm256_set1_epi8(static_cast<char>(bounded_upper)),
          unsigned_bias);
      const __m256i failed = _mm256_or_si256(
          _mm256_cmpgt_epi8(query_lo, center_dist),
          _mm256_cmpgt_epi8(center_dist, query_hi));
      alive = _mm256_and_si256(
          alive, _mm256_cmpeq_epi8(failed, zero));
      if (_mm256_movemask_epi8(alive) == 0) break;
    }

    alignas(32) int8_t lane_alive[kWidth];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < kWidth; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(static_cast<uint32_t>(child_idx + lane));
      }
    }
  }

  if (child_idx + 16 <= child_count) {
    if (stats) ++stats->simd_batches;
    __m256i alive = _mm256_set1_epi16(-1);
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const int64_t query_lower =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_upper =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (query_upper < 0 ||
          query_lower > std::numeric_limits<uint8_t>::max()) {
        alive = zero;
        break;
      }
      const int16_t bounded_lower = static_cast<int16_t>(
          std::max<int64_t>(0, query_lower));
      const int16_t bounded_upper = static_cast<int16_t>(
          std::min<int64_t>(
              std::numeric_limits<uint8_t>::max(), query_upper));
      const size_t flat = dim_idx * child_count + child_idx;
      const __m256i center_dist = _mm256_cvtepu8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(
              center_dist_by_dim + flat)));
      const __m256i failed = _mm256_or_si256(
          _mm256_cmpgt_epi16(
              _mm256_set1_epi16(bounded_lower), center_dist),
          _mm256_cmpgt_epi16(
              center_dist, _mm256_set1_epi16(bounded_upper)));
      alive = _mm256_and_si256(
          alive, _mm256_cmpeq_epi16(failed, zero));
      if (_mm256_movemask_epi8(alive) == 0) break;
    }
    alignas(32) int16_t lane_alive[16];
    _mm256_store_si256(
        reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < 16; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(
            static_cast<uint32_t>(child_idx + lane));
      }
    }
    child_idx += 16;
  }

  if (child_idx + 8 <= child_count) {
    if (stats) ++stats->simd_batches;
    __m256i alive = _mm256_set1_epi32(-1);
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const int64_t query_lower =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_upper =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (query_upper < 0 ||
          query_lower > std::numeric_limits<uint8_t>::max()) {
        alive = zero;
        break;
      }
      const size_t flat = dim_idx * child_count + child_idx;
      const __m256i center_dist = _mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
              center_dist_by_dim + flat)));
      const __m256i query_lo = _mm256_set1_epi32(
          static_cast<int32_t>(std::max<int64_t>(0, query_lower)));
      const __m256i query_hi = _mm256_set1_epi32(
          static_cast<int32_t>(std::min<int64_t>(
              std::numeric_limits<uint8_t>::max(), query_upper)));
      const __m256i failed = _mm256_or_si256(
          _mm256_cmpgt_epi32(query_lo, center_dist),
          _mm256_cmpgt_epi32(center_dist, query_hi));
      alive = _mm256_and_si256(
          alive, _mm256_cmpeq_epi32(failed, zero));
      if (_mm256_movemask_epi8(alive) == 0) break;
    }
    alignas(32) int32_t lane_alive[8];
    _mm256_store_si256(
        reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < 8; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(
            static_cast<uint32_t>(child_idx + lane));
      }
    }
    child_idx += 8;
  }

  if (child_idx < child_count) {
    if (stats) stats->scalar_checks += child_count - child_idx;
    for (size_t tail_idx = child_idx; tail_idx < child_count; ++tail_idx) {
      bool ok = true;
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        const size_t flat = dim_idx * child_count + tail_idx;
        const int64_t query_lo =
            static_cast<int64_t>(query_beacon_dists[dim_idx]) -
            tolerance;
        const int64_t query_hi =
            static_cast<int64_t>(query_beacon_dists[dim_idx]) +
            tolerance;
        if (center_dist_by_dim[flat] < query_lo ||
            center_dist_by_dim[flat] > query_hi) {
          ok = false;
          break;
        }
      }
      if (ok) survivors.push_back(static_cast<uint32_t>(tail_idx));
    }
  }
  return survivors;
}

__attribute__((target("avx2")))
std::vector<uint32_t> filter_leaf_avx2(const uint8_t* dist_by_dim,
                                       size_t leaf_count,
                                       size_t dim,
                                       const int* query_beacon_dists,
                                       int32_t tolerance,
                                       LeafBeaconFilterSimdStats* stats) {
  constexpr size_t kWidth = 32;
  std::vector<uint32_t> survivors;
  survivors.reserve(leaf_count);

  const __m256i zero = _mm256_setzero_si256();
  const __m256i all_true = _mm256_set1_epi8(-1);
  const __m256i unsigned_bias =
      _mm256_set1_epi8(static_cast<char>(0x80));
  size_t leaf_idx = 0;
  for (; leaf_idx + kWidth <= leaf_count; leaf_idx += kWidth) {
    if (stats) ++stats->simd_batches;
    __m256i alive = all_true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const int64_t query_lower =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_upper =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (query_upper < 0 ||
          query_lower >
              std::numeric_limits<uint8_t>::max()) {
        alive = zero;
        break;
      }
      const uint8_t bounded_lower = static_cast<uint8_t>(
          std::max<int64_t>(0, query_lower));
      const uint8_t bounded_upper = static_cast<uint8_t>(
          std::min<int64_t>(
              std::numeric_limits<uint8_t>::max(), query_upper));
      const size_t flat = dim_idx * leaf_count + leaf_idx;
      const __m256i dist = _mm256_xor_si256(
          _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(
                  dist_by_dim + flat)),
          unsigned_bias);
      const __m256i query_lo = _mm256_xor_si256(
          _mm256_set1_epi8(static_cast<char>(bounded_lower)),
          unsigned_bias);
      const __m256i query_hi = _mm256_xor_si256(
          _mm256_set1_epi8(static_cast<char>(bounded_upper)),
          unsigned_bias);
      const __m256i failed = _mm256_or_si256(
          _mm256_cmpgt_epi8(query_lo, dist),
          _mm256_cmpgt_epi8(dist, query_hi));
      alive = _mm256_and_si256(
          alive, _mm256_cmpeq_epi8(failed, zero));
      if (_mm256_movemask_epi8(alive) == 0) break;
    }

    alignas(32) int8_t lane_alive[kWidth];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < kWidth; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(static_cast<uint32_t>(leaf_idx + lane));
      }
    }
  }

  if (leaf_idx + 16 <= leaf_count) {
    if (stats) ++stats->simd_batches;
    __m256i alive = _mm256_set1_epi16(-1);
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const int64_t query_lower =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_upper =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (query_upper < 0 ||
          query_lower > std::numeric_limits<uint8_t>::max()) {
        alive = zero;
        break;
      }
      const int16_t bounded_lower = static_cast<int16_t>(
          std::max<int64_t>(0, query_lower));
      const int16_t bounded_upper = static_cast<int16_t>(
          std::min<int64_t>(
              std::numeric_limits<uint8_t>::max(), query_upper));
      const size_t flat = dim_idx * leaf_count + leaf_idx;
      const __m256i dist = _mm256_cvtepu8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(
              dist_by_dim + flat)));
      const __m256i failed = _mm256_or_si256(
          _mm256_cmpgt_epi16(
              _mm256_set1_epi16(bounded_lower), dist),
          _mm256_cmpgt_epi16(
              dist, _mm256_set1_epi16(bounded_upper)));
      alive = _mm256_and_si256(
          alive, _mm256_cmpeq_epi16(failed, zero));
      if (_mm256_movemask_epi8(alive) == 0) break;
    }
    alignas(32) int16_t lane_alive[16];
    _mm256_store_si256(
        reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < 16; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(
            static_cast<uint32_t>(leaf_idx + lane));
      }
    }
    leaf_idx += 16;
  }

  if (leaf_idx + 8 <= leaf_count) {
    if (stats) ++stats->simd_batches;
    __m256i alive = _mm256_set1_epi32(-1);
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const int64_t query_lower =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) -
          tolerance;
      const int64_t query_upper =
          static_cast<int64_t>(query_beacon_dists[dim_idx]) +
          tolerance;
      if (query_upper < 0 ||
          query_lower > std::numeric_limits<uint8_t>::max()) {
        alive = zero;
        break;
      }
      const size_t flat = dim_idx * leaf_count + leaf_idx;
      const __m256i dist = _mm256_cvtepu8_epi32(
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
              dist_by_dim + flat)));
      const __m256i failed = _mm256_or_si256(
          _mm256_cmpgt_epi32(
              _mm256_set1_epi32(static_cast<int32_t>(
                  std::max<int64_t>(0, query_lower))),
              dist),
          _mm256_cmpgt_epi32(
              dist,
              _mm256_set1_epi32(static_cast<int32_t>(
                  std::min<int64_t>(
                      std::numeric_limits<uint8_t>::max(),
                      query_upper)))));
      alive = _mm256_and_si256(
          alive, _mm256_cmpeq_epi32(failed, zero));
      if (_mm256_movemask_epi8(alive) == 0) break;
    }
    alignas(32) int32_t lane_alive[8];
    _mm256_store_si256(
        reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < 8; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(
            static_cast<uint32_t>(leaf_idx + lane));
      }
    }
    leaf_idx += 8;
  }

  if (leaf_idx < leaf_count) {
    if (stats) stats->scalar_checks += leaf_count - leaf_idx;
    for (size_t tail_idx = leaf_idx; tail_idx < leaf_count; ++tail_idx) {
      bool ok = true;
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        const size_t flat = dim_idx * leaf_count + tail_idx;
        const int64_t query_lower =
            static_cast<int64_t>(query_beacon_dists[dim_idx]) -
            tolerance;
        const int64_t query_upper =
            static_cast<int64_t>(query_beacon_dists[dim_idx]) +
            tolerance;
        if (dist_by_dim[flat] < query_lower ||
            dist_by_dim[flat] > query_upper) {
          ok = false;
          break;
        }
      }
      if (ok) survivors.push_back(static_cast<uint32_t>(tail_idx));
    }
  }
  return survivors;
}
#endif

}  // namespace

const char* simd_mode_name(SimdMode mode) {
  switch (mode) {
    case SimdMode::Auto:
      return "auto";
    case SimdMode::Scalar:
      return "scalar";
    case SimdMode::AVX2:
      return "avx2";
    case SimdMode::AVX512:
      return "avx512";
  }
  return "auto";
}

SimdMode parse_simd_mode(const std::string& value) {
  if (value == "auto") return SimdMode::Auto;
  if (value == "scalar") return SimdMode::Scalar;
  if (value == "avx2") return SimdMode::AVX2;
  if (value == "avx512") return SimdMode::AVX512;
  throw std::invalid_argument("unknown SIMD mode: " + value);
}

bool simd_avx2_runtime_supported() {
#if NAVIGAMER_HAS_AVX2_TARGET
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

std::vector<uint32_t> filter_mbb_survivors(
    const uint8_t* center_dist_by_dim,
    size_t child_count,
    size_t dim,
    const int* query_beacon_dists,
    int32_t child_radius,
    int32_t tolerance,
    SimdMode mode,
    MBBFilterSimdStats* stats,
    uint32_t packed_bits) {
  validate_inputs(center_dist_by_dim, child_count, dim,
                  query_beacon_dists);
  if (packed_bits == 0 || packed_bits > 8) {
    throw std::invalid_argument("MBB packed width must be in [1, 8]");
  }
  if (child_radius < 0) {
    throw std::invalid_argument("MBB child radius must be nonnegative");
  }
  const int32_t effective_tolerance = static_cast<int32_t>(
      std::min<int64_t>(
          std::numeric_limits<int32_t>::max(),
          static_cast<int64_t>(child_radius) + tolerance));
  if (packed_bits != 8) {
    if (stats && mode != SimdMode::Scalar) ++stats->simd_fallbacks;
    return filter_packed_scalar(
        center_dist_by_dim, child_count, dim, query_beacon_dists,
        effective_tolerance, packed_bits, stats);
  }
  if (mode == SimdMode::Scalar || child_count == 0) {
    return filter_scalar(center_dist_by_dim, child_count, dim,
                         query_beacon_dists, effective_tolerance, stats);
  }

#if NAVIGAMER_HAS_AVX2_TARGET
  if ((mode == SimdMode::Auto || mode == SimdMode::AVX2) &&
      simd_avx2_runtime_supported()) {
    return filter_avx2(center_dist_by_dim, child_count, dim,
                       query_beacon_dists, effective_tolerance, stats);
  }
#endif

  if (stats) ++stats->simd_fallbacks;
  return filter_scalar(center_dist_by_dim, child_count, dim,
                       query_beacon_dists, effective_tolerance, stats);
}

std::vector<uint32_t> filter_leaf_beacon_survivors(
    const uint8_t* dist_by_dim,
    size_t leaf_count,
    size_t dim,
    const int* query_beacon_dists,
    int32_t tolerance,
    SimdMode mode,
    LeafBeaconFilterSimdStats* stats) {
  validate_leaf_inputs(dist_by_dim, leaf_count, dim, query_beacon_dists);
  if (mode == SimdMode::Scalar || leaf_count == 0) {
    return filter_leaf_scalar(dist_by_dim, leaf_count, dim, query_beacon_dists,
                              tolerance, stats);
  }

#if NAVIGAMER_HAS_AVX2_TARGET
  if ((mode == SimdMode::Auto || mode == SimdMode::AVX2) &&
      simd_avx2_runtime_supported()) {
    return filter_leaf_avx2(dist_by_dim, leaf_count, dim, query_beacon_dists,
                            tolerance, stats);
  }
#endif

  if (stats) ++stats->simd_fallbacks;
  return filter_leaf_scalar(dist_by_dim, leaf_count, dim, query_beacon_dists,
                            tolerance, stats);
}

}  // namespace navigamer
