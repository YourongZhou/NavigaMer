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

void validate_inputs(const int32_t* lo_by_dim,
                     const int32_t* hi_by_dim,
                     size_t child_count,
                     size_t dim,
                     const int32_t* query_beacon_dists) {
  if (child_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::invalid_argument("MBB child_count exceeds uint32_t output range");
  }
  if (child_count == 0 || dim == 0) return;
  if (!lo_by_dim || !hi_by_dim || !query_beacon_dists) {
    throw std::invalid_argument("MBB filter received null input");
  }
}

void validate_leaf_inputs(const int32_t* dist_by_dim,
                          size_t leaf_count,
                          size_t dim,
                          const int32_t* query_beacon_dists) {
  if (leaf_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::invalid_argument("leaf_count exceeds uint32_t output range");
  }
  if (leaf_count == 0 || dim == 0) return;
  if (!dist_by_dim || !query_beacon_dists) {
    throw std::invalid_argument("leaf beacon filter received null input");
  }
}

std::vector<uint32_t> filter_scalar(const int32_t* lo_by_dim,
                                    const int32_t* hi_by_dim,
                                    size_t child_count,
                                    size_t dim,
                                    const int32_t* query_beacon_dists,
                                    int32_t tolerance,
                                    MBBFilterSimdStats* stats) {
  if (stats) stats->scalar_checks += child_count;
  std::vector<uint32_t> survivors;
  survivors.reserve(child_count);
  for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * child_count + child_idx;
      const int32_t query_lo = query_beacon_dists[dim_idx] - tolerance;
      const int32_t query_hi = query_beacon_dists[dim_idx] + tolerance;
      if (hi_by_dim[flat] < query_lo || lo_by_dim[flat] > query_hi) {
        ok = false;
        break;
      }
    }
    if (ok) survivors.push_back(static_cast<uint32_t>(child_idx));
  }
  return survivors;
}

std::vector<uint32_t> filter_leaf_scalar(const int32_t* dist_by_dim,
                                         size_t leaf_count,
                                         size_t dim,
                                         const int32_t* query_beacon_dists,
                                         int32_t tolerance,
                                         LeafBeaconFilterSimdStats* stats) {
  if (stats) stats->scalar_checks += leaf_count;
  std::vector<uint32_t> survivors;
  survivors.reserve(leaf_count);
  for (size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
    bool ok = true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * leaf_count + leaf_idx;
      const int32_t diff =
          std::abs(query_beacon_dists[dim_idx] - dist_by_dim[flat]);
      if (diff > tolerance) {
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
std::vector<uint32_t> filter_avx2(const int32_t* lo_by_dim,
                                  const int32_t* hi_by_dim,
                                  size_t child_count,
                                  size_t dim,
                                  const int32_t* query_beacon_dists,
                                  int32_t tolerance,
                                  MBBFilterSimdStats* stats) {
  constexpr size_t kWidth = 8;
  std::vector<uint32_t> survivors;
  survivors.reserve(child_count);

  const __m256i zero = _mm256_setzero_si256();
  const __m256i all_true = _mm256_set1_epi32(-1);
  size_t child_idx = 0;
  for (; child_idx + kWidth <= child_count; child_idx += kWidth) {
    if (stats) ++stats->simd_batches;
    __m256i alive = all_true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * child_count + child_idx;
      const __m256i lo_v =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lo_by_dim + flat));
      const __m256i hi_v =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hi_by_dim + flat));
      const __m256i query_lo =
          _mm256_set1_epi32(query_beacon_dists[dim_idx] - tolerance);
      const __m256i query_hi =
          _mm256_set1_epi32(query_beacon_dists[dim_idx] + tolerance);
      const __m256i hi_too_low = _mm256_cmpgt_epi32(query_lo, hi_v);
      const __m256i lo_too_high = _mm256_cmpgt_epi32(lo_v, query_hi);
      const __m256i failed = _mm256_or_si256(hi_too_low, lo_too_high);
      const __m256i passed = _mm256_cmpeq_epi32(failed, zero);
      alive = _mm256_and_si256(alive, passed);
      if (_mm256_movemask_epi8(alive) == 0) break;
    }

    alignas(32) int32_t lane_alive[kWidth];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < kWidth; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(static_cast<uint32_t>(child_idx + lane));
      }
    }
  }

  if (child_idx < child_count) {
    if (stats) stats->scalar_checks += child_count - child_idx;
    for (size_t tail_idx = child_idx; tail_idx < child_count; ++tail_idx) {
      bool ok = true;
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        const size_t flat = dim_idx * child_count + tail_idx;
        const int32_t query_lo = query_beacon_dists[dim_idx] - tolerance;
        const int32_t query_hi = query_beacon_dists[dim_idx] + tolerance;
        if (hi_by_dim[flat] < query_lo || lo_by_dim[flat] > query_hi) {
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
std::vector<uint32_t> filter_leaf_avx2(const int32_t* dist_by_dim,
                                       size_t leaf_count,
                                       size_t dim,
                                       const int32_t* query_beacon_dists,
                                       int32_t tolerance,
                                       LeafBeaconFilterSimdStats* stats) {
  constexpr size_t kWidth = 8;
  std::vector<uint32_t> survivors;
  survivors.reserve(leaf_count);

  const __m256i zero = _mm256_setzero_si256();
  const __m256i all_true = _mm256_set1_epi32(-1);
  const __m256i tol = _mm256_set1_epi32(tolerance);
  size_t leaf_idx = 0;
  for (; leaf_idx + kWidth <= leaf_count; leaf_idx += kWidth) {
    if (stats) ++stats->simd_batches;
    __m256i alive = all_true;
    for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
      const size_t flat = dim_idx * leaf_count + leaf_idx;
      const __m256i dist_v =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dist_by_dim + flat));
      const __m256i query_v = _mm256_set1_epi32(query_beacon_dists[dim_idx]);
      const __m256i delta = _mm256_sub_epi32(query_v, dist_v);
      const __m256i neg_delta = _mm256_sub_epi32(zero, delta);
      const __m256i abs_delta = _mm256_max_epi32(delta, neg_delta);
      const __m256i failed = _mm256_cmpgt_epi32(abs_delta, tol);
      const __m256i passed = _mm256_cmpeq_epi32(failed, zero);
      alive = _mm256_and_si256(alive, passed);
      if (_mm256_movemask_epi8(alive) == 0) break;
    }

    alignas(32) int32_t lane_alive[kWidth];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane_alive), alive);
    for (size_t lane = 0; lane < kWidth; ++lane) {
      if (lane_alive[lane] != 0) {
        survivors.push_back(static_cast<uint32_t>(leaf_idx + lane));
      }
    }
  }

  if (leaf_idx < leaf_count) {
    if (stats) stats->scalar_checks += leaf_count - leaf_idx;
    for (size_t tail_idx = leaf_idx; tail_idx < leaf_count; ++tail_idx) {
      bool ok = true;
      for (size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
        const size_t flat = dim_idx * leaf_count + tail_idx;
        const int32_t diff =
            std::abs(query_beacon_dists[dim_idx] - dist_by_dim[flat]);
        if (diff > tolerance) {
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
    const int32_t* lo_by_dim,
    const int32_t* hi_by_dim,
    size_t child_count,
    size_t dim,
    const int32_t* query_beacon_dists,
    int32_t tolerance,
    SimdMode mode,
    MBBFilterSimdStats* stats) {
  validate_inputs(lo_by_dim, hi_by_dim, child_count, dim, query_beacon_dists);
  if (mode == SimdMode::Scalar || child_count == 0) {
    return filter_scalar(lo_by_dim, hi_by_dim, child_count, dim,
                         query_beacon_dists, tolerance, stats);
  }

#if NAVIGAMER_HAS_AVX2_TARGET
  if ((mode == SimdMode::Auto || mode == SimdMode::AVX2) &&
      simd_avx2_runtime_supported()) {
    return filter_avx2(lo_by_dim, hi_by_dim, child_count, dim,
                       query_beacon_dists, tolerance, stats);
  }
#endif

  if (stats) ++stats->simd_fallbacks;
  return filter_scalar(lo_by_dim, hi_by_dim, child_count, dim,
                       query_beacon_dists, tolerance, stats);
}

std::vector<uint32_t> filter_leaf_beacon_survivors(
    const int32_t* dist_by_dim,
    size_t leaf_count,
    size_t dim,
    const int32_t* query_beacon_dists,
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
