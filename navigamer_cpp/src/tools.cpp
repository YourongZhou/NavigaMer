#include "tools.hpp"
#include "edlib.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <utility>

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define NAVIGAMER_HAS_MYERS_BATCH4_AVX2 1
#else
#define NAVIGAMER_HAS_MYERS_BATCH4_AVX2 0
#endif

namespace navigamer {

#if defined(__GNUC__) || defined(__clang__)
#define NAVIGAMER_DISTANCE_HOT_ALIGN __attribute__((aligned(128)))
#else
#define NAVIGAMER_DISTANCE_HOT_ALIGN
#endif

const char* distance_mode_name(DistanceMode mode) {
  switch (mode) {
    case DistanceMode::DP:
      return "dp";
    case DistanceMode::Myers:
      return "myers";
    case DistanceMode::Edlib:
      return "edlib";
    case DistanceMode::Auto:
      return "auto";
  }
  return "dp";
}

DistanceMode parse_distance_mode(const std::string& value) {
  if (value == "dp") return DistanceMode::DP;
  if (value == "myers") return DistanceMode::Myers;
  if (value == "edlib") return DistanceMode::Edlib;
  if (value == "auto") return DistanceMode::Auto;
  throw std::invalid_argument("distance mode must be dp, myers, edlib, or auto");
}

NAVIGAMER_DISTANCE_HOT_ALIGN
int compute_distance(std::string_view a, std::string_view b) {
  const size_t m = a.size();
  const size_t n = b.size();
  if (m == 0) return static_cast<int>(n);
  if (n == 0) return static_cast<int>(m);

  std::vector<int> prev(n + 1), curr(n + 1);
  for (size_t j = 0; j <= n; ++j) prev[j] = static_cast<int>(j);
  for (size_t i = 1; i <= m; ++i) {
    curr[0] = static_cast<int>(i);
    for (size_t j = 1; j <= n; ++j) {
      if (a[i - 1] == b[j - 1])
        curr[j] = prev[j - 1];
      else
        curr[j] = 1 + std::min({prev[j - 1], prev[j], curr[j - 1]});
    }
    std::swap(prev, curr);
  }
  return prev[n];
}

int compute_distance(const BioSequence& a, const BioSequence& b) {
  return compute_distance(a.seq, b.seq);
}

int compute_distance_bounded_dp(std::string_view a, std::string_view b,
                                int tau) {
  if (tau < 0) throw std::invalid_argument("edit-distance threshold must be non-negative");

  const int m = static_cast<int>(a.size());
  const int n = static_cast<int>(b.size());
  if (std::abs(m - n) > tau) return tau + 1;
  if (m == 0) return n <= tau ? n : tau + 1;
  if (n == 0) return m <= tau ? m : tau + 1;

  const int outside = tau + 1;
  std::vector<int> prev(static_cast<size_t>(n + 1), outside);
  std::vector<int> curr(static_cast<size_t>(n + 1), outside);
  for (int j = 0; j <= std::min(n, tau); ++j) prev[static_cast<size_t>(j)] = j;

  for (int i = 1; i <= m; ++i) {
    std::fill(curr.begin(), curr.end(), outside);
    if (i <= tau) curr[0] = i;

    const int j_begin = std::max(1, i - tau);
    const int j_end = std::min(n, i + tau);
    int row_min = curr[0];
    for (int j = j_begin; j <= j_end; ++j) {
      int value = std::min({
          prev[static_cast<size_t>(j)] + 1,
          curr[static_cast<size_t>(j - 1)] + 1,
          prev[static_cast<size_t>(j - 1)] +
              (a[static_cast<size_t>(i - 1)] == b[static_cast<size_t>(j - 1)] ? 0 : 1)});
      curr[static_cast<size_t>(j)] = std::min(value, outside);
      row_min = std::min(row_min, curr[static_cast<size_t>(j)]);
    }
    if (row_min > tau) return tau + 1;
    std::swap(prev, curr);
  }

  return prev[static_cast<size_t>(n)] <= tau
             ? prev[static_cast<size_t>(n)]
             : tau + 1;
}

int compute_distance_edlib(std::string_view a, std::string_view b) {
  EdlibAlignConfig config =
      edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_DISTANCE, nullptr, 0);
  EdlibAlignResult result =
      edlibAlign(a.data(), static_cast<int>(a.size()), b.data(),
                 static_cast<int>(b.size()), config);
  if (result.status != EDLIB_STATUS_OK) {
    edlibFreeAlignResult(result);
    return compute_distance(a, b);
  }
  const int distance = result.editDistance;
  edlibFreeAlignResult(result);
  return distance;
}

int compute_distance_bounded_edlib(std::string_view a, std::string_view b,
                                   int tau) {
  if (tau < 0) throw std::invalid_argument("edit-distance threshold must be non-negative");
  const int m = static_cast<int>(a.size());
  const int n = static_cast<int>(b.size());
  if (std::abs(m - n) > tau) return tau + 1;

  EdlibAlignConfig config =
      edlibNewAlignConfig(tau, EDLIB_MODE_NW, EDLIB_TASK_DISTANCE, nullptr, 0);
  EdlibAlignResult result =
      edlibAlign(a.data(), m, b.data(), n, config);
  if (result.status != EDLIB_STATUS_OK) {
    edlibFreeAlignResult(result);
    return compute_distance_bounded_dp(a, b, tau);
  }
  const int distance = result.editDistance < 0 ? tau + 1 : result.editDistance;
  edlibFreeAlignResult(result);
  return distance <= tau ? distance : tau + 1;
}

PreparedEdlibDnaPattern::~PreparedEdlibDnaPattern() {
  if (handle) {
    edlibDnaPreparedFree(static_cast<EdlibDnaPrepared*>(handle));
  }
}

PreparedEdlibDnaPattern::PreparedEdlibDnaPattern(
    PreparedEdlibDnaPattern&& other) noexcept
    : pattern(std::move(other.pattern)), handle(other.handle) {
  other.handle = nullptr;
}

PreparedEdlibDnaPattern& PreparedEdlibDnaPattern::operator=(
    PreparedEdlibDnaPattern&& other) noexcept {
  if (this == &other) return *this;
  if (handle) {
    edlibDnaPreparedFree(static_cast<EdlibDnaPrepared*>(handle));
  }
  pattern = std::move(other.pattern);
  handle = other.handle;
  other.handle = nullptr;
  return *this;
}

PreparedEdlibDnaPattern prepare_edlib_dna_pattern(
    std::string_view pattern) {
  PreparedEdlibDnaPattern prepared;
  prepared.pattern.assign(pattern.data(), pattern.size());
  EdlibDnaPrepared* handle = edlibDnaPrepare(
      prepared.pattern.data(), static_cast<int>(prepared.pattern.size()));
  if (handle) {
    prepared.handle = handle;
  }
  return prepared;
}

int compute_distance_bounded_edlib_prepared(
    const PreparedEdlibDnaPattern& pattern,
    std::string_view text,
    int tau) {
  if (tau < 0) {
    throw std::invalid_argument(
        "edit-distance threshold must be non-negative");
  }
  if (!pattern.handle) {
    return compute_distance_bounded_edlib(pattern.pattern, text, tau);
  }
  const int distance = edlibDnaPreparedBoundedDistance(
      static_cast<const EdlibDnaPrepared*>(pattern.handle),
      text.data(), static_cast<int>(text.size()), tau);
  if (distance >= 0) return distance;
  if (distance == -1) return tau + 1;
  return compute_distance_bounded_edlib(pattern.pattern, text, tau);
}

int compute_distance_edlib_prepared(
    const PreparedEdlibDnaPattern& pattern,
    std::string_view text) {
  if (!pattern.handle) {
    return compute_distance_edlib(pattern.pattern, text);
  }
  const int distance = edlibDnaPreparedDistance(
      static_cast<const EdlibDnaPrepared*>(pattern.handle),
      text.data(), static_cast<int>(text.size()));
  return distance >= 0
             ? distance
             : compute_distance_edlib(pattern.pattern, text);
}

namespace {

constexpr size_t kMyersMaxPatternBits = 256;

int dna_base_index(char c) {
  switch (c) {
    case 'A':
      return 0;
    case 'C':
      return 1;
    case 'G':
      return 2;
    case 'T':
      return 3;
    default:
      return -1;
  }
}

bool is_acgt_string(std::string_view value) {
  for (char c : value) {
    if (dna_base_index(c) < 0) return false;
  }
  return true;
}

uint64_t valid_block_mask(size_t block, size_t pattern_length) {
  const size_t full_blocks = pattern_length / 64;
  const size_t tail_bits = pattern_length % 64;
  if (block < full_blocks) return ~uint64_t{0};
  if (tail_bits == 0) return ~uint64_t{0};
  return (uint64_t{1} << tail_bits) - 1;
}

template <typename PreparedPattern>
void prepare_myers_pattern_bits(
    std::string_view pattern, PreparedPattern& prepared) {
  const size_t m = pattern.size();
  if (m > kMyersMaxPatternBits) return;
  prepared.block_count = (m + 63) / 64;
  for (size_t block = 0; block < prepared.block_count; ++block) {
    prepared.masks[block] = valid_block_mask(block, m);
  }
  for (size_t i = 0; i < m; ++i) {
    const size_t block = i / 64;
    const size_t bit = i % 64;
    const int base = dna_base_index(pattern[i]);
    if (base < 0) return;
    prepared.peq[block][static_cast<size_t>(base)] |=
        (uint64_t{1} << bit);
  }
  prepared.supported = true;
}

PreparedMyersPattern prepare_myers_pattern_impl(
    std::string_view pattern) {
  PreparedMyersPattern prepared;
  prepared.pattern.assign(pattern.data(), pattern.size());
  prepare_myers_pattern_bits(pattern, prepared);
  return prepared;
}

int compute_distance_bounded_myers_single_word(
    const PreparedMyersPattern& pattern,
    std::string_view text,
    int tau) {
  const size_t m = pattern.pattern.size();
  const uint64_t valid_mask = pattern.masks[0];
  const uint64_t top_bit = uint64_t{1} << (m - 1);
  uint64_t vp = valid_mask;
  uint64_t vn = 0;
  int score = static_cast<int>(m);

  for (char c : text) {
    const int base = dna_base_index(c);
    if (base < 0) {
      return compute_distance_bounded_dp(pattern.pattern, text, tau);
    }
    const uint64_t eq =
        pattern.peq[0][static_cast<size_t>(base)];
    const uint64_t x = eq | vn;
    const uint64_t d0 = (((x & vp) + vp) ^ vp) | x;
    uint64_t hp = vn | ~(d0 | vp);
    uint64_t hn = d0 & vp;

    if (hp & top_bit) {
      ++score;
    } else if (hn & top_bit) {
      --score;
    }

    hp = ((hp << 1) | uint64_t{1}) & valid_mask;
    hn = (hn << 1) & valid_mask;
    vn = hp & d0;
    vp = hn | ~(hp | d0);
    vn &= valid_mask;
    vp &= valid_mask;
  }

  return score <= tau ? score : tau + 1;
}

int compute_distance_bounded_myers_multiword(
    const PreparedMyersPattern& pattern,
    std::string_view text,
    int tau) {
  const size_t m = pattern.pattern.size();
  const size_t block_count = pattern.block_count;
  std::array<uint64_t, 4> pv = {};
  std::array<uint64_t, 4> mv = {};
  std::array<uint64_t, 4> xv = {};
  std::array<uint64_t, 4> sum_words = {};
  std::array<uint64_t, 4> ph = {};
  std::array<uint64_t, 4> mh = {};

  for (size_t block = 0; block < block_count; ++block) {
    pv[block] = pattern.masks[block];
  }

  const size_t last_block = block_count - 1;
  const uint64_t top_bit = uint64_t{1} << ((m - 1) % 64);
  int score = static_cast<int>(m);

  for (char c : text) {
    const int base_index = dna_base_index(c);
    if (base_index < 0) {
      return compute_distance_bounded_dp(pattern.pattern, text, tau);
    }
    const size_t base = static_cast<size_t>(base_index);
    uint64_t carry = 0;
    for (size_t block = 0; block < block_count; ++block) {
      xv[block] = pattern.peq[block][base] | mv[block];
      const uint64_t addend = xv[block] & pv[block];
      const unsigned __int128 sum =
          static_cast<unsigned __int128>(addend) + pv[block] + carry;
      sum_words[block] = static_cast<uint64_t>(sum);
      carry = static_cast<uint64_t>(sum >> 64);
    }

    for (size_t block = 0; block < block_count; ++block) {
      const uint64_t xh = ((sum_words[block] ^ pv[block]) | xv[block]) &
                          pattern.masks[block];
      ph[block] =
          (mv[block] | ~(xh | pv[block])) & pattern.masks[block];
      mh[block] = (pv[block] & xh) & pattern.masks[block];
    }

    if (ph[last_block] & top_bit) {
      ++score;
    } else if (mh[last_block] & top_bit) {
      --score;
    }

    uint64_t ph_carry = 1;
    uint64_t mh_carry = 0;
    for (size_t block = 0; block < block_count; ++block) {
      const uint64_t next_ph_carry = ph[block] >> 63;
      const uint64_t next_mh_carry = mh[block] >> 63;
      ph[block] =
          ((ph[block] << 1) | ph_carry) & pattern.masks[block];
      mh[block] =
          ((mh[block] << 1) | mh_carry) & pattern.masks[block];
      ph_carry = next_ph_carry;
      mh_carry = next_mh_carry;
    }

    for (size_t block = 0; block < block_count; ++block) {
      mv[block] =
          (ph[block] & xv[block]) & pattern.masks[block];
      pv[block] =
          (mh[block] | ~(ph[block] | xv[block])) & pattern.masks[block];
    }
  }

  return score <= tau ? score : tau + 1;
}

}  // namespace

PreparedMyersPattern prepare_myers_pattern(std::string_view pattern) {
  return prepare_myers_pattern_impl(pattern);
}

PreparedMyersDnaPattern prepare_myers_dna_pattern(
    std::string_view pattern) {
  PreparedMyersDnaPattern prepared;
  prepared.pattern_length = pattern.size();
  prepare_myers_pattern_bits(pattern, prepared);
  return prepared;
}

int compute_distance_bounded_myers_prepared(
    const PreparedMyersPattern& pattern,
    std::string_view text,
    int tau) {
  if (tau < 0) {
    throw std::invalid_argument(
        "edit-distance threshold must be non-negative");
  }

  const int m = static_cast<int>(pattern.pattern.size());
  const int n = static_cast<int>(text.size());
  if (std::abs(m - n) > tau) return tau + 1;
  if (m == 0) return n <= tau ? n : tau + 1;
  if (n == 0) return m <= tau ? m : tau + 1;
  if (!pattern.supported) {
    return compute_distance_bounded_dp(pattern.pattern, text, tau);
  }
  if (pattern.pattern.size() <= 64) {
    return compute_distance_bounded_myers_single_word(pattern, text, tau);
  }
  return compute_distance_bounded_myers_multiword(pattern, text, tau);
}

#if NAVIGAMER_HAS_MYERS_BATCH4_AVX2
template <size_t BlockCount>
__attribute__((target("avx2"), always_inline))
inline bool compute_distance_bounded_myers_prepared_batch4_avx2_fixed(
    const PreparedMyersDnaPattern& pattern,
    const std::array<std::string_view, 4>& texts,
    int tau,
    std::array<int, 4>& distances) {
  const size_t pattern_length = pattern.pattern_length;
  const size_t text_length = texts[0].size();
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi64x(1);
  const __m256i sign = _mm256_set1_epi64x(
      static_cast<long long>(uint64_t{1} << 63));
  const __m256i all_ones = _mm256_set1_epi64x(-1);
  __m256i pv[BlockCount] = {};
  __m256i mv[BlockCount] = {};
  __m256i xv[BlockCount] = {};
  __m256i ph[BlockCount] = {};
  __m256i mh[BlockCount] = {};
  __m256i masks[BlockCount] = {};
#pragma GCC unroll 4
  for (size_t block = 0; block < BlockCount; ++block) {
    masks[block] = _mm256_set1_epi64x(
        static_cast<long long>(pattern.masks[block]));
    pv[block] = masks[block];
    mv[block] = zero;
  }

  __m256i scores = _mm256_set1_epi64x(
      static_cast<long long>(pattern_length));
  constexpr size_t last_block = BlockCount - 1;
  const __m256i top_bit = _mm256_set1_epi64x(
      static_cast<long long>(
          uint64_t{1} << ((pattern_length - 1) % 64)));

  for (size_t text_idx = 0; text_idx < text_length; ++text_idx) {
    const auto code = [&](size_t lane) -> size_t {
      const unsigned char base =
          static_cast<unsigned char>(texts[lane][text_idx]);
      return static_cast<size_t>(((base >> 1) ^ (base >> 2)) & 3);
    };
    const size_t code0 = code(0);
    const size_t code1 = code(1);
    const size_t code2 = code(2);
    const size_t code3 = code(3);

    __m256i carry = zero;
#pragma GCC unroll 4
    for (size_t block = 0; block < BlockCount; ++block) {
      const __m256i eq = _mm256_set_epi64x(
          static_cast<long long>(pattern.peq[block][code3]),
          static_cast<long long>(pattern.peq[block][code2]),
          static_cast<long long>(pattern.peq[block][code1]),
          static_cast<long long>(pattern.peq[block][code0]));
      xv[block] = _mm256_or_si256(eq, mv[block]);
      const __m256i addend = _mm256_and_si256(xv[block], pv[block]);
      const __m256i sum1 = _mm256_add_epi64(addend, pv[block]);
      const __m256i carry1 = _mm256_cmpgt_epi64(
          _mm256_xor_si256(addend, sign),
          _mm256_xor_si256(sum1, sign));
      const __m256i sum2 = _mm256_add_epi64(sum1, carry);
      const __m256i carry2 = _mm256_cmpgt_epi64(
          _mm256_xor_si256(sum1, sign),
          _mm256_xor_si256(sum2, sign));
      carry = _mm256_and_si256(
          _mm256_or_si256(carry1, carry2), one);

      __m256i xh = _mm256_or_si256(
          _mm256_xor_si256(sum2, pv[block]), xv[block]);
      if (block == last_block) {
        xh = _mm256_and_si256(xh, masks[block]);
      }
      ph[block] = _mm256_or_si256(
          mv[block],
          _mm256_andnot_si256(
              _mm256_or_si256(xh, pv[block]), all_ones));
      mh[block] = _mm256_and_si256(pv[block], xh);
      if (block == last_block) {
        ph[block] = _mm256_and_si256(ph[block], masks[block]);
        mh[block] = _mm256_and_si256(mh[block], masks[block]);
      }
    }

    const __m256i ph_zero = _mm256_cmpeq_epi64(
        _mm256_and_si256(ph[last_block], top_bit), zero);
    const __m256i mh_zero = _mm256_cmpeq_epi64(
        _mm256_and_si256(mh[last_block], top_bit), zero);
    scores = _mm256_add_epi64(
        scores, _mm256_andnot_si256(ph_zero, one));
    scores = _mm256_sub_epi64(
        scores, _mm256_andnot_si256(mh_zero, one));

    __m256i ph_carry = one;
    __m256i mh_carry = zero;
#pragma GCC unroll 4
    for (size_t block = 0; block < BlockCount; ++block) {
      const __m256i next_ph_carry =
          _mm256_srli_epi64(ph[block], 63);
      const __m256i next_mh_carry =
          _mm256_srli_epi64(mh[block], 63);
      ph[block] = _mm256_or_si256(
          _mm256_slli_epi64(ph[block], 1), ph_carry);
      mh[block] = _mm256_or_si256(
          _mm256_slli_epi64(mh[block], 1), mh_carry);
      if (block == last_block) {
        ph[block] = _mm256_and_si256(ph[block], masks[block]);
        mh[block] = _mm256_and_si256(mh[block], masks[block]);
      }
      ph_carry = next_ph_carry;
      mh_carry = next_mh_carry;
    }

#pragma GCC unroll 4
    for (size_t block = 0; block < BlockCount; ++block) {
      mv[block] = _mm256_and_si256(ph[block], xv[block]);
      pv[block] = _mm256_or_si256(
          mh[block],
          _mm256_andnot_si256(
              _mm256_or_si256(ph[block], xv[block]), all_ones));
      if (block == last_block) {
        mv[block] = _mm256_and_si256(mv[block], masks[block]);
        pv[block] = _mm256_and_si256(pv[block], masks[block]);
      }
    }

    // The current score is ED(pattern, text_prefix). Appending the remaining
    // text characters can reduce that distance by at most one per character,
    // so score - remaining is an exact lower bound on the final distance.
    // Check only periodically to keep the accepted/near-candidate hot path
    // cheap, and stop only when every SIMD lane is provably outside tau.
    if ((text_idx & 7U) == 7U) {
      const __m256i remaining = _mm256_set1_epi64x(
          static_cast<long long>(text_length - text_idx - 1));
      const __m256i threshold = _mm256_set1_epi64x(tau);
      const __m256i rejected = _mm256_cmpgt_epi64(
          _mm256_sub_epi64(scores, remaining), threshold);
      if (_mm256_movemask_pd(_mm256_castsi256_pd(rejected)) == 0x0f) {
        distances.fill(tau + 1);
        return true;
      }
    }
  }

  alignas(32) std::array<int64_t, 4> score_lanes{};
  _mm256_store_si256(
      reinterpret_cast<__m256i*>(score_lanes.data()), scores);
  for (size_t lane = 0; lane < 4; ++lane) {
    distances[lane] = score_lanes[lane] <= tau
                          ? static_cast<int>(score_lanes[lane])
                          : tau + 1;
  }
  return true;
}

__attribute__((target("avx2")))
bool compute_distance_bounded_myers_prepared_batch4_avx2(
    const PreparedMyersDnaPattern& pattern,
    const std::array<std::string_view, 4>& texts,
    int tau,
    std::array<int, 4>& distances) {
  switch (pattern.block_count) {
    case 1:
      return compute_distance_bounded_myers_prepared_batch4_avx2_fixed<1>(
          pattern, texts, tau, distances);
    case 2:
      return compute_distance_bounded_myers_prepared_batch4_avx2_fixed<2>(
          pattern, texts, tau, distances);
    case 3:
      return compute_distance_bounded_myers_prepared_batch4_avx2_fixed<3>(
          pattern, texts, tau, distances);
    case 4:
      return compute_distance_bounded_myers_prepared_batch4_avx2_fixed<4>(
          pattern, texts, tau, distances);
    default:
      return false;
  }
}
#endif

bool myers_batch4_avx2_runtime_supported() {
#if NAVIGAMER_HAS_MYERS_BATCH4_AVX2
  static const bool supported = [] {
    __builtin_cpu_init();
    return static_cast<bool>(__builtin_cpu_supports("avx2"));
  }();
  return supported;
#else
  return false;
#endif
}

bool compute_distance_bounded_myers_prepared_batch4_trusted_acgt(
    const PreparedMyersDnaPattern& pattern,
    const std::array<std::string_view, 4>& texts,
    int tau,
    std::array<int, 4>& distances) {
  if (tau < 0) {
    throw std::invalid_argument(
        "edit-distance threshold must be non-negative");
  }
  if (!myers_batch4_avx2_runtime_supported() ||
      !pattern.supported || pattern.pattern_length == 0 ||
      pattern.block_count == 0 || pattern.block_count > 4) {
    return false;
  }
  const size_t text_length = texts[0].size();
  if (text_length != pattern.pattern_length) return false;
  for (size_t lane = 1; lane < 4; ++lane) {
    if (texts[lane].size() != text_length) return false;
  }
#if NAVIGAMER_HAS_MYERS_BATCH4_AVX2
  return compute_distance_bounded_myers_prepared_batch4_avx2(
      pattern, texts, tau, distances);
#else
  (void)texts;
  (void)distances;
  return false;
#endif
}

bool compute_distance_bounded_myers_supported(std::string_view a,
                                              std::string_view b) {
  if (a.empty() || b.empty()) return true;
  const size_t shorter = std::min(a.size(), b.size());
  return shorter <= kMyersMaxPatternBits && is_acgt_string(a) &&
         is_acgt_string(b);
}

int compute_distance_bounded_myers(std::string_view a, std::string_view b,
                                   int tau) {
  if (tau < 0) throw std::invalid_argument("edit-distance threshold must be non-negative");

  const int m_input = static_cast<int>(a.size());
  const int n_input = static_cast<int>(b.size());
  if (std::abs(m_input - n_input) > tau) return tau + 1;
  if (a.empty()) return n_input <= tau ? n_input : tau + 1;
  if (b.empty()) return m_input <= tau ? m_input : tau + 1;

  std::string_view pattern = a;
  std::string_view text = b;
  if (pattern.size() > text.size()) std::swap(pattern, text);
  const size_t m = pattern.size();
  if (!compute_distance_bounded_myers_supported(a, b)) {
    return compute_distance_bounded_dp(a, b, tau);
  }
  const PreparedMyersPattern prepared =
      prepare_myers_pattern_impl(pattern);
  if (m <= 64) {
    return compute_distance_bounded_myers_single_word(prepared, text, tau);
  }
  return compute_distance_bounded_myers_multiword(prepared, text, tau);
}

int compute_distance_bounded_with_mode(std::string_view a,
                                       std::string_view b, int tau,
                                       DistanceMode mode) {
  switch (mode) {
    case DistanceMode::DP:
      return compute_distance_bounded_dp(a, b, tau);
    case DistanceMode::Myers:
      return compute_distance_bounded_myers(a, b, tau);
    case DistanceMode::Edlib:
      return compute_distance_bounded_edlib(a, b, tau);
    case DistanceMode::Auto:
      if (compute_distance_bounded_myers_supported(a, b)) {
        return compute_distance_bounded_myers(a, b, tau);
      }
      return compute_distance_bounded_edlib(a, b, tau);
  }
  return compute_distance_bounded_dp(a, b, tau);
}

int compute_distance_bounded(std::string_view a, std::string_view b,
                             int tau) {
  return compute_distance_bounded_dp(a, b, tau);
}

static int node_distance(const std::shared_ptr<WorldNode>& na,
                         const std::shared_ptr<WorldNode>& nb) {
  if (!na->center_ptr || !nb->center_ptr) return 0;
  return compute_distance(na->center_ptr->seq, nb->center_ptr->seq);
}

static int seq_distance(const std::shared_ptr<BioSequence>& sa,
                        const std::shared_ptr<BioSequence>& sb) {
  return compute_distance(sa->seq, sb->seq);
}

template <typename T, typename DistFunc>
std::vector<size_t> fps_impl(const std::vector<T>& items, size_t k, DistFunc dist_fn) {
  if (items.empty() || k == 0) return {};
  k = std::min(k, items.size());
  std::vector<size_t> chosen;
  std::vector<bool> used(items.size(), false);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, items.size() - 1);
  chosen.push_back(dis(gen));
  used[chosen[0]] = true;

  while (chosen.size() < k) {
    int best_idx = -1;
    int best_min_dist = -1;
    for (size_t i = 0; i < items.size(); ++i) {
      if (used[i]) continue;
      int min_d = std::numeric_limits<int>::max();
      for (size_t j : chosen)
        min_d = std::min(min_d, dist_fn(items[i], items[j]));
      if (min_d > best_min_dist) {
        best_min_dist = min_d;
        best_idx = static_cast<int>(i);
      }
    }
    if (best_idx < 0) break;
    chosen.push_back(static_cast<size_t>(best_idx));
    used[static_cast<size_t>(best_idx)] = true;
  }
  return chosen;
}

std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<WorldNode>>& nodes, size_t k) {
  return fps_impl(nodes, k, node_distance);
}

std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<BioSequence>>& sequences, size_t k) {
  return fps_impl(sequences, k, seq_distance);
}

void shuffle_indices(std::vector<size_t>& indices, unsigned seed) {
  std::mt19937 gen(seed);
  std::shuffle(indices.begin(), indices.end(), gen);
}

#undef NAVIGAMER_DISTANCE_HOT_ALIGN

}  // namespace navigamer
