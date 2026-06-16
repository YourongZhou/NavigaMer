#include "tools.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace navigamer {

const char* distance_mode_name(DistanceMode mode) {
  switch (mode) {
    case DistanceMode::DP:
      return "dp";
    case DistanceMode::Myers:
      return "myers";
    case DistanceMode::Auto:
      return "auto";
  }
  return "dp";
}

DistanceMode parse_distance_mode(const std::string& value) {
  if (value == "dp") return DistanceMode::DP;
  if (value == "myers") return DistanceMode::Myers;
  if (value == "auto") return DistanceMode::Auto;
  throw std::invalid_argument("distance mode must be dp, myers, or auto");
}

int compute_distance(const std::string& a, const std::string& b) {
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

int compute_distance_bounded_dp(const std::string& a, const std::string& b,
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

bool is_acgt_string(const std::string& value) {
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

int compute_distance_bounded_myers_single_word(const std::string& pattern,
                                               const std::string& text,
                                               int tau) {
  const size_t m = pattern.size();
  std::array<uint64_t, 4> peq = {0, 0, 0, 0};
  for (size_t i = 0; i < m; ++i) {
    const int base = dna_base_index(pattern[i]);
    peq[static_cast<size_t>(base)] |= (uint64_t{1} << i);
  }

  const uint64_t valid_mask = m == 64 ? ~uint64_t{0} : ((uint64_t{1} << m) - 1);
  const uint64_t top_bit = uint64_t{1} << (m - 1);
  uint64_t vp = valid_mask;
  uint64_t vn = 0;
  int score = static_cast<int>(m);

  for (char c : text) {
    const uint64_t eq = peq[static_cast<size_t>(dna_base_index(c))];
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

int compute_distance_bounded_myers_multiword(const std::string& pattern,
                                             const std::string& text,
                                             int tau) {
  const size_t m = pattern.size();
  const size_t block_count = (m + 63) / 64;
  std::array<std::array<uint64_t, 4>, 4> peq = {};
  std::array<uint64_t, 4> masks = {};
  std::array<uint64_t, 4> pv = {};
  std::array<uint64_t, 4> mv = {};
  std::array<uint64_t, 4> xv = {};
  std::array<uint64_t, 4> sum_words = {};
  std::array<uint64_t, 4> ph = {};
  std::array<uint64_t, 4> mh = {};

  for (size_t block = 0; block < block_count; ++block) {
    masks[block] = valid_block_mask(block, m);
    pv[block] = masks[block];
  }
  for (size_t i = 0; i < m; ++i) {
    const size_t block = i / 64;
    const size_t bit = i % 64;
    const int base = dna_base_index(pattern[i]);
    peq[block][static_cast<size_t>(base)] |= (uint64_t{1} << bit);
  }

  const size_t last_block = block_count - 1;
  const uint64_t top_bit = uint64_t{1} << ((m - 1) % 64);
  int score = static_cast<int>(m);

  for (char c : text) {
    const size_t base = static_cast<size_t>(dna_base_index(c));
    uint64_t carry = 0;
    for (size_t block = 0; block < block_count; ++block) {
      xv[block] = peq[block][base] | mv[block];
      const uint64_t addend = xv[block] & pv[block];
      const unsigned __int128 sum =
          static_cast<unsigned __int128>(addend) + pv[block] + carry;
      sum_words[block] = static_cast<uint64_t>(sum);
      carry = static_cast<uint64_t>(sum >> 64);
    }

    for (size_t block = 0; block < block_count; ++block) {
      const uint64_t xh = ((sum_words[block] ^ pv[block]) | xv[block]) &
                          masks[block];
      ph[block] = (mv[block] | ~(xh | pv[block])) & masks[block];
      mh[block] = (pv[block] & xh) & masks[block];
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
      ph[block] = ((ph[block] << 1) | ph_carry) & masks[block];
      mh[block] = ((mh[block] << 1) | mh_carry) & masks[block];
      ph_carry = next_ph_carry;
      mh_carry = next_mh_carry;
    }

    for (size_t block = 0; block < block_count; ++block) {
      mv[block] = (ph[block] & xv[block]) & masks[block];
      pv[block] = (mh[block] | ~(ph[block] | xv[block])) & masks[block];
    }
  }

  return score <= tau ? score : tau + 1;
}

}  // namespace

bool compute_distance_bounded_myers_supported(const std::string& a,
                                              const std::string& b) {
  if (a.empty() || b.empty()) return true;
  const size_t shorter = std::min(a.size(), b.size());
  return shorter <= kMyersMaxPatternBits && is_acgt_string(a) &&
         is_acgt_string(b);
}

int compute_distance_bounded_myers(const std::string& a, const std::string& b,
                                   int tau) {
  if (tau < 0) throw std::invalid_argument("edit-distance threshold must be non-negative");

  const int m_input = static_cast<int>(a.size());
  const int n_input = static_cast<int>(b.size());
  if (std::abs(m_input - n_input) > tau) return tau + 1;
  if (a.empty()) return n_input <= tau ? n_input : tau + 1;
  if (b.empty()) return m_input <= tau ? m_input : tau + 1;

  const std::string* pattern = &a;
  const std::string* text = &b;
  if (pattern->size() > text->size()) std::swap(pattern, text);
  const size_t m = pattern->size();
  if (!compute_distance_bounded_myers_supported(a, b)) {
    return compute_distance_bounded_dp(a, b, tau);
  }
  if (m <= 64) {
    return compute_distance_bounded_myers_single_word(*pattern, *text, tau);
  }
  return compute_distance_bounded_myers_multiword(*pattern, *text, tau);
}

int compute_distance_bounded_with_mode(const std::string& a,
                                       const std::string& b, int tau,
                                       DistanceMode mode) {
  switch (mode) {
    case DistanceMode::DP:
      return compute_distance_bounded_dp(a, b, tau);
    case DistanceMode::Myers:
      return compute_distance_bounded_myers(a, b, tau);
    case DistanceMode::Auto:
      return compute_distance_bounded_dp(a, b, tau);
  }
  return compute_distance_bounded_dp(a, b, tau);
}

int compute_distance_bounded(const std::string& a, const std::string& b,
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

}  // namespace navigamer
