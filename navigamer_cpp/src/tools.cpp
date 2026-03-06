#include "tools.hpp"
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>

namespace navigamer {

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
