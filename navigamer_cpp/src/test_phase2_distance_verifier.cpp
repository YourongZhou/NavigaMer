#include "phase2_distance_verifier.hpp"
#include "tools.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace {

void test_cpu_batch_matches_direct_distance() {
  using navigamer::DistanceMode;
  using navigamer::Phase2DistancePair;

  const std::vector<std::string> parents = {
      "ACGTACGT",
      "AAAAACCC",
      "GGGGTTTT"};
  const std::vector<std::string> children = {
      "ACGTTCGT",
      "TTTTTTTT",
      "AAAAACCA",
      "GGGGTTTA"};
  const std::vector<std::string_view> parent_views = {
      parents[0], parents[1], parents[2]};
  const std::vector<std::string_view> child_views = {
      children[0], children[1], children[2], children[3]};
  const std::vector<Phase2DistancePair> pairs = {
      {0, 0},
      {0, 1},
      {1, 2},
      {2, 3}};

  auto verifier =
      navigamer::make_phase2_distance_verifier(DistanceMode::Edlib);
  constexpr int tau = 1;
  const auto result = verifier->verify(parent_views, child_views, pairs, tau);

  std::vector<uint32_t> expected;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto& pair = pairs[i];
    const int dist = navigamer::compute_distance_bounded_with_mode(
        parents[pair.parent_idx], children[pair.child_idx], tau,
        DistanceMode::Edlib);
    if (dist <= tau) expected.push_back(i);
  }

  assert(result.accepted_pair_indices == expected);
}

}  // namespace

int main() {
  test_cpu_batch_matches_direct_distance();
  return 0;
}
