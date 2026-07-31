#ifndef NAVIGAMER_PHASE2_DISTANCE_VERIFIER_HPP
#define NAVIGAMER_PHASE2_DISTANCE_VERIFIER_HPP

#include "tools.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace navigamer {

struct Phase2DistancePair {
  uint32_t parent_idx = 0;
  uint32_t child_idx = 0;
};
static_assert(sizeof(Phase2DistancePair) == 8,
              "phase2 distance pair must remain a compact 8-byte value");

struct Phase2DistanceBatchResult {
  std::vector<uint32_t> accepted_pair_indices;
};

class Phase2DistanceVerifier {
 public:
  virtual ~Phase2DistanceVerifier() = default;

  virtual Phase2DistanceBatchResult verify(
      const std::vector<std::string_view>& parent_sequences,
      const std::vector<std::string_view>& child_sequences,
      const std::vector<Phase2DistancePair>& pairs,
      int tau) = 0;
};

std::unique_ptr<Phase2DistanceVerifier> make_phase2_distance_verifier(
    DistanceMode distance_mode);

}  // namespace navigamer

#endif  // NAVIGAMER_PHASE2_DISTANCE_VERIFIER_HPP
