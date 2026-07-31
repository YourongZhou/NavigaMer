#ifndef NAVIGAMER_PHASE2_DISTANCE_VERIFIER_HPP
#define NAVIGAMER_PHASE2_DISTANCE_VERIFIER_HPP

#include "tools.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace navigamer {

struct Phase2DistancePair {
  size_t parent_idx = 0;
  size_t child_idx = 0;
  int tau = 0;
};

struct Phase2DistanceBatchResult {
  std::vector<size_t> accepted_pair_indices;
};

class Phase2DistanceVerifier {
 public:
  virtual ~Phase2DistanceVerifier() = default;

  virtual Phase2DistanceBatchResult verify(
      const std::vector<const std::string*>& parent_sequences,
      const std::vector<const std::string*>& child_sequences,
      const std::vector<Phase2DistancePair>& pairs) = 0;
};

std::unique_ptr<Phase2DistanceVerifier> make_phase2_distance_verifier(
    DistanceMode distance_mode);

}  // namespace navigamer

#endif  // NAVIGAMER_PHASE2_DISTANCE_VERIFIER_HPP
