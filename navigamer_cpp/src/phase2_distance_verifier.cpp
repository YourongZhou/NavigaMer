#include "phase2_distance_verifier.hpp"

#include <stdexcept>

namespace navigamer {

namespace {

class CpuPhase2DistanceVerifier final : public Phase2DistanceVerifier {
 public:
  explicit CpuPhase2DistanceVerifier(DistanceMode distance_mode)
      : distance_mode_(distance_mode) {}

  Phase2DistanceBatchResult verify(
      const std::vector<std::string>& parent_sequences,
      const std::vector<std::string>& child_sequences,
      const std::vector<Phase2DistancePair>& pairs) override {
    Phase2DistanceBatchResult result;
    result.accepted_pair_indices.reserve(pairs.size());

    for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
      const auto& pair = pairs[pair_idx];
      if (pair.parent_idx >= parent_sequences.size() ||
          pair.child_idx >= child_sequences.size()) {
        throw std::out_of_range("phase2 distance pair index out of range");
      }
      const int dist = compute_distance_bounded_with_mode(
          parent_sequences[pair.parent_idx],
          child_sequences[pair.child_idx],
          pair.tau,
          distance_mode_);
      if (dist <= pair.tau) result.accepted_pair_indices.push_back(pair_idx);
    }
    return result;
  }

 private:
  DistanceMode distance_mode_;
};

}  // namespace

std::unique_ptr<Phase2DistanceVerifier> make_phase2_distance_verifier(
    DistanceMode distance_mode) {
  return std::unique_ptr<Phase2DistanceVerifier>(
      new CpuPhase2DistanceVerifier(distance_mode));
}

}  // namespace navigamer
