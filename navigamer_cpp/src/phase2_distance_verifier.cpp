#include "phase2_distance_verifier.hpp"

#include <limits>
#include <stdexcept>

namespace navigamer {

namespace {

class CpuPhase2DistanceVerifier final : public Phase2DistanceVerifier {
 public:
  explicit CpuPhase2DistanceVerifier(DistanceMode distance_mode)
      : distance_mode_(distance_mode) {}

  Phase2DistanceBatchResult verify(
      const std::vector<const std::string*>& parent_sequences,
      const std::vector<const std::string*>& child_sequences,
      const std::vector<Phase2DistancePair>& pairs) override {
    Phase2DistanceBatchResult result;
    result.accepted_pair_indices.reserve(pairs.size());

    bool prepare_parent = false;
    if (distance_mode_ == DistanceMode::Edlib && !pairs.empty()) {
      size_t parent_runs = 1;
      size_t child_runs = 1;
      for (size_t pair_idx = 1; pair_idx < pairs.size(); ++pair_idx) {
        parent_runs +=
            pairs[pair_idx].parent_idx != pairs[pair_idx - 1].parent_idx;
        child_runs +=
            pairs[pair_idx].child_idx != pairs[pair_idx - 1].child_idx;
      }
      prepare_parent = parent_runs <= child_runs;
    }
    size_t prepared_idx = std::numeric_limits<size_t>::max();
    PreparedEdlibDnaPattern prepared;

    for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
      const auto& pair = pairs[pair_idx];
      if (pair.parent_idx >= parent_sequences.size() ||
          pair.child_idx >= child_sequences.size() ||
          !parent_sequences[pair.parent_idx] ||
          !child_sequences[pair.child_idx]) {
        throw std::out_of_range("phase2 distance pair index out of range");
      }
      int dist = 0;
      if (distance_mode_ == DistanceMode::Edlib) {
        const size_t pattern_idx =
            prepare_parent ? pair.parent_idx : pair.child_idx;
        if (pattern_idx != prepared_idx) {
          prepared = prepare_edlib_dna_pattern(
              prepare_parent ? *parent_sequences[pair.parent_idx]
                             : *child_sequences[pair.child_idx]);
          prepared_idx = pattern_idx;
        }
        dist = compute_distance_bounded_edlib_prepared(
            prepared,
            prepare_parent ? *child_sequences[pair.child_idx]
                           : *parent_sequences[pair.parent_idx],
            pair.tau);
      } else {
        dist = compute_distance_bounded_with_mode(
            *parent_sequences[pair.parent_idx],
            *child_sequences[pair.child_idx],
            pair.tau,
            distance_mode_);
      }
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
