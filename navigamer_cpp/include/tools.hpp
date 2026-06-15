#ifndef NAVIGAMER_TOOLS_HPP
#define NAVIGAMER_TOOLS_HPP

#include "structure.hpp"
#include <vector>
#include <cstddef>

namespace navigamer {

// Levenshtein edit distance.
int compute_distance(const std::string& a, const std::string& b);
int compute_distance(const BioSequence& a, const BioSequence& b);
int compute_distance_bounded(const std::string& a, const std::string& b, int tau);

// Farthest Point Sampling: choose k dispersed candidates.
std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<WorldNode>>& nodes, size_t k);
std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<BioSequence>>& sequences, size_t k);

// Shuffle the [0, n) index range with a fixed seed.
void shuffle_indices(std::vector<size_t>& indices, unsigned seed);

}  // namespace navigamer

#endif  // NAVIGAMER_TOOLS_HPP
