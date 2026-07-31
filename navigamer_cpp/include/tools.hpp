#ifndef NAVIGAMER_TOOLS_HPP
#define NAVIGAMER_TOOLS_HPP

#include "structure.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace navigamer {

// Levenshtein edit distance.
enum class DistanceMode {
  DP,
  Myers,
  Edlib,
  Auto,
};

const char* distance_mode_name(DistanceMode mode);
DistanceMode parse_distance_mode(const std::string& value);

int compute_distance(const std::string& a, const std::string& b);
int compute_distance(const BioSequence& a, const BioSequence& b);
int compute_distance_bounded_dp(const std::string& a, const std::string& b,
                                int tau);
int compute_distance_edlib(const std::string& a, const std::string& b);
int compute_distance_bounded_edlib(const std::string& a, const std::string& b,
                                   int tau);

struct PreparedEdlibDnaPattern {
  std::string pattern;
  void* handle = nullptr;

  PreparedEdlibDnaPattern() = default;
  ~PreparedEdlibDnaPattern();
  PreparedEdlibDnaPattern(PreparedEdlibDnaPattern&& other) noexcept;
  PreparedEdlibDnaPattern& operator=(
      PreparedEdlibDnaPattern&& other) noexcept;
  PreparedEdlibDnaPattern(const PreparedEdlibDnaPattern&) = delete;
  PreparedEdlibDnaPattern& operator=(
      const PreparedEdlibDnaPattern&) = delete;
};

PreparedEdlibDnaPattern prepare_edlib_dna_pattern(
    const std::string& pattern);
int compute_distance_bounded_edlib_prepared(
    const PreparedEdlibDnaPattern& pattern,
    const std::string& text,
    int tau);
int compute_distance_edlib_prepared(
    const PreparedEdlibDnaPattern& pattern,
    const std::string& text);
bool compute_distance_bounded_myers_supported(const std::string& a,
                                              const std::string& b);
int compute_distance_bounded_myers(const std::string& a, const std::string& b,
                                   int tau);

struct PreparedMyersPattern {
  std::string pattern;
  std::array<std::array<uint64_t, 4>, 4> peq{};
  std::array<uint64_t, 4> masks{};
  size_t block_count = 0;
  bool supported = false;
};

PreparedMyersPattern prepare_myers_pattern(const std::string& pattern);
int compute_distance_bounded_myers_prepared(
    const PreparedMyersPattern& pattern,
    const std::string& text,
    int tau);

int compute_distance_bounded_with_mode(const std::string& a,
                                       const std::string& b, int tau,
                                       DistanceMode mode);
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
