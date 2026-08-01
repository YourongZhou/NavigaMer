#ifndef NAVIGAMER_TOOLS_HPP
#define NAVIGAMER_TOOLS_HPP

#include "structure.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
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

int compute_distance(std::string_view a, std::string_view b);
int compute_distance(const BioSequence& a, const BioSequence& b);
int compute_distance_bounded_dp(std::string_view a, std::string_view b,
                                int tau);
int compute_distance_edlib(std::string_view a, std::string_view b);
int compute_distance_bounded_edlib(std::string_view a, std::string_view b,
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
    std::string_view pattern);
int compute_distance_bounded_edlib_prepared(
    const PreparedEdlibDnaPattern& pattern,
    std::string_view text,
    int tau);
int compute_distance_edlib_prepared(
    const PreparedEdlibDnaPattern& pattern,
    std::string_view text);
bool compute_distance_bounded_myers_supported(std::string_view a,
                                              std::string_view b);
int compute_distance_bounded_myers(std::string_view a, std::string_view b,
                                   int tau);

struct PreparedMyersPattern {
  std::string pattern;
  std::array<std::array<uint64_t, 4>, 4> peq{};
  std::array<uint64_t, 4> masks{};
  size_t block_count = 0;
  bool supported = false;
};

PreparedMyersPattern prepare_myers_pattern(std::string_view pattern);
int compute_distance_bounded_myers_prepared(
    const PreparedMyersPattern& pattern,
    std::string_view text,
    int tau);
bool myers_batch4_avx2_runtime_supported();
// Computes four exact bounded distances. The caller guarantees that the
// shared pattern and all texts contain only uppercase A/C/G/T. Returns false
// when AVX2 or the equal-length <=256 bp shape is unavailable, in which case
// callers must use the scalar verifier.
bool compute_distance_bounded_myers_prepared_batch4_trusted_acgt(
    const PreparedMyersPattern& pattern,
    const std::array<std::string_view, 4>& texts,
    int tau,
    std::array<int, 4>& distances);

int compute_distance_bounded_with_mode(std::string_view a,
                                       std::string_view b, int tau,
                                       DistanceMode mode);
int compute_distance_bounded(std::string_view a, std::string_view b, int tau);

// Farthest Point Sampling: choose k dispersed candidates.
std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<WorldNode>>& nodes, size_t k);
std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<BioSequence>>& sequences, size_t k);

// Shuffle the [0, n) index range with a fixed seed.
void shuffle_indices(std::vector<size_t>& indices, unsigned seed);

}  // namespace navigamer

#endif  // NAVIGAMER_TOOLS_HPP
