#ifndef DNA_EDIT_DISTRIBUTION_HPP
#define DNA_EDIT_DISTRIBUTION_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include <wavefront/wfa.h>
}

namespace dna_edit_distribution {

struct HistogramResult {
  std::vector<std::uint64_t> counts;
  int actual_threads = 0;
};

void generate_pair(std::size_t length,
                   std::uint64_t seed,
                   std::uint64_t pair_index,
                   std::string& first,
                   std::string& second);

int levenshtein_dp(std::string_view first, std::string_view second);

class ExactWfaAligner {
 public:
  ExactWfaAligner();
  ~ExactWfaAligner();

  ExactWfaAligner(const ExactWfaAligner&) = delete;
  ExactWfaAligner& operator=(const ExactWfaAligner&) = delete;

  int distance(std::string_view first, std::string_view second);

 private:
  wavefront_aligner_t* aligner_ = nullptr;
};

HistogramResult compute_histogram(std::size_t length,
                                  std::uint64_t pairs,
                                  std::uint64_t seed,
                                  int threads);

}  // namespace dna_edit_distribution

#endif
