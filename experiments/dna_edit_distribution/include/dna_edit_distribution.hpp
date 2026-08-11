#ifndef DNA_EDIT_DISTRIBUTION_HPP
#define DNA_EDIT_DISTRIBUTION_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
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

struct Summary {
  double mean = 0.0;
  double standard_deviation = 0.0;
  int min = 0;
  int median = 0;
  int max = 0;
  int mode = 0;
  int q05 = 0;
  int q95 = 0;
};

struct RunParameters {
  std::size_t length = 150;
  std::uint64_t pairs = 1000000;
  std::uint64_t seed = 20260719;
  int requested_threads = 1;
  std::filesystem::path output_dir = "results";
};

struct RunTiming {
  std::string started_at_utc;
  std::string finished_at_utc;
  double elapsed_seconds = 0.0;
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

Summary summarize(const std::vector<std::uint64_t>& counts);

void write_results(const RunParameters& parameters,
                   const HistogramResult& histogram,
                   const Summary& summary,
                   const RunTiming& timing);

}  // namespace dna_edit_distribution

#endif
