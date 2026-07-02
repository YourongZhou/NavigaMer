#ifndef NAVIGAMER_CANDIDATE_VERIFIER_HPP
#define NAVIGAMER_CANDIDATE_VERIFIER_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace navigamer {

enum class CandidateTruthMode {
  Source,
  Exhaustive,
};

struct CandidateVerifierConfig {
  std::string reference_input;
  std::string reads_fastq_path;
  std::string candidates_tsv_path;
  std::string detail_tsv_path;
  std::string summary_tsv_path;
  int tolerance = 0;
  int window_length = 150;
  int stride = 1;
  CandidateTruthMode truth_mode = CandidateTruthMode::Source;
};

struct CandidateVerifierSummary {
  size_t query_count = 0;
  size_t raw_candidate_count = 0;
  size_t verified_match_count = 0;
  size_t truth_match_count = 0;
  size_t tp_count = 0;
  size_t fp_count = 0;
  size_t fn_count = 0;
  double verify_ms = 0.0;
  double truth_ms = 0.0;
};

CandidateTruthMode parse_candidate_truth_mode(const std::string& value);

CandidateVerifierSummary run_candidate_verifier(
    const CandidateVerifierConfig& config);

}  // namespace navigamer

#endif  // NAVIGAMER_CANDIDATE_VERIFIER_HPP
