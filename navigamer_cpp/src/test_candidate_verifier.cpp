#include "candidate_verifier.hpp"

#include <cassert>
#include <fstream>
#include <string>

namespace {

void write_file(const std::string& path, const std::string& contents) {
  std::ofstream out(path);
  out << contents;
}

}  // namespace

int main() {
  const std::string ref_path = "/tmp/navigamer_candidate_verifier_ref.fa";
  const std::string reads_path = "/tmp/navigamer_candidate_verifier_reads.fq";
  const std::string candidates_path =
      "/tmp/navigamer_candidate_verifier_candidates.tsv";
  const std::string detail_path = "/tmp/navigamer_candidate_verifier_detail.tsv";
  const std::string summary_path =
      "/tmp/navigamer_candidate_verifier_summary.tsv";

  write_file(ref_path, ">ref\nAAAACCCCGGGGTTTT\n");
  write_file(reads_path,
             "@q0 source_pos=4\n"
             "CCCC\n"
             "+\n"
             "IIII\n"
             "@q1 source_pos=8\n"
             "GGGA\n"
             "+\n"
             "IIII\n"
	             "@q2 source_pos=12\n"
	             "TTTT\n"
	             "+\n"
	             "IIII\n");
  write_file(candidates_path,
	             "read_id\ttau\traw_candidate_count\tcandidate_window_ids\n"
	             "q0\t1\t3\t4,0,5\n"
	             "q1\t1\t1\t8\n"
	             "q2\t1\t1\t0\n");

  navigamer::CandidateVerifierConfig source_config;
  source_config.reference_input = ref_path;
  source_config.reads_fastq_path = reads_path;
  source_config.candidates_tsv_path = candidates_path;
  source_config.detail_tsv_path = detail_path;
  source_config.summary_tsv_path = summary_path;
  source_config.tolerance = 1;
  source_config.window_length = 4;
  source_config.stride = 1;
  source_config.truth_mode = navigamer::CandidateTruthMode::Source;

  const auto source_summary =
      navigamer::run_candidate_verifier(source_config);
  assert(source_summary.query_count == 3);
  assert(source_summary.raw_candidate_count == 5);
  assert(source_summary.verified_match_count == 3);
  assert(source_summary.truth_match_count == 3);
  assert(source_summary.tp_count == 2);
  assert(source_summary.fp_count == 1);
  assert(source_summary.fn_count == 1);

  navigamer::CandidateVerifierConfig exhaustive_config = source_config;
  exhaustive_config.summary_tsv_path =
      "/tmp/navigamer_candidate_verifier_summary_exhaustive.tsv";
  exhaustive_config.truth_mode = navigamer::CandidateTruthMode::Exhaustive;
	  const auto exhaustive_summary =
	      navigamer::run_candidate_verifier(exhaustive_config);
	  assert(exhaustive_summary.truth_match_count >= 3);
	  assert(exhaustive_summary.tp_count == exhaustive_summary.verified_match_count);
	  assert(exhaustive_summary.fp_count == 0);
	  assert(exhaustive_summary.fn_count > 0);

  assert(navigamer::parse_candidate_truth_mode("source") ==
         navigamer::CandidateTruthMode::Source);
  assert(navigamer::parse_candidate_truth_mode("exhaustive") ==
         navigamer::CandidateTruthMode::Exhaustive);

  return 0;
}
