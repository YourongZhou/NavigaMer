#pragma once

#include "index_persistence.hpp"
#include "reference_windows.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct OracleMetrics {
  std::optional<uint32_t> true_neighbor_count;
  std::optional<uint32_t> false_negative_count;
  std::optional<double> recall;
  std::optional<double> raw_candidate_blowup;
  std::optional<double> accepted_candidate_blowup;
};

struct PerReadResult {
  std::string read_id;
  uint32_t tolerance = 0;
  std::vector<uint32_t> raw_candidate_ids;
  std::vector<uint32_t> verified_candidate_ids;
  std::vector<uint32_t> accepted_candidate_ids;
  uint32_t raw_candidate_count = 0;
  uint32_t verified_candidate_count = 0;
  uint32_t accepted_candidate_count = 0;
  double retrieval_milliseconds = 0.0;
  double verification_milliseconds = 0.0;
  double total_milliseconds = 0.0;
  std::optional<double> source_recovery;
  std::optional<OracleMetrics> oracle;
};

struct SummaryStats {
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
};

using DistanceVerifier =
    std::function<int(const std::string&, const std::string&, int)>;
using CandidateRetriever = std::function<std::vector<uint32_t>()>;

struct BuildMatrixRequest {
  std::filesystem::path reference_path;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  std::filesystem::path out_dir;
  bool rebuild = false;
};

struct BuildSummaryRow {
  IndexManifest manifest;
  std::filesystem::path index_dir;
  std::filesystem::path index_path;
  std::string variant;
  bool reused = false;
  double wall_seconds = 0.0;
};

struct QueryRead {
  std::string read_id;
  std::string sequence;
};

struct ComparisonRequest {
  std::filesystem::path reference_path;
  std::filesystem::path reads_path;
  std::vector<QueryRead> reads;
  uint32_t window_length = 0;
  uint32_t stride = 0;
  uint32_t tolerance = 0;
  std::filesystem::path out_dir;
  bool oracle_enabled = true;
  bool rebuild = false;
  std::filesystem::path navigamer_binary;
  std::filesystem::path navigamer_index_path;
  std::size_t tensor_top_k = 64;
};

struct ComparisonPerReadRow {
  std::string method;
  std::string variant;
  PerReadResult result;
};

struct ComparisonSummaryRow {
  std::string method;
  std::string variant;
  uint32_t read_count = 0;
  SummaryStats raw_candidate_count;
  SummaryStats accepted_candidate_count;
  SummaryStats retrieval_milliseconds;
  SummaryStats verification_milliseconds;
  SummaryStats total_milliseconds;
  uint32_t oracle_read_count = 0;
  uint32_t true_neighbor_count_total = 0;
  uint32_t false_negative_count_total = 0;
  std::optional<double> mean_recall;
  std::optional<double> mean_raw_candidate_blowup;
  std::optional<double> mean_accepted_candidate_blowup;
};

struct ComparisonReport {
  std::vector<BuildSummaryRow> build_rows;
  std::vector<ComparisonPerReadRow> per_read_rows;
  std::vector<ComparisonSummaryRow> summary_rows;
};

PerReadResult evaluate_candidates(const std::string& read_id,
                                  const std::string& query_sequence,
                                  const std::vector<uint32_t>& raw_candidates,
                                  const ReferenceWindows& reference,
                                  uint32_t tolerance,
                                  DistanceVerifier verifier = {});
PerReadResult evaluate_candidates(const std::string& read_id,
                                  const std::string& query_sequence,
                                  const CandidateRetriever& retriever,
                                  const ReferenceWindows& reference,
                                  uint32_t tolerance,
                                  DistanceVerifier verifier = {});

SummaryStats summarize_samples(std::vector<double> samples);
std::string render_oracle_metrics_tsv(const OracleMetrics& oracle);
std::string render_build_summary_row_tsv(const BuildSummaryRow& row);
void write_build_summary_tsv(const std::filesystem::path& path,
                            const std::vector<BuildSummaryRow>& rows);

std::vector<BuildSummaryRow> build_candidate_matrix(
    const BuildMatrixRequest& request);
ComparisonReport run_comparison(const ComparisonRequest& request);
