#include "evaluation.hpp"

#include "candidate_indexes.hpp"
#include "reference_windows.hpp"

#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class TempDirectory {
 public:
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("navigamer_evaluation_" + std::to_string(random()) + "_" +
               std::to_string(random()));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error) {
        throw std::runtime_error("unable to create temporary directory: " +
                                 error.message());
      }
    }
    throw std::runtime_error("unable to allocate temporary directory");
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  std::filesystem::path file(std::string_view name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

class TempFasta {
 public:
  TempFasta(std::string_view name, std::string_view contents)
      : directory_(std::filesystem::temp_directory_path() /
                   ("navigamer_evaluation_fasta_" + std::string(name) + "_" +
                    std::to_string(std::random_device{}()))),
        path_(directory_ / (std::string(name) + ".fa")) {
    std::error_code error;
    std::filesystem::create_directory(directory_, error);
    if (error) {
      throw std::runtime_error("unable to create temporary FASTA directory: " +
                               error.message());
    }
    std::ofstream output(path_);
    if (!output) {
      throw std::runtime_error("unable to create temporary FASTA");
    }
    output << contents;
    output.close();
    if (!output) {
      throw std::runtime_error("unable to write temporary FASTA");
    }
  }

  ~TempFasta() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  const std::string path() const { return path_.string(); }

 private:
  std::filesystem::path directory_;
  std::filesystem::path path_;
};

void assert_throws(const std::function<void()>& function,
                   std::string_view substring) {
  try {
    function();
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find(substring) == std::string::npos) {
      throw std::runtime_error(std::string("unexpected message: ") +
                               error.what());
    }
    return;
  }
  throw std::runtime_error("expected exception was not thrown");
}

std::vector<uint32_t> brute_force_accepted(const ReferenceWindows& reference,
                                          std::string_view query,
                                          uint32_t tau,
                                          const std::vector<uint32_t>& raw) {
  std::vector<uint32_t> accepted;
  for (uint32_t window_id : raw) {
    const int distance = navigamer::compute_distance_bounded_edlib(
        std::string(query), std::string(reference.window(window_id)),
        static_cast<int>(tau));
    if (distance <= static_cast<int>(tau)) {
      accepted.push_back(window_id);
    }
  }
  return accepted;
}

void test_every_raw_candidate_is_verified_by_bounded_edlib() {
  TempFasta fasta("verify", ">ref\nACGTACGTACGT\n");
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(fasta.path(), 4, 1);

  const std::vector<uint32_t> raw_candidates{0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<std::pair<std::string, std::string>> verifier_calls;
  auto verifier = [&](const std::string& query, const std::string& window,
                      int tau) {
    verifier_calls.emplace_back(query, window);
    return navigamer::compute_distance_bounded_edlib(query, window, tau);
  };

  const PerReadResult result = evaluate_candidates(
      "read0", "ACGT", raw_candidates, reference, 1, verifier);

  assert(result.raw_candidate_count == raw_candidates.size());
  assert(result.verified_candidate_count == raw_candidates.size());
  assert(verifier_calls.size() == raw_candidates.size());
}

void test_accepted_ids_match_bruteforce_on_small_case() {
  TempFasta fasta("accepted", ">ref\nACGTACGTACGT\n");
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(fasta.path(), 4, 1);

  const std::vector<uint32_t> raw_candidates{0, 1, 2, 3, 4, 5, 6, 7, 8};
  const PerReadResult result =
      evaluate_candidates("read1", "ACGA", raw_candidates, reference, 1);

  assert(result.accepted_candidate_ids ==
         brute_force_accepted(reference, "ACGA", 1, raw_candidates));
}

void test_percentile_nearest_rank_is_fixed() {
  std::vector<double> samples(20);
  std::iota(samples.begin(), samples.end(), 1.0);
  const SummaryStats stats = summarize_samples(samples);
  assert(stats.mean == 10.5);
  assert(stats.median == 10.5);
  assert(stats.p95 == 19.0);
  assert(stats.p99 == 20.0);
}

void test_na_oracle_fields_remain_literal_na() {
  const OracleMetrics oracle;
  const std::string rendered = render_oracle_metrics_tsv(oracle);
  assert(rendered == "NA\tNA\tNA\tNA\tNA");
}

void test_build_matrix_reuse_and_rebuild_behaviour() {
  TempFasta fasta(
      "matrix",
      ">ref\n"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT\n");
  TempDirectory temp;

  const BuildMatrixRequest request{
      fasta.path(),
      80,
      1,
      temp.file("matrix"),
      false,
  };

  const std::vector<BuildSummaryRow> first = build_candidate_matrix(request);
  assert(!first.empty());
  for (const BuildSummaryRow& row : first) {
    assert(!row.reused);
    if (row.manifest.method == "ts::Tensor") {
      assert(std::filesystem::exists(row.index_dir / "manifest.meta"));
      assert(std::filesystem::exists(row.index_dir / "hnsw.bin"));
    } else {
      assert(std::filesystem::exists(row.index_dir / "index.bin"));
      assert(std::filesystem::exists(row.index_dir / "manifest.json"));
    }
    assert(std::filesystem::exists(row.index_dir / "build_summary.tsv"));
  }

  const std::vector<BuildSummaryRow> second = build_candidate_matrix(request);
  for (const BuildSummaryRow& row : second) {
    assert(row.reused);
  }

  {
    std::ofstream output(fasta.path(), std::ios::trunc);
    output << ">ref\n"
           << "ACGTACGTACGTACGTACGTACGTACGTACGT"
           << "ACGTACGTACGTACGTACGTACGTACGTACGT"
           << "ACGTACGTACGTACGTACGTACGTACGTACGT"
           << "ACGTACGTACGTACGTACGTACGTACGTACGA\n";
  }

  assert_throws(
      [&] { build_candidate_matrix(request); }, "incompatible existing index");

  const BuildMatrixRequest rebuild_request{
      fasta.path(),
      80,
      1,
      temp.file("matrix"),
      true,
  };
  const std::vector<BuildSummaryRow> rebuilt =
      build_candidate_matrix(rebuild_request);
  for (const BuildSummaryRow& row : rebuilt) {
    assert(!row.reused);
  }
}

void test_build_matrix_rebuild_recovers_incomplete_tensor_cache() {
  TempFasta fasta(
      "matrix_rebuild",
      ">ref\n"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT\n");
  TempDirectory temp;

  const BuildMatrixRequest request{
      fasta.path(),
      80,
      1,
      temp.file("matrix_rebuild"),
      false,
  };

  const std::vector<BuildSummaryRow> initial = build_candidate_matrix(request);
  const auto tensor_it = std::find_if(
      initial.begin(), initial.end(),
      [](const BuildSummaryRow& row) { return row.manifest.method == "ts::Tensor"; });
  assert(tensor_it != initial.end());
  assert(std::filesystem::exists(tensor_it->index_dir / "manifest.meta"));
  assert(std::filesystem::exists(tensor_it->index_dir / "hnsw.bin"));

  std::filesystem::remove(tensor_it->index_dir / "hnsw.bin");
  assert(std::filesystem::exists(tensor_it->index_dir / "manifest.meta"));
  assert(!std::filesystem::exists(tensor_it->index_dir / "hnsw.bin"));

  assert_throws([&] { build_candidate_matrix(request); }, "open file");

  const BuildMatrixRequest rebuild_request{
      fasta.path(),
      80,
      1,
      temp.file("matrix_rebuild"),
      true,
  };
  const std::vector<BuildSummaryRow> rebuilt =
      build_candidate_matrix(rebuild_request);
  const auto rebuilt_tensor_it = std::find_if(
      rebuilt.begin(), rebuilt.end(),
      [](const BuildSummaryRow& row) { return row.manifest.method == "ts::Tensor"; });
  assert(rebuilt_tensor_it != rebuilt.end());
  assert(!rebuilt_tensor_it->reused);
  assert(std::filesystem::exists(rebuilt_tensor_it->index_dir / "manifest.meta"));
  assert(std::filesystem::exists(rebuilt_tensor_it->index_dir / "hnsw.bin"));
  assert(std::filesystem::exists(temp.file("matrix_rebuild") /
                                 "build_summary.tsv"));
}

void test_compare_methods_includes_navigamer_bridge_results() {
  TempFasta fasta(
      "compare",
      ">ref\n"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT\n");
  TempDirectory temp;

  const std::string read0 =
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT";
  const std::string read1(read0.size(), 'T');

  {
    std::ofstream reads(temp.file("reads.fq"));
    reads << "@read0\n" << read0 << "\n+\n"
          << std::string(read0.size(), 'I') << "\n";
    reads << "@read1\n" << read1 << "\n+\n"
          << std::string(read1.size(), 'I') << "\n";
  }

  const std::filesystem::path fake_navigamer = temp.file("fake_navigamer.sh");
  {
    std::ofstream script(fake_navigamer);
    script
        << "#!/usr/bin/env bash\n"
        << "set -euo pipefail\n"
        << "out=\"\"\n"
        << "while [[ $# -gt 0 ]]; do\n"
        << "  if [[ \"$1\" == \"--out\" ]]; then\n"
        << "    out=\"$2\"\n"
        << "    shift 2\n"
        << "  else\n"
        << "    shift\n"
        << "  fi\n"
        << "done\n"
        << "cat >\"$out\" <<'EOF'\n"
        << "query_id\thit_id\tleaf_verify_count\tcandidate_count_for_prune\tquery_time_ms\n"
        << "read0\tref_0\t2\t3\t1.5\n"
        << "read1\t\t1\t2\t0.5\n"
        << "EOF\n";
  }
  std::filesystem::permissions(
      fake_navigamer,
      std::filesystem::perms::owner_exec |
          std::filesystem::perms::group_exec |
          std::filesystem::perms::others_exec,
      std::filesystem::perm_options::add);
  const std::filesystem::path navigamer_index = temp.file("main.navidx");
  {
    std::ofstream index_out(navigamer_index);
    index_out << "stub\n";
  }

  ComparisonRequest request;
  request.reference_path = fasta.path();
  request.reads_path = temp.file("reads.fq");
  request.reads = {
      {"read0", read0},
      {"read1", read1},
  };
  request.window_length = 128;
  request.stride = 1;
  request.tolerance = 2;
  request.out_dir = temp.file("compare_out");
  request.navigamer_binary = fake_navigamer;
  request.navigamer_index_path = navigamer_index;
  request.tensor_top_k = 8;

  const ComparisonReport report = run_comparison(request);
  assert(report.build_rows.size() == 16);
  assert(std::filesystem::exists(request.out_dir / "per_read.tsv"));
  assert(std::filesystem::exists(request.out_dir / "summary.tsv"));

  const auto navigamer_read0 = std::find_if(
      report.per_read_rows.begin(), report.per_read_rows.end(),
      [](const ComparisonPerReadRow& row) {
        return row.method == "NavigaMer" && row.variant == "adaptive" &&
               row.result.read_id == "read0";
      });
  assert(navigamer_read0 != report.per_read_rows.end());
  assert(navigamer_read0->result.raw_candidate_count == 3);
  assert(navigamer_read0->result.verified_candidate_count == 2);
  assert(navigamer_read0->result.accepted_candidate_ids ==
         std::vector<uint32_t>{0});

  const auto navigamer_summary = std::find_if(
      report.summary_rows.begin(), report.summary_rows.end(),
      [](const ComparisonSummaryRow& row) {
        return row.method == "NavigaMer" && row.variant == "adaptive";
      });
  assert(navigamer_summary != report.summary_rows.end());
  assert(navigamer_summary->read_count == 2);
}

void test_compare_methods_can_skip_oracle_and_keep_na_fields() {
  TempFasta fasta(
      "compare_no_oracle",
      ">ref\n"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT\n");
  TempDirectory temp;

  const std::string read0 =
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGT";
  {
    std::ofstream reads(temp.file("reads.fq"));
    reads << "@read0\n" << read0 << "\n+\n"
          << std::string(read0.size(), 'I') << "\n";
  }

  const std::filesystem::path fake_navigamer = temp.file("fake_navigamer.sh");
  {
    std::ofstream script(fake_navigamer);
    script
        << "#!/usr/bin/env bash\n"
        << "set -euo pipefail\n"
        << "out=\"\"\n"
        << "while [[ $# -gt 0 ]]; do\n"
        << "  if [[ \"$1\" == \"--out\" ]]; then\n"
        << "    out=\"$2\"\n"
        << "    shift 2\n"
        << "  else\n"
        << "    shift\n"
        << "  fi\n"
        << "done\n"
        << "cat >\"$out\" <<'EOF'\n"
        << "query_id\thit_id\tleaf_verify_count\tcandidate_count_for_prune\tquery_time_ms\n"
        << "read0\tref_0\t2\t3\t1.5\n"
        << "EOF\n";
  }
  std::filesystem::permissions(
      fake_navigamer,
      std::filesystem::perms::owner_exec |
          std::filesystem::perms::group_exec |
          std::filesystem::perms::others_exec,
      std::filesystem::perm_options::add);
  const std::filesystem::path navigamer_index = temp.file("main.navidx");
  {
    std::ofstream index_out(navigamer_index);
    index_out << "stub\n";
  }

  ComparisonRequest request;
  request.reference_path = fasta.path();
  request.reads_path = temp.file("reads.fq");
  request.reads = {{"read0", read0}};
  request.window_length = 128;
  request.stride = 1;
  request.tolerance = 2;
  request.out_dir = temp.file("compare_out");
  request.oracle_enabled = false;
  request.navigamer_binary = fake_navigamer;
  request.navigamer_index_path = navigamer_index;
  request.tensor_top_k = 8;

  const ComparisonReport report = run_comparison(request);
  const auto navigamer_row = std::find_if(
      report.per_read_rows.begin(), report.per_read_rows.end(),
      [](const ComparisonPerReadRow& row) {
        return row.method == "NavigaMer" && row.variant == "adaptive" &&
               row.result.read_id == "read0";
      });
  assert(navigamer_row != report.per_read_rows.end());
  assert(!navigamer_row->result.oracle.has_value());

  const auto navigamer_summary = std::find_if(
      report.summary_rows.begin(), report.summary_rows.end(),
      [](const ComparisonSummaryRow& row) {
        return row.method == "NavigaMer" && row.variant == "adaptive";
      });
  assert(navigamer_summary != report.summary_rows.end());
  assert(navigamer_summary->oracle_read_count == 0);
  assert(!navigamer_summary->mean_recall.has_value());

  std::ifstream per_read(request.out_dir / "per_read.tsv");
  assert(per_read.good());
  std::string header;
  std::getline(per_read, header);
  std::string row;
  std::getline(per_read, row);
  assert(row.find("\tNA\tNA\tNA\tNA\tNA") != std::string::npos);
}

}  // namespace

int main() {
  test_every_raw_candidate_is_verified_by_bounded_edlib();
  test_accepted_ids_match_bruteforce_on_small_case();
  test_percentile_nearest_rank_is_fixed();
  test_na_oracle_fields_remain_literal_na();
  test_build_matrix_reuse_and_rebuild_behaviour();
  test_build_matrix_rebuild_recovers_incomplete_tensor_cache();
  test_compare_methods_includes_navigamer_bridge_results();
  test_compare_methods_can_skip_oracle_and_keep_na_fields();
  std::cout << "evaluation tests passed\n";
  return 0;
}
