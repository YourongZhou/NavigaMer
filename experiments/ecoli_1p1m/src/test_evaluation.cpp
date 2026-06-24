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

}  // namespace

int main() {
  test_every_raw_candidate_is_verified_by_bounded_edlib();
  test_accepted_ids_match_bruteforce_on_small_case();
  test_percentile_nearest_rank_is_fixed();
  test_na_oracle_fields_remain_literal_na();
  test_build_matrix_reuse_and_rebuild_behaviour();
  std::cout << "evaluation tests passed\n";
  return 0;
}
