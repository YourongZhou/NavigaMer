#include "candidate_indexes.hpp"
#include "reference_windows.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TempDirectory {
 public:
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("navigamer_seed_indexes_" + std::to_string(random()) + "_" +
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

  std::filesystem::path file(std::string_view name) const {
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

class TempFasta {
 public:
  TempFasta(std::string_view name, std::string_view contents)
      : directory_(std::filesystem::temp_directory_path() /
                   ("navigamer_seed_fasta_" + std::string(name) + "_" +
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

std::string random_dna(std::mt19937_64& rng, std::size_t length) {
  static constexpr char kAlphabet[] = {'A', 'C', 'G', 'T'};
  std::string sequence;
  sequence.reserve(length);
  std::uniform_int_distribution<int> distribution(0, 3);
  for (std::size_t index = 0; index < length; ++index) {
    sequence.push_back(kAlphabet[distribution(rng)]);
  }
  return sequence;
}

std::vector<uint32_t> naive_candidates(const ReferenceWindows& reference,
                                       std::string_view query, uint32_t k) {
  if (query.size() < k) {
    return {};
  }
  std::vector<uint32_t> candidates;
  for (uint32_t window_id = 0; window_id < reference.size(); ++window_id) {
    const std::string_view window = reference.window(window_id);
    bool shared = false;
    for (std::size_t query_offset = 0; query_offset + k <= query.size();
         ++query_offset) {
      const std::string_view query_kmer = query.substr(query_offset, k);
      for (std::size_t window_offset = 0; window_offset + k <= window.size();
           ++window_offset) {
        if (window.substr(window_offset, k) == query_kmer) {
          shared = true;
          break;
        }
      }
      if (shared) {
        break;
      }
    }
    if (shared) {
      candidates.push_back(window_id);
    }
  }
  return candidates;
}

void expect_equal(const std::vector<uint32_t>& lhs,
                  const std::vector<uint32_t>& rhs) {
  assert(lhs == rhs);
}

void test_random_queries_match_naive_for_k_5_and_7() {
  std::mt19937_64 rng(0x5eed1234ull);
  for (uint32_t k : {5U, 7U}) {
    for (int trial = 0; trial < 50; ++trial) {
      const std::string reference_sequence = random_dna(rng, 80);
      const std::string query_sequence = random_dna(rng, 20);
      TempFasta fasta("ref", ">ref\n" + reference_sequence + "\n");
      const ReferenceWindows reference =
          ReferenceWindows::from_fasta(fasta.path(), 20, 1);
      const ContiguousIndexConfig config{fasta.path(), 20, 1, k};
      const ContiguousIndex index = ContiguousIndex::build(config);

      const std::vector<uint32_t> expected =
          naive_candidates(reference, query_sequence, k);
      expect_equal(index.query(query_sequence), expected);
    }
  }
}

void test_round_trip_preserves_candidate_ids() {
  TempFasta fasta("roundtrip",
                  ">ref\nACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n");
  TempDirectory temp;
  const ContiguousIndexConfig config{fasta.path(), 20, 1, 5};
  const ContiguousIndex index = ContiguousIndex::build(config);
  const std::string query_sequence = "ACGTACGTACGTACGTACGT";
  const std::vector<uint32_t> before = index.query(query_sequence);

  const std::filesystem::path out_dir = temp.file("index");
  std::filesystem::create_directory(out_dir);
  index.save(out_dir);
  const ContiguousIndex loaded = ContiguousIndex::load(out_dir / "index.bin");
  const std::vector<uint32_t> after = loaded.query(query_sequence);

  expect_equal(before, after);
}

}  // namespace

int main() {
  test_random_queries_match_naive_for_k_5_and_7();
  test_round_trip_preserves_candidate_ids();
  std::cout << "seed index tests passed\n";
  return 0;
}
