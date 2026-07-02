#include "candidate_indexes.hpp"
#include "reference_windows.hpp"

#include "edlib.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

class TempDirectory {
 public:
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("navigamer_pigeonhole_" + std::to_string(random()) + "_" +
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
                   ("navigamer_pigeonhole_fasta_" + std::string(name) + "_" +
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

int edlib_distance(std::string_view lhs, std::string_view rhs) {
  const EdlibAlignConfig config =
      edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_DISTANCE, nullptr, 0);
  const EdlibAlignResult result =
      edlibAlign(lhs.data(), static_cast<int>(lhs.size()), rhs.data(),
                 static_cast<int>(rhs.size()), config);
  if (result.status != EDLIB_STATUS_OK) {
    edlibFreeAlignResult(result);
    throw std::runtime_error("edlib alignment failed");
  }
  const int distance = result.editDistance;
  edlibFreeAlignResult(result);
  return distance;
}

std::vector<std::string> enumerate_one_and_two_edit_variants(std::string seed) {
  std::vector<std::string> variants;
  std::unordered_set<std::string> seen;
  auto add_variant = [&](const std::string& value) {
    if (seen.insert(value).second) {
      variants.push_back(value);
    }
  };

  add_variant(seed);

  static constexpr char kBases[] = {'A', 'C', 'G', 'T'};

  const std::size_t original_size = seed.size();
  for (std::size_t i = 0; i < original_size; ++i) {
    const char original = seed[i];
    for (char base : kBases) {
      if (base == original) {
        continue;
      }
      std::string substituted = seed;
      substituted[i] = base;
      add_variant(substituted);
    }
  }
  for (std::size_t i = 0; i <= original_size; ++i) {
    for (char base : kBases) {
      std::string inserted = seed;
      inserted.insert(inserted.begin() + static_cast<std::ptrdiff_t>(i), base);
      add_variant(inserted);
    }
  }
  for (std::size_t i = 0; i < original_size; ++i) {
    std::string deleted = seed;
    deleted.erase(deleted.begin() + static_cast<std::ptrdiff_t>(i));
    add_variant(deleted);
  }

  const std::vector<std::string> first_shell = variants;
  for (const std::string& intermediate : first_shell) {
    const std::size_t intermediate_size = intermediate.size();
    for (std::size_t i = 0; i < intermediate_size; ++i) {
      const char original = intermediate[i];
      for (char base : kBases) {
        if (base == original) {
          continue;
        }
        std::string substituted = intermediate;
        substituted[i] = base;
        add_variant(substituted);
      }
    }
    for (std::size_t i = 0; i <= intermediate_size; ++i) {
      for (char base : kBases) {
        std::string inserted = intermediate;
        inserted.insert(inserted.begin() + static_cast<std::ptrdiff_t>(i),
                        base);
        add_variant(inserted);
      }
    }
    for (std::size_t i = 0; i < intermediate_size; ++i) {
      std::string deleted = intermediate;
      deleted.erase(deleted.begin() + static_cast<std::ptrdiff_t>(i));
      add_variant(deleted);
    }
  }

  return variants;
}

void expect_superset(const std::vector<uint32_t>& actual,
                     const std::vector<uint32_t>& expected) {
  for (uint32_t window_id : expected) {
    assert(std::binary_search(actual.begin(), actual.end(), window_id));
  }
}

void test_nominal_read_length_can_differ_from_window_length() {
  TempFasta fasta(
      "nominal_mismatch",
      ">ref\n"
      "ACGTACGTGTCAGTACGTACGTGTCAGTACGTACGTGTCAGTACGTACGT\n");
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(fasta.path(), 12, 1);

  PigeonholeIndexConfig config;
  config.reference_path = fasta.path();
  config.window_length = 12;
  config.stride = 1;
  config.tau = 1;
  config.nominal_read_length = 8;

  const PigeonholeIndex index = PigeonholeIndex::build(config);
  const std::string query = std::string(reference.window(0)).substr(0, 11);
  const std::vector<uint32_t> actual = index.query(query, 1);
  std::vector<uint32_t> expected;
  for (uint32_t window_id = 0; window_id < reference.size(); ++window_id) {
    const int distance = edlib_distance(query, reference.window(window_id));
    if (distance <= 1) {
      expected.push_back(window_id);
    }
  }
  expect_superset(actual, expected);
}

}  // namespace

int main() {
  test_nominal_read_length_can_differ_from_window_length();
  TempFasta fasta(
      "reference",
      ">ref\n"
      "ACGTACGTGTCAGTACGTACGTGTCAGTACGTACGTGTCAGTACGTACGT\n");
  const ReferenceWindows reference = ReferenceWindows::from_fasta(fasta.path(), 12, 1);
  const TempDirectory temp;
  const std::vector<std::size_t> source_starts = {0, 5, 10, 15, 20};

  for (uint32_t tau : {1U, 2U}) {
    PigeonholeIndexConfig config;
    config.reference_path = fasta.path();
    config.window_length = 12;
    config.stride = 1;
    config.tau = tau;
    config.nominal_read_length = 12;

    const PigeonholeIndex index = PigeonholeIndex::build(config);
    const std::filesystem::path out_dir = temp.file("tau_" + std::to_string(tau));
    index.save(out_dir);
    const PigeonholeIndex loaded = PigeonholeIndex::load(out_dir / "index.bin");

    for (std::size_t start : source_starts) {
      const std::string source = std::string(reference.window(
          reference.window_id_for_start(static_cast<uint32_t>(start))));
      const std::vector<std::string> variants =
          enumerate_one_and_two_edit_variants(source);
      for (const std::string& query : variants) {
        const int source_distance = edlib_distance(query, source);
        if (source_distance > static_cast<int>(tau)) {
          continue;
        }
        const std::vector<uint32_t> actual = loaded.query(query, tau);
        std::vector<uint32_t> expected;
        for (uint32_t window_id = 0; window_id < reference.size();
             ++window_id) {
          const int distance = edlib_distance(query, reference.window(window_id));
          if (distance <= static_cast<int>(tau)) {
            expected.push_back(window_id);
          }
        }
        expect_superset(actual, expected);
      }
    }
  }

  return 0;
}
