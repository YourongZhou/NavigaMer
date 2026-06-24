#include "candidate_indexes.hpp"
#include "reference_windows.hpp"

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
#include <unordered_map>
#include <vector>

namespace {

class TempDirectory {
 public:
  TempDirectory() {
    std::random_device random;
    for (int attempt = 0; attempt < 64; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("navigamer_qgram_safe_" + std::to_string(random()) + "_" +
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
                   ("navigamer_qgram_safe_fasta_" + std::string(name) + "_" +
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

std::unordered_map<std::string, uint32_t> qgram_counts(std::string_view sequence,
                                                       uint32_t q) {
  std::unordered_map<std::string, uint32_t> counts;
  if (q == 0 || sequence.size() < q) {
    return counts;
  }
  for (std::size_t offset = 0; offset + q <= sequence.size(); ++offset) {
    counts[std::string(sequence.substr(offset, q))]++;
  }
  return counts;
}

uint32_t full_qgram_l1(std::string_view lhs, std::string_view rhs, uint32_t q) {
  const std::unordered_map<std::string, uint32_t> lhs_counts =
      qgram_counts(lhs, q);
  const std::unordered_map<std::string, uint32_t> rhs_counts =
      qgram_counts(rhs, q);

  uint32_t l1 = 0;
  for (const auto& entry : lhs_counts) {
    const auto rhs_it = rhs_counts.find(entry.first);
    const uint32_t rhs_count =
        rhs_it == rhs_counts.end() ? 0U : rhs_it->second;
    l1 += entry.second > rhs_count ? entry.second - rhs_count
                                   : rhs_count - entry.second;
  }
  for (const auto& entry : rhs_counts) {
    if (lhs_counts.find(entry.first) == lhs_counts.end()) {
      l1 += entry.second;
    }
  }
  return l1;
}

std::vector<uint32_t> naive_qgram_safe_candidates(const ReferenceWindows& reference,
                                                  std::string_view query,
                                                  uint32_t q, uint32_t tau) {
  std::vector<uint32_t> window_ids;
  for (uint32_t window_id = 0; window_id < reference.size(); ++window_id) {
    const std::string_view window = reference.window(window_id);
    if (full_qgram_l1(query, window, q) <= 2U * q * tau) {
      window_ids.push_back(window_id);
    }
  }
  return window_ids;
}

int edit_distance(std::string_view lhs, std::string_view rhs) {
  const std::size_t m = lhs.size();
  const std::size_t n = rhs.size();
  std::vector<int> previous(n + 1);
  std::vector<int> current(n + 1);
  for (std::size_t j = 0; j <= n; ++j) {
    previous[j] = static_cast<int>(j);
  }
  for (std::size_t i = 0; i < m; ++i) {
    current[0] = static_cast<int>(i + 1);
    for (std::size_t j = 0; j < n; ++j) {
      const int substitution = previous[j] + (lhs[i] == rhs[j] ? 0 : 1);
      const int insertion = current[j] + 1;
      const int deletion = previous[j + 1] + 1;
      current[j + 1] = std::min({substitution, insertion, deletion});
    }
    previous.swap(current);
  }
  return previous[n];
}

std::string mutate_with_substitutions(std::string value, uint32_t edits) {
  static constexpr char kAlternates[] = {'C', 'G', 'T', 'A'};
  for (uint32_t edit = 0; edit < edits && edit < value.size(); ++edit) {
    const char original = value[edit];
    for (char candidate : kAlternates) {
      if (candidate != original) {
        value[edit] = candidate;
        break;
      }
    }
  }
  return value;
}

std::string mutate_with_insertion(std::string value, uint32_t edits) {
  for (uint32_t edit = 0; edit < edits; ++edit) {
    value.insert(value.begin() + static_cast<std::ptrdiff_t>(edit % (value.size() + 1)),
                 'A');
  }
  return value;
}

std::string mutate_with_deletion(std::string value, uint32_t edits) {
  for (uint32_t edit = 0; edit < edits && !value.empty(); ++edit) {
    value.erase(value.begin() + static_cast<std::ptrdiff_t>(edit % value.size()));
  }
  return value;
}

void expect_equal(const std::vector<uint32_t>& lhs,
                  const std::vector<uint32_t>& rhs) {
  assert(lhs == rhs);
}

void test_naive_candidate_sets_match_for_q_3_and_q_4() {
  TempFasta fasta("reference",
                  ">ref\nACGTACGTNNNACGTACGTTTACGTACGNACGTACGT\n");
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(fasta.path(), 12, 1);
  const TempDirectory temp;

  for (uint32_t q : {3U, 4U}) {
    const QgramSafeIndexConfig config{fasta.path(), 12, 1, q};
    const QgramSafeIndex index = QgramSafeIndex::build(config);
    const std::filesystem::path out_dir = temp.file("qgram_" + std::to_string(q));
    index.save(out_dir);
    const QgramSafeIndex loaded = QgramSafeIndex::load(out_dir / "index.bin");

    const std::vector<std::string> queries = {
        "ACGTACGT",
        "ACGTN",
        "NNNAC",
        "TGCATGCATGCA",
        "ACG",
        "ACGTACGTACGT",
        mutate_with_substitutions(std::string("ACGTACGT"), 2),
    };

    for (const std::string& query : queries) {
      for (uint32_t tau = 0; tau <= 3; ++tau) {
        const std::vector<uint32_t> expected =
            naive_qgram_safe_candidates(reference, query, q, tau);
        const std::vector<uint32_t> actual = loaded.query(query, tau);
        expect_equal(actual, expected);
      }
    }
  }
}

void test_mutated_windows_keep_all_tau_neighbors() {
  TempFasta fasta("mutated_reference",
                  ">ref\nACGTACGTNNNACGTACGTTTACGTACGNACGTACGT\n");
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(fasta.path(), 12, 1);
  const TempDirectory temp;

  for (uint32_t q : {3U, 4U}) {
    const QgramSafeIndexConfig config{fasta.path(), 12, 1, q};
    const QgramSafeIndex index = QgramSafeIndex::build(config);
    const std::vector<uint32_t> source_windows = {0, 2, 5, 8};

    for (uint32_t window_id : source_windows) {
      const std::string source = std::string(reference.window(window_id));
      const std::vector<std::pair<std::string, uint32_t>> mutated_queries = {
          {mutate_with_substitutions(source, 1), 1},
          {mutate_with_substitutions(source, 2), 2},
          {mutate_with_insertion(source, 1), 1},
          {mutate_with_deletion(source, 1), 1},
      };

      for (const auto& [query, tau] : mutated_queries) {
        const std::vector<uint32_t> actual = index.query(query, tau);
        for (uint32_t candidate_id = 0; candidate_id < reference.size();
             ++candidate_id) {
          const int dist = edit_distance(query, reference.window(candidate_id));
          if (dist <= static_cast<int>(tau)) {
            assert(std::binary_search(actual.begin(), actual.end(), candidate_id));
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  test_naive_candidate_sets_match_for_q_3_and_q_4();
  test_mutated_windows_keep_all_tau_neighbors();
  return 0;
}
