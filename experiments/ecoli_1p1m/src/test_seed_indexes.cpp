#include "candidate_indexes.hpp"
#include "reference_windows.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <limits>
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

uint64_t encode_masked_key(std::string_view sequence,
                           const std::vector<uint8_t>& mask_bits) {
  uint64_t key = 0;
  for (std::size_t index = 0; index < mask_bits.size(); ++index) {
    if (!mask_bits[index]) {
      continue;
    }
    key <<= 2U;
    switch (sequence[index]) {
      case 'A':
      case 'a':
        break;
      case 'C':
      case 'c':
        key |= 1U;
        break;
      case 'G':
      case 'g':
        key |= 2U;
        break;
      case 'T':
      case 't':
        key |= 3U;
        break;
      default:
        return std::numeric_limits<uint64_t>::max();
    }
  }
  return key;
}

std::vector<uint32_t> naive_spaced_candidates(
    const ReferenceWindows& reference, std::string_view query,
    const std::vector<SpacedMask>& masks) {
  std::vector<uint8_t> marked(reference.size(), 0);
  for (uint8_t mask_id = 0; mask_id < masks.size(); ++mask_id) {
    const SpacedMask& mask = masks[mask_id];
    if (query.size() < mask.span) {
      continue;
    }
    for (std::size_t query_start = 0; query_start + mask.span <= query.size();
         ++query_start) {
      const uint64_t query_key =
          encode_masked_key(query.substr(query_start, mask.span), mask.bits);
      if (query_key == std::numeric_limits<uint64_t>::max()) {
        continue;
      }
      for (uint32_t ref_start = 0;
           ref_start + mask.span <= reference.sequence().size(); ++ref_start) {
        const uint64_t ref_key =
            encode_masked_key(reference.sequence().substr(ref_start, mask.span),
                              mask.bits);
        if (ref_key != query_key) {
          continue;
        }
        const std::vector<uint32_t> window_ids =
            reference.covering_window_ids(ref_start, mask.span);
        for (uint32_t window_id : window_ids) {
          marked[window_id] = 1;
        }
      }
    }
  }

  std::vector<uint32_t> candidates;
  for (uint32_t window_id = 0; window_id < marked.size(); ++window_id) {
    if (marked[window_id]) {
      candidates.push_back(window_id);
    }
  }
  return candidates;
}

std::string mask_bits_to_string(const std::vector<uint8_t>& bits) {
  std::string value;
  value.reserve(bits.size());
  for (uint8_t bit : bits) {
    value.push_back(bit ? '1' : '0');
  }
  return value;
}

void test_spaced_masks_are_distinct_and_weighted() {
  const std::vector<SpacedMask> masks = make_spaced_masks(15);
  assert(masks.size() == 4);
  const std::vector<uint32_t> expected_spans{24, 26, 29, 32};
  std::unordered_set<std::string> observed_patterns;
  for (std::size_t index = 0; index < masks.size(); ++index) {
    assert(masks[index].span == expected_spans[index]);
    assert(std::count(masks[index].bits.begin(), masks[index].bits.end(), 1) ==
           15);
    observed_patterns.insert(mask_bits_to_string(masks[index].bits));
  }
  assert(observed_patterns.size() == masks.size());
}

void test_spaced_seed_round_trip_matches_naive_extraction() {
  TempFasta fasta("spaced",
                  ">ref\n"
                  "ACGTACGTGTCAGTACGTACGTGTCAGTACGTACGTGTCAGTACGTACGT"
                  "ACGTACGTGTCAGTACGTACGTGTCAGTACGTACGTGTCAGTACGTACGT\n");
  TempDirectory temp;
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(fasta.path(), 80, 1);
  const std::vector<SpacedMask> masks = make_spaced_masks(15);
  const SpacedSeedIndexConfig config{fasta.path(), 80, 1, 15};
  const SpacedSeedIndex index = SpacedSeedIndex::build(config);

  const std::vector<std::string> queries{
      "ACGTACGTGTCAGTACGTACGTGTCAGTACGT",
      "GTACGTGTCAGTACGTACGTGTCAGTACGTAC",
      "TACGTGTCAGTACGTACGTGTCAGTACGTACG",
  };
  for (const std::string& query : queries) {
    expect_equal(index.query(query), naive_spaced_candidates(reference, query, masks));
  }

  const std::filesystem::path out_dir = temp.file("spaced_index");
  std::filesystem::create_directory(out_dir);
  index.save(out_dir);
  const SpacedSeedIndex loaded = SpacedSeedIndex::load(out_dir / "index.bin");
  for (const std::string& query : queries) {
    expect_equal(loaded.query(query), index.query(query));
  }
}

void test_randstrobe_seed_stability_changes_with_seed() {
  const std::string sequence =
      "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
  const std::vector<uint64_t> first =
      randstrobe_composite_keys(sequence, 15, 20, 50, 1234);
  const std::vector<uint64_t> same =
      randstrobe_composite_keys(sequence, 15, 20, 50, 1234);
  const std::vector<uint64_t> different =
      randstrobe_composite_keys(sequence, 15, 20, 50, 4321);

  assert(first == same);
  assert(!first.empty());
  assert(first != different);

  TempFasta fasta("randstrobe",
                  ">ref\n"
                  "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                  "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n");
  TempDirectory temp;
  const RandstrobeIndexConfig config{fasta.path(), 80, 1, 15, 20, 50, 1234};
  const RandstrobeIndex index = RandstrobeIndex::build(config);
  const std::string query =
      "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
      "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
  const std::vector<uint32_t> baseline = index.query(query);
  const std::filesystem::path out_dir = temp.file("randstrobe_index");
  std::filesystem::create_directory(out_dir);
  index.save(out_dir);
  const RandstrobeIndex loaded = RandstrobeIndex::load(out_dir / "index.bin");
  expect_equal(baseline, loaded.query(query));
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
  const std::string query_sequence = "ACGTACGTACGTACGTACGT";

  for (uint32_t k : {5U, 7U}) {
    const std::filesystem::path out_dir = temp.file("index_" + std::to_string(k));
    std::filesystem::create_directory(out_dir);
    const ContiguousIndexConfig config{fasta.path(), 20, 1, k};
    const ContiguousIndex index = ContiguousIndex::build(config);
    const std::vector<uint32_t> before = index.query(query_sequence);

    index.save(out_dir);
    const ContiguousIndex loaded = ContiguousIndex::load(out_dir / "index.bin");
    const std::vector<uint32_t> after = loaded.query(query_sequence);

    expect_equal(before, after);
  }
}

}  // namespace

int main() {
  test_random_queries_match_naive_for_k_5_and_7();
  test_round_trip_preserves_candidate_ids();
  test_spaced_masks_are_distinct_and_weighted();
  test_spaced_seed_round_trip_matches_naive_extraction();
  test_randstrobe_seed_stability_changes_with_seed();
  std::cout << "seed index tests passed\n";
  return 0;
}
