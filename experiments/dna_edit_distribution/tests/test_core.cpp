#include "dna_edit_distribution.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: "          \
                << #condition << '\n';                                          \
      ++failures;                                                               \
    }                                                                           \
  } while (false)

void test_known_distances() {
  dna_edit_distribution::ExactWfaAligner aligner;
  CHECK(aligner.distance("ACGT", "ACGT") == 0);
  CHECK(aligner.distance("ACGT", "AGGT") == 1);
  CHECK(aligner.distance("ACGT", "ACGGT") == 1);
  CHECK(aligner.distance("ACGT", "ACT") == 1);
}

void test_wfa_matches_dynamic_programming() {
  dna_edit_distribution::ExactWfaAligner aligner;
  std::string first;
  std::string second;
  for (std::uint64_t pair_index = 0; pair_index < 100; ++pair_index) {
    dna_edit_distribution::generate_pair(
        25, 20260719, pair_index, first, second);
    CHECK(aligner.distance(first, second) ==
          dna_edit_distribution::levenshtein_dp(first, second));
  }
}

void test_generator_is_pair_index_deterministic() {
  std::string first_a;
  std::string second_a;
  std::string first_b;
  std::string second_b;
  dna_edit_distribution::generate_pair(
      37, 1234, 99, first_a, second_a);
  dna_edit_distribution::generate_pair(
      37, 1234, 99, first_b, second_b);
  CHECK(first_a == first_b);
  CHECK(second_a == second_b);
  CHECK(first_a.size() == 37);
  CHECK(second_a.size() == 37);
  CHECK(first_a.find_first_not_of("ACGT") == std::string::npos);
  CHECK(second_a.find_first_not_of("ACGT") == std::string::npos);
}

void test_histogram_is_thread_count_independent() {
  const auto serial =
      dna_edit_distribution::compute_histogram(31, 200, 7, 1);
  const auto parallel =
      dna_edit_distribution::compute_histogram(31, 200, 7, 4);
  CHECK(serial.counts == parallel.counts);
  CHECK(serial.actual_threads == 1);
  CHECK(parallel.actual_threads >= 1);
}

}  // namespace

int main() {
  test_known_distances();
  test_wfa_matches_dynamic_programming();
  test_generator_is_pair_index_deterministic();
  test_histogram_is_thread_count_independent();
  if (failures != 0) {
    std::cerr << failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all core tests passed\n";
  return 0;
}
