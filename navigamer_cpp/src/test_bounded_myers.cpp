#include "tools.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string random_dna(size_t length, std::mt19937& gen) {
  static const std::string alphabet = "ACGT";
  std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
  std::string out;
  out.reserve(length);
  for (size_t i = 0; i < length; ++i) out.push_back(alphabet[pick(gen)]);
  return out;
}

void assert_bounded_semantics(const std::string& a, const std::string& b,
                              int tau) {
  const int full = navigamer::compute_distance(a, b);
  const int myers = navigamer::compute_distance_bounded_myers(a, b, tau);
  const int edlib = navigamer::compute_distance_bounded_edlib(a, b, tau);
  if (full <= tau) {
    assert(myers == full);
    assert(edlib == full);
  } else {
    assert(myers > tau);
    assert(edlib > tau);
  }
}

void assert_matches_dp_fallback(const std::string& a, const std::string& b,
                                int tau) {
  const int dp = navigamer::compute_distance_bounded_dp(a, b, tau);
  const int myers = navigamer::compute_distance_bounded_myers(a, b, tau);
  assert(myers == dp);
}

void test_parse_and_names() {
  assert(navigamer::parse_distance_mode("dp") == navigamer::DistanceMode::DP);
  assert(navigamer::parse_distance_mode("myers") ==
         navigamer::DistanceMode::Myers);
  assert(navigamer::parse_distance_mode("edlib") ==
         navigamer::DistanceMode::Edlib);
  assert(navigamer::parse_distance_mode("auto") ==
         navigamer::DistanceMode::Auto);
  assert(std::string(navigamer::distance_mode_name(
             navigamer::DistanceMode::DP)) == "dp");
  assert(std::string(navigamer::distance_mode_name(
             navigamer::DistanceMode::Myers)) == "myers");
  assert(std::string(navigamer::distance_mode_name(
             navigamer::DistanceMode::Edlib)) == "edlib");
  assert(std::string(navigamer::distance_mode_name(
             navigamer::DistanceMode::Auto)) == "auto");

  bool threw = false;
  try {
    (void)navigamer::parse_distance_mode("fast");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

void test_known_indels_and_substitutions() {
  for (int tau : {0, 1, 2, 3, 5, 10}) {
    assert_bounded_semantics("ACGT", "ACGT", tau);
    assert_bounded_semantics("ACGT", "ACGA", tau);
    assert_bounded_semantics("ACGT", "ACGTT", tau);
    assert_bounded_semantics("ACGTT", "ACGT", tau);
    assert_bounded_semantics("ACGTACGT", "ACGACGT", tau);
    assert_bounded_semantics("AAAAACCCCC", "AAAAGCCCCG", tau);
    assert_bounded_semantics("", "", tau);
    assert_bounded_semantics("", "ACGT", tau);
    assert_bounded_semantics("ACGT", "", tau);
  }
}

void test_random_against_full_dp() {
  std::mt19937 gen(20260616);
  for (size_t a_len : {1U, 2U, 5U, 20U, 50U, 64U, 65U, 100U, 150U, 250U}) {
    for (size_t b_len : {1U, 2U, 5U, 20U, 50U, 64U, 65U, 100U, 150U, 250U}) {
      for (int trial = 0; trial < 12; ++trial) {
        const std::string a = random_dna(a_len, gen);
        const std::string b = random_dna(b_len, gen);
        for (int tau : {0, 1, 2, 5, 10, 20, 50}) {
          assert_bounded_semantics(a, b, tau);
        }
      }
    }
  }
}

void test_non_acgt_fallback() {
  const std::vector<std::pair<std::string, std::string>> pairs = {
      {"ACGTNACGT", "ACGTGACGT"},
      {"acgtacgt", "acgtccgt"},
      {"ACGT-ACGT", "ACGTAACGT"},
      {"NNNN", "ACGT"},
      {"ACGT", "NNNN"},
  };
  for (const auto& pair : pairs) {
    for (int tau : {0, 1, 2, 5, 10}) {
      assert_matches_dp_fallback(pair.first, pair.second, tau);
      assert(navigamer::compute_distance_bounded_edlib(
                 pair.first, pair.second, tau) ==
             navigamer::compute_distance_bounded_dp(
                 pair.first, pair.second, tau));
    }
  }
}

void test_multiword_supports_250bp_acgt() {
  std::mt19937 gen(20260617);
  const std::string a = random_dna(250, gen);
  const std::string b = random_dna(250, gen);
  assert(navigamer::compute_distance_bounded_myers_supported(a, b));
  assert(!navigamer::compute_distance_bounded_myers_supported(
      a, std::string("ACGTN") + b.substr(5)));
  assert(!navigamer::compute_distance_bounded_myers_supported(
      std::string(300, 'A'), std::string(300, 'A')));
}

void test_mode_wrapper() {
  for (int tau : {0, 1, 2, 5}) {
    const std::string a = "ACGTACGTACGT";
    const std::string b = "ACGTTCGTACGA";
    assert(navigamer::compute_distance_bounded_with_mode(
               a, b, tau, navigamer::DistanceMode::DP) ==
           navigamer::compute_distance_bounded_dp(a, b, tau));
    assert(navigamer::compute_distance_bounded_with_mode(
               a, b, tau, navigamer::DistanceMode::Myers) ==
           navigamer::compute_distance_bounded_myers(a, b, tau));
    assert(navigamer::compute_distance_bounded_with_mode(
               a, b, tau, navigamer::DistanceMode::Edlib) ==
           navigamer::compute_distance_bounded_edlib(a, b, tau));
    assert(navigamer::compute_distance_bounded_with_mode(
               a, b, tau, navigamer::DistanceMode::Auto) ==
           navigamer::compute_distance_bounded_dp(a, b, tau));
  }
}

}  // namespace

int main() {
  test_parse_and_names();
  test_known_indels_and_substitutions();
  test_random_against_full_dp();
  test_non_acgt_fallback();
  test_multiword_supports_250bp_acgt();
  test_mode_wrapper();

  std::cout << "bounded Myers edit distance tests passed\n";
  return 0;
}
