#include "tools.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::string random_string(size_t length, const std::string& alphabet,
                          std::mt19937& gen) {
  std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
  std::string out;
  out.reserve(length);
  for (size_t i = 0; i < length; ++i) out.push_back(alphabet[pick(gen)]);
  return out;
}

void check_pair(const std::string& a, const std::string& b, int tau) {
  int full = navigamer::compute_distance(a, b);
  int bounded = navigamer::compute_distance_bounded(a, b, tau);
  if (full <= tau) {
    assert(bounded == full);
  } else {
    assert(bounded > tau);
  }
}

}  // namespace

int main() {
  for (int tau : {0, 1, 2, 5, 10, 20}) {
    check_pair("", "", tau);
    check_pair("", "ACGT", tau);
    check_pair("ACGT", "", tau);
    check_pair("ACGT", "ACGT", tau);
    check_pair("ACGT", "TGCA", tau);
  }

  std::mt19937 gen(20260613);
  for (const auto& alphabet : {std::string("ACGT"), std::string("AC")}) {
    for (size_t length : {20U, 50U, 100U, 250U}) {
      std::uniform_int_distribution<int> delta(-12, 12);
      for (int trial = 0; trial < 80; ++trial) {
        std::string a = random_string(length, alphabet, gen);
        int other_length = std::max(0, static_cast<int>(length) + delta(gen));
        std::string b = random_string(static_cast<size_t>(other_length), alphabet, gen);
        for (int tau : {0, 1, 2, 5, 10, 20}) check_pair(a, b, tau);
      }
    }
  }

  std::cout << "bounded edit distance tests passed\n";
  return 0;
}
