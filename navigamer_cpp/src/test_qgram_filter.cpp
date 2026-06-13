#include "qgram_filter.hpp"
#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::string random_dna(size_t length, std::mt19937& gen) {
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> pick(0, 3);
  std::string out;
  out.reserve(length);
  for (size_t i = 0; i < length; ++i) out.push_back(bases[pick(gen)]);
  return out;
}

std::string mutate_with_indels(std::string value, int edits, std::mt19937& gen) {
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> operation(0, 2);
  std::uniform_int_distribution<int> base(0, 3);
  for (int i = 0; i < edits; ++i) {
    int op = operation(gen);
    if (op == 0 && !value.empty()) {
      std::uniform_int_distribution<size_t> pos(0, value.size() - 1);
      value[pos(gen)] = bases[base(gen)];
    } else if (op == 1 && !value.empty()) {
      std::uniform_int_distribution<size_t> pos(0, value.size() - 1);
      value.erase(pos(gen), 1);
    } else {
      std::uniform_int_distribution<size_t> pos(0, value.size());
      value.insert(pos(gen), 1, bases[base(gen)]);
    }
  }
  return value;
}

bool contains(const std::vector<size_t>& ids, size_t id) {
  return std::binary_search(ids.begin(), ids.end(), id);
}

void test_qgram_counts_basic() {
  auto counts = navigamer::compute_qgram_counts("ACGTAC", 2);
  assert(counts.size() == 4);
  assert(counts.at("AC") == 2);
  assert(counts.at("CG") == 1);
  assert(counts.at("GT") == 1);
  assert(counts.at("TA") == 1);
  assert(navigamer::qgram_total("ACGT", 2) == 3);
  assert(navigamer::qgram_total("ACGTAC", 2) == 5);
  assert(navigamer::qgram_total("A", 2) == 0);
}

void test_qgram_l1_basic() {
  assert(navigamer::compute_qgram_l1("ACGT", "ACGT", 2) == 0);
  assert(navigamer::compute_qgram_l1("ACGT", "ACGA", 2) == 2);
  assert(navigamer::compute_qgram_l1("ANNT", "ANNT", 2) == 0);
}

void test_qgram_l1_bound_no_false_negative() {
  std::mt19937 gen(12345);
  for (size_t length : {size_t{20}, size_t{50}, size_t{100}, size_t{250}}) {
    for (int q : {3, 4, 5}) {
      for (int tau : {0, 1, 2, 5, 10, 20}) {
        for (int trial = 0; trial < 8; ++trial) {
          std::string lhs = random_dna(length, gen);
          std::string rhs = mutate_with_indels(lhs, tau, gen);
          int distance = navigamer::compute_distance(lhs, rhs);
          if (distance <= tau) {
            assert(navigamer::compute_qgram_l1(lhs, rhs, q) <=
                   static_cast<size_t>(2 * q * tau));
          }
        }
      }
    }
  }

  const std::string ambiguous_a = "AACNNGTACN";
  const std::string ambiguous_b = "AACNAGTACN";
  assert(navigamer::compute_distance(ambiguous_a, ambiguous_b) == 1);
  assert(navigamer::compute_qgram_l1(ambiguous_a, ambiguous_b, 3) <= 6);
}

void test_qgram_index_no_false_negative() {
  std::mt19937 gen(777);
  std::vector<navigamer::QGramCountIndex::Item> items;
  for (size_t i = 0; i < 80; ++i) {
    items.push_back({i, random_dna(40 + i % 7, gen)});
  }
  items.push_back({80, "ANNT"});
  items.push_back({81, "A"});

  navigamer::QGramCountIndex index(5);
  index.build(items);
  assert(index.q() == 5);
  assert(index.size() == items.size());

  for (int tau : {0, 1, 2, 5, 10, 20}) {
    for (size_t query_idx = 0; query_idx < 20; ++query_idx) {
      std::string query = mutate_with_indels(items[query_idx].sequence, tau, gen);
      navigamer::QGramCountIndex::QueryStats stats;
      auto candidates = index.query(query, tau, &stats);
      assert(std::is_sorted(candidates.begin(), candidates.end()));
      assert(std::adjacent_find(candidates.begin(), candidates.end()) ==
             candidates.end());
      assert(stats.total_items == items.size());
      for (const auto& item : items) {
        if (navigamer::compute_distance(query, item.sequence) <= tau) {
          assert(contains(candidates, item.item_id));
        }
      }
    }
  }

  auto ambiguous_candidates = index.query("ANNT", 0);
  assert(contains(ambiguous_candidates, 80));
  auto short_candidates = index.query("A", 0);
  assert(contains(short_candidates, 81));
}

}  // namespace

int main() {
  test_qgram_counts_basic();
  test_qgram_l1_basic();
  test_qgram_l1_bound_no_false_negative();
  test_qgram_index_no_false_negative();
  std::cout << "qgram filter tests passed\n";
  return 0;
}
