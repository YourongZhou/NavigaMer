#include "qgram_filter.hpp"
#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
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

void test_qgram_signature_basic() {
  auto short_sig = navigamer::compute_qgram_signature("ACGT", 2);
  assert(short_sig.safe_for_pruning);
  assert(short_sig.q == 2);
  assert(short_sig.sequence_length == 4);
  assert(short_sig.total_qgrams == 3);
  assert(short_sig.entries.size() == 3);
  assert(short_sig.entries[0].code == 1 && short_sig.entries[0].count == 1);
  assert(short_sig.entries[1].code == 6 && short_sig.entries[1].count == 1);
  assert(short_sig.entries[2].code == 11 && short_sig.entries[2].count == 1);

  auto repeated_sig = navigamer::compute_qgram_signature("ACGTAC", 2);
  assert(repeated_sig.safe_for_pruning);
  assert(repeated_sig.total_qgrams == 5);
  assert(repeated_sig.entries.size() == 4);
  assert(repeated_sig.entries[0].code == 1 && repeated_sig.entries[0].count == 2);
  assert(repeated_sig.entries[1].code == 6 && repeated_sig.entries[1].count == 1);
  assert(repeated_sig.entries[2].code == 11 && repeated_sig.entries[2].count == 1);
  assert(repeated_sig.entries[3].code == 12 && repeated_sig.entries[3].count == 1);
}

void test_qgram_signature_safe_fallbacks() {
  auto empty = navigamer::compute_qgram_signature("A", 2);
  assert(empty.safe_for_pruning);
  assert(empty.total_qgrams == 0);
  assert(empty.entries.empty());

  auto ambiguous = navigamer::compute_qgram_signature("ACNT", 2);
  assert(!ambiguous.safe_for_pruning);
  auto invalid_q = navigamer::compute_qgram_signature("ACGT", 0);
  assert(!invalid_q.safe_for_pruning);
  auto unsupported_q = navigamer::compute_qgram_signature("ACGT", 33);
  assert(!unsupported_q.safe_for_pruning);

  auto safe = navigamer::compute_qgram_signature("ACGT", 2);
  assert(!navigamer::qgram_can_prune_edit_distance(ambiguous, safe, 0));
  assert(!navigamer::qgram_can_prune_edit_distance(invalid_q, safe, 0));
  assert(!navigamer::qgram_can_prune_edit_distance(safe, safe, -1));
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
            auto lhs_sig = navigamer::compute_qgram_signature(lhs, q);
            auto rhs_sig = navigamer::compute_qgram_signature(rhs, q);
            assert(!navigamer::qgram_can_prune_edit_distance(
                lhs_sig, rhs_sig, tau));
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

void test_qgram_index_reuses_sparse_workspace() {
  std::mt19937 gen(2026);
  std::vector<navigamer::QGramCountIndex::Item> items;
  for (size_t i = 0; i < 128; ++i) {
    items.push_back({i, random_dna(60 + i % 5, gen)});
  }
  items.push_back({128, "A"});

  navigamer::QGramCountIndex index(5);
  index.build(items);
  navigamer::QGramQueryWorkspace workspace;

  navigamer::QGramCountIndex::QueryStats first_stats;
  auto first = index.query(items[0].sequence, 2, &first_stats, &workspace);
  assert(contains(first, 0));
  assert(workspace.shared.size() == items.size());
  assert(workspace.seen_epoch.size() == items.size());
  const auto* shared_storage = workspace.shared.data();
  const auto* seen_storage = workspace.seen_epoch.data();

  navigamer::QGramCountIndex::QueryStats second_stats;
  auto second = index.query(items[1].sequence, 2, &second_stats, &workspace);
  assert(contains(second, 1));
  assert(workspace.shared.data() == shared_storage);
  assert(workspace.seen_epoch.data() == seen_storage);
  assert(workspace.shared.size() == items.size());
  assert(workspace.seen_epoch.size() == items.size());

  navigamer::QGramCountIndex::QueryStats short_stats;
  auto short_candidates = index.query("A", 0, &short_stats, &workspace);
  assert(short_stats.full_scan_fallbacks == 1);
  assert(contains(short_candidates, 128));

  auto without_workspace = index.query(items[2].sequence, 3);
  navigamer::QGramCountIndex::QueryStats with_workspace_stats;
  auto with_workspace =
      index.query(items[2].sequence, 3, &with_workspace_stats, &workspace);
  assert(with_workspace == without_workspace);
}

void test_qgram_index_sorts_nonmonotonic_item_ids() {
  const std::vector<navigamer::QGramCountIndex::Item> items = {
      {9, "ACGTACGTACGT"},
      {3, "ACGTACGTACGT"},
      {9, "ACGTACGTACGT"},
  };
  navigamer::QGramCountIndex index(5);
  index.build(items);
  const auto candidates = index.query("ACGTACGTACGT", 0);
  assert((candidates == std::vector<size_t>{3, 9}));
}

void test_qgram_dense_postings_use_wide_fallbacks() {
  constexpr size_t item_count =
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 2;
  std::vector<navigamer::QGramCountIndex::Item> many_items;
  many_items.reserve(item_count);
  for (size_t idx = 0; idx < item_count; ++idx) {
    many_items.push_back({idx, "ACGTA"});
  }
  navigamer::QGramCountIndex many_index(5);
  many_index.build(many_items);
  const auto many_candidates = many_index.query("ACGTA", 0);
  assert(many_candidates.size() == item_count);
  assert(many_candidates.front() == 0);
  assert(many_candidates.back() == item_count - 1);

  const std::string long_sequence(
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 8, 'A');
  navigamer::QGramCountIndex long_index(5);
  long_index.build({{7, long_sequence}});
  const auto long_candidates = long_index.query(long_sequence, 0);
  assert((long_candidates == std::vector<size_t>{7}));
}

}  // namespace

int main() {
  test_qgram_counts_basic();
  test_qgram_l1_basic();
  test_qgram_signature_basic();
  test_qgram_signature_safe_fallbacks();
  test_qgram_l1_bound_no_false_negative();
  test_qgram_index_no_false_negative();
  test_qgram_index_reuses_sparse_workspace();
  test_qgram_index_sorts_nonmonotonic_item_ids();
  test_qgram_dense_postings_use_wide_fallbacks();
  std::cout << "qgram filter tests passed\n";
  return 0;
}
