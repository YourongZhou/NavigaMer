#include "phase1_seed_index.hpp"
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
  std::string sequence;
  sequence.reserve(length);
  for (size_t i = 0; i < length; ++i) sequence.push_back(bases[pick(gen)]);
  return sequence;
}

std::string mutate_with_indels(std::string sequence, int edits,
                               std::mt19937& gen) {
  static const char bases[] = "ACGT";
  std::uniform_int_distribution<int> operation(0, 2);
  std::uniform_int_distribution<int> base(0, 3);
  for (int edit = 0; edit < edits; ++edit) {
    const int op = operation(gen);
    if (op == 0 && !sequence.empty()) {
      std::uniform_int_distribution<size_t> pos(0, sequence.size() - 1);
      sequence[pos(gen)] = bases[base(gen)];
    } else if (op == 1 && !sequence.empty()) {
      std::uniform_int_distribution<size_t> pos(0, sequence.size() - 1);
      sequence.erase(pos(gen), 1);
    } else {
      std::uniform_int_distribution<size_t> pos(0, sequence.size());
      sequence.insert(pos(gen), 1, bases[base(gen)]);
    }
  }
  return sequence;
}

void assert_contains_all_matches(
    navigamer::IncrementalPigeonholeIndex& index,
    const std::vector<std::string>& candidates,
    const std::string& query, int tau) {
  const auto result = index.query(query, tau);
  assert(result.safe);
  assert(std::is_sorted(result.candidate_indices.begin(),
                        result.candidate_indices.end()));
  assert(std::adjacent_find(result.candidate_indices.begin(),
                            result.candidate_indices.end()) ==
         result.candidate_indices.end());
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    if (navigamer::compute_distance(query, candidates[idx]) <= tau) {
      assert(std::binary_search(result.candidate_indices.begin(),
                                result.candidate_indices.end(), idx));
    }
  }
}

void test_incremental_queries_are_recall_safe() {
  std::mt19937 gen(99173);
  std::vector<std::string> candidates;
  for (size_t idx = 0; idx < 180; ++idx) {
    candidates.push_back(random_dna(80 + idx % 21, gen));
  }

  navigamer::IncrementalPigeonholeIndex index({8, 20});
  for (size_t idx = 0; idx < 90; ++idx) index.append(idx, candidates[idx]);

  for (int tau : {1, 2, 4, 6}) {
    const std::string query =
        mutate_with_indels(candidates[static_cast<size_t>(tau * 7)], tau, gen);
    assert(navigamer::compute_distance(
               query, candidates[static_cast<size_t>(tau * 7)]) <= tau);
    assert_contains_all_matches(index,
                                std::vector<std::string>(candidates.begin(),
                                                         candidates.begin() + 90),
                                query, tau);
  }

  // Append after seed-length postings have already been materialized.
  for (size_t idx = 90; idx < candidates.size(); ++idx) {
    index.append(idx, candidates[idx]);
  }
  assert(index.size() == candidates.size());

  for (int tau : {1, 2, 4, 6}) {
    const size_t source_idx = 120 + static_cast<size_t>(tau);
    const std::string query =
        mutate_with_indels(candidates[source_idx], tau, gen);
    assert(navigamer::compute_distance(query, candidates[source_idx]) <= tau);
    assert_contains_all_matches(index, candidates, query, tau);
  }
}

void test_unindexable_candidates_are_not_dropped() {
  navigamer::IncrementalPigeonholeIndex index({4, 12});
  const std::vector<std::string> candidates = {
      "ACGTACGTACGTACGT",
      "ACGTACGTNCGTACGT",
      "TTTTTTTTTTTTTTTT",
  };
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    index.append(idx, candidates[idx]);
  }

  const auto result = index.query("ACGTACGTACGTACGT", 1);
  assert(result.safe);
  assert(std::binary_search(result.candidate_indices.begin(),
                            result.candidate_indices.end(), size_t{0}));
  assert(std::binary_search(result.candidate_indices.begin(),
                            result.candidate_indices.end(), size_t{1}));
}

void test_unsafe_queries_request_fallback() {
  navigamer::IncrementalPigeonholeIndex index({8, 20});
  index.append(0, std::string(80, 'A'));

  assert(!index.query("ACGTNNNNACGT", 1).safe);
  assert(!index.query(std::string(40, 'A'), 20).safe);
}

void test_far_seed_occurrences_are_filtered_by_position() {
  navigamer::IncrementalPigeonholeIndex index({4, 12});
  const std::string query = "AAAACCCCGGGGTTTTACGTACGT";
  index.append(0, query);
  index.append(1, "TTTTACGTACGTAAAACCCCGGGG");

  const auto result = index.query(query, 1);
  assert(result.safe);
  assert(std::binary_search(result.candidate_indices.begin(),
                            result.candidate_indices.end(), size_t{0}));
  assert(!std::binary_search(result.candidate_indices.begin(),
                             result.candidate_indices.end(), size_t{1}));
}

void test_fixed_layout_sparse_postings_are_recall_safe() {
  constexpr size_t sequence_length = 250;
  constexpr int tau = 5;
  std::mt19937 gen(17191);
  const std::string query = random_dna(sequence_length, gen);
  std::vector<std::string> candidates;
  candidates.reserve(260);
  candidates.push_back(query);

  std::string substitutions = query;
  for (size_t idx : {size_t{0}, size_t{41}, size_t{82},
                     size_t{164}, size_t{249}}) {
    substitutions[idx] = substitutions[idx] == 'A' ? 'C' : 'A';
  }
  candidates.push_back(std::move(substitutions));

  std::string balanced_indels = query;
  balanced_indels.erase(17, 1);
  balanced_indels.insert(211, 1, 'A');
  candidates.push_back(std::move(balanced_indels));
  for (size_t idx = candidates.size(); idx < 260; ++idx) {
    candidates.push_back(random_dna(sequence_length, gen));
  }

  navigamer::IncrementalPigeonholeIndex index(
      {8, 20, sequence_length, tau});
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    index.append(idx, candidates[idx]);
  }
  assert_contains_all_matches(index, candidates, query, tau);
  assert_contains_all_matches(index, candidates, query, tau - 1);
  assert(!index.query(query, tau + 1).safe);
  assert(!index.query(query.substr(1), tau).safe);
}

void test_fixed_layout_matches_dense_candidate_sets() {
  constexpr size_t sequence_length = 250;
  constexpr int maximum_tau = 30;
  std::mt19937 gen(44109);
  std::vector<std::string> candidates;
  candidates.reserve(320);
  for (size_t idx = 0; idx < 320; ++idx) {
    candidates.push_back(random_dna(sequence_length, gen));
  }

  navigamer::IncrementalPigeonholeIndex dense({8, 20});
  navigamer::IncrementalPigeonholeIndex compact(
      {8, 20, sequence_length, maximum_tau});
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    dense.append(idx, candidates[idx]);
    compact.append(idx, candidates[idx]);
  }

  for (int tau : {0, 1, 4, 5, 10, 15, 22, 30}) {
    for (size_t query_idx = 0; query_idx < 12; ++query_idx) {
      const std::string query = mutate_with_indels(
          candidates[(query_idx * 23 + static_cast<size_t>(tau)) %
                     candidates.size()],
          0, gen);
      const auto dense_result = dense.query(query, tau);
      const auto compact_result = compact.query(query, tau);
      assert(dense_result.safe);
      assert(compact_result.safe);
      assert(dense_result.candidate_indices ==
             compact_result.candidate_indices);
    }
  }
  assert(compact.posting_entry_count() <
         dense.posting_entry_count());
}

void test_packed_postings_promote_without_losing_candidates() {
  navigamer::IncrementalPigeonholeIndex packed_items({4, 4});
  const std::string short_sequence = "ACGT";
  for (size_t idx = 0; idx < 300; ++idx) {
    packed_items.append(idx, short_sequence);
  }
  const auto packed_result = packed_items.query(short_sequence, 0);
  assert(packed_result.safe);
  assert(packed_result.candidate_indices.size() == 300);

  // A 21-mer all-T code exceeds 40 bits. This exercises the exact wide-key
  // head-table fallback and then the Compact24 -> Packed32 posting promotion.
  navigamer::IncrementalPigeonholeIndex many_items({21, 21});
  const std::string wide_key_sequence(21, 'T');
  constexpr size_t item_count =
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 2;
  for (size_t idx = 0; idx < item_count; ++idx) {
    many_items.append(idx, wide_key_sequence);
  }
  const auto many_result = many_items.query(wide_key_sequence, 0);
  assert(many_result.safe);
  assert(many_result.candidate_indices.size() == item_count);
  assert(many_result.candidate_indices.front() == 0);
  assert(many_result.candidate_indices.back() == item_count - 1);

  navigamer::IncrementalPigeonholeIndex long_position({4, 4});
  const std::string long_sequence(
      static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 8, 'A');
  long_position.append(0, long_sequence);
  const auto long_result = long_position.query(long_sequence, 0);
  assert(long_result.safe);
  assert((long_result.candidate_indices == std::vector<size_t>{0}));
}

}  // namespace

int main() {
  test_incremental_queries_are_recall_safe();
  test_unindexable_candidates_are_not_dropped();
  test_unsafe_queries_request_fallback();
  test_far_seed_occurrences_are_filtered_by_position();
  test_fixed_layout_sparse_postings_are_recall_safe();
  test_fixed_layout_matches_dense_candidate_sets();
  test_packed_postings_promote_without_losing_candidates();
  std::cout << "phase1 seed index tests passed\n";
  return 0;
}
