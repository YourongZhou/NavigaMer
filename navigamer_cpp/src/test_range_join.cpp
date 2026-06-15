#include "range_join.hpp"
#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
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

std::string mutate(std::string value, int edits, std::mt19937& gen) {
  if (value.empty()) return value;
  std::uniform_int_distribution<size_t> pos(0, value.size() - 1);
  for (int i = 0; i < edits; ++i) {
    size_t p = pos(gen);
    value[p] = value[p] == 'A' ? 'C' : 'A';
  }
  return value;
}

std::string mutate_with_indels(std::string value, int edits, std::mt19937& gen) {
  std::uniform_int_distribution<int> operation(0, 2);
  std::uniform_int_distribution<int> base(0, 3);
  static const char bases[] = "ACGT";
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

}  // namespace

int main() {
  using navigamer::ExactRangeJoinIndex;
  using navigamer::RangeJoinConfig;
  using navigamer::RangeJoinItem;

  std::mt19937 gen(42);
  std::vector<RangeJoinItem> items;
  for (size_t i = 0; i < 160; ++i) items.push_back({i, random_dna(100, gen)});

  ExactRangeJoinIndex index(RangeJoinConfig{8, 20});
  index.build(items);

  auto adaptive = index.query(items[0].sequence, 2);
  assert(!adaptive.used_full_scan);
  assert(adaptive.block_len == 33);
  assert(adaptive.seed_len == 20);
  assert(contains(adaptive.candidate_item_ids, items[0].item_id));
  assert(std::adjacent_find(adaptive.candidate_item_ids.begin(),
                            adaptive.candidate_item_ids.end()) ==
         adaptive.candidate_item_ids.end());

  auto fallback = index.query(items[0].sequence, 20);
  assert(fallback.used_full_scan);
  assert(fallback.block_len == 4);
  assert(fallback.seed_len == 4);
  assert(fallback.candidate_item_ids.size() == items.size());

  for (int tau : {0, 1, 2, 5, 10, 20}) {
    for (size_t q_idx = 0; q_idx < 40; ++q_idx) {
      std::string query = mutate(items[q_idx].sequence, std::min(tau, 5), gen);
      auto result = index.query(query, tau);
      std::unordered_set<size_t> verified;
      for (size_t item_id : result.candidate_item_ids) {
        int distance = navigamer::compute_distance_bounded(
            query, items[item_id].sequence, tau);
        if (distance <= tau) verified.insert(item_id);
      }

      for (const auto& item : items) {
        int full = navigamer::compute_distance(query, item.sequence);
        if (full <= tau) {
          assert(contains(result.candidate_item_ids, item.item_id));
          assert(verified.count(item.item_id) == 1);
        } else {
          assert(verified.count(item.item_id) == 0);
        }
      }
    }
  }

  for (int tau : {1, 2, 5, 10}) {
    for (size_t item_idx = 0; item_idx < 30; ++item_idx) {
      std::string query = mutate_with_indels(items[item_idx].sequence, tau, gen);
      auto result = index.query(query, tau);
      assert(navigamer::compute_distance(query, items[item_idx].sequence) <= tau);
      assert(contains(result.candidate_item_ids, items[item_idx].item_id));
    }
  }

  std::cout << "range join tests passed\n";
  return 0;
}
