#include "range_join.hpp"
#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
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

std::vector<size_t> intersection(
    const std::vector<size_t>& lhs, const std::vector<size_t>& rhs) {
  std::vector<size_t> out;
  std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::back_inserter(out));
  return out;
}

navigamer::RangeJoinConfig config_for(navigamer::RangeCandidateMode mode) {
  navigamer::RangeJoinConfig config;
  config.min_seed_len = 8;
  config.max_seed_len = 20;
  config.qgram_q = 5;
  config.candidate_mode = mode;
  return config;
}

std::vector<navigamer::RangeJoinItem> posting_explosion_items() {
  std::vector<navigamer::RangeJoinItem> items;
  const std::string shared_prefix(20, 'A');
  for (size_t i = 0; i < 4100; ++i) {
    std::string suffix(80, static_cast<char>('C' + (i % 2)));
    suffix[i % suffix.size()] = i % 3 == 0 ? 'G' : 'T';
    items.push_back({i, shared_prefix + suffix});
  }
  items.push_back({4100, std::string(100, 'A')});
  return items;
}

}  // namespace

int main() {
  using navigamer::ExactRangeJoinIndex;
  using navigamer::RangeCandidateMode;
  using navigamer::RangeJoinConfig;
  using navigamer::RangeJoinItem;

  std::mt19937 gen(42);
  std::vector<RangeJoinItem> items;
  for (size_t i = 0; i < 160; ++i) items.push_back({i, random_dna(100, gen)});

  ExactRangeJoinIndex index(config_for(RangeCandidateMode::Auto));
  index.build(items);

  auto adaptive = index.query(items[0].sequence, 2);
  assert(!adaptive.used_full_scan);
  assert(adaptive.mode_used == RangeCandidateMode::PigeonholeOnly);
  assert(adaptive.block_len == 33);
  assert(adaptive.seed_len == 20);
  assert(contains(adaptive.candidate_item_ids, items[0].item_id));
  assert(std::adjacent_find(adaptive.candidate_item_ids.begin(),
                            adaptive.candidate_item_ids.end()) ==
         adaptive.candidate_item_ids.end());

  auto fallback_index =
      ExactRangeJoinIndex(config_for(RangeCandidateMode::PigeonholeOnly));
  fallback_index.build(items);
  auto fallback = fallback_index.query(items[0].sequence, 20);
  assert(fallback.used_full_scan);
  assert(fallback.block_len == 4);
  assert(fallback.seed_len == 4);
  assert(fallback.candidate_item_ids.size() == items.size());

  auto auto_qgram = index.query(items[0].sequence, 20);
  assert(!auto_qgram.used_full_scan);
  assert(auto_qgram.mode_used == RangeCandidateMode::QGramOnly);

  ExactRangeJoinIndex pigeonhole_index(
      config_for(RangeCandidateMode::PigeonholeOnly));
  ExactRangeJoinIndex qgram_index(config_for(RangeCandidateMode::QGramOnly));
  ExactRangeJoinIndex hybrid_index(config_for(RangeCandidateMode::Hybrid));
  ExactRangeJoinIndex full_index(config_for(RangeCandidateMode::FullScan));
  pigeonhole_index.build(items);
  qgram_index.build(items);
  hybrid_index.build(items);
  full_index.build(items);

  for (int tau : {0, 1, 2, 5, 10, 20}) {
    for (size_t q_idx = 0; q_idx < 40; ++q_idx) {
      std::string query = mutate(items[q_idx].sequence, std::min(tau, 5), gen);
      auto result = index.query(query, tau);
      auto pigeonhole = pigeonhole_index.query(query, tau);
      auto qgram = qgram_index.query(query, tau);
      auto hybrid = hybrid_index.query(query, tau);
      auto full = full_index.query(query, tau);
      assert(hybrid.candidate_item_ids ==
             intersection(pigeonhole.candidate_item_ids,
                          qgram.candidate_item_ids));
      assert(full.mode_used == RangeCandidateMode::FullScan);
      assert(qgram.mode_used == RangeCandidateMode::QGramOnly);
      assert(hybrid.mode_used == RangeCandidateMode::Hybrid);
      auto verified_ids = [&](const navigamer::RangeJoinQueryResult& candidates) {
        std::unordered_set<size_t> verified;
        for (size_t item_id : candidates.candidate_item_ids) {
          int distance = navigamer::compute_distance_bounded(
              query, items[item_id].sequence, tau);
          if (distance <= tau) verified.insert(item_id);
        }
        return verified;
      };
      auto verified = verified_ids(result);
      auto verified_pigeonhole = verified_ids(pigeonhole);
      auto verified_qgram = verified_ids(qgram);
      auto verified_hybrid = verified_ids(hybrid);
      auto verified_full = verified_ids(full);
      std::unordered_set<size_t> true_matches;

      for (const auto& item : items) {
        int full_distance = navigamer::compute_distance(query, item.sequence);
        if (full_distance <= tau) {
          true_matches.insert(item.item_id);
          assert(contains(result.candidate_item_ids, item.item_id));
          assert(contains(pigeonhole.candidate_item_ids, item.item_id));
          assert(contains(qgram.candidate_item_ids, item.item_id));
          assert(contains(hybrid.candidate_item_ids, item.item_id));
          assert(verified.count(item.item_id) == 1);
        } else {
          assert(verified.count(item.item_id) == 0);
        }
      }
      assert(verified == true_matches);
      assert(verified_pigeonhole == true_matches);
      assert(verified_qgram == true_matches);
      assert(verified_hybrid == true_matches);
      assert(verified_full == true_matches);
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

  std::vector<RangeJoinItem> ambiguous_items = {
      {0, "AACNNGTACN"}, {1, "AACNAGTACN"}, {2, "TTTTTTTTTT"}};
  ExactRangeJoinIndex ambiguous_index(config_for(RangeCandidateMode::Hybrid));
  ambiguous_index.build(ambiguous_items);
  auto ambiguous = ambiguous_index.query("AACNNGTACN", 1);
  assert(contains(ambiguous.candidate_item_ids, 0));
  assert(contains(ambiguous.candidate_item_ids, 1));

  auto explosion_items = posting_explosion_items();
  auto auto_config = config_for(RangeCandidateMode::Auto);
  ExactRangeJoinIndex explosion_auto(auto_config);
  ExactRangeJoinIndex explosion_pigeonhole(
      config_for(RangeCandidateMode::PigeonholeOnly));
  ExactRangeJoinIndex explosion_qgram(config_for(RangeCandidateMode::QGramOnly));
  ExactRangeJoinIndex explosion_hybrid(config_for(RangeCandidateMode::Hybrid));
  explosion_auto.build(explosion_items);
  explosion_pigeonhole.build(explosion_items);
  explosion_qgram.build(explosion_items);
  explosion_hybrid.build(explosion_items);
  const std::string explosion_query = std::string(100, 'A');
  auto selected = explosion_auto.query(explosion_query, 4);
  auto explosion_p = explosion_pigeonhole.query(explosion_query, 4);
  auto explosion_q = explosion_qgram.query(explosion_query, 4);
  auto explosion_h = explosion_hybrid.query(explosion_query, 4);
  assert(explosion_p.candidate_item_ids.size() >
         auto_config.auto_pigeonhole_max_candidates);
  assert(explosion_q.candidate_item_ids.size() <
         explosion_p.candidate_item_ids.size());
  assert(selected.candidate_item_ids == explosion_h.candidate_item_ids);
  assert(selected.mode_used == RangeCandidateMode::Hybrid);
  assert(selected.auto_pigeonhole_rejected_large_candidates == 1);
  assert(selected.auto_qgram_invoked == 1);
  assert(selected.auto_hybrid_invoked == 1);
  assert(selected.auto_final_candidate_pairs == selected.candidate_item_ids.size());

  auto no_hybrid_config = auto_config;
  no_hybrid_config.auto_hybrid_on_large_candidates = false;
  ExactRangeJoinIndex explosion_no_hybrid(no_hybrid_config);
  explosion_no_hybrid.build(explosion_items);
  auto selected_qgram = explosion_no_hybrid.query(explosion_query, 4);
  assert(selected_qgram.candidate_item_ids == explosion_q.candidate_item_ids);
  assert(selected_qgram.mode_used == RangeCandidateMode::QGramOnly);
  assert(selected_qgram.auto_qgram_invoked == 1);
  assert(selected_qgram.auto_hybrid_invoked == 0);

  auto old_auto_config = auto_config;
  old_auto_config.auto_pigeonhole_max_candidates =
      std::numeric_limits<size_t>::max();
  old_auto_config.auto_pigeonhole_max_ratio = 1.0;
  ExactRangeJoinIndex old_auto(old_auto_config);
  old_auto.build(explosion_items);
  auto selected_pigeonhole = old_auto.query(explosion_query, 4);
  assert(selected_pigeonhole.candidate_item_ids ==
         explosion_p.candidate_item_ids);
  assert(selected_pigeonhole.mode_used == RangeCandidateMode::PigeonholeOnly);
  assert(selected_pigeonhole.auto_pigeonhole_accepted == 1);
  assert(selected_pigeonhole.auto_qgram_invoked == 0);
  assert(selected_pigeonhole.auto_hybrid_invoked == 0);

  std::cout << "range join tests passed\n";
  return 0;
}
