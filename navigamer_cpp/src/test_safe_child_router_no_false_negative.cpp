#include "index_builder.hpp"
#include "search_engine.hpp"
#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using SequencePtr = std::shared_ptr<navigamer::BioSequence>;

std::vector<SequencePtr> make_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a0", "ACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("a1", "ACGTACGTTCGT"),
      std::make_shared<navigamer::BioSequence>("a2", "ACGTTCGTACGT"),
      std::make_shared<navigamer::BioSequence>("a3", "ACGTACGTAGGT"),
      std::make_shared<navigamer::BioSequence>("c0", "CCCCGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("c1", "CCCCGGGACCCC"),
      std::make_shared<navigamer::BioSequence>("c2", "CCCCGCGGCCCC"),
      std::make_shared<navigamer::BioSequence>("g0", "GGGGCCCCGGGG"),
      std::make_shared<navigamer::BioSequence>("g1", "GGGGCCCCAGGG"),
      std::make_shared<navigamer::BioSequence>("g2", "GGGGACCCGGGG"),
      std::make_shared<navigamer::BioSequence>("t0", "TTTTAAAATTTT"),
      std::make_shared<navigamer::BioSequence>("t1", "TTTTAAAAGTTT"),
      std::make_shared<navigamer::BioSequence>("t2", "TTTGAAAATTTT"),
      std::make_shared<navigamer::BioSequence>("m0", "AAAACCCCGGGG"),
      std::make_shared<navigamer::BioSequence>("m1", "AAAACCCCGGGA"),
      std::make_shared<navigamer::BioSequence>("n0", "TTTTGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("n1", "TTTTGGGGCCCA"),
      std::make_shared<navigamer::BioSequence>("o0", "ACACACACACAC"),
      std::make_shared<navigamer::BioSequence>("o1", "ACACACACACAT"),
      std::make_shared<navigamer::BioSequence>("p0", "TGTGTGTGTGTG"),
      std::make_shared<navigamer::BioSequence>("p1", "TGTGTGTGTGTA"),
  };
}

std::set<std::string> ids(const std::vector<SequencePtr>& hits) {
  std::set<std::string> out;
  for (const auto& hit : hits) {
    if (hit) out.insert(hit->id);
  }
  return out;
}

navigamer::SearchConfig baseline_config() {
  navigamer::SearchConfig config;
  config.graph_view_mode = navigamer::GraphViewMode::Flat;
  config.visited_mode = navigamer::VisitedMode::Epoch;
  config.query_profile = true;
  return config;
}

navigamer::SearchConfig safe_config() {
  auto config = baseline_config();
  config.safe_child_router_enabled = true;
  config.safe_child_router_min_fanout = 1;
  config.safe_child_router_max_candidates = 4096;
  config.safe_child_router_max_ratio = 1.0;
  config.safe_child_router_min_seed_len = 2;
  config.safe_child_router_mode = "qgram";
  config.safe_child_router_validate = true;
  return config;
}

navigamer::SearchConfig safe_mbb_config() {
  auto config = safe_config();
  config.safe_child_router_mode = "mbb";
  config.safe_child_router_validate = false;
  return config;
}

std::shared_ptr<navigamer::WorldNode> first_parent_with_children(
    const navigamer::BioGeometryIndexBuilder& builder) {
  for (int layer = builder.coarsest_primary_layer_index();
       layer < builder.finest_primary_layer_index(); ++layer) {
    for (const auto& node : builder.primary_layer(layer)) {
      if (node && !node->child_nodes.empty()) return node;
    }
  }
  return nullptr;
}

}  // namespace

int main() {
  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({12, 6, 2}), build_config);
  builder.build(make_sequences());

  navigamer::BioGeometrySearchEngine baseline(builder, baseline_config());
  navigamer::BioGeometrySearchEngine safe(builder, safe_config());

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTTCGT"),
      navigamer::BioSequence("q2", "CCCCGGGGCCCC"),
      navigamer::BioSequence("q3", "TTTTAAAATTTT"),
      navigamer::BioSequence("q4", "GGGGCCCCGGGG"),
      navigamer::BioSequence("q5", "ACACACACACAC"),
  };

  size_t invoked = 0;
  for (const auto& query : queries) {
    auto [baseline_hits, baseline_stats] = baseline.search_adaptive(query, 2);
    auto [safe_hits, safe_stats] = safe.search_adaptive(query, 2);
    if (ids(safe_hits) != ids(baseline_hits)) {
      std::cerr << "safe child router mismatch for " << query.id << "\n";
      std::cerr << "baseline:";
      for (const auto& id : ids(baseline_hits)) std::cerr << " " << id;
      std::cerr << "\nsafe:";
      for (const auto& id : ids(safe_hits)) std::cerr << " " << id;
      std::cerr << "\ninvoked=" << safe_stats.safe_child_router_invoked_count
                << " candidates=" << safe_stats.safe_child_router_candidate_count
                << " pruned_not_candidate="
                << safe_stats.safe_child_router_pruned_by_not_candidate_count
                << " fallback=" << safe_stats.safe_child_router_fallback_count
                << "\n";
      assert(false);
    }
    assert(safe_stats.result_count == baseline_stats.result_count);
    assert(safe_stats.safe_child_router_exact_verify_count >=
           safe_stats.safe_child_router_invoked_count);
    if (safe_stats.safe_child_router_candidate_count > 0) {
      assert(safe_stats.child_count_before_router > 0);
      assert(safe_stats.post_mbb_survivor_count > 0);
      assert(safe_stats.safe_router_candidate_count ==
             safe_stats.safe_child_router_candidate_count);
      assert(safe_stats.children_actually_processed <=
             safe_stats.child_count_before_router);
      assert(safe_stats.center_checks_saved ==
             safe_stats.child_count_before_router -
                 safe_stats.children_actually_processed);
      assert(safe_stats.candidate_ratio_to_all_children > 0.0);
      assert(safe_stats.candidate_ratio_to_post_mbb_survivors > 0.0);
    }
    if (safe_stats.safe_child_router_invoked_count > 0 &&
        safe_stats.safe_child_router_fallback_count == 0) {
      assert(safe_stats.safe_child_router_exact_pruned_count > 0);
      assert(safe_stats.safe_child_router_center_distance_reused_count > 0);
    }
    invoked += safe_stats.safe_child_router_invoked_count;
  }
  assert(invoked > 0);

  auto parent = first_parent_with_children(builder);
  assert(parent);
  bool used_router = false;
  auto candidate_ids = safe.debug_safe_child_router_candidate_ids(
      parent->node_id, queries[0], 2, &used_router);
  assert(used_router);
  std::set<std::string> candidate_set(candidate_ids.begin(), candidate_ids.end());
  for (const auto& child : parent->child_nodes) {
    assert(child && child->center_ptr);
    const int tau = 2 + child->radius;
    const int dist = navigamer::compute_distance_bounded_with_mode(
        queries[0].seq, child->center_ptr->seq, tau,
        navigamer::DistanceMode::Myers);
    if (dist <= tau) {
      assert(candidate_set.count(child->node_id) == 1);
    }
  }

  auto fallback_config = safe_config();
  fallback_config.safe_child_router_max_ratio = 0.0;
  navigamer::BioGeometrySearchEngine fallback(builder, fallback_config);
  auto [fallback_hits, fallback_stats] =
      fallback.search_adaptive(queries[0], 2);
  auto [baseline_hits, baseline_stats] =
      baseline.search_adaptive(queries[0], 2);
  assert(ids(fallback_hits) == ids(baseline_hits));
  assert(fallback_stats.result_count == baseline_stats.result_count);
  assert(fallback_stats.safe_child_router_fallback_count > 0);

  navigamer::BioGeometrySearchEngine safe_mbb(builder, safe_mbb_config());
  size_t mbb_saved = 0;
  size_t mbb_invoked = 0;
  for (const auto& query : queries) {
    auto [baseline_mbb_hits, baseline_mbb_stats] =
        baseline.search_adaptive(query, 2);
    auto [safe_mbb_hits, safe_mbb_stats] =
        safe_mbb.search_adaptive(query, 2);
    assert(ids(safe_mbb_hits) == ids(baseline_mbb_hits));
    assert(safe_mbb_stats.result_count == baseline_mbb_stats.result_count);
    mbb_invoked += safe_mbb_stats.safe_child_router_invoked_count;
    mbb_saved += safe_mbb_stats.center_checks_saved;
    if (safe_mbb_stats.safe_child_router_candidate_count > 0) {
      assert(safe_mbb_stats.safe_router_candidate_count ==
             safe_mbb_stats.safe_child_router_candidate_count);
      assert(safe_mbb_stats.children_actually_processed <=
             safe_mbb_stats.child_count_before_router);
      assert(safe_mbb_stats.center_checks_saved ==
             safe_mbb_stats.child_count_before_router -
                 safe_mbb_stats.children_actually_processed);
      assert(safe_mbb_stats.candidate_ratio_to_all_children > 0.0);
      assert(safe_mbb_stats.candidate_ratio_to_post_mbb_survivors > 0.0);
    }
  }
  assert(mbb_invoked > 0);
  assert(mbb_saved > 0);

  std::cout << "safe child router no-false-negative tests passed\n";
  return 0;
}
