#include "index_builder.hpp"
#include "search_engine.hpp"

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
      std::make_shared<navigamer::BioSequence>("c0", "CCCCGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("c1", "CCCCGGGACCCC"),
      std::make_shared<navigamer::BioSequence>("g0", "GGGGCCCCGGGG"),
      std::make_shared<navigamer::BioSequence>("g1", "GGGGCCCCAGGG"),
      std::make_shared<navigamer::BioSequence>("t0", "TTTTAAAATTTT"),
      std::make_shared<navigamer::BioSequence>("t1", "TTTTAAAAGTTT"),
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

navigamer::SearchConfig optimized_planner_config(size_t router_min_fanout) {
  auto config = baseline_config();
  config.search_qgram_prefilter = true;
  config.search_qgram_q = 3;
  config.router_hint_enabled = true;
  config.router_hint_qgram_q = 3;
  config.router_hint_minimizer_k = 3;
  config.router_hint_minimizer_w = 5;
  config.local_router_enabled = true;
  config.best_first_enabled = true;
  config.safe_child_router_enabled = true;
  config.safe_child_router_min_fanout = 1;
  config.safe_child_router_max_ratio = 1.0;
  config.safe_child_router_min_seed_len = 2;
  config.safe_child_router_mode = "qgram";
  config.query_planner_enabled = true;
  config.planner_router_min_fanout = router_min_fanout;
  config.planner_safe_child_router_min_fanout = router_min_fanout;
  return config;
}

}  // namespace

int main() {
  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({12, 6, 2}), build_config);
  builder.build(make_sequences());

  navigamer::BioGeometrySearchEngine baseline(builder, baseline_config());
  navigamer::BioGeometrySearchEngine low_fanout_planner(
      builder, optimized_planner_config(1000));
  navigamer::BioGeometrySearchEngine high_fanout_planner(
      builder, optimized_planner_config(1));

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGT"),
      navigamer::BioSequence("q1", "CCCCGGGGCCCC"),
      navigamer::BioSequence("q2", "TTTTAAAATTTT"),
  };

  for (const auto& query : queries) {
    auto [baseline_hits, baseline_stats] = baseline.search_adaptive(query, 2);
    auto [low_hits, low_stats] = low_fanout_planner.search_adaptive(query, 2);
    auto [high_hits, high_stats] = high_fanout_planner.search_adaptive(query, 2);

    assert(ids(low_hits) == ids(baseline_hits));
    assert(ids(high_hits) == ids(baseline_hits));
    assert(low_stats.result_count == baseline_stats.result_count);
    assert(high_stats.result_count == baseline_stats.result_count);

    assert(low_stats.planner_invoked_count == 1);
    assert(low_stats.planner_strategy_baseline_count == 1);
    assert(low_stats.planner_strategy_direct_qgram_count == 0);
    assert(low_stats.planner_strategy_router_count == 0);
    assert(low_stats.search_qgram_prefilter_enabled == false);
    assert(low_stats.router_hint_invoked_count == 0);
    assert(low_stats.local_router_invoked_count == 0);
    assert(low_stats.best_first_invoked_count == 0);
    assert(low_stats.safe_child_router_invoked_count == 0);
    assert(low_stats.safe_child_router_build_ms == 0.0);

    assert(high_stats.planner_invoked_count == 1);
    assert(high_stats.planner_strategy_direct_qgram_count == 0);
    assert(high_stats.planner_strategy_safe_child_router_count == 1);
    assert(high_stats.planner_disable_router_ordering);
    if (high_stats.safe_child_router_invoked_count > 0 &&
        high_stats.safe_child_router_fallback_count == 0) {
      assert(high_stats.router_hint_invoked_count == 0);
      assert(high_stats.local_router_invoked_count == 0);
      assert(high_stats.best_first_invoked_count == 0);
    }
  }

  std::cout << "query planner no-false-negative tests passed\n";
  return 0;
}
