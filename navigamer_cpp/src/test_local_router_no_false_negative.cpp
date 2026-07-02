#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<std::shared_ptr<navigamer::BioSequence>> build_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a0", "ACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("a1", "ACGTACGTTCGT"),
      std::make_shared<navigamer::BioSequence>("a2", "ACGTTCGTACGT"),
      std::make_shared<navigamer::BioSequence>("c0", "CCCCGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("c1", "CCCCGGGACCCC"),
      std::make_shared<navigamer::BioSequence>("t0", "TTTTAAAATTTT"),
      std::make_shared<navigamer::BioSequence>("t1", "TTTTAAAAGTTT"),
      std::make_shared<navigamer::BioSequence>("g0", "GGGGCCCCGGGG"),
  };
}

std::vector<std::string> ids(
    const std::vector<std::shared_ptr<navigamer::BioSequence>>& hits) {
  std::vector<std::string> out;
  for (const auto& hit : hits) {
    if (hit) out.push_back(hit->id);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

int main() {
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({12, 6, 2}));
  builder.build(build_sequences());

  navigamer::SearchConfig baseline_config;
  baseline_config.graph_view_mode = navigamer::GraphViewMode::Flat;
  baseline_config.visited_mode = navigamer::VisitedMode::Epoch;
  baseline_config.query_profile = true;
  navigamer::BioGeometrySearchEngine baseline(builder, baseline_config);

  navigamer::SearchConfig router_config = baseline_config;
  router_config.local_router_enabled = true;
  router_config.local_router_max_anchors = 4;
  router_config.local_router_max_children = 2;
  router_config.local_router_score_mode = "anchor-envelope";
  navigamer::BioGeometrySearchEngine routed(builder, router_config);

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTTCGT"),
      navigamer::BioSequence("q2", "CCCCGGGGCCCC"),
      navigamer::BioSequence("q3", "TTTTAAAATTTT"),
      navigamer::BioSequence("q4", "GGGGCCCCGGGG"),
  };

  size_t invoked = 0;
  size_t shortlisted = 0;
  for (const auto& query : queries) {
    auto [baseline_hits, baseline_stats] = baseline.search_adaptive(query, 1);
    auto [router_hits, router_stats] = routed.search_adaptive(query, 1);
    assert(ids(router_hits) == ids(baseline_hits));
    assert(router_stats.result_count == baseline_stats.result_count);
    assert(router_stats.local_router_enabled_count > 0);
    assert(router_stats.local_router_invoked_count >=
           router_stats.local_router_enabled_count);
    invoked += router_stats.local_router_invoked_count;
    shortlisted += router_stats.local_router_shortlist_child_count;
  }

  assert(invoked > 0);
  assert(shortlisted > 0);
  std::cout << "local router no-false-negative smoke passed\n";
  return 0;
}
