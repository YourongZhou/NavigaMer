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
      std::make_shared<navigamer::BioSequence>("a3", "ACGTACGTAGGT"),
      std::make_shared<navigamer::BioSequence>("c0", "CCCCGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("c1", "CCCCGGGACCCC"),
      std::make_shared<navigamer::BioSequence>("t0", "TTTTAAAATTTT"),
      std::make_shared<navigamer::BioSequence>("t1", "TTTTAAAAGTTT"),
      std::make_shared<navigamer::BioSequence>("g0", "GGGGCCCCGGGG"),
      std::make_shared<navigamer::BioSequence>("g1", "GGGGCCCCAGGG"),
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

  navigamer::SearchConfig best_first_config = baseline_config;
  best_first_config.best_first_enabled = true;
  navigamer::BioGeometrySearchEngine best_first(builder, best_first_config);

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTTCGT"),
      navigamer::BioSequence("q2", "ACGTTCGTACGT"),
      navigamer::BioSequence("q3", "CCCCGGGGCCCC"),
      navigamer::BioSequence("q4", "TTTTAAAATTTT"),
      navigamer::BioSequence("q5", "GGGGCCCCGGGG"),
  };

  size_t invoked = 0;
  size_t bounded = 0;
  for (const auto& query : queries) {
    auto [baseline_hits, baseline_stats] = baseline.search_adaptive(query, 2);
    auto [best_first_hits, best_first_stats] =
        best_first.search_adaptive(query, 2);
    assert(ids(best_first_hits) == ids(baseline_hits));
    assert(best_first_stats.result_count == baseline_stats.result_count);
    assert(best_first_stats.best_first_enabled_count > 0);
    assert(best_first_stats.best_first_invoked_count >=
           best_first_stats.best_first_enabled_count);
    invoked += best_first_stats.best_first_invoked_count;
    bounded += best_first_stats.best_first_bound_candidate_count;
  }

  assert(invoked > 0);
  assert(bounded > 0);
  std::cout << "best-first no-false-negative smoke passed\n";
  return 0;
}
