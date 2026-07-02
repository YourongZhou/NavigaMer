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

std::vector<std::shared_ptr<navigamer::BioSequence>> build_anchor_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a", "AAAAAA"),
      std::make_shared<navigamer::BioSequence>("b", "AAAATA"),
      std::make_shared<navigamer::BioSequence>("c", "AAATTA"),
      std::make_shared<navigamer::BioSequence>("d", "TTTTTT"),
  };
}

std::vector<std::shared_ptr<navigamer::BioSequence>>
build_wide_fanout_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a0", "ACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("a1", "ACGTACGTTCGT"),
      std::make_shared<navigamer::BioSequence>("a2", "ACGTTCGTACGT"),
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
  {
    navigamer::BioGeometryIndexBuilder builder(
        navigamer::HierarchyConfig({20, 12, 6}));
    builder.build(build_anchor_sequences());

    navigamer::SearchConfig baseline_config;
    baseline_config.graph_view_mode = navigamer::GraphViewMode::Flat;
    baseline_config.visited_mode = navigamer::VisitedMode::Epoch;
    baseline_config.query_profile = true;
    navigamer::BioGeometrySearchEngine baseline(builder, baseline_config);

    navigamer::SearchConfig reuse_config = baseline_config;
    reuse_config.path_reuse_enabled = true;
    navigamer::BioGeometrySearchEngine reused(builder, reuse_config);

    const std::vector<navigamer::BioSequence> queries = {
        navigamer::BioSequence("q0", "AAAATA"),
        navigamer::BioSequence("q1", "AAAATA"),
    };

    size_t attempts = 0;
    size_t hits = 0;
    size_t productive_reuse_hits = 0;
    for (const auto& query : queries) {
      auto [baseline_hits, baseline_stats] = baseline.search_adaptive(query, 2);
      auto [reuse_hits, reuse_stats] = reused.search_adaptive(query, 2);
      assert(ids(reuse_hits) == ids(baseline_hits));
      assert(reuse_stats.result_count == baseline_stats.result_count);
      attempts += reuse_stats.path_reuse_attempt_count;
      hits += reuse_stats.path_reuse_hit_count;
      productive_reuse_hits += reuse_stats.productive_world_reuse_hit_count;
    }

    assert(attempts > 0);
    assert(hits > 0);
    assert(productive_reuse_hits > 0);
  }

  {
    navigamer::BuildRangeConfig build_config;
    build_config.min_rect_index_fanout = 1;
    navigamer::BioGeometryIndexBuilder builder(
        navigamer::HierarchyConfig({12, 6, 2}), build_config);
    builder.build(build_wide_fanout_sequences());

    navigamer::SearchConfig baseline_config;
    baseline_config.graph_view_mode = navigamer::GraphViewMode::Flat;
    baseline_config.visited_mode = navigamer::VisitedMode::Epoch;
    baseline_config.query_profile = true;
    navigamer::BioGeometrySearchEngine baseline(builder, baseline_config);

    navigamer::SearchConfig reuse_config = baseline_config;
    reuse_config.path_reuse_enabled = true;
    navigamer::BioGeometrySearchEngine reused(builder, reuse_config);

    const std::vector<navigamer::BioSequence> queries = {
        navigamer::BioSequence("q0", "ACGTACGTACGT"),
        navigamer::BioSequence("q1", "ACGTACGTACGA"),
        navigamer::BioSequence("q2", "ACGTACGTTCGT"),
        navigamer::BioSequence("q3", "ACGTACGTTCGA"),
    };

    size_t shortlist_hits = 0;
    for (const auto& query : queries) {
      auto [baseline_hits, baseline_stats] = baseline.search_adaptive(query, 8);
      auto [reuse_hits, reuse_stats] = reused.search_adaptive(query, 8);
      assert(ids(reuse_hits) == ids(baseline_hits));
      assert(reuse_stats.result_count == baseline_stats.result_count);
      shortlist_hits += reuse_stats.child_shortlist_reuse_hit_count;
    }

    assert(shortlist_hits > 0);
  }

  std::cout << "path reuse no-false-negative smoke passed\n";
  return 0;
}
