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

std::vector<std::shared_ptr<navigamer::BioSequence>> build_small_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a", "AAAAAA"),
      std::make_shared<navigamer::BioSequence>("b", "AAAATA"),
      std::make_shared<navigamer::BioSequence>("c", "AAATTA"),
      std::make_shared<navigamer::BioSequence>("d", "TTTTTT"),
  };
}

std::vector<std::shared_ptr<navigamer::BioSequence>> build_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a0", "ACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("a1", "ACGTACGTTCGT"),
      std::make_shared<navigamer::BioSequence>("a2", "ACGTTCGTACGT"),
      std::make_shared<navigamer::BioSequence>("a3", "ACGTACGTAGGT"),
      std::make_shared<navigamer::BioSequence>("c0", "CCCCGGGGCCCC"),
      std::make_shared<navigamer::BioSequence>("c1", "CCCCGGGACCCC"),
      std::make_shared<navigamer::BioSequence>("c2", "CCCCGCGGCCCC"),
      std::make_shared<navigamer::BioSequence>("t0", "TTTTAAAATTTT"),
      std::make_shared<navigamer::BioSequence>("t1", "TTTTAAAAGTTT"),
      std::make_shared<navigamer::BioSequence>("g0", "GGGGCCCCGGGG"),
      std::make_shared<navigamer::BioSequence>("g1", "GGGGCCCCAGGG"),
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
    navigamer::BioGeometryIndexBuilder small_builder(
        navigamer::HierarchyConfig({20, 12, 6}));
    small_builder.build(build_small_sequences());

    navigamer::SearchConfig small_baseline_config;
    small_baseline_config.graph_view_mode = navigamer::GraphViewMode::Flat;
    small_baseline_config.visited_mode = navigamer::VisitedMode::Epoch;
    navigamer::BioGeometrySearchEngine small_baseline(small_builder,
                                                      small_baseline_config);

    navigamer::SearchConfig small_hint_config = small_baseline_config;
    small_hint_config.router_hint_enabled = true;
    small_hint_config.router_hint_qgram_q = 3;
    small_hint_config.router_hint_minimizer_k = 3;
    small_hint_config.router_hint_minimizer_w = 5;
    navigamer::BioGeometrySearchEngine small_hinted(small_builder,
                                                    small_hint_config);

    navigamer::BioSequence small_query("small_q", "AAAATA");
    auto [small_baseline_hits, small_baseline_stats] =
        small_baseline.search_adaptive(small_query, 2);
    auto [small_hint_hits, small_hint_stats] =
        small_hinted.search_adaptive(small_query, 2);
    assert(ids(small_hint_hits) == ids(small_baseline_hits));
    assert(small_hint_stats.result_count == small_baseline_stats.result_count);
    assert(small_hint_stats.router_hint_invoked_count == 0);
    assert(small_hint_stats.router_qgram_signature_build_count == 0);
    assert(small_hint_stats.router_minimizer_signature_build_count == 0);
    assert(small_hint_stats.router_candidate_count == 0);
  }

  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({12, 6, 2}), build_config);
  builder.build(build_sequences());

  navigamer::SearchConfig baseline_config;
  baseline_config.graph_view_mode = navigamer::GraphViewMode::Flat;
  baseline_config.visited_mode = navigamer::VisitedMode::Epoch;
  baseline_config.query_profile = true;
  navigamer::BioGeometrySearchEngine baseline(builder, baseline_config);

  navigamer::SearchConfig hint_config = baseline_config;
  hint_config.router_hint_enabled = true;
  hint_config.router_hint_qgram_q = 3;
  hint_config.router_hint_minimizer_k = 3;
  hint_config.router_hint_minimizer_w = 5;
  navigamer::BioGeometrySearchEngine hinted(builder, hint_config);

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTTCGT"),
      navigamer::BioSequence("q2", "CCCCGGGGCCCC"),
      navigamer::BioSequence("q3", "TTTTAAAATTTT"),
      navigamer::BioSequence("q4", "GGGGCCCCGGGG"),
  };

  size_t invoked = 0;
  size_t predicted_hits = 0;
  for (const auto& query : queries) {
    auto [baseline_hits, baseline_stats] = baseline.search_adaptive(query, 2);
    auto [hint_hits, hint_stats] = hinted.search_adaptive(query, 2);
    assert(ids(hint_hits) == ids(baseline_hits));
    assert(hint_stats.result_count == baseline_stats.result_count);
    assert(hint_stats.router_candidate_count >=
           hint_stats.router_candidate_hit_count);
    invoked += hint_stats.router_hint_invoked_count;
    predicted_hits += hint_stats.router_candidate_hit_count;
  }

  {
    navigamer::BuildRangeConfig wide_build_config;
    wide_build_config.min_rect_index_fanout = 1;
    navigamer::BioGeometryIndexBuilder wide_builder(
        navigamer::HierarchyConfig({12, 6, 2}), wide_build_config);
    wide_builder.build(build_wide_fanout_sequences());

    navigamer::SearchConfig wide_baseline_config = baseline_config;
    navigamer::BioGeometrySearchEngine wide_baseline(wide_builder,
                                                     wide_baseline_config);

    navigamer::SearchConfig wide_hint_config = wide_baseline_config;
    wide_hint_config.router_hint_enabled = true;
    wide_hint_config.router_hint_qgram_q = 3;
    wide_hint_config.router_hint_minimizer_k = 3;
    wide_hint_config.router_hint_minimizer_w = 5;
    navigamer::BioGeometrySearchEngine wide_hinted(wide_builder,
                                                   wide_hint_config);

    navigamer::BioSequence wide_query("wide_q", "ACGTACGTACGT");
    auto [wide_baseline_hits, wide_baseline_stats] =
        wide_baseline.search_adaptive(wide_query, 8);
    auto [wide_hint_hits, wide_hint_stats] =
        wide_hinted.search_adaptive(wide_query, 8);
    assert(ids(wide_hint_hits) == ids(wide_baseline_hits));
    assert(wide_hint_stats.result_count == wide_baseline_stats.result_count);
    assert(wide_hint_stats.router_hint_invoked_count > 0);
    assert(wide_hint_stats.router_qgram_signature_build_count > 0);
    assert(wide_hint_stats.router_minimizer_signature_build_count > 0);
    assert(wide_hint_stats.router_qgram_ranked_count > 0);
    assert(wide_hint_stats.router_minimizer_ranked_count > 0);
    invoked += wide_hint_stats.router_hint_invoked_count;
    predicted_hits += wide_hint_stats.router_candidate_hit_count;
  }

  assert(invoked > 0);
  assert(predicted_hits > 0);
  std::cout << "router hints no-false-negative smoke passed\n";
  return 0;
}
