#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"
#include "tools.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

std::vector<std::shared_ptr<navigamer::BioSequence>> build_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a0", "ACGTACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("a1", "ACGTACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("a2", "ACGTACGTACGTACGTACAA"),
      std::make_shared<navigamer::BioSequence>("b0", "TTTTACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("b1", "TTTTACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("c0", "GGGGACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("c1", "GGGGACGTACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("n0", "ACGTNACGTACGTACGTACG"),
      std::make_shared<navigamer::BioSequence>("r0", "AAAAAAAAAAAAAAAAAAAA"),
      std::make_shared<navigamer::BioSequence>("r1", "AAAAAAAAAAAAAAAAAAAT"),
  };
}

std::set<std::string> ids(
    const navigamer::SearchResult& hits) {
  std::set<std::string> out;
  for (const auto& hit : hits) out.insert(hit->id);
  return out;
}

void assert_mode_equivalence(navigamer::GraphViewMode graph_mode,
                             navigamer::VisitedMode visited_mode,
                             bool qgram_enabled) {
  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 10, 3}), build_config);
  builder.build(build_sequences());

  navigamer::SearchConfig dp_config;
  dp_config.mbb_filter_mode = navigamer::MBBFilterMode::Scan;
  dp_config.visited_mode = visited_mode;
  dp_config.graph_view_mode = graph_mode;
  dp_config.simd_mode = navigamer::SimdMode::Scalar;
  dp_config.search_qgram_prefilter = qgram_enabled;
  dp_config.search_qgram_q = 3;
  dp_config.distance_mode = navigamer::DistanceMode::DP;

  navigamer::SearchConfig myers_config = dp_config;
  myers_config.distance_mode = navigamer::DistanceMode::Myers;
  navigamer::SearchConfig edlib_config = dp_config;
  edlib_config.distance_mode = navigamer::DistanceMode::Edlib;

  navigamer::BioGeometrySearchEngine dp_engine(builder, dp_config);
  navigamer::BioGeometrySearchEngine myers_engine(builder, myers_config);
  navigamer::BioGeometrySearchEngine edlib_engine(builder, edlib_config);

  const std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTACGTACGTACAA"),
      navigamer::BioSequence("q2", "TTTTACGTACGTACGTACGC"),
      navigamer::BioSequence("q3", "ACGTNACGTACGTACGTACG"),
      navigamer::BioSequence("q4", "AAAAAAAAAAAAAAAAAAAA"),
      navigamer::BioSequence("q5", "CCCCCCCCCCCCCCCCCCCC"),
      navigamer::BioSequence("q6", "ACGTACGTACGTACGTACGTA"),
      navigamer::BioSequence("q7", "ACGTACGTACGTACGTACG"),
  };

  for (int tolerance : {0, 1, 2, 5}) {
    for (const auto& query : queries) {
      auto [dp_hits, dp_stats] = dp_engine.search_adaptive(query, tolerance);
      auto [myers_hits, myers_stats] =
          myers_engine.search_adaptive(query, tolerance);
      auto [edlib_hits, edlib_stats] =
          edlib_engine.search_adaptive(query, tolerance);
      assert(ids(dp_hits) == ids(myers_hits));
      assert(ids(dp_hits) == ids(edlib_hits));
      assert(dp_stats.result_count == myers_stats.result_count);
      assert(dp_stats.result_count == edlib_stats.result_count);
    }
  }
}

}  // namespace

int main() {
  assert(navigamer::SearchConfig{}.distance_mode == navigamer::DistanceMode::Myers);

  for (navigamer::GraphViewMode graph_mode :
       {navigamer::GraphViewMode::Original, navigamer::GraphViewMode::Flat}) {
    for (navigamer::VisitedMode visited_mode :
         {navigamer::VisitedMode::StringSet, navigamer::VisitedMode::Epoch}) {
      for (bool qgram_enabled : {false, true}) {
        assert_mode_equivalence(graph_mode, visited_mode, qgram_enabled);
      }
    }
  }

  std::cout << "search distance mode tests passed\n";
  return 0;
}
