#include "index_builder.hpp"
#include "search_engine.hpp"

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
      std::make_shared<navigamer::BioSequence>("n0", "ACGTNACGTACGTACGTACG"),
      std::make_shared<navigamer::BioSequence>("r0", "AAAAAAAAAAAAAAAAAAAA"),
      std::make_shared<navigamer::BioSequence>("r1", "AAAAAAAAAAAAAAAAAAAT"),
  };
}

std::set<navigamer::LeafId> ids(
    const navigamer::SearchResult& hits) {
  return {hits.begin(), hits.end()};
}

std::vector<std::set<std::string>> edge_signature(
    const navigamer::BioGeometryIndexBuilder& builder) {
  std::vector<std::set<std::string>> out;
  const auto& view = builder.search_graph_view();
  for (navigamer::NodeId node_id = 0;
       node_id < view.node_records.size(); ++node_id) {
    const auto& node = view.node_records[node_id];
    std::set<std::string> row;
    if (node_id >= view.layer_begin.back()) {
      for (uint32_t offset = 0; offset < node.leaf_count(); ++offset) {
        row.insert(
            "leaf:" +
            view.sequences[
                view.leaf_ids[node.leaf_begin() + offset]].id);
      }
    } else {
      for (uint32_t offset = 0; offset < node.child_count(); ++offset) {
        const auto& child =
            view.node_records[
                view.child_ids[node.child_begin() + offset]];
        row.insert(view.sequences[child.center_sequence_id].id);
      }
    }
    out.push_back(std::move(row));
  }
  return out;
}

}  // namespace

int main() {
  assert(navigamer::build_distance_mode_name(
             navigamer::BuildDistanceMode::DP) == std::string("dp"));
  assert(navigamer::build_distance_mode_name(
             navigamer::BuildDistanceMode::Edlib) == std::string("edlib"));
  assert(navigamer::build_distance_mode_name(
             navigamer::BuildDistanceMode::Auto) == std::string("auto"));
  assert(navigamer::parse_build_distance_mode("dp") ==
         navigamer::BuildDistanceMode::DP);
  assert(navigamer::parse_build_distance_mode("edlib") ==
         navigamer::BuildDistanceMode::Edlib);
  assert(navigamer::parse_build_distance_mode("auto") ==
         navigamer::BuildDistanceMode::Auto);
  assert(navigamer::BuildRangeConfig{}.distance_mode ==
         navigamer::BuildDistanceMode::Edlib);

  navigamer::BuildRangeConfig dp_config;
  dp_config.min_rect_index_fanout = 1;
  dp_config.distance_mode = navigamer::BuildDistanceMode::DP;
  navigamer::BuildRangeConfig edlib_config = dp_config;
  edlib_config.distance_mode = navigamer::BuildDistanceMode::Edlib;

  navigamer::BioGeometryIndexBuilder dp_builder(
      navigamer::HierarchyConfig({20, 10, 3}), dp_config);
  navigamer::BioGeometryIndexBuilder edlib_builder(
      navigamer::HierarchyConfig({20, 10, 3}), edlib_config);
  const auto sequences = build_sequences();
  dp_builder.build(sequences);
  edlib_builder.build(sequences);

  assert(dp_builder.validate_integer_ids());
  assert(edlib_builder.validate_integer_ids());
  assert(dp_builder.validate_search_graph_view());
  assert(edlib_builder.validate_search_graph_view());
  assert(dp_builder.num_world_nodes() == edlib_builder.num_world_nodes());
  assert(dp_builder.num_sequences() == edlib_builder.num_sequences());
  assert(edge_signature(dp_builder) == edge_signature(edlib_builder));

  navigamer::SearchConfig search_config;
  search_config.distance_mode = navigamer::DistanceMode::DP;
  navigamer::BioGeometrySearchEngine dp_engine(dp_builder, search_config);
  navigamer::BioGeometrySearchEngine edlib_engine(edlib_builder, search_config);
  for (int tolerance : {0, 1, 2, 5}) {
    for (const auto& query : {
             navigamer::BioSequence("q0", "ACGTACGTACGTACGTACGT"),
             navigamer::BioSequence("q1", "ACGTNACGTACGTACGTACG"),
             navigamer::BioSequence("q2", "CCCCCCCCCCCCCCCCCCCC"),
         }) {
      auto [dp_hits, dp_stats] = dp_engine.search_adaptive(query, tolerance);
      auto [edlib_hits, edlib_stats] =
          edlib_engine.search_adaptive(query, tolerance);
      assert(ids(dp_hits) == ids(edlib_hits));
      assert(dp_stats.result_count == edlib_stats.result_count);
    }
  }

  std::cout << "build distance mode tests passed\n";
  return 0;
}
