#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <cassert>
#include <cstdint>
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
      std::make_shared<navigamer::BioSequence>("d0", "CCCCACGTACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("d1", "CCCCACGTACGTACGTACGA"),
  };
}

void assert_view_equivalent_to_original() {
  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 10, 3}), build_config);
  builder.build(build_sequences());
  assert(builder.validate_integer_ids());
  assert(builder.validate_search_graph_view());

  const auto& view = builder.search_graph_view();
  assert(view.nodes.size() == builder.num_world_nodes());
  assert(view.leaves.size() == builder.num_sequences());
  assert(view.child_begin.size() == builder.num_world_nodes());
  assert(view.child_end.size() == builder.num_world_nodes());
  assert(view.leaf_begin.size() == builder.num_world_nodes());
  assert(view.leaf_end.size() == builder.num_world_nodes());

  for (const auto& layer : builder.primary_layers()) {
    for (const auto& node : layer) {
      const navigamer::NodeId node_id = node->integer_id;
      assert(view.nodes[node_id] == node);

      uint32_t child_begin = view.child_begin[node_id];
      uint32_t child_end = view.child_end[node_id];
      assert(child_end >= child_begin);
      assert(child_end - child_begin == node->child_nodes.size());
      for (size_t child_idx = 0; child_idx < node->child_nodes.size(); ++child_idx) {
        assert(view.child_ids[child_begin + child_idx] ==
               node->child_nodes[child_idx]->integer_id);
      }

      uint32_t leaf_begin = view.leaf_begin[node_id];
      uint32_t leaf_end = view.leaf_end[node_id];
      assert(leaf_end >= leaf_begin);
      assert(leaf_end - leaf_begin == node->child_leaves.size());
      for (size_t leaf_idx = 0; leaf_idx < node->child_leaves.size(); ++leaf_idx) {
        assert(view.leaf_ids[leaf_begin + leaf_idx] ==
               node->child_leaves[leaf_idx]->sequence_id);
      }

      assert(view.mbb_dim[node_id] == node->beacons.size());
      const uint32_t mbb_offset = view.mbb_begin[node_id];
      const size_t child_count = node->child_nodes.size();
      for (size_t child_idx = 0; child_idx < node->child_beacon_mbbs.size(); ++child_idx) {
        for (size_t dim = 0; dim < node->child_beacon_mbbs[child_idx].size(); ++dim) {
          const size_t flat = mbb_offset + dim * child_count + child_idx;
          assert(view.mbb_lo[flat] == node->child_beacon_mbbs[child_idx][dim].min_dist);
          assert(view.mbb_hi[flat] == node->child_beacon_mbbs[child_idx][dim].max_dist);
        }
      }

      assert(view.leaf_beacon_dim[node_id] == node->beacons.size());
      const uint32_t leaf_beacon_offset = view.leaf_beacon_begin[node_id];
      const size_t leaf_count = node->child_leaves.size();
      for (size_t leaf_idx = 0; leaf_idx < node->leaf_beacon_dists.size(); ++leaf_idx) {
        for (size_t dim = 0; dim < node->leaf_beacon_dists[leaf_idx].size(); ++dim) {
          const size_t flat = leaf_beacon_offset + dim * leaf_count + leaf_idx;
          assert(view.leaf_beacon_dists[flat] ==
                 node->leaf_beacon_dists[leaf_idx][dim]);
        }
      }
    }
  }
}

std::set<std::string> ids(
    const std::vector<std::shared_ptr<navigamer::BioSequence>>& hits) {
  std::set<std::string> out;
  for (const auto& hit : hits) out.insert(hit->id);
  return out;
}

void assert_flat_search_matches_original() {
  assert(navigamer::parse_graph_view_mode("original") ==
         navigamer::GraphViewMode::Original);
  assert(navigamer::parse_graph_view_mode("flat") ==
         navigamer::GraphViewMode::Flat);
  assert(std::string(navigamer::graph_view_mode_name(
             navigamer::GraphViewMode::Original)) == "original");
  assert(std::string(navigamer::graph_view_mode_name(
             navigamer::GraphViewMode::Flat)) == "flat");

  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 10, 3}), build_config);
  builder.build(build_sequences());

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTACGTACGTACAA"),
      navigamer::BioSequence("q2", "TTTTACGTACGTACGTACGT"),
      navigamer::BioSequence("q3", "AAAAGGGGAAAAGGGGAAAA"),
  };

  for (navigamer::MBBFilterMode mbb_mode :
       {navigamer::MBBFilterMode::Scan, navigamer::MBBFilterMode::RectIndex}) {
    for (bool qgram_enabled : {false, true}) {
      for (navigamer::VisitedMode visited_mode :
           {navigamer::VisitedMode::StringSet, navigamer::VisitedMode::Epoch}) {
        navigamer::SearchConfig original_config;
        original_config.mbb_filter_mode = mbb_mode;
        original_config.search_qgram_prefilter = qgram_enabled;
        original_config.search_qgram_q = 3;
        original_config.visited_mode = visited_mode;
        original_config.graph_view_mode = navigamer::GraphViewMode::Original;

        navigamer::SearchConfig flat_config = original_config;
        flat_config.graph_view_mode = navigamer::GraphViewMode::Flat;

        navigamer::BioGeometrySearchEngine original_engine(builder, original_config);
        navigamer::BioGeometrySearchEngine flat_engine(builder, flat_config);

        for (const auto& query : queries) {
          auto [original_hits, original_stats] =
              original_engine.search_adaptive(query, 2);
          auto [flat_hits, flat_stats] = flat_engine.search_adaptive(query, 2);
          assert(ids(original_hits) == ids(flat_hits));
          assert(original_stats.result_count == flat_stats.result_count);
        }
      }
    }
  }
}

}  // namespace

int main() {
  assert_view_equivalent_to_original();
  assert_flat_search_matches_original();
  std::cout << "search graph view tests passed\n";
  return 0;
}
