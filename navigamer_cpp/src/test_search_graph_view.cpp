#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
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

}  // namespace

int main() {
  assert_view_equivalent_to_original();
  std::cout << "search graph view tests passed\n";
  return 0;
}

