#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <vector>

namespace {

std::vector<std::shared_ptr<navigamer::BioSequence>> build_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a", "ACGTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("b", "ACGTACGTACGA"),
      std::make_shared<navigamer::BioSequence>("c", "TTTTACGTACGT"),
      std::make_shared<navigamer::BioSequence>("d", "GGGGACGTACGT"),
      std::make_shared<navigamer::BioSequence>("e", "CCCCACGTACGT"),
  };
}

void assert_integer_ids_unique() {
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({12, 6, 2}));
  builder.build(build_sequences());

  assert(builder.num_world_nodes() > 0);
  assert(builder.num_sequences() == builder.unique_sequences.size());
  assert(builder.validate_integer_ids());

  std::vector<bool> seen_nodes(builder.num_world_nodes(), false);
  for (const auto& layer : builder.primary_layers()) {
    for (const auto& node : layer) {
      assert(node->integer_id < builder.num_world_nodes());
      assert(!seen_nodes[node->integer_id]);
      seen_nodes[node->integer_id] = true;
    }
  }
  for (bool seen : seen_nodes) assert(seen);

  std::vector<bool> seen_sequences(builder.num_sequences(), false);
  for (const auto& entry : builder.unique_sequences) {
    const auto& sequence = entry.second;
    assert(sequence->sequence_id < builder.num_sequences());
    assert(!seen_sequences[sequence->sequence_id]);
    seen_sequences[sequence->sequence_id] = true;
  }
  for (bool seen : seen_sequences) assert(seen);
}

void assert_epoch_visited_basic() {
  navigamer::SearchScratch scratch;
  scratch.begin_query(4);
  assert(scratch.current_epoch == 1);
  assert(scratch.visited_epoch.size() == 4);

  assert(scratch.mark_visited(2));
  assert(!scratch.mark_visited(2));
  assert(scratch.mark_visited(3));

  scratch.begin_query(4);
  assert(scratch.current_epoch == 2);
  assert(scratch.mark_visited(2));
  assert(!scratch.mark_visited(2));

  scratch.current_epoch = std::numeric_limits<uint32_t>::max();
  scratch.visited_epoch[1] = scratch.current_epoch;
  scratch.begin_query(4);
  assert(scratch.current_epoch == 1);
  assert(scratch.mark_visited(1));
  assert(!scratch.mark_visited(1));
}

}  // namespace

int main() {
  assert_integer_ids_unique();
  assert_epoch_visited_basic();
  std::cout << "epoch visited tests passed\n";
  return 0;
}
