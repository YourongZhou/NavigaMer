#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"

#include <cassert>
#include <iostream>
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

}  // namespace

int main() {
  assert_integer_ids_unique();
  std::cout << "epoch visited tests passed\n";
  return 0;
}
