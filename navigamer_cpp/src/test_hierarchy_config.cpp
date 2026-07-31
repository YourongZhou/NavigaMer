#include "index_builder.hpp"
#include "structure.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using navigamer::BioGeometryIndexBuilder;
using navigamer::BioSequence;
using navigamer::HierarchyConfig;

std::vector<std::shared_ptr<BioSequence>> toy_sequences() {
  return {
      std::make_shared<BioSequence>("s0", "AAAA"),
      std::make_shared<BioSequence>("s1", "AAAT"),
      std::make_shared<BioSequence>("s2", "AATT"),
      std::make_shared<BioSequence>("s3", "TTTT"),
  };
}

void expect_invalid_config(const std::vector<int>& primary_radii) {
  bool threw = false;
  try {
    BioGeometryIndexBuilder builder{HierarchyConfig(primary_radii)};
    builder.build(toy_sequences());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw && "expected invalid hierarchy config to throw");
}

void expect_long_sequence_rejected() {
  bool threw = false;
  try {
    BioGeometryIndexBuilder builder{HierarchyConfig({20, 8})};
    builder.build({
        std::make_shared<BioSequence>(
            "too_long", std::string(256, 'A')),
    });
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw && "expected a sequence longer than 255 to throw");

  threw = false;
  try {
    BioGeometryIndexBuilder builder{HierarchyConfig({20, 8})};
    builder.build_reference_windows(
        "ref", std::string(300, 'A'), 256, 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw && "expected a reference window longer than 255 to throw");
}

void validate_primary_layout(const std::vector<int>& primary_radii) {
  BioGeometryIndexBuilder builder{HierarchyConfig(primary_radii)};
  builder.build(toy_sequences());

  assert(builder.num_primary_layers() == static_cast<int>(primary_radii.size()));
  assert(builder.num_expanded_layers() == static_cast<int>(primary_radii.size() * 2 - 1));
  assert(builder.coarsest_primary_layer_index() == 0);
  assert(builder.finest_primary_layer_index() == static_cast<int>(primary_radii.size() - 1));

  for (int i = 0; i < builder.num_primary_layers(); ++i) {
    const auto& view = builder.search_graph_view();
    const size_t layer = static_cast<size_t>(i);
    if (i == builder.finest_primary_layer_index()) {
      for (uint32_t node_id = view.layer_begin[layer];
           node_id < view.layer_end[layer]; ++node_id) {
        const auto& node = view.node_records[node_id];
        assert(node.child_count == 0);
        assert(node.leaf_count > 0);
      }
    } else {
      bool saw_child = false;
      for (uint32_t node_id = view.layer_begin[layer];
           node_id < view.layer_end[layer]; ++node_id) {
        const auto& node = view.node_records[node_id];
        assert(node.beacon_count > 0);
        if (node.child_count > 0) saw_child = true;
      }
      assert(saw_child && "expected at least one folded child edge in non-finest layer");
    }
  }
}

}  // namespace

int main() {
  std::cerr << "=== NavigaMer HierarchyConfig Test ===\n";

  expect_invalid_config({});
  expect_invalid_config({10});
  expect_invalid_config({10, 10});
  expect_invalid_config({8, 12});
  expect_invalid_config({65536, 1});
  expect_long_sequence_rejected();

  validate_primary_layout({20, 8});
  validate_primary_layout({30, 15, 5});
  validate_primary_layout({40, 28, 18, 10});
  validate_primary_layout({50, 35, 24, 16, 8});

  std::cout << "ALL PASSED\n";
  return 0;
}
