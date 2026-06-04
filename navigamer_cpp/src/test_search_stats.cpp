#include "experiment_utils.hpp"
#include "index_builder.hpp"
#include "search_engine.hpp"
#include "structure.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace {

std::vector<std::shared_ptr<navigamer::BioSequence>> build_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("a", "AAAAAA"),
      std::make_shared<navigamer::BioSequence>("b", "AAAATA"),
      std::make_shared<navigamer::BioSequence>("c", "AAATTA"),
      std::make_shared<navigamer::BioSequence>("d", "TTTTTT"),
  };
}

}  // namespace

int main() {
  {
    auto radii = navigamer::generate_geometric_radius_schedule(4, 8, 0.5);
    assert((radii == std::vector<int>{64, 32, 16, 8}));
  }

  {
    auto radii = navigamer::generate_geometric_radius_schedule(4, 8, 0.7);
    assert((radii == std::vector<int>{23, 16, 11, 8}));
    assert(navigamer::join_radius_schedule(radii) == "23|16|11|8");
  }

  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 12, 6}));
  builder.build(build_sequences());
  navigamer::BioGeometrySearchEngine engine(builder);
  navigamer::BioSequence query("q", "AAAATA");

  auto [results, stats] = engine.search_adaptive(query, 2);
  assert(!results.empty());
  assert(stats.world_access_count > 0);
  assert(stats.node_access_count > 0);
  assert(stats.edge_access_count > 0);
  assert(stats.anchor_distance_count > 0);
  assert(stats.bound_check_count > 0);
  assert(stats.candidate_count >= stats.candidate_verify_count);
  assert(stats.candidate_verify_count == stats.leaf_verify_count);

  std::cout << "ALL PASSED\n";
  return 0;
}
