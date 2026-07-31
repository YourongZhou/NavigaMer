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

std::vector<std::shared_ptr<navigamer::BioSequence>> build_clustered_sequences() {
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

std::vector<std::shared_ptr<navigamer::BioSequence>> build_diverse_sequences() {
  return {
      std::make_shared<navigamer::BioSequence>("d0", "AAAAAAAAAAAAAAAAAAAA"),
      std::make_shared<navigamer::BioSequence>("d1", "CCCCCCCCCCCCCCCCCCCC"),
      std::make_shared<navigamer::BioSequence>("d2", "GGGGGGGGGGGGGGGGGGGG"),
      std::make_shared<navigamer::BioSequence>("d3", "TTTTTTTTTTTTTTTTTTTT"),
      std::make_shared<navigamer::BioSequence>("d4", "ACACACACACACACACACAC"),
      std::make_shared<navigamer::BioSequence>("d5", "AGAGAGAGAGAGAGAGAGAG"),
      std::make_shared<navigamer::BioSequence>("d6", "ATATATATATATATATATAT"),
      std::make_shared<navigamer::BioSequence>("d7", "CGCGCGCGCGCGCGCGCGCG"),
      std::make_shared<navigamer::BioSequence>("d8", "CTCTCTCTCTCTCTCTCTCT"),
      std::make_shared<navigamer::BioSequence>("d9", "GTGTGTGTGTGTGTGTGTGT"),
  };
}

std::set<navigamer::LeafId> ids(
    const navigamer::SearchResult& hits) {
  return {hits.begin(), hits.end()};
}

void assert_integer_ids_unique() {
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({12, 6, 2}));
  builder.build(build_sequences());

  assert(builder.num_world_nodes() > 0);
  assert(builder.num_sequences() == builder.sequence_store().size());
  assert(builder.validate_integer_ids());

  std::vector<bool> seen_nodes(builder.num_world_nodes(), false);
  for (navigamer::NodeId node_id = 0;
       node_id < builder.search_graph_view().node_records.size(); ++node_id) {
    assert(!seen_nodes[node_id]);
    seen_nodes[node_id] = true;
  }
  for (bool seen : seen_nodes) assert(seen);

  std::vector<bool> seen_sequences(builder.num_sequences(), false);
  for (const auto& sequence : builder.sequence_store().records) {
    assert(sequence.sequence_id < builder.num_sequences());
    assert(!seen_sequences[sequence.sequence_id]);
    seen_sequences[sequence.sequence_id] = true;
  }
  for (bool seen : seen_sequences) assert(seen);
}

void assert_epoch_visited_basic() {
  assert(navigamer::parse_visited_mode("string") ==
         navigamer::VisitedMode::StringSet);
  assert(navigamer::parse_visited_mode("epoch") ==
         navigamer::VisitedMode::Epoch);
  assert(std::string(navigamer::visited_mode_name(
             navigamer::VisitedMode::StringSet)) == "string");
  assert(std::string(navigamer::visited_mode_name(
             navigamer::VisitedMode::Epoch)) == "epoch");

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

  scratch.begin_query(8);
  assert(scratch.current_epoch == 1);
  assert(scratch.visited_epoch.size() == 8);
  assert(scratch.mark_visited(7));

  scratch.begin_query(3);
  assert(scratch.current_epoch == 2);
  assert(scratch.visited_epoch.size() == 8);
  assert(scratch.mark_visited(2));
  assert(!scratch.mark_visited(2));

  scratch.current_epoch = std::numeric_limits<uint32_t>::max();
  scratch.visited_epoch[1] = scratch.current_epoch;
  scratch.begin_query(3);
  assert(scratch.current_epoch == 1);
  assert(scratch.mark_visited(1));
  assert(!scratch.mark_visited(1));
}

void assert_epoch_search_matches_string_baseline() {
  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({20, 10, 3}), build_config);
  builder.build(build_clustered_sequences());

  std::vector<navigamer::BioSequence> queries = {
      navigamer::BioSequence("q0", "ACGTACGTACGTACGTACGT"),
      navigamer::BioSequence("q1", "ACGTACGTACGTACGTACAA"),
      navigamer::BioSequence("q2", "TTTTACGTACGTACGTACGT"),
      navigamer::BioSequence("q3", "AAAAGGGGAAAAGGGGAAAA"),
  };

  for (navigamer::MBBFilterMode mbb_mode :
       {navigamer::MBBFilterMode::Scan, navigamer::MBBFilterMode::RectIndex}) {
    for (bool qgram_enabled : {false, true}) {
      navigamer::SearchConfig string_config;
      string_config.mbb_filter_mode = mbb_mode;
      string_config.search_qgram_prefilter = qgram_enabled;
      string_config.search_qgram_q = 3;
      string_config.visited_mode = navigamer::VisitedMode::StringSet;

      navigamer::SearchConfig epoch_config = string_config;
      epoch_config.visited_mode = navigamer::VisitedMode::Epoch;

      navigamer::BioGeometrySearchEngine string_engine(builder, string_config);
      navigamer::BioGeometrySearchEngine epoch_engine(builder, epoch_config);

      for (const auto& query : queries) {
        auto [string_hits, string_stats] = string_engine.search_adaptive(query, 2);
        auto [epoch_hits, epoch_stats] = epoch_engine.search_adaptive(query, 2);
        assert(ids(string_hits) == ids(epoch_hits));
        assert(string_stats.result_count == epoch_stats.result_count);
      }
    }
  }
}

void assert_epoch_search_survives_alternating_index_sizes() {
  navigamer::BioGeometryIndexBuilder small_builder(
      navigamer::HierarchyConfig({12, 6, 2}));
  small_builder.build(build_sequences());

  navigamer::BuildRangeConfig build_config;
  build_config.min_rect_index_fanout = 1;
  navigamer::BioGeometryIndexBuilder large_builder(
      navigamer::HierarchyConfig({20, 10, 3}), build_config);
  large_builder.build(build_diverse_sequences());
  assert(small_builder.num_world_nodes() != large_builder.num_world_nodes());

  navigamer::SearchConfig epoch_config;
  epoch_config.visited_mode = navigamer::VisitedMode::Epoch;
  navigamer::BioGeometrySearchEngine small_engine(small_builder, epoch_config);
  navigamer::BioGeometrySearchEngine large_engine(large_builder, epoch_config);

  const navigamer::BioSequence small_query("small", "ACGTACGTACGT");
  const navigamer::BioSequence large_query(
      "large", "ACGTACGTACGTACGTACGT");

  for (int iteration = 0; iteration < 3; ++iteration) {
    auto [small_hits, small_stats] =
        small_engine.search_adaptive(small_query, 1);
    auto [large_hits, large_stats] =
        large_engine.search_adaptive(large_query, 2);
    assert(ids(small_hits) == ids(small_engine.search_brute_force(
                                  small_query, 1).first));
    assert(ids(large_hits) == ids(large_engine.search_brute_force(
                                  large_query, 2).first));
    assert(small_stats.result_count == small_hits.size());
    assert(large_stats.result_count == large_hits.size());
  }
}

}  // namespace

int main() {
  assert_integer_ids_unique();
  assert_epoch_visited_basic();
  assert_epoch_search_matches_string_baseline();
  assert_epoch_search_survives_alternating_index_sizes();
  std::cout << "epoch visited tests passed\n";
  return 0;
}
