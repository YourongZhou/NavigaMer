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

std::set<std::string> ids(
    const std::vector<std::shared_ptr<navigamer::BioSequence>>& hits) {
  std::set<std::string> out;
  for (const auto& hit : hits) out.insert(hit->id);
  return out;
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

  scratch.current_epoch = std::numeric_limits<uint32_t>::max();
  scratch.visited_epoch[1] = scratch.current_epoch;
  scratch.begin_query(4);
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

}  // namespace

int main() {
  assert_integer_ids_unique();
  assert_epoch_visited_basic();
  assert_epoch_search_matches_string_baseline();
  std::cout << "epoch visited tests passed\n";
  return 0;
}
