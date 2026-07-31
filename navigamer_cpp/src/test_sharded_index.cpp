#include "sharded_index.hpp"
#include "search_engine.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {

using Occurrence =
    std::tuple<std::string, uint32_t, std::string>;

std::set<Occurrence> occurrences(
    const navigamer::SequenceStore& store) {
  std::set<Occurrence> result;
  for (navigamer::LeafId id = 0; id < store.size(); ++id) {
    const std::string sequence(store.sequence(id));
    store.for_each_occurrence(
        id, [&](uint32_t source_pos) {
          const auto& contig =
              store.contig_for_position(source_pos);
          const uint32_t original_pos =
              contig.source_begin + source_pos - contig.begin;
          result.emplace(contig.id, original_pos, sequence);
        });
  }
  return result;
}

std::set<std::string> matching_sequences(
    const navigamer::BioGeometryIndexBuilder& builder,
    const navigamer::BioSequence& query,
    int tolerance) {
  navigamer::BioGeometrySearchEngine engine(builder);
  auto [hits, stats] =
      engine.search_adaptive(query, tolerance);
  (void)stats;
  std::set<std::string> result;
  for (navigamer::LeafId hit : hits) {
    result.insert(
        std::string(builder.sequence_store().sequence(hit)));
  }
  return result;
}

std::set<std::string> matching_sequences(
    const std::vector<navigamer::LoadedIndex>& shards,
    const navigamer::BioSequence& query,
    int tolerance) {
  std::set<std::string> result;
  for (const auto& shard : shards) {
    const auto matches =
        matching_sequences(shard.builder, query, tolerance);
    result.insert(matches.begin(), matches.end());
  }
  return result;
}

void test_sharded_round_trip_and_no_false_negatives() {
  const std::string first =
      "ACGTACGTAAAACCCCACGTACGTNNNNGGGG";
  const std::string second =
      "TTTTACGTACGTCCCCAAAATTTT";
  const std::string reference = first + second;
  const std::vector<navigamer::ReferenceContig> contigs = {
      {"chrA", 0, static_cast<uint32_t>(first.size()), 0},
      {"chrB", static_cast<uint32_t>(first.size()),
       static_cast<uint32_t>(reference.size()), 0}};
  constexpr size_t window = 8;
  constexpr size_t stride = 1;
  constexpr size_t shard_windows = 4;

  navigamer::HierarchyConfig hierarchy({8, 4, 2});
  navigamer::BuildRangeConfig range_config;
  navigamer::BioGeometryIndexBuilder monolithic(
      hierarchy, range_config);
  monolithic.build_reference_windows(
      "reference", reference, window, stride, contigs);

  const auto directory =
      std::filesystem::temp_directory_path() /
      ("navigamer-sharded-test-" +
       std::to_string(static_cast<unsigned long long>(
           ::getpid())));
  std::filesystem::create_directories(directory);
  const auto bundle = directory / "reference.navshard";
  const auto manifest =
      navigamer::build_sharded_reference_index(
          bundle.string(), "literal-reference", "reference",
          reference, contigs, window, stride, shard_windows,
          hierarchy, range_config, 4);

  assert(navigamer::is_sharded_index(bundle.string()));
  const auto reloaded_manifest =
      navigamer::read_sharded_index_manifest(bundle.string());
  assert(reloaded_manifest.window_length == window);
  assert(reloaded_manifest.stride == stride);
  assert(reloaded_manifest.shards.size() ==
         manifest.shards.size());
  assert(reloaded_manifest.shards.size() > 2);
  for (const auto& shard : reloaded_manifest.shards) {
    assert(shard.window_count <= shard_windows);
    assert(std::filesystem::exists(
        navigamer::resolve_index_shard_path(
            bundle.string(), shard.path)));
  }

  std::vector<std::filesystem::file_time_type> shard_times;
  shard_times.reserve(reloaded_manifest.shards.size());
  for (const auto& shard : reloaded_manifest.shards) {
    shard_times.push_back(std::filesystem::last_write_time(
        navigamer::resolve_index_shard_path(
            bundle.string(), shard.path)));
  }
  const auto resumed =
      navigamer::build_sharded_reference_index(
          bundle.string(), "literal-reference", "reference",
          reference, contigs, window, stride, shard_windows,
          hierarchy, range_config, 4);
  assert(resumed.shards.size() == reloaded_manifest.shards.size());
  for (size_t shard_idx = 0;
       shard_idx < resumed.shards.size(); ++shard_idx) {
    assert(std::filesystem::last_write_time(
               navigamer::resolve_index_shard_path(
                   bundle.string(),
                   resumed.shards[shard_idx].path)) ==
           shard_times[shard_idx]);
  }

  const std::string damaged_path =
      navigamer::resolve_index_shard_path(
          bundle.string(), resumed.shards.front().path);
  {
    std::ofstream damaged(damaged_path, std::ios::binary);
    damaged << "damaged";
  }
  const auto repaired =
      navigamer::build_sharded_reference_index(
          bundle.string(), "literal-reference", "reference",
          reference, contigs, window, stride, shard_windows,
          hierarchy, range_config, 4);
  assert(repaired.shards.size() == resumed.shards.size());

  {
    std::fstream damaged_manifest(
        bundle, std::ios::in | std::ios::out | std::ios::binary);
    damaged_manifest.seekg(-1, std::ios::end);
    char byte = 0;
    damaged_manifest.get(byte);
    damaged_manifest.seekp(-1, std::ios::end);
    damaged_manifest.put(static_cast<char>(byte ^ 1));
  }
  bool checksum_rejected = false;
  try {
    (void)navigamer::read_sharded_index_manifest(
        bundle.string());
  } catch (const std::runtime_error&) {
    checksum_rejected = true;
  }
  assert(checksum_rejected);
  (void)navigamer::build_sharded_reference_index(
      bundle.string(), "literal-reference", "reference",
      reference, contigs, window, stride, shard_windows,
      hierarchy, range_config, 4);

  const auto loaded = navigamer::load_sharded_index(
      bundle.string(),
      navigamer::read_sharded_index_manifest(bundle.string()));
  std::set<Occurrence> sharded_occurrences;
  for (const auto& shard : loaded) {
    const auto part =
        occurrences(shard.builder.sequence_store());
    sharded_occurrences.insert(part.begin(), part.end());
  }
  assert(sharded_occurrences ==
         occurrences(monolithic.sequence_store()));

  size_t query_ordinal = 0;
  for (const auto& occurrence : sharded_occurrences) {
    const std::string& sequence = std::get<2>(occurrence);
    navigamer::BioSequence exact(
        "exact_" + std::to_string(query_ordinal), sequence);
    assert(matching_sequences(monolithic, exact, 0) ==
           matching_sequences(loaded, exact, 0));

    std::string mutated = sequence;
    mutated[query_ordinal % mutated.size()] =
        mutated[query_ordinal % mutated.size()] == 'A'
            ? 'C'
            : 'A';
    navigamer::BioSequence one_edit(
        "mutated_" + std::to_string(query_ordinal), mutated);
    assert(matching_sequences(monolithic, one_edit, 1) ==
           matching_sequences(loaded, one_edit, 1));
    ++query_ordinal;
  }

  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  test_sharded_round_trip_and_no_false_negatives();
  std::cout << "sharded index tests passed\n";
  return 0;
}
