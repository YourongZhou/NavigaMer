#include "sharded_index.hpp"
#include "io_utils.hpp"
#include "search_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
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

bool verify_implicit_dense_leaf_layout(
    const std::vector<navigamer::LoadedIndex>& shards) {
  bool saw_layout = false;
  for (const auto& shard : shards) {
    const auto& view = shard.builder.search_graph_view();
    if (!view.node_records.leaf_layout()
             .has_implicit_dense_leaf_fields()) {
      continue;
    }
    saw_layout = true;
    assert(view.dense_leaf_mbb_ternary);
    assert(view.implicit_consecutive_leaf_ids);
    assert(view.implicit_shift_leaf_mbb);
    assert(view.leaf_beacon_dists.empty());
    assert(view.node_records.leaf_layout().record_bytes == 0);
    for (navigamer::NodeId node_id =
             view.node_records.finest_node_begin();
         node_id < view.node_records.size(); ++node_id) {
      const auto node = view.node_records[node_id];
      const navigamer::LeafId center =
          view.center_sequence_id(node_id);
      const uint32_t leaf_count =
          view.link_count(node_id, node, center);
      assert(leaf_count == view.implicit_consecutive_leaf_count(center));
      const navigamer::LeafId leaf_begin =
          center > view.implicit_consecutive_leaf_radius
              ? center - view.implicit_consecutive_leaf_radius
              : 0;
      for (uint32_t leaf_offset = 0; leaf_offset < leaf_count;
           ++leaf_offset) {
        const navigamer::LeafId leaf_id = leaf_begin + leaf_offset;
        const uint32_t delta =
            leaf_id > center ? leaf_id - center : center - leaf_id;
        assert(view.leaf_beacon_distance(node_id, leaf_offset) ==
               delta * 2);
      }
    }
  }
  return saw_layout;
}

std::set<Occurrence> matching_occurrences(
    const std::vector<navigamer::LoadedIndex>& shards,
    const navigamer::BioSequence& query,
    int tolerance,
    const std::vector<uint32_t>* selected_shards = nullptr) {
  std::set<Occurrence> result;
  for (size_t shard_idx = 0; shard_idx < shards.size(); ++shard_idx) {
    if (selected_shards &&
        !std::binary_search(
            selected_shards->begin(), selected_shards->end(),
            static_cast<uint32_t>(shard_idx))) {
      continue;
    }
    const auto& builder = shards[shard_idx].builder;
    navigamer::BioGeometrySearchEngine engine(builder);
    const auto [hits, stats] =
        engine.search_brute_force(query, tolerance);
    (void)stats;
    const auto& store = builder.sequence_store();
    for (navigamer::LeafId hit : hits) {
      const std::string sequence(store.sequence(hit));
      store.for_each_occurrence(
          hit, [&](uint32_t source_pos) {
            const auto& contig =
                store.contig_for_position(source_pos);
            const uint32_t original_pos =
                contig.source_begin + source_pos - contig.begin;
            result.emplace(contig.id, original_pos, sequence);
          });
    }
  }
  return result;
}

std::string deterministic_dna(size_t size) {
  std::string sequence;
  sequence.reserve(size);
  uint32_t state = 0x9e3779b9U;
  constexpr char bases[] = "ACGT";
  for (size_t idx = 0; idx < size; ++idx) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    sequence.push_back(bases[state & 3U]);
  }
  return sequence;
}

bool files_equal(const std::filesystem::path& left,
                 const std::filesystem::path& right) {
  if (std::filesystem::file_size(left) !=
      std::filesystem::file_size(right)) {
    return false;
  }
  std::ifstream left_in(left, std::ios::binary);
  std::ifstream right_in(right, std::ios::binary);
  return std::equal(
      std::istreambuf_iterator<char>(left_in),
      std::istreambuf_iterator<char>(),
      std::istreambuf_iterator<char>(right_in));
}

void test_indexed_reference_file_slices() {
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("navigamer-reference-file-test-" +
       std::to_string(static_cast<unsigned long long>(
           ::getpid())));
  std::filesystem::create_directories(directory);
  const auto fasta = directory / "reference.fa";
  const std::string first =
      deterministic_dna((size_t{1} << 20) + 257);
  {
    std::ofstream out(fasta, std::ios::binary);
    out << ">chrA description\r\n";
    for (size_t begin = 0; begin < first.size(); begin += 61) {
      const size_t count = std::min<size_t>(61, first.size() - begin);
      std::string line = first.substr(begin, count);
      std::transform(
          line.begin(), line.end(), line.begin(),
          [](unsigned char base) {
            return static_cast<char>(std::tolower(base));
          });
      out << line << "\r\n";
    }
    out << "\r\n>chrB\tmetadata\r\n"
        << "nN nN\r\n"
        << "a c g t\r\n";
  }

  const auto loaded =
      navigamer::load_reference_genome(fasta.string());
  const auto indexed =
      navigamer::index_reference_genome_file(fasta.string());
  const auto densely_indexed =
      navigamer::index_reference_genome_file(fasta.string(), 4096);
  assert(indexed.id == loaded.id);
  assert(indexed.sequence_size == loaded.sequence.size());
  assert(indexed.contigs.size() == loaded.contigs.size());
  for (size_t idx = 0; idx < indexed.contigs.size(); ++idx) {
    assert(indexed.contigs[idx].id == loaded.contigs[idx].id);
    assert(indexed.contigs[idx].begin == loaded.contigs[idx].begin);
    assert(indexed.contigs[idx].end == loaded.contigs[idx].end);
    assert(indexed.contigs[idx].source_begin ==
           loaded.contigs[idx].source_begin);
  }
  const std::vector<std::pair<size_t, size_t>> slices = {
      {0, 1},
      {0, 4097},
      {(size_t{1} << 20) - 127, (size_t{1} << 20) + 127},
      {first.size() - 257, first.size()},
      {first.size(), first.size() + 8}};
  for (const auto& slice : slices) {
    assert(indexed.slice(slice.first, slice.second) ==
           loaded.sequence.substr(
               slice.first, slice.second - slice.first));
    assert(densely_indexed.slice(slice.first, slice.second) ==
           loaded.sequence.substr(
               slice.first, slice.second - slice.first));
  }
  bool crossing_rejected = false;
  try {
    (void)indexed.slice(first.size() - 1, first.size() + 1);
  } catch (const std::out_of_range&) {
    crossing_rejected = true;
  }
  assert(crossing_rejected);

  const auto fai_fasta = directory / "indexed.fa";
  {
    std::ofstream out(fai_fasta, std::ios::binary);
    out << ">chrF\nACGT\nTGCA\nAA\n";
  }
  {
    std::ofstream out(fai_fasta.string() + ".fai");
    out << "chrF\t10\t6\t4\t5\n";
  }
  const auto fai_indexed =
      navigamer::index_reference_genome_file(fai_fasta.string(), 3);
  assert(fai_indexed.sequence_size == 10);
  assert(fai_indexed.contigs.size() == 1);
  assert(fai_indexed.contigs.front().id == "chrF");
  assert(fai_indexed.slice(0, 10) == "ACGTTGCAAA");
  assert(fai_indexed.slice(2, 9) == "GTTGCAA");

  const auto lowercase_fastq = directory / "lowercase.fq";
  {
    std::ofstream out(lowercase_fastq);
    out << "@lowercase\nacgttgcaaa\n+\nIIIIIIIIII\n";
  }
  navigamer::QuerySequenceReader lowercase_reader(
      lowercase_fastq.string());
  navigamer::QuerySequence lowercase_query;
  assert(lowercase_reader.next(&lowercase_query));
  assert(lowercase_query.seq == "ACGTTGCAAA");

  navigamer::HierarchyConfig hierarchy({8, 4, 2});
  navigamer::BuildRangeConfig range_config;
  range_config.emit_build_output = false;
  const auto packed_bundle = directory / "multi-contig.navshard";
  const auto packed_manifest =
      navigamer::build_sharded_reference_index(
          packed_bundle.string(), fasta.string(), indexed,
          24, 1, 5000, hierarchy, range_config, 2, false);
  const auto packed_reference =
      navigamer::load_packed_reference_file(
          packed_bundle.string(), packed_manifest);
  assert(packed_reference.sequence_size == loaded.sequence.size());
  assert(packed_reference.slice(0, first.size()) == first);
  assert(packed_reference.slice(
             first.size(), loaded.sequence.size()) ==
         loaded.sequence.substr(first.size()));

  std::filesystem::remove_all(directory);
}

uint32_t shard_id_bits(uint32_t shard_count) {
  uint32_t bits = 0;
  uint32_t largest_id = shard_count - 1;
  do {
    ++bits;
    largest_id >>= 1;
  } while (largest_id != 0);
  return bits;
}

std::vector<uint8_t> pack_shard_ids(
    const std::vector<uint32_t>& shard_ids, uint32_t bits) {
  std::vector<uint8_t> packed(
      (shard_ids.size() * bits + 7) / 8, 0);
  size_t bit_offset = 0;
  for (uint32_t shard_id : shard_ids) {
    for (uint32_t bit = 0; bit < bits; ++bit) {
      if ((shard_id & (uint32_t{1} << bit)) != 0) {
        packed[(bit_offset + bit) / 8] |= static_cast<uint8_t>(
            uint8_t{1} << ((bit_offset + bit) % 8));
      }
    }
    bit_offset += bits;
  }
  return packed;
}

void test_bit_packed_shard_id_boundaries() {
  const std::vector<uint32_t> shard_counts = {
      1, 2, 3, 4, 255, 256, 257,
      65535, 65536, 65537, uint32_t{1} << 20};
  for (uint32_t shard_count : shard_counts) {
    std::vector<uint32_t> expected = {
        0, shard_count / 2, shard_count - 1};
    std::sort(expected.begin(), expected.end());
    expected.erase(
        std::unique(expected.begin(), expected.end()),
        expected.end());

    navigamer::ShardedSeedRouter router;
    router.k = 1;
    router.window = 1;
    router.shard_count = shard_count;
    router.shard_id_bits = shard_id_bits(shard_count);
    router.minimizer_codes.assign(expected.size(), 0);
    router.packed_shard_ids.set_owned(
        pack_shard_ids(expected, router.shard_id_bits));
    const auto selected = router.select("A", 0);
    assert(selected.enabled);
    assert(selected.shard_ids == expected);
  }
}

void test_router_merges_sorted_code_ranges() {
  navigamer::ShardedSeedRouter router;
  router.k = 1;
  router.window = 1;
  router.shard_count = 100;
  router.shard_id_bits = shard_id_bits(router.shard_count);
  router.minimizer_codes = {0, 0, 0, 1, 1, 1};
  router.packed_shard_ids.set_owned(
      pack_shard_ids({1, 3, 5, 0, 3, 4}, router.shard_id_bits));

  std::vector<uint32_t> selected = {99};
  assert(router.append_selected_shards("AC", 1, &selected));
  assert((selected == std::vector<uint32_t>{99, 0, 1, 3, 4, 5}));

  navigamer::ShardedSeedRouter large_router;
  large_router.k = 1;
  large_router.window = 1;
  large_router.shard_count = 3000;
  large_router.shard_id_bits = shard_id_bits(large_router.shard_count);
  std::vector<uint32_t> large_shard_ids;
  for (uint32_t code = 0; code < 2; ++code) {
    for (uint32_t shard_id = 0;
         shard_id < large_router.shard_count; ++shard_id) {
      large_router.minimizer_codes.push_back(code);
      large_shard_ids.push_back(shard_id);
    }
  }
  large_router.packed_shard_ids.set_owned(
      pack_shard_ids(large_shard_ids, large_router.shard_id_bits));
  large_shard_ids.resize(large_router.shard_count);
  std::iota(
      large_shard_ids.begin(), large_shard_ids.end(), uint32_t{0});
  const auto large_selection = large_router.select("AC", 1);
  assert(large_selection.enabled);
  assert(large_selection.shard_ids == large_shard_ids);

  navigamer::ShardedSeedRouter long_router;
  long_router.k = 3;
  long_router.window = 3;
  long_router.shard_count = 300;
  long_router.shard_id_bits = shard_id_bits(long_router.shard_count);
  std::string long_query;
  std::vector<uint32_t> long_shard_ids;
  const std::array<char, 4> bases = {'A', 'C', 'G', 'T'};
  for (uint32_t code = 0; code < 17; ++code) {
    long_query.push_back(bases[(code >> 4) & 3]);
    long_query.push_back(bases[(code >> 2) & 3]);
    long_query.push_back(bases[code & 3]);
    for (uint32_t shard_id = 0;
         shard_id < long_router.shard_count; ++shard_id) {
      long_router.minimizer_codes.push_back(code);
      long_shard_ids.push_back(shard_id);
    }
  }
  long_router.packed_shard_ids.set_owned(
      pack_shard_ids(long_shard_ids, long_router.shard_id_bits));
  long_shard_ids.resize(long_router.shard_count);
  std::iota(
      long_shard_ids.begin(), long_shard_ids.end(), uint32_t{0});
  const auto long_selection = long_router.select(long_query, 16);
  assert(long_selection.enabled);
  assert(long_selection.shard_ids == long_shard_ids);
}

void test_router_intersects_minimizers_within_one_partition() {
  navigamer::ShardedSeedRouter router;
  router.k = 1;
  router.window = 2;
  router.shard_count = 5;
  router.shard_id_bits = shard_id_bits(router.shard_count);
  router.minimizer_codes = {0, 0, 0, 1, 1, 1};
  router.packed_shard_ids.set_owned(
      pack_shard_ids({1, 2, 3, 2, 3, 4}, router.shard_id_bits));

  const auto selected = router.select("AAAACC", 0);
  assert(selected.enabled);
  assert((selected.shard_ids == std::vector<uint32_t>{2, 3}));
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
  const auto packed_reference =
      navigamer::load_packed_reference_file(
          bundle.string(), reloaded_manifest);
  assert(packed_reference.sequence_size == reference.size());
  assert(packed_reference.slice(0, first.size()) == first);
  assert(packed_reference.slice(
             first.size(), reference.size()) == second);
  assert(reloaded_manifest.window_length == window);
  assert(reloaded_manifest.format_version == 20);
  assert(reloaded_manifest.stride == stride);
  assert(reloaded_manifest.shards.size() ==
         manifest.shards.size());
  assert(reloaded_manifest.shards.size() > 2);
  for (const auto& shard : reloaded_manifest.shards) {
    assert(shard.window_count <= shard_windows);
    assert(shard.pack_id < reloaded_manifest.pack_paths.size());
    assert(shard.file_offset % 64 == 0);
    assert(shard.file_size > 0);
  }

  std::vector<std::filesystem::file_time_type> pack_times;
  pack_times.reserve(reloaded_manifest.pack_paths.size());
  for (const auto& pack_path : reloaded_manifest.pack_paths) {
    const auto resolved = navigamer::resolve_index_shard_path(
        bundle.string(), pack_path);
    assert(std::filesystem::exists(resolved));
    pack_times.push_back(std::filesystem::last_write_time(
        resolved));
  }
  const auto resumed =
      navigamer::build_sharded_reference_index(
          bundle.string(), "literal-reference", "reference",
          reference, contigs, window, stride, shard_windows,
          hierarchy, range_config, 4);
  assert(resumed.shards.size() == reloaded_manifest.shards.size());
  assert(resumed.pack_paths.size() == pack_times.size());
  for (size_t pack_idx = 0;
       pack_idx < resumed.pack_paths.size(); ++pack_idx) {
    assert(std::filesystem::last_write_time(
        navigamer::resolve_index_shard_path(
            bundle.string(), resumed.pack_paths[pack_idx])) ==
           pack_times[pack_idx]);
  }

  const std::string damaged_path =
      navigamer::resolve_index_shard_path(
          bundle.string(), resumed.pack_paths.front());
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

void test_seed_router_no_false_negatives() {
  constexpr size_t window = 250;
  constexpr size_t stride = 1;
  constexpr size_t shard_windows = 100;
  const std::string reference = deterministic_dna(650);
  const std::vector<navigamer::ReferenceContig> contigs = {
      {"chrR", 0, static_cast<uint32_t>(reference.size()), 0}};
  navigamer::HierarchyConfig hierarchy({8, 4, 2});
  navigamer::BuildRangeConfig range_config;

  const auto directory =
      std::filesystem::temp_directory_path() /
      ("navigamer-router-test-" +
       std::to_string(static_cast<unsigned long long>(
           ::getpid())));
  std::filesystem::create_directories(directory);
  const auto bundle = directory / "reference.navshard";
  (void)navigamer::build_sharded_reference_index(
      bundle.string(), "literal-router-reference", "chrR",
      reference, contigs, window, stride, shard_windows,
      hierarchy, range_config, 2);

  const auto manifest =
      navigamer::read_sharded_index_manifest(bundle.string());
  assert(manifest.shards.size() > 3);
  assert(manifest.router_k == 16);
  assert(manifest.router_window == 24);
  assert(manifest.router_entry_count > 0);
  const size_t raw_router_bytes =
      48 + manifest.router_entry_count * sizeof(uint32_t) +
      (manifest.router_entry_count * 3 + 7) / 8;
  const size_t compact_router_bytes = static_cast<size_t>(
      std::filesystem::file_size(bundle.string() + ".route"));
  assert(compact_router_bytes < raw_router_bytes);
  assert(compact_router_bytes <
         40 + manifest.router_entry_count * sizeof(uint64_t));
  const auto rebuilt_manifest =
      navigamer::build_sharded_reference_index(
          bundle.string(), "literal-router-reference", "chrR",
          reference, contigs, window, stride, shard_windows,
          hierarchy, range_config, 2);
  assert(rebuilt_manifest.router_entry_count ==
         manifest.router_entry_count);
  assert(rebuilt_manifest.router_checksum ==
         manifest.router_checksum);
  assert(!std::filesystem::exists(
      bundle.string() + ".route.packed.tmp"));
  assert(!std::filesystem::exists(
      bundle.string() + ".route.codes.tmp"));

  const auto fasta = directory / "reference.fa";
  {
    std::ofstream out(fasta);
    out << ">chrR source\n";
    for (size_t begin = 0; begin < reference.size(); begin += 53) {
      std::string line = reference.substr(
          begin, std::min<size_t>(53, reference.size() - begin));
      std::transform(
          line.begin(), line.end(), line.begin(),
          [](unsigned char base) {
            return static_cast<char>(std::tolower(base));
          });
      out << line << '\n';
    }
  }
  const auto indexed_reference =
      navigamer::index_reference_genome_file(fasta.string());
  const auto file_bundle = directory / "reference-file.navshard";
  const auto file_manifest =
      navigamer::build_sharded_reference_index(
          file_bundle.string(), "literal-router-reference",
          indexed_reference, window, stride, shard_windows,
          hierarchy, range_config, 2);
  assert(file_manifest.router_checksum ==
         rebuilt_manifest.router_checksum);
  assert(file_manifest.shards.size() ==
         rebuilt_manifest.shards.size());
  assert(file_manifest.pack_paths.size() ==
         rebuilt_manifest.pack_paths.size());
  for (size_t pack_idx = 0;
       pack_idx < file_manifest.pack_paths.size(); ++pack_idx) {
    assert(files_equal(
        navigamer::resolve_index_shard_path(
            bundle.string(), rebuilt_manifest.pack_paths[pack_idx]),
        navigamer::resolve_index_shard_path(
            file_bundle.string(), file_manifest.pack_paths[pack_idx])));
  }
  assert(files_equal(
      bundle.string() + ".route",
      file_bundle.string() + ".route"));
  const auto packed_file_reference =
      navigamer::load_packed_reference_file(
          file_bundle.string(), file_manifest);
  assert(packed_file_reference.sequence_size == reference.size());
  assert(packed_file_reference.slice(0, reference.size()) == reference);

  const auto router_only_bundle =
      directory / "reference-router-only.navshard";
  const auto router_only_manifest =
      navigamer::build_sharded_reference_index(
          router_only_bundle.string(), fasta.string(), indexed_reference,
          window, stride, shard_windows, hierarchy, range_config, 2,
          false);
  assert(router_only_manifest.pack_paths.empty());
  assert(router_only_manifest.total_window_count ==
         file_manifest.total_window_count);
  assert(router_only_manifest.total_sequence_count == 0);
  assert(router_only_manifest.total_world_node_count == 0);
  assert(router_only_manifest.router_entry_count ==
         file_manifest.router_entry_count);
  assert(router_only_manifest.router_checksum ==
         file_manifest.router_checksum);
  assert(files_equal(
      router_only_bundle.string() + ".route",
      file_bundle.string() + ".route"));
  for (const auto& shard : router_only_manifest.shards) {
    assert(shard.pack_id == 0);
    assert(shard.file_offset == 0);
    assert(shard.file_size == 0);
    assert(shard.sequence_count == 0);
    assert(shard.world_node_count == 0);
  }
  bool router_only_graph_load_rejected = false;
  try {
    (void)navigamer::load_sharded_index(
        router_only_bundle.string(), router_only_manifest);
  } catch (const std::runtime_error&) {
    router_only_graph_load_rejected = true;
  }
  assert(router_only_graph_load_rejected);
  const auto router_only_reloaded =
      navigamer::read_sharded_index_manifest(
          router_only_bundle.string());
  const auto router_only_router =
      navigamer::load_sharded_seed_router(
          router_only_bundle.string(), router_only_reloaded);
  assert(router_only_router.enabled());

  const auto router = navigamer::load_sharded_seed_router(
      bundle.string(), rebuilt_manifest);
  assert(router.enabled());
  assert(router.shard_id_bits == 3);
  assert(router.minimizer_code_bases.is_mapped());
  assert(router.minimizer_code_widths.is_mapped());
  assert(router.minimizer_code_group_offsets.is_mapped());
  assert(router.packed_minimizer_code_deltas.is_mapped());
  assert(router.packed_shard_ids.is_mapped());
  assert(router.minimizer_code_count() ==
         rebuilt_manifest.router_entry_count);
  std::vector<uint32_t> decoded_router_codes;
  decoded_router_codes.reserve(router.minimizer_code_count());
  for (size_t code_idx = 1;
       code_idx < router.minimizer_code_count(); ++code_idx) {
    assert(router.minimizer_code_at(code_idx - 1) <=
           router.minimizer_code_at(code_idx));
  }
  for (size_t code_idx = 0;
       code_idx < router.minimizer_code_count(); ++code_idx) {
    decoded_router_codes.push_back(router.minimizer_code_at(code_idx));
  }
  std::vector<uint32_t> bound_probes = decoded_router_codes;
  bound_probes.push_back(0);
  bound_probes.push_back(UINT32_MAX);
  for (uint32_t code : decoded_router_codes) {
    if (code != 0) bound_probes.push_back(code - 1);
    if (code != UINT32_MAX) bound_probes.push_back(code + 1);
  }
  for (uint32_t code : bound_probes) {
    const size_t expected_lower = static_cast<size_t>(
        std::lower_bound(
            decoded_router_codes.begin(), decoded_router_codes.end(), code) -
        decoded_router_codes.begin());
    const size_t expected_upper = static_cast<size_t>(
        std::upper_bound(
            decoded_router_codes.begin(), decoded_router_codes.end(), code) -
        decoded_router_codes.begin());
    assert(router.lower_bound_minimizer_code(code) == expected_lower);
    assert(router.upper_bound_minimizer_code(code) == expected_upper);
    assert(router.equal_range_minimizer_code(code) ==
           std::make_pair(expected_lower, expected_upper));
  }
  const auto shards = navigamer::load_sharded_index(
      bundle.string(), rebuilt_manifest);
  const auto subset = navigamer::load_sharded_index(
      bundle.string(), rebuilt_manifest,
      std::vector<uint32_t>{1, 3});
  assert(subset.size() == 2);
  assert(subset[0].builder.num_sequences() ==
         rebuilt_manifest.shards[1].sequence_count);
  assert(subset[1].builder.num_sequences() ==
         rebuilt_manifest.shards[3].sequence_count);

  const std::vector<size_t> source_positions = {
      0, 1, 50, 98, 99, 100, 101, 198, 199,
      200, 201, 298, 299, 300, 301, 399, 400};
  size_t ordinal = 0;
  for (size_t source_pos : source_positions) {
    const std::string exact = reference.substr(source_pos, window);
    for (int mutation = 0; mutation < 6; ++mutation) {
      std::string sequence = exact;
      if (mutation == 1) {
        sequence[7] = sequence[7] == 'A' ? 'C' : 'A';
        sequence[151] = sequence[151] == 'G' ? 'T' : 'G';
      } else if (mutation == 2) {
        sequence.insert(sequence.begin() + 95, 'A');
      } else if (mutation == 3) {
        sequence.erase(sequence.begin() + 96);
      } else if (mutation == 4) {
        sequence.insert(sequence.begin() + 33, 'C');
        sequence.erase(sequence.begin() + 177);
      } else if (mutation == 5) {
        for (size_t pos : {3, 45, 87, 129, 171}) {
          sequence[pos] = sequence[pos] == 'A' ? 'C' : 'A';
        }
      }
      const auto route = router.select(sequence, 5);
      assert(route.enabled);
      assert(!route.shard_ids.empty());
      navigamer::BioSequence query(
          "router_" + std::to_string(ordinal++), sequence);
      assert(matching_occurrences(shards, query, 5) ==
             matching_occurrences(
                 shards, query, 5, &route.shard_ids));
    }
  }

  // A 150 bp target at d=5 can lose five bases, leaving six 24 bp blocks.
  // Both substitution and deletion cases must route without falling back to
  // every shard, while retaining every exact result.
  const auto short_bundle = directory / "reference-150.navshard";
  const auto short_manifest = navigamer::build_sharded_reference_index(
      short_bundle.string(), "literal-router-reference", "chrR",
      reference, contigs, 150, stride, shard_windows,
      hierarchy, range_config, 2);
  const auto short_router = navigamer::load_sharded_seed_router(
      short_bundle.string(), short_manifest);
  const auto short_shards = navigamer::load_sharded_index(
      short_bundle.string(), short_manifest);
  const auto short_router_only_bundle =
      directory / "reference-150-router-only.navshard";
  const auto short_router_only_manifest =
      navigamer::build_sharded_reference_index(
          short_router_only_bundle.string(), fasta.string(),
          indexed_reference, 150, stride, shard_windows,
          hierarchy, range_config, 2, false);
  const auto short_router_only =
      navigamer::load_sharded_seed_router(
          short_router_only_bundle.string(),
          short_router_only_manifest);
  const auto short_packed_reference =
      navigamer::load_packed_reference_file(
          short_router_only_bundle.string(),
          short_router_only_manifest);
  for (size_t source_pos : {size_t{0}, size_t{98}, size_t{300}}) {
    const std::string exact = reference.substr(source_pos, 150);
    for (size_t edit_case = 0; edit_case < 4; ++edit_case) {
      std::string sequence = exact;
      if (edit_case == 0) {
        for (size_t mutation : {size_t{3}, size_t{45}, size_t{87},
                                size_t{129}, size_t{147}}) {
          sequence[mutation] = sequence[mutation] == 'A' ? 'C' : 'A';
        }
      } else {
        const std::array<size_t, 5> positions = {
            size_t{143}, size_t{120}, size_t{96},
            size_t{72}, size_t{48}};
        if (edit_case == 1) {
          for (size_t mutation : positions) {
            sequence.erase(sequence.begin() + mutation);
          }
        } else if (edit_case == 2) {
          for (size_t mutation : positions) {
            sequence.insert(sequence.begin() + mutation, 'A');
          }
        } else {
          sequence.erase(sequence.begin() + 120);
          sequence.erase(sequence.begin() + 72);
          sequence.insert(sequence.begin() + 96, 'C');
          sequence.insert(sequence.begin() + 48, 'G');
          sequence[12] = sequence[12] == 'A' ? 'C' : 'A';
        }
      }
      const auto route = short_router.select(sequence, 5);
      assert(route.enabled);
      assert(!route.shard_ids.empty());
      const auto router_only_route =
          short_router_only.select(sequence, 5);
      assert(router_only_route.enabled);
      assert(router_only_route.shard_ids == route.shard_ids);
      const auto direct =
          navigamer::verify_selected_shards_by_exact_blocks(
              sequence, 5, short_router_only_manifest,
              short_packed_reference,
              router_only_route.shard_ids.data(),
              router_only_route.shard_ids.data() +
                  router_only_route.shard_ids.size());
      assert(direct.enabled);
      navigamer::BioSequence query(
          "router_150_" + std::to_string(source_pos) + "_" +
          std::to_string(edit_case), sequence);
      assert(matching_occurrences(short_shards, query, 5) ==
             matching_occurrences(
                 short_shards, query, 5, &route.shard_ids));
      std::set<Occurrence> direct_occurrences;
      for (const auto& occurrence : direct.occurrences) {
        direct_occurrences.emplace(
            short_manifest.contig_ids[occurrence.contig_id],
            occurrence.source_start, occurrence.sequence);
      }
      assert(matching_occurrences(short_shards, query, 5) ==
             direct_occurrences);
      if (source_pos == 0 && edit_case == 0) {
        std::string lowercase = sequence;
        std::transform(
            lowercase.begin(), lowercase.end(), lowercase.begin(),
            [](unsigned char base) {
              return static_cast<char>(std::tolower(base));
            });
        const auto lowercase_direct =
            navigamer::verify_selected_shards_by_exact_blocks(
                lowercase, 5, short_manifest, indexed_reference,
                route.shard_ids.data(),
                route.shard_ids.data() + route.shard_ids.size());
        assert(lowercase_direct.enabled);
        std::set<Occurrence> lowercase_occurrences;
        for (const auto& occurrence : lowercase_direct.occurrences) {
          lowercase_occurrences.emplace(
              short_manifest.contig_ids[occurrence.contig_id],
              occurrence.source_start, occurrence.sequence);
        }
        assert(lowercase_occurrences == direct_occurrences);
      }
    }
  }

  std::vector<std::string> batch_sequences = {
      reference.substr(98, 150), reference.substr(300, 150)};
  batch_sequences[0].erase(batch_sequences[0].begin() + 70);
  batch_sequences[1].insert(batch_sequences[1].begin() + 80, 'G');
  for (size_t duplicate = 0; duplicate < 6; ++duplicate) {
    batch_sequences.push_back(batch_sequences[0]);
  }
  std::vector<std::vector<uint32_t>> batch_routes;
  for (const auto& sequence : batch_sequences) {
    auto route = short_router.select(sequence, 5);
    assert(route.enabled && !route.shard_ids.empty());
    batch_routes.push_back(std::move(route.shard_ids));
  }
  std::vector<navigamer::ExactBlockVerificationRequest> batch_requests;
  for (size_t query_idx = 0; query_idx < batch_sequences.size();
       ++query_idx) {
    batch_requests.push_back(
        {batch_sequences[query_idx], batch_routes[query_idx].data(),
         batch_routes[query_idx].data() + batch_routes[query_idx].size()});
  }
  const auto batch_results =
      navigamer::verify_selected_shards_by_exact_blocks_batch(
          5, short_router_only_manifest, short_packed_reference,
          batch_requests);
  assert(batch_results.size() == batch_sequences.size());
  for (size_t query_idx = 0; query_idx < batch_results.size();
       ++query_idx) {
    assert(batch_results[query_idx].enabled);
    std::set<Occurrence> batch_occurrences;
    for (const auto& occurrence : batch_results[query_idx].occurrences) {
      batch_occurrences.emplace(
          short_manifest.contig_ids[occurrence.contig_id],
          occurrence.source_start, occurrence.sequence);
    }
    navigamer::BioSequence query(
        "router_batch_" + std::to_string(query_idx),
        batch_sequences[query_idx]);
    assert(matching_occurrences(short_shards, query, 5) ==
           batch_occurrences);
  }

  const std::string exact = reference.substr(98, window);
  std::string ambiguous = exact;
  ambiguous[10] = 'N';
  assert(!router.select(ambiguous, 5).enabled);
  assert(!router.select(exact.substr(0, 100), 5).enabled);
  assert(!router.select(exact, -1).enabled);

  {
    std::fstream damaged(
        short_router_only_bundle.string() + ".ref2",
        std::ios::in | std::ios::out | std::ios::binary);
    assert(damaged.good());
    damaged.seekg(static_cast<std::streamoff>(
        short_packed_reference.payload_offset));
    char byte = 0;
    damaged.get(byte);
    damaged.seekp(static_cast<std::streamoff>(
        short_packed_reference.payload_offset));
    damaged.put(static_cast<char>(byte ^ 1));
  }
  const auto damaged_packed_reference =
      navigamer::load_packed_reference_file(
          short_router_only_bundle.string(),
          short_router_only_manifest);
  bool damaged_block_rejected = false;
  try {
    (void)damaged_packed_reference.slice(0, 1);
  } catch (const std::runtime_error&) {
    damaged_block_rejected = true;
  }
  assert(damaged_block_rejected);

  std::filesystem::remove_all(directory);
}

void test_implicit_dense_leaf_fields() {
  constexpr size_t window = 250;
  constexpr size_t shard_windows = 5000;
  const std::string reference = deterministic_dna(
      shard_windows + window - 1);
  const std::vector<navigamer::ReferenceContig> contigs = {
      {"chrDense", 0, static_cast<uint32_t>(reference.size()), 0}};
  navigamer::HierarchyConfig hierarchy({30, 15, 5});
  navigamer::BuildRangeConfig range_config;
  range_config.emit_build_output = false;

  const auto directory =
      std::filesystem::temp_directory_path() /
      ("navigamer-dense-leaf-test-" +
       std::to_string(static_cast<unsigned long long>(::getpid())));
  std::filesystem::create_directories(directory);
  const auto bundle = directory / "reference.navshard";
  const auto manifest = navigamer::build_sharded_reference_index(
      bundle.string(), "dense-leaf-reference", "chrDense",
      reference, contigs, window, 1, shard_windows,
      hierarchy, range_config, 1);
  const auto shards = navigamer::load_sharded_index(
      bundle.string(), manifest);
  assert(shards.size() == 1);
  assert(verify_implicit_dense_leaf_layout(shards));

  const auto& builder = shards.front().builder;
  const auto& view = builder.search_graph_view();
  assert(view.periodic_center_layers.size() == 3);
  assert(view.periodic_center_layers[0].period == 1);
  assert(view.periodic_center_layers[1].period == 3);
  assert(view.periodic_center_layers[2].period == 7);
  assert(view.periodic_center_layers[0].pattern ==
         navigamer::PeriodicCenterLayer::Linear16);
  assert(view.periodic_center_layers[1].pattern ==
         navigamer::PeriodicCenterLayer::DefaultPeriod3);
  assert(view.periodic_center_layers[2].pattern ==
         navigamer::PeriodicCenterLayer::DefaultPeriod7);
  assert(view.periodic_center_offsets.size() == 11);
  assert(view.center_id_block_bases16.empty());
  assert(view.center_id_block_bases.empty());
  assert(view.center_id_block_deltas.empty());
  for (size_t layer = 0; layer < view.layer_begin.size(); ++layer) {
    navigamer::LeafId previous = 0;
    for (navigamer::NodeId node_id = view.layer_begin[layer];
         node_id < view.layer_end[layer]; ++node_id) {
      const navigamer::LeafId center =
          view.center_sequence_id(node_id, layer);
      if (node_id != view.layer_begin[layer]) assert(center > previous);
      previous = center;
    }
  }
  assert(view.node_records.bytes().size() ==
         static_cast<size_t>(view.node_records.finest_node_begin()) *
             view.node_records.child_layout().record_bytes);
  assert(view.node_records.record_data(
             view.node_records.finest_node_begin()) == nullptr);
  navigamer::BioGeometrySearchEngine engine(builder);
  for (size_t source_pos : {size_t{0}, size_t{2000}, size_t{4999}}) {
    std::string query_sequence = reference.substr(source_pos, window);
    for (size_t mutation = 0; mutation < 2; ++mutation) {
      if (mutation != 0) {
        query_sequence[73] =
            query_sequence[73] == 'A' ? 'C' : 'A';
      }
      navigamer::BioSequence query(
          "dense_" + std::to_string(source_pos) + "_" +
              std::to_string(mutation),
          query_sequence);
      const auto [adaptive, adaptive_stats] =
          engine.search_adaptive(query, 1);
      const auto [brute_force, brute_stats] =
          engine.search_brute_force(query, 1);
      (void)adaptive_stats;
      (void)brute_stats;
      assert(std::set<navigamer::LeafId>(adaptive.begin(), adaptive.end()) ==
             std::set<navigamer::LeafId>(
                 brute_force.begin(), brute_force.end()));
    }
  }
  std::filesystem::remove_all(directory);
}

void test_child_mbb_width_fallback_for_three_widths() {
  const std::string informative =
      "TGCTGGTCTGGTTCTCTTTCTCTCCGTGACCTATAGAGCAAGGTGGAGGGGTAGGAGGGGGACACCCAGTGAAGGG"
      "TCCTTTGGCCTTGTAGTTTCTTAGAGGCTTCTTCTGGGAACATGTACTGGGAGCTGGGGTGGGTCCTGCACCTGCA"
      "TGGGGCCATTTCCCTTCGTGGGCCCACAGACAACTGTTCCCCACCACGGAGGGAAGGAGACGCACAGGGCCTGGGC"
      "CTTCTTCTCTGAGAACACTCTCAAGCAGAACTCGCCGTCTTTGAAGGGTTCAAATGTGGATGGCACCACCAGGTAC"
      "TCCCCAGGGGGCAGCCGGGCCCGGCCAGAGACCTCCCGCAGGTTGACGTAGGTGCTGGTGCGGGCTGAGGGCTGGT"
      "AGGCCAGGAAGAAATCCCGGCCCAAGTGTGCGTCCGTGTGACTCTCCAGCTGCACGAAACAATAAGCAGAGTCAAT"
      "TTCTTGTTAAATCCTGGAAGATGAGAGCCCAAGAGTTCAGCTTTATTGTGCTGATTTAGGAATTATTGATTTTTAC"
      "CATTGCACCAAGAATCAGGAGGCCGTGGATTCTGTTGTGAACTCACTGTATGTCAATCATCAAGTGTATTTTCAGT"
      "GCCTTCTGGGTGCCAGGCCCTGTTTGAGGCATTGATCTTGACTGTGTGACCTTGACCTCTGGGCCTCCCCAGTTAA"
      "ACGAAGGTTGAGGGACAGGGTCTCTAGTGTGCGCTCAGCTTCTCTCTGGATATTTTTCCTCTCTAATCCAATAGGC"
      "TCTTTCATTCTGCAGCTGTCTCTGGGAGTGTGGCATTGATCTTCCCAGTACAGGCCCAAGGCTGGAGAAAAGAGCT"
      "TAAATCCTAGTCCTCAAGCAAAAGCTGCCACAAAAACTTGTTCACCTTTGACTATCTGTGAAAACTGCTCTGCAAA"
      "GAGACTAGGATGGTCAGTCTGGGAACCCAGGGAGGCAGCTCTAAAAGAGACAGGCAGAGGGGGCGGCCACAGCTAG"
      "GCCTAGTCTGGGGTCCCCCCAGCCTCCCCAGGGCCTCGACTCTCCTCAACAGGCGACGATGCTCTCCAAGACCCAC"
      "TTATTTGTTGCGGGGAGGTGGGAGGCTGTTGGTGCATGACACAGGTTAATTAGTGACTGCAGAGTGCTTCCAAAAC"
      "ACAGGTGCCAAAGTTATTATTGCTGTAATTAAGCCTCCCCATAACACTGTGTCTTTACCATCAATCTTCATCAA";
  const std::string reference = std::string(3935, 'N') + informative;
  assert(reference.size() == 5149);
  const std::vector<navigamer::ReferenceContig> contigs = {
      {"chrThreeWidths", 0, static_cast<uint32_t>(reference.size()), 0}};
  navigamer::HierarchyConfig hierarchy({40, 20, 8});
  navigamer::BuildRangeConfig range_config;
  range_config.emit_build_output = false;

  const auto directory =
      std::filesystem::temp_directory_path() /
      ("navigamer-three-child-mbb-widths-test-" +
       std::to_string(static_cast<unsigned long long>(::getpid())));
  std::filesystem::create_directories(directory);
  const auto bundle = directory / "reference.navshard";
  const auto manifest = navigamer::build_sharded_reference_index(
      bundle.string(), "three-child-mbb-widths-reference",
      "chrThreeWidths", reference, contigs, 150, 1, 5000,
      hierarchy, range_config, 1);
  const auto shards = navigamer::load_sharded_index(
      bundle.string(), manifest);
  assert(shards.size() == 1);
  const auto& builder = shards.front().builder;
  const auto& view = builder.search_graph_view();
  assert(!view.implicit_child_mbb_widths);
  assert(view.implicit_child_mbb_exception_bits == 0);
  assert(view.child_mbb_width_exceptions.empty());
  assert(builder.validate_search_graph_view());

  navigamer::BioGeometrySearchEngine engine(builder);
  for (size_t source_pos : {size_t{3935}, size_t{4467}, size_t{4999}}) {
    navigamer::BioSequence query(
        "three_widths_" + std::to_string(source_pos),
        reference.substr(source_pos, 150));
    const auto [adaptive, adaptive_stats] =
        engine.search_adaptive(query, 5);
    const auto [brute_force, brute_force_stats] =
        engine.search_brute_force(query, 5);
    (void)adaptive_stats;
    (void)brute_force_stats;
    assert(std::set<navigamer::LeafId>(adaptive.begin(), adaptive.end()) ==
           std::set<navigamer::LeafId>(
               brute_force.begin(), brute_force.end()));
  }
  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  test_indexed_reference_file_slices();
  test_bit_packed_shard_id_boundaries();
  test_router_merges_sorted_code_ranges();
  test_router_intersects_minimizers_within_one_partition();
  test_sharded_round_trip_and_no_false_negatives();
  test_seed_router_no_false_negatives();
  test_implicit_dense_leaf_fields();
  test_child_mbb_width_fallback_for_three_widths();
  std::cout << "sharded index tests passed\n";
  return 0;
}
