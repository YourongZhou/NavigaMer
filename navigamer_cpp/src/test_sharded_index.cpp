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
  }
  bool crossing_rejected = false;
  try {
    (void)indexed.slice(first.size() - 1, first.size() + 1);
  } catch (const std::out_of_range&) {
    crossing_rejected = true;
  }
  assert(crossing_rejected);

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
  assert(reloaded_manifest.format_version == 4);
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
  assert(manifest.router_window == 32);
  assert(manifest.router_entry_count > 0);
  const size_t expected_router_bytes =
      48 + manifest.router_entry_count * sizeof(uint32_t) +
      (manifest.router_entry_count * 3 + 7) / 8;
  assert(std::filesystem::file_size(
             bundle.string() + ".route") ==
         expected_router_bytes);
  assert(expected_router_bytes <
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

  const auto router = navigamer::load_sharded_seed_router(
      bundle.string(), rebuilt_manifest);
  assert(router.enabled());
  assert(router.shard_id_bits == 3);
  assert(router.minimizer_codes.is_mapped());
  assert(router.packed_shard_ids.is_mapped());
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

  const std::string exact = reference.substr(98, window);
  std::string ambiguous = exact;
  ambiguous[10] = 'N';
  assert(!router.select(ambiguous, 5).enabled);
  assert(!router.select(exact.substr(0, 100), 5).enabled);
  assert(!router.select(exact, -1).enabled);

  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  test_indexed_reference_file_slices();
  test_bit_packed_shard_id_boundaries();
  test_sharded_round_trip_and_no_false_negatives();
  test_seed_router_no_false_negatives();
  std::cout << "sharded index tests passed\n";
  return 0;
}
