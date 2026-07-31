#include "sharded_index.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace navigamer {

namespace {

constexpr std::array<char, 8> kShardMagic = {
    'N', 'G', 'S', 'H', 'R', 'D', '0', '1'};
constexpr uint32_t kShardFormatVersion = 1;
constexpr uint64_t kMaxShardCount = uint64_t{1} << 20;
constexpr uint64_t kMaxStringLength = uint64_t{1} << 30;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

template <typename T>
void write_pod(std::ostream& out, T value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "binary field must be trivially copyable");
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) {
    throw std::runtime_error("failed to write sharded index manifest");
  }
}

template <typename T>
T read_pod(std::istream& in, const char* field) {
  static_assert(std::is_trivially_copyable<T>::value,
                "binary field must be trivially copyable");
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error(
        std::string("truncated sharded index field: ") + field);
  }
  return value;
}

void write_string(std::ostream& out, const std::string& value) {
  write_pod<uint64_t>(out, value.size());
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!out) {
    throw std::runtime_error("failed to write sharded index string");
  }
}

std::string read_string(std::istream& in, const char* field) {
  const uint64_t size = read_pod<uint64_t>(in, field);
  if (size > kMaxStringLength) {
    throw std::runtime_error(
        std::string("oversized sharded index field: ") + field);
  }
  std::string value(static_cast<size_t>(size), '\0');
  in.read(value.data(), static_cast<std::streamsize>(value.size()));
  if (!in) {
    throw std::runtime_error(
        std::string("truncated sharded index field: ") + field);
  }
  return value;
}

size_t read_size(std::istream& in, const char* field) {
  const uint64_t value = read_pod<uint64_t>(in, field);
  if (value > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(
        std::string("sharded index size exceeds this platform: ") + field);
  }
  return static_cast<size_t>(value);
}

void hash_bytes(
    uint64_t* hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t idx = 0; idx < size; ++idx) {
    *hash ^= bytes[idx];
    *hash *= kFnvPrime;
  }
}

template <typename T>
void hash_pod(uint64_t* hash, T value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "manifest hash field must be trivially copyable");
  hash_bytes(hash, &value, sizeof(value));
}

void hash_string(uint64_t* hash, const std::string& value) {
  hash_pod<uint64_t>(hash, value.size());
  hash_bytes(hash, value.data(), value.size());
}

uint64_t manifest_checksum(
    const ShardedIndexManifest& manifest) {
  uint64_t hash = kFnvOffset;
  hash_pod<uint32_t>(&hash, manifest.format_version);
  hash_pod<uint64_t>(&hash, manifest.window_length);
  hash_pod<uint64_t>(&hash, manifest.stride);
  hash_pod<uint64_t>(&hash, manifest.total_window_count);
  hash_pod<uint64_t>(&hash, manifest.total_sequence_count);
  hash_pod<uint64_t>(&hash, manifest.total_world_node_count);
  hash_string(&hash, manifest.part_signature);
  hash_pod<uint64_t>(&hash, manifest.shards.size());
  for (const auto& shard : manifest.shards) {
    hash_string(&hash, shard.path);
    hash_string(&hash, shard.ref_id);
    hash_pod<uint32_t>(&hash, shard.source_begin);
    hash_pod<uint32_t>(&hash, shard.source_end);
    hash_pod<uint64_t>(&hash, shard.window_count);
    hash_pod<uint64_t>(&hash, shard.sequence_count);
    hash_pod<uint64_t>(&hash, shard.world_node_count);
  }
  return hash;
}

std::filesystem::path shard_output_path(
    const std::filesystem::path& bundle_path,
    size_t shard_ordinal) {
  std::ostringstream name;
  name << bundle_path.filename().string()
       << ".shard" << std::setw(6) << std::setfill('0')
       << shard_ordinal << ".navidx";
  return bundle_path.parent_path() / name.str();
}

struct ShardBuildSpec {
  std::filesystem::path part_path;
  std::string ref_id;
  size_t slice_begin = 0;
  size_t slice_end = 0;
  uint32_t source_begin = 0;
  uint32_t source_end = 0;
  size_t window_count = 0;
};

bool load_reusable_shard(
    const std::filesystem::path& part_path,
    const IndexBuildManifest& expected_manifest,
    const std::string& reference_id,
    const std::string& reference_slice,
    const ReferenceContig& expected_contig,
    size_t window_length,
    LoadedIndex* loaded) {
  std::string reason;
  if (!index_matches_manifest(
          part_path.string(), expected_manifest, nullptr, &reason)) {
    return false;
  }
  try {
    LoadedIndex candidate = load_index(part_path.string());
    const auto& store = candidate.builder.sequence_store();
    if (!store.reference_backed ||
        store.reference_id != reference_id ||
        store.fixed_sequence_length != window_length ||
        store.reference_sequence != reference_slice ||
        store.reference_contigs.size() != 1) {
      return false;
    }
    const auto& contig = store.reference_contigs.front();
    if (contig.id != expected_contig.id ||
        contig.begin != expected_contig.begin ||
        contig.end != expected_contig.end ||
        contig.source_begin != expected_contig.source_begin) {
      return false;
    }
    *loaded = std::move(candidate);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

void install_shard_atomically(
    const std::filesystem::path& part_path,
    const BioGeometryIndexBuilder& builder,
    const IndexBuildManifest& manifest) {
  const std::filesystem::path temporary =
      part_path.string() + ".tmp";
  std::error_code error;
  std::filesystem::remove(temporary, error);
  error.clear();
  save_index(temporary.string(), builder, manifest);
  std::filesystem::rename(temporary, part_path, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error(
        "unable to install index shard: " + error.message());
  }
}

void validate_manifest(const ShardedIndexManifest& manifest) {
  if (manifest.format_version != kShardFormatVersion) {
    throw std::runtime_error(
        "unsupported NavigaMer sharded index format version");
  }
  if (manifest.window_length == 0 || manifest.stride == 0) {
    throw std::runtime_error(
        "sharded index has invalid window configuration");
  }
  if (manifest.part_signature.empty() ||
      manifest.shards.empty()) {
    throw std::runtime_error("sharded index contains no shards");
  }
  size_t total_windows = 0;
  size_t total_sequences = 0;
  size_t total_nodes = 0;
  const IndexShardDescriptor* previous = nullptr;
  for (const auto& shard : manifest.shards) {
    if (shard.path.empty() || shard.ref_id.empty() ||
        shard.source_end < shard.source_begin ||
        shard.window_count == 0) {
      throw std::runtime_error(
          "sharded index contains an invalid shard descriptor");
    }
    const uint64_t expected_source_end =
        static_cast<uint64_t>(shard.source_begin) +
        (static_cast<uint64_t>(shard.window_count) - 1) *
            manifest.stride +
        manifest.window_length;
    if (expected_source_end >
            std::numeric_limits<uint32_t>::max() ||
        shard.source_end != expected_source_end) {
      throw std::runtime_error(
          "sharded index has inconsistent shard coordinates");
    }
    if (previous && previous->ref_id == shard.ref_id) {
      const uint64_t expected_source_begin =
          static_cast<uint64_t>(previous->source_begin) +
          static_cast<uint64_t>(previous->window_count) *
              manifest.stride;
      if (expected_source_begin != shard.source_begin) {
        throw std::runtime_error(
            "sharded index has a missing or duplicate window range");
      }
    }
    if (shard.window_count >
            std::numeric_limits<size_t>::max() - total_windows ||
        shard.sequence_count >
            std::numeric_limits<size_t>::max() - total_sequences ||
        shard.world_node_count >
            std::numeric_limits<size_t>::max() - total_nodes) {
      throw std::runtime_error(
          "sharded index aggregate counts overflow");
    }
    total_windows += shard.window_count;
    total_sequences += shard.sequence_count;
    total_nodes += shard.world_node_count;
    previous = &shard;
  }
  if (total_windows != manifest.total_window_count ||
      total_sequences != manifest.total_sequence_count ||
      total_nodes != manifest.total_world_node_count) {
    throw std::runtime_error(
        "sharded index aggregate counts are inconsistent");
  }
}

}  // namespace

bool is_sharded_index(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  return in && magic == kShardMagic;
}

void save_sharded_index_manifest(
    const std::string& path,
    const ShardedIndexManifest& manifest) {
  validate_manifest(manifest);
  const std::filesystem::path output(path);
  const std::filesystem::path temporary =
      output.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary);
    if (!out) {
      throw std::runtime_error(
          "unable to open sharded index manifest output: " + path);
    }
    out.write(kShardMagic.data(),
              static_cast<std::streamsize>(kShardMagic.size()));
    write_pod<uint32_t>(out, kShardFormatVersion);
    write_pod<uint64_t>(out, manifest.window_length);
    write_pod<uint64_t>(out, manifest.stride);
    write_pod<uint64_t>(out, manifest.total_window_count);
    write_pod<uint64_t>(out, manifest.total_sequence_count);
    write_pod<uint64_t>(out, manifest.total_world_node_count);
    write_string(out, manifest.part_signature);
    write_pod<uint64_t>(out, manifest.shards.size());
    for (const auto& shard : manifest.shards) {
      write_string(out, shard.path);
      write_string(out, shard.ref_id);
      write_pod<uint32_t>(out, shard.source_begin);
      write_pod<uint32_t>(out, shard.source_end);
      write_pod<uint64_t>(out, shard.window_count);
      write_pod<uint64_t>(out, shard.sequence_count);
      write_pod<uint64_t>(out, shard.world_node_count);
    }
    write_pod<uint64_t>(out, manifest_checksum(manifest));
    out.close();
    if (!out) {
      throw std::runtime_error(
          "failed to finalize sharded index manifest: " + path);
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, output, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error(
        "unable to install sharded index manifest: " + error.message());
  }
}

ShardedIndexManifest read_sharded_index_manifest(
    const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(
        "unable to open sharded index manifest: " + path);
  }
  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!in || magic != kShardMagic) {
    throw std::runtime_error("invalid NavigaMer sharded index magic");
  }
  ShardedIndexManifest manifest;
  manifest.format_version =
      read_pod<uint32_t>(in, "format_version");
  manifest.window_length = read_size(in, "window_length");
  manifest.stride = read_size(in, "stride");
  manifest.total_window_count =
      read_size(in, "total_window_count");
  manifest.total_sequence_count =
      read_size(in, "total_sequence_count");
  manifest.total_world_node_count =
      read_size(in, "total_world_node_count");
  manifest.part_signature =
      read_string(in, "part_signature");
  const uint64_t shard_count =
      read_pod<uint64_t>(in, "shard_count");
  if (shard_count == 0 || shard_count > kMaxShardCount) {
    throw std::runtime_error("invalid sharded index shard count");
  }
  manifest.shards.reserve(static_cast<size_t>(shard_count));
  for (uint64_t shard_idx = 0; shard_idx < shard_count;
       ++shard_idx) {
    IndexShardDescriptor shard;
    shard.path = read_string(in, "shard.path");
    shard.ref_id = read_string(in, "shard.ref_id");
    shard.source_begin =
        read_pod<uint32_t>(in, "shard.source_begin");
    shard.source_end =
        read_pod<uint32_t>(in, "shard.source_end");
    shard.window_count = read_size(in, "shard.window_count");
    shard.sequence_count =
        read_size(in, "shard.sequence_count");
    shard.world_node_count =
        read_size(in, "shard.world_node_count");
    manifest.shards.push_back(std::move(shard));
  }
  const uint64_t stored_checksum =
      read_pod<uint64_t>(in, "manifest_checksum");
  if (stored_checksum != manifest_checksum(manifest)) {
    throw std::runtime_error(
        "sharded index manifest checksum mismatch");
  }
  if (in.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error(
        "sharded index manifest contains trailing data");
  }
  validate_manifest(manifest);
  return manifest;
}

std::string resolve_index_shard_path(
    const std::string& manifest_path,
    const std::string& shard_path) {
  const std::filesystem::path part(shard_path);
  if (part.is_absolute()) return part.string();
  return (std::filesystem::path(manifest_path).parent_path() /
          part).string();
}

ShardedIndexManifest build_sharded_reference_index(
    const std::string& bundle_path,
    const std::string& ref_input,
    const std::string& reference_id,
    const std::string& reference_sequence,
    const std::vector<ReferenceContig>& reference_contigs,
    size_t window_length,
    size_t stride,
    size_t max_shard_windows,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config,
    size_t build_jobs) {
  if (bundle_path.empty()) {
    throw std::invalid_argument(
        "sharded index output path must not be empty");
  }
  if (window_length == 0 || stride == 0 ||
      max_shard_windows == 0) {
    throw std::invalid_argument(
        "sharded index sizes must be positive");
  }
  if (window_length >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      stride >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "sharded index window or stride exceeds manifest storage");
  }
  if (reference_sequence.empty() || reference_contigs.empty()) {
    throw std::invalid_argument(
        "sharded index reference must not be empty");
  }

  const std::filesystem::path bundle(bundle_path);
  const IndexBuildManifest part_manifest =
      make_reference_window_index_manifest(
          ref_input, reference_sequence.size(),
          static_cast<int>(window_length),
          static_cast<int>(stride), hierarchy, range_config);

  std::vector<ShardBuildSpec> specs;
  size_t shard_ordinal = 0;
  uint32_t expected_contig_begin = 0;
  for (const auto& contig : reference_contigs) {
    if (contig.id.empty() || contig.begin != expected_contig_begin ||
        contig.end < contig.begin ||
        contig.end > reference_sequence.size()) {
      throw std::invalid_argument(
          "reference contigs must be contiguous and in bounds");
    }
    expected_contig_begin = contig.end;
    const size_t contig_length =
        static_cast<size_t>(contig.end - contig.begin);
    if (contig_length < window_length) continue;
    const size_t contig_window_count =
        1 + (contig_length - window_length) / stride;
    for (size_t first_window = 0;
         first_window < contig_window_count;
         first_window += max_shard_windows) {
      const size_t shard_window_count =
          std::min(max_shard_windows,
                   contig_window_count - first_window);
      const size_t slice_begin =
          static_cast<size_t>(contig.begin) +
          first_window * stride;
      const size_t slice_end =
          slice_begin + (shard_window_count - 1) * stride +
          window_length;
      const size_t source_begin =
          static_cast<size_t>(contig.source_begin) +
          slice_begin - contig.begin;
      const size_t source_end =
          source_begin + slice_end - slice_begin;
      if (source_end > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            "reference shard coordinate exceeds 32-bit storage");
      }

      specs.push_back({
          shard_output_path(bundle, shard_ordinal), contig.id,
          slice_begin, slice_end,
          static_cast<uint32_t>(source_begin),
          static_cast<uint32_t>(source_end), shard_window_count});
      ++shard_ordinal;
    }
  }
  if (expected_contig_begin != reference_sequence.size()) {
    throw std::invalid_argument(
        "reference contigs do not cover the reference");
  }
  if (specs.empty()) {
    throw std::invalid_argument(
        "sharded index reference contains no complete windows");
  }

  const size_t available_threads = static_cast<size_t>(
      std::max(1, omp_get_max_threads()));
  // A few concurrently built parts expose the mostly serial sketch phase
  // without taking one full build peak per hardware thread. Leave each part a
  // share of the thread budget for its parallel rebinding/attachment phases.
  const size_t automatic_jobs = std::max<size_t>(
      1, std::min<size_t>(
             4, static_cast<size_t>(
                    std::sqrt(static_cast<double>(available_threads)))));
  const size_t requested_jobs =
      build_jobs == 0 ? automatic_jobs : build_jobs;
  const size_t job_count_size = std::min(
      {requested_jobs, specs.size(), available_threads});
  if (job_count_size >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "sharded index build job count is too large");
  }
  const int job_count = static_cast<int>(job_count_size);
  const int threads_per_job = static_cast<int>(
      std::max<size_t>(1, available_threads / job_count_size));

  std::vector<IndexShardDescriptor> descriptors(specs.size());
  std::vector<std::exception_ptr> errors(specs.size());
  const auto build_one = [&](size_t spec_idx) {
    try {
      const auto& spec = specs[spec_idx];
      std::string slice = reference_sequence.substr(
          spec.slice_begin, spec.slice_end - spec.slice_begin);
      ReferenceContig slice_contig{
          spec.ref_id, 0, static_cast<uint32_t>(slice.size()),
          spec.source_begin};

      LoadedIndex reusable;
      const bool reused = load_reusable_shard(
          spec.part_path, part_manifest, reference_id, slice,
          slice_contig, window_length, &reusable);

      size_t sequence_count = 0;
      size_t world_node_count = 0;
      if (reused) {
        sequence_count = reusable.builder.num_sequences();
        world_node_count = reusable.builder.num_world_nodes();
      } else {
        BuildRangeConfig shard_range_config = range_config;
        if (job_count > 1) {
          shard_range_config.progress_interval_seconds = 0;
        }
        BioGeometryIndexBuilder builder(hierarchy, shard_range_config);
        builder.build_reference_windows(
            reference_id, std::move(slice), window_length, stride,
            {slice_contig});
        install_shard_atomically(
            spec.part_path, builder, part_manifest);
        sequence_count = builder.num_sequences();
        world_node_count = builder.num_world_nodes();
      }

      descriptors[spec_idx] = {
          spec.part_path.filename().string(), spec.ref_id,
          spec.source_begin, spec.source_end, spec.window_count,
          sequence_count, world_node_count};
    } catch (...) {
      errors[spec_idx] = std::current_exception();
    }
  };

  if (job_count == 1) {
    for (size_t spec_idx = 0; spec_idx < specs.size(); ++spec_idx) {
      build_one(spec_idx);
    }
  } else {
    const int previous_active_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(std::max(2, previous_active_levels));
#pragma omp parallel num_threads(job_count)
    {
      // Bound the product of concurrent shards and each builder's nested team
      // by the original OpenMP thread budget.
      const int previous_nested_threads = omp_get_max_threads();
      omp_set_num_threads(threads_per_job);
#pragma omp for schedule(dynamic, 1)
      for (size_t spec_idx = 0; spec_idx < specs.size(); ++spec_idx) {
        build_one(spec_idx);
      }
      omp_set_num_threads(previous_nested_threads);
    }
    omp_set_max_active_levels(previous_active_levels);
  }
  for (const auto& error : errors) {
    if (error) std::rethrow_exception(error);
  }

  ShardedIndexManifest manifest;
  manifest.window_length = window_length;
  manifest.stride = stride;
  manifest.part_signature = part_manifest.signature;
  manifest.shards.reserve(descriptors.size());
  for (auto& descriptor : descriptors) {
    manifest.total_window_count += descriptor.window_count;
    manifest.total_sequence_count += descriptor.sequence_count;
    manifest.total_world_node_count += descriptor.world_node_count;
    manifest.shards.push_back(std::move(descriptor));
  }
  save_sharded_index_manifest(bundle_path, manifest);
  return manifest;
}

std::vector<LoadedIndex> load_sharded_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest) {
  validate_manifest(manifest);
  std::vector<LoadedIndex> loaded;
  loaded.reserve(manifest.shards.size());
  for (const auto& descriptor : manifest.shards) {
    loaded.push_back(load_index(
        resolve_index_shard_path(
            manifest_path, descriptor.path),
        IndexLoadValidation::Structural));
    const auto& store = loaded.back().builder.sequence_store();
    if (!store.reference_backed ||
        loaded.back().manifest.signature !=
            manifest.part_signature ||
        store.fixed_sequence_length != manifest.window_length ||
        store.reference_contigs.size() != 1) {
      throw std::runtime_error(
          "sharded index part has incompatible sequence storage");
    }
    const auto& contig = store.reference_contigs.front();
    const size_t source_end =
        static_cast<size_t>(contig.source_begin) +
        contig.end - contig.begin;
    if (contig.id != descriptor.ref_id ||
        contig.source_begin != descriptor.source_begin ||
        source_end != descriptor.source_end ||
        store.size() != descriptor.sequence_count ||
        loaded.back().builder.num_world_nodes() !=
            descriptor.world_node_count) {
      throw std::runtime_error(
          "sharded index part does not match its descriptor");
    }
  }
  return loaded;
}

}  // namespace navigamer
