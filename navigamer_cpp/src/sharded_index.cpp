#include "sharded_index.hpp"
#include "io_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace navigamer {

namespace {

constexpr std::array<char, 8> kShardMagic = {
    'N', 'G', 'S', 'H', 'R', 'D', '1', '3'};
constexpr std::array<char, 8> kShardPackMagic = {
    'N', 'G', 'P', 'A', 'C', 'K', '1', '2'};
constexpr std::array<char, 8> kRouterMagic = {
    'N', 'G', 'R', 'O', 'U', 'T', '0', '4'};
constexpr uint32_t kShardFormatVersion = 19;
constexpr uint32_t kShardPackFormatVersion = 12;
constexpr uint32_t kRouterFormatVersion = 5;
constexpr size_t kRouterHeaderBytes = 80;
constexpr std::streamoff kRouterCodePayloadSizeOffset = 60;
constexpr std::streamoff kRouterChecksumOffset = 68;
constexpr uint32_t kRouterK = 16;
// A 150 bp reference hit at d=5 can lose five bases, leaving six 24 bp
// pigeonhole blocks. Keeping the router window at this exact floor routes
// those lossless indel cases without materializing every k-mer in reference.
constexpr uint32_t kRouterWindow = 24;
constexpr uint32_t kRouterCodeBlockSize = 16;
static_assert(
    static_cast<uint64_t>(kRouterCodeGroupsPerSupergroup) *
        kRouterCodeBlocksPerGroup * (kRouterCodeBlockSize - 1) *
        sizeof(uint32_t) <=
    std::numeric_limits<uint16_t>::max(),
    "router supergroup offsets must fit uint16_t");
constexpr uint64_t kMaxShardCount =
    std::numeric_limits<uint32_t>::max();
constexpr uint64_t kMaxStringLength = uint64_t{1} << 30;
constexpr size_t kShardsPerPack = 1024;
constexpr uint64_t kShardPackAlignment = 64;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

class MappedRouterFile {
 public:
  MappedRouterFile(void* address, size_t size)
      : address_(address), size_(size) {}
  ~MappedRouterFile() {
#if defined(__unix__) || defined(__APPLE__)
    if (address_ && size_ != 0) munmap(address_, size_);
#endif
  }
  const uint8_t* data() const {
    return static_cast<const uint8_t*>(address_);
  }
  size_t size() const { return size_; }

 private:
  void* address_ = nullptr;
  size_t size_ = 0;
};

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

void hash_build_manifest(
    uint64_t* hash, const IndexBuildManifest& manifest) {
  std::ostringstream encoded(std::ios::out | std::ios::binary);
  write_index_build_manifest(encoded, manifest);
  const std::string bytes = encoded.str();
  hash_pod<uint64_t>(hash, bytes.size());
  hash_bytes(hash, bytes.data(), bytes.size());
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
  hash_pod<uint32_t>(&hash, manifest.router_k);
  hash_pod<uint32_t>(&hash, manifest.router_window);
  hash_pod<uint64_t>(&hash, manifest.router_entry_count);
  hash_pod<uint64_t>(&hash, manifest.router_checksum);
  hash_build_manifest(&hash, manifest.part_manifest);
  hash_pod<uint64_t>(&hash, manifest.pack_paths.size());
  for (const auto& path : manifest.pack_paths) {
    hash_string(&hash, path);
  }
  hash_pod<uint64_t>(&hash, manifest.contig_ids.size());
  for (const auto& id : manifest.contig_ids) {
    hash_string(&hash, id);
  }
  hash_pod<uint64_t>(&hash, manifest.shards.size());
  for (const auto& shard : manifest.shards) {
    hash_pod<uint32_t>(&hash, shard.pack_id);
    hash_pod<uint64_t>(&hash, shard.file_offset);
    hash_pod<uint64_t>(&hash, shard.file_size);
    hash_pod<uint32_t>(&hash, shard.contig_id);
    hash_pod<uint32_t>(&hash, shard.source_begin);
    hash_pod<uint64_t>(&hash, shard.window_count);
    hash_pod<uint64_t>(&hash, shard.sequence_count);
    hash_pod<uint64_t>(&hash, shard.world_node_count);
  }
  return hash;
}

std::filesystem::path router_output_path(
    const std::filesystem::path& bundle_path) {
  return bundle_path.string() + ".route";
}

uint32_t required_shard_id_bits(uint32_t shard_count) {
  if (shard_count == 0) return 0;
  uint32_t bits = 0;
  uint32_t largest_id = shard_count - 1;
  do {
    ++bits;
    largest_id >>= 1;
  } while (largest_id != 0);
  return bits;
}

size_t packed_shard_byte_count(
    size_t entry_count, uint32_t shard_id_bits) {
  if (shard_id_bits == 0 || shard_id_bits > 32 ||
      entry_count >
          (std::numeric_limits<size_t>::max() - 7) /
              shard_id_bits) {
    throw std::runtime_error("shard router bit size overflow");
  }
  return (entry_count * shard_id_bits + 7) / 8;
}

size_t checked_router_add(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::runtime_error("shard router size overflow");
  }
  return left + right;
}

size_t checked_router_multiply(size_t left, size_t right) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error("shard router size overflow");
  }
  return left * right;
}

size_t align_router_size(size_t value, size_t alignment) {
  const size_t mask = alignment - 1;
  if (value > std::numeric_limits<size_t>::max() - mask) {
    throw std::runtime_error("shard router size overflow");
  }
  return (value + mask) & ~mask;
}

struct RouterStorageLayout {
  size_t code_block_count = 0;
  size_t code_group_count = 0;
  size_t code_supergroup_count = 0;
  size_t bases_begin = 0;
  size_t widths_begin = 0;
  size_t group_offsets_begin = 0;
  size_t supergroup_offsets_begin = 0;
  size_t payload_begin = 0;
  size_t shard_ids_begin = 0;
  size_t total_size = 0;
};

RouterStorageLayout router_storage_layout(
    size_t entry_count, uint64_t payload_size,
    size_t packed_shard_ids_size) {
  if (entry_count == 0 ||
      payload_size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("shard router size overflow");
  }
  RouterStorageLayout layout;
  layout.code_block_count =
      (entry_count + kRouterCodeBlockSize - 1) / kRouterCodeBlockSize;
  layout.code_group_count =
      (layout.code_block_count + kRouterCodeBlocksPerGroup - 1) /
      kRouterCodeBlocksPerGroup;
  layout.code_supergroup_count =
      (layout.code_group_count + kRouterCodeGroupsPerSupergroup - 1) /
      kRouterCodeGroupsPerSupergroup;
  layout.bases_begin = kRouterHeaderBytes;
  layout.widths_begin = checked_router_add(
      layout.bases_begin,
      checked_router_multiply(layout.code_block_count, sizeof(uint32_t)));
  layout.group_offsets_begin = align_router_size(
      checked_router_add(layout.widths_begin, layout.code_block_count),
      alignof(uint16_t));
  layout.supergroup_offsets_begin = align_router_size(
      checked_router_add(
          layout.group_offsets_begin,
          checked_router_multiply(
              layout.code_group_count, sizeof(uint16_t))),
      alignof(uint64_t));
  layout.payload_begin = checked_router_add(
      layout.supergroup_offsets_begin,
      checked_router_multiply(
          layout.code_supergroup_count, sizeof(uint64_t)));
  layout.shard_ids_begin = checked_router_add(
      layout.payload_begin, static_cast<size_t>(payload_size));
  layout.total_size = checked_router_add(
      layout.shard_ids_begin, packed_shard_ids_size);
  return layout;
}

uint8_t minimizer_delta_width(uint32_t delta) {
  uint8_t width = 1;
  while (delta >>= 1) ++width;
  return width;
}

uint64_t begin_router_checksum(
    uint32_t k, uint32_t window, uint32_t shard_count,
    uint32_t shard_id_bits, size_t entry_count) {
  uint64_t hash = kFnvOffset;
  hash_pod<uint32_t>(&hash, kRouterFormatVersion);
  hash_pod<uint32_t>(&hash, k);
  hash_pod<uint32_t>(&hash, window);
  hash_pod<uint32_t>(&hash, shard_count);
  hash_pod<uint32_t>(&hash, shard_id_bits);
  hash_pod<uint64_t>(&hash, entry_count);
  return hash;
}

int dna_code(char base) {
  switch (base) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default: return -1;
  }
}

void reference_minimizers(
    std::string_view sequence, uint32_t k, uint32_t window,
    std::vector<uint32_t>* minimizers) {
  if (!minimizers) {
    throw std::invalid_argument("router minimizer output must not be null");
  }
  minimizers->clear();
  if (k == 0 || k > 16 || window < k ||
      sequence.size() < window) {
    return;
  }
  const size_t qmers_per_window = window - k + 1;
  const uint64_t mask =
      k == 16 ? UINT32_MAX : ((uint64_t{1} << (2 * k)) - 1);
  std::deque<std::pair<size_t, uint32_t>> minimum_queue;
  uint64_t code = 0;
  size_t valid_bases = 0;
  for (size_t pos = 0; pos < sequence.size(); ++pos) {
    const int value = dna_code(sequence[pos]);
    if (value < 0) {
      code = 0;
      valid_bases = 0;
      minimum_queue.clear();
      continue;
    }
    code = ((code << 2) | static_cast<uint64_t>(value)) & mask;
    ++valid_bases;
    if (valid_bases < k) continue;
    const size_t qmer_start = pos + 1 - k;
    const uint32_t qmer_code = static_cast<uint32_t>(code);
    while (!minimum_queue.empty() &&
           minimum_queue.back().second >= qmer_code) {
      minimum_queue.pop_back();
    }
    minimum_queue.emplace_back(qmer_start, qmer_code);
    const size_t first_qmer =
        qmer_start + 1 >= qmers_per_window
            ? qmer_start + 1 - qmers_per_window
            : 0;
    while (!minimum_queue.empty() &&
           minimum_queue.front().first < first_qmer) {
      minimum_queue.pop_front();
    }
    if (valid_bases >= window &&
        (minimizers->empty() ||
         minimizers->back() != minimum_queue.front().second)) {
      minimizers->push_back(minimum_queue.front().second);
    }
  }
  std::sort(minimizers->begin(), minimizers->end());
  minimizers->erase(
      std::unique(minimizers->begin(), minimizers->end()),
      minimizers->end());
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

std::filesystem::path shard_pack_output_path(
    const std::filesystem::path& bundle_path,
    size_t pack_ordinal) {
  std::ostringstream name;
  name << bundle_path.filename().string()
       << ".pack" << std::setw(6) << std::setfill('0')
       << pack_ordinal << ".navpack";
  return bundle_path.parent_path() / name.str();
}

struct RouterBuildData {
  explicit RouterBuildData(std::filesystem::path path)
      : spool_path(std::move(path)) {}
  RouterBuildData(const RouterBuildData&) = delete;
  RouterBuildData& operator=(const RouterBuildData&) = delete;
  RouterBuildData(RouterBuildData&& other) noexcept
      : spool_path(std::move(other.spool_path)),
        shard_counts(std::move(other.shard_counts)),
        spool_size(other.spool_size),
        entry_count(other.entry_count) {
    other.spool_path.clear();
  }
  ~RouterBuildData() {
    if (!spool_path.empty()) {
      std::error_code ignored;
      std::filesystem::remove(spool_path, ignored);
    }
  }

  std::filesystem::path spool_path;
  std::vector<uint32_t> shard_counts;
  size_t spool_size = 0;
  size_t entry_count = 0;
};

bool reusable_shard_matches(
    const LoadedIndex& candidate,
    const std::string& expected_signature,
    const std::string& reference_id,
    const std::string& reference_slice,
    const ReferenceContig& expected_contig,
    size_t window_length) {
  if (candidate.manifest.signature != expected_signature) {
    return false;
  }
  const auto& store = candidate.builder.sequence_store();
  if (!store.reference_backed ||
      store.reference_id != reference_id ||
      store.fixed_sequence_length != window_length ||
      store.reference_view() != reference_slice ||
      store.reference_contigs.size() != 1) {
    return false;
  }
  const auto& contig = store.reference_contigs.front();
  return contig.id == expected_contig.id &&
         contig.begin == expected_contig.begin &&
         contig.end == expected_contig.end &&
         contig.source_begin == expected_contig.source_begin;
}

bool load_reusable_shard(
    const std::filesystem::path& part_path,
    const IndexBuildManifest& expected_manifest,
    const std::string& reference_id,
    const std::string& reference_slice,
    const ReferenceContig& expected_contig,
    size_t window_length,
    LoadedIndex* loaded) {
  try {
    LoadedIndex candidate = load_index_payload(
        part_path.string(), expected_manifest);
    if (!reusable_shard_matches(
            candidate, expected_manifest.signature, reference_id,
            reference_slice, expected_contig, window_length)) {
      return false;
    }
    *loaded = std::move(candidate);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

struct ShardPackEntry {
  uint64_t offset = 0;
  uint64_t size = 0;
};

uint64_t align_shard_pack_offset(uint64_t offset) {
  const uint64_t remainder = offset % kShardPackAlignment;
  if (remainder == 0) return offset;
  const uint64_t padding = kShardPackAlignment - remainder;
  if (offset > std::numeric_limits<uint64_t>::max() - padding) {
    throw std::runtime_error("shard pack offset overflow");
  }
  return offset + padding;
}

std::vector<ShardPackEntry> read_shard_pack_directory(
    const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("unable to open shard pack");
  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!in || magic != kShardPackMagic) {
    throw std::runtime_error("invalid shard pack magic");
  }
  const uint32_t version = read_pod<uint32_t>(in, "pack.version");
  const uint32_t count = read_pod<uint32_t>(in, "pack.count");
  if (version != kShardPackFormatVersion || count == 0 ||
      count > kShardsPerPack) {
    throw std::runtime_error("invalid shard pack directory");
  }
  std::vector<ShardPackEntry> entries(count);
  for (auto& entry : entries) {
    entry.offset = read_pod<uint64_t>(in, "pack.offset");
    entry.size = read_pod<uint64_t>(in, "pack.size");
  }
  std::error_code error;
  const uint64_t file_size = std::filesystem::file_size(path, error);
  if (error) throw std::runtime_error("unable to stat shard pack");
  uint64_t expected_offset = align_shard_pack_offset(
      16 + static_cast<uint64_t>(entries.size()) * 16);
  for (const auto& entry : entries) {
    if (entry.offset != expected_offset || entry.size == 0 ||
        entry.offset > file_size ||
        entry.size > file_size - entry.offset) {
      throw std::runtime_error("invalid shard pack range");
    }
    expected_offset = align_shard_pack_offset(entry.offset + entry.size);
  }
  if (entries.back().offset + entries.back().size != file_size) {
    throw std::runtime_error("shard pack has truncated or trailing data");
  }
  return entries;
}

uint64_t save_router_sidecar(
    const std::filesystem::path& path,
    uint32_t k, uint32_t window, uint32_t shard_count,
    RouterBuildData& data) {
  if (data.shard_counts.size() != shard_count ||
      data.entry_count == 0 || data.spool_size == 0) {
    throw std::runtime_error("invalid shard router build data");
  }
  const std::filesystem::path temporary = path.string() + ".tmp";
  const std::filesystem::path packed_temporary =
      path.string() + ".packed.tmp";
  const uint32_t shard_id_bits =
      required_shard_id_bits(shard_count);
  const size_t packed_size =
      packed_shard_byte_count(data.entry_count, shard_id_bits);
  const RouterStorageLayout layout = router_storage_layout(
      data.entry_count, 0, packed_size);
  uint64_t checksum = begin_router_checksum(
      k, window, shard_count, shard_id_bits, data.entry_count);
  try {
#if defined(__unix__) || defined(__APPLE__)
    const int spool_fd = open(data.spool_path.c_str(), O_RDONLY);
    if (spool_fd < 0) {
      throw std::runtime_error("unable to open shard router code spool");
    }
    struct stat spool_status {};
    if (fstat(spool_fd, &spool_status) != 0 ||
        spool_status.st_size < 0 ||
        static_cast<uint64_t>(spool_status.st_size) !=
            data.spool_size) {
      close(spool_fd);
      throw std::runtime_error("invalid shard router code spool size");
    }
    void* spool_address = mmap(
        nullptr, data.spool_size, PROT_READ, MAP_PRIVATE,
        spool_fd, 0);
    close(spool_fd);
    if (spool_address == MAP_FAILED) {
      throw std::runtime_error("unable to map shard router code spool");
    }
#if defined(MADV_RANDOM)
    (void)madvise(spool_address, data.spool_size, MADV_RANDOM);
#endif
    auto spool_mapping = std::make_shared<MappedRouterFile>(
        spool_address, data.spool_size);
    const uint8_t* spool_bytes = spool_mapping->data();
#else
    std::ifstream spool_in(data.spool_path, std::ios::binary);
    if (!spool_in) {
      throw std::runtime_error("unable to open shard router code spool");
    }
    std::vector<uint8_t> owned_spool(data.spool_size);
    spool_in.read(
        reinterpret_cast<char*>(owned_spool.data()),
        static_cast<std::streamsize>(owned_spool.size()));
    if (!spool_in ||
        spool_in.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error("invalid shard router code spool");
    }
    const uint8_t* spool_bytes = owned_spool.data();
#endif
    size_t counted_entries = 0;
    size_t nonempty_shard_count = 0;
    for (uint32_t count : data.shard_counts) {
      if (count >
          std::numeric_limits<size_t>::max() - counted_entries) {
        throw std::runtime_error("invalid shard router code range");
      }
      counted_entries += count;
      nonempty_shard_count += count != 0;
    }
    if (counted_entries != data.entry_count ||
        counted_entries >
            std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
        counted_entries * sizeof(uint32_t) != data.spool_size) {
      throw std::runtime_error("invalid shard router code range");
    }
    const auto code_at = [&](size_t entry_index) {
      uint32_t code = 0;
      std::memcpy(
          &code,
          spool_bytes + entry_index * sizeof(uint32_t),
          sizeof(code));
      return code;
    };

    std::ofstream out(temporary, std::ios::binary);
    std::ofstream packed_out(packed_temporary, std::ios::binary);
    if (!out || !packed_out) {
      throw std::runtime_error("unable to open shard router output");
    }
    out.write(kRouterMagic.data(),
              static_cast<std::streamsize>(kRouterMagic.size()));
    write_pod<uint32_t>(out, kRouterFormatVersion);
    write_pod<uint32_t>(out, k);
    write_pod<uint32_t>(out, window);
    write_pod<uint32_t>(out, shard_count);
    write_pod<uint32_t>(out, shard_id_bits);
    write_pod<uint32_t>(out, kRouterCodeBlockSize);
    write_pod<uint32_t>(out, 0);
    write_pod<uint64_t>(out, data.entry_count);
    write_pod<uint64_t>(out, layout.code_block_count);
    write_pod<uint64_t>(out, layout.code_group_count);
    write_pod<uint64_t>(out, 0);
    write_pod<uint64_t>(out, 0);
    write_pod<uint32_t>(out, 0);
    if (layout.payload_begin == 0) {
      throw std::runtime_error("invalid shard router layout");
    }
    out.seekp(static_cast<std::streamoff>(layout.payload_begin - 1),
              std::ios::beg);
    out.put('\0');
    if (!out) {
      throw std::runtime_error("failed to reserve shard router metadata");
    }

    struct Cursor32 {
      uint32_t code = 0;
      uint32_t shard_id = 0;
      uint32_t entry_index = 0;
    };
    static_assert(sizeof(Cursor32) == 12,
                  "32-bit router cursor must remain 12 bytes");
    struct Cursor32Greater {
      bool operator()(const Cursor32& left,
                      const Cursor32& right) const {
        if (left.code != right.code) return left.code > right.code;
        return left.shard_id > right.shard_id;
      }
    };
    struct Cursor64 {
      uint32_t code = 0;
      uint32_t shard_id = 0;
      size_t entry_index = 0;
    };
    struct Cursor64Greater {
      bool operator()(const Cursor64& left,
                      const Cursor64& right) const {
        if (left.code != right.code) return left.code > right.code;
        return left.shard_id > right.shard_id;
      }
    };

    std::vector<uint32_t> code_bases;
    std::vector<uint8_t> code_widths;
    std::vector<uint16_t> code_group_offsets;
    std::vector<uint64_t> code_supergroup_offsets;
    code_bases.reserve(layout.code_block_count);
    code_widths.reserve(layout.code_block_count);
    code_group_offsets.reserve(layout.code_group_count);
    code_supergroup_offsets.reserve(layout.code_supergroup_count);
    std::vector<uint8_t> code_payload_buffer(65536);
    size_t code_payload_buffered = 0;
    size_t code_payload_written = 0;
    const auto flush_code_payload = [&]() {
      if (code_payload_buffered == 0) return;
      out.write(
          reinterpret_cast<const char*>(code_payload_buffer.data()),
          static_cast<std::streamsize>(code_payload_buffered));
      if (!out) {
        throw std::runtime_error("failed to write shard router code payload");
      }
      hash_bytes(&checksum, code_payload_buffer.data(),
                 code_payload_buffered);
      code_payload_written = checked_router_add(
          code_payload_written, code_payload_buffered);
      code_payload_buffered = 0;
    };
    const auto emit_code_payload_byte = [&](uint8_t byte) {
      code_payload_buffer[code_payload_buffered++] = byte;
      if (code_payload_buffered == code_payload_buffer.size()) {
        flush_code_payload();
      }
    };
    std::array<uint32_t, kRouterCodeBlockSize> code_block{};
    size_t code_block_size = 0;
    const auto flush_code_block = [&]() {
      if (code_block_size == 0) return;
      const uint32_t base = code_block[0];
      uint32_t maximum_delta = 0;
      for (size_t idx = 1; idx < code_block_size; ++idx) {
        maximum_delta = std::max(
            maximum_delta, code_block[idx] - code_block[idx - 1]);
      }
      const uint8_t width = minimizer_delta_width(maximum_delta);
      if (code_bases.size() % kRouterCodeBlocksPerGroup == 0) {
        const uint64_t payload_offset = checked_router_add(
            code_payload_written, code_payload_buffered);
        if (code_group_offsets.size() % kRouterCodeGroupsPerSupergroup ==
            0) {
          code_supergroup_offsets.push_back(payload_offset);
        }
        const uint64_t supergroup_offset =
            code_supergroup_offsets.back();
        if (payload_offset < supergroup_offset ||
            payload_offset - supergroup_offset >
                std::numeric_limits<uint16_t>::max()) {
          throw std::runtime_error(
              "shard router compact group offset overflow");
        }
        code_group_offsets.push_back(static_cast<uint16_t>(
            payload_offset - supergroup_offset));
      }
      code_bases.push_back(base);
      code_widths.push_back(width);
      uint64_t pending_bits = 0;
      uint32_t pending_bit_count = 0;
      for (size_t idx = 1; idx < code_block_size; ++idx) {
        const uint32_t delta = code_block[idx] - code_block[idx - 1];
        pending_bits |= static_cast<uint64_t>(delta) << pending_bit_count;
        pending_bit_count += width;
        while (pending_bit_count >= 8) {
          emit_code_payload_byte(static_cast<uint8_t>(pending_bits));
          pending_bits >>= 8;
          pending_bit_count -= 8;
        }
      }
      if (pending_bit_count != 0) {
        emit_code_payload_byte(static_cast<uint8_t>(pending_bits));
      }
      code_block_size = 0;
    };
    std::vector<uint8_t> shard_buffer(
        std::min<size_t>(65536, packed_size));
    size_t shard_buffered = 0;
    size_t packed_written = 0;
    const auto flush_shards = [&]() {
      if (shard_buffered == 0) return;
      packed_out.write(
          reinterpret_cast<const char*>(shard_buffer.data()),
          static_cast<std::streamsize>(shard_buffered));
      packed_written += shard_buffered;
      shard_buffered = 0;
    };
    const auto emit_packed_byte = [&](uint8_t byte) {
      shard_buffer[shard_buffered++] = byte;
      if (shard_buffered == shard_buffer.size()) flush_shards();
    };
    uint64_t pending_shard_bits = 0;
    uint32_t pending_bit_count = 0;
    size_t emitted = 0;
    const auto emit_router_entry = [&](uint32_t code,
                                       uint32_t shard_id) {
      code_block[code_block_size++] = code;
      if (code_block_size == code_block.size()) flush_code_block();

      pending_shard_bits |=
          static_cast<uint64_t>(shard_id) << pending_bit_count;
      pending_bit_count += shard_id_bits;
      while (pending_bit_count >= 8) {
        emit_packed_byte(static_cast<uint8_t>(pending_shard_bits));
        pending_shard_bits >>= 8;
        pending_bit_count -= 8;
      }
      ++emitted;
    };

    if (data.entry_count <= std::numeric_limits<uint32_t>::max()) {
      // Reuse the count array as exclusive entry ends. Human-scale routers
      // remain below 2^32 entries, so every live heap cursor is only 12 bytes
      // and the hot merge loop performs no per-entry metadata writes.
      std::vector<Cursor32> cursor_storage;
      cursor_storage.reserve(nonempty_shard_count);
      std::priority_queue<
          Cursor32, std::vector<Cursor32>, Cursor32Greater> queue(
              Cursor32Greater{}, std::move(cursor_storage));
      uint32_t entry_begin = 0;
      for (uint32_t shard_id = 0; shard_id < shard_count; ++shard_id) {
        const uint32_t count = data.shard_counts[shard_id];
        const uint32_t entry_end = static_cast<uint32_t>(
            static_cast<uint64_t>(entry_begin) + count);
        data.shard_counts[shard_id] = entry_end;
        if (count != 0) {
          queue.push({code_at(entry_begin), shard_id, entry_begin});
        }
        entry_begin = entry_end;
      }
      while (!queue.empty()) {
        const Cursor32 cursor = queue.top();
        queue.pop();
        emit_router_entry(cursor.code, cursor.shard_id);
        const uint32_t next_entry_index = cursor.entry_index + 1;
        if (next_entry_index < data.shard_counts[cursor.shard_id]) {
          queue.push({code_at(next_entry_index), cursor.shard_id,
                      next_entry_index});
        }
      }
    } else {
      // Exceptionally large routers retain size_t entry indices. Counts are
      // consumed in place so this path still needs no offset array.
      std::vector<Cursor64> cursor_storage;
      cursor_storage.reserve(nonempty_shard_count);
      std::priority_queue<
          Cursor64, std::vector<Cursor64>, Cursor64Greater> queue(
              Cursor64Greater{}, std::move(cursor_storage));
      size_t entry_begin = 0;
      for (uint32_t shard_id = 0; shard_id < shard_count; ++shard_id) {
        const uint32_t count = data.shard_counts[shard_id];
        if (count != 0) {
          queue.push({code_at(entry_begin), shard_id, entry_begin});
          data.shard_counts[shard_id] = count - 1;
        }
        entry_begin += count;
      }
      while (!queue.empty()) {
        const Cursor64 cursor = queue.top();
        queue.pop();
        emit_router_entry(cursor.code, cursor.shard_id);
        if (data.shard_counts[cursor.shard_id] != 0) {
          const size_t next_entry_index = cursor.entry_index + 1;
          --data.shard_counts[cursor.shard_id];
          queue.push({code_at(next_entry_index), cursor.shard_id,
                      next_entry_index});
        }
      }
    }
    flush_code_block();
    flush_code_payload();
    if (pending_bit_count != 0) {
      emit_packed_byte(static_cast<uint8_t>(pending_shard_bits));
    }
    flush_shards();
    packed_out.close();
    if (!packed_out || emitted != data.entry_count ||
        packed_written != packed_size) {
      throw std::runtime_error("failed to pack shard router IDs");
    }

    if (code_bases.size() != layout.code_block_count ||
        code_widths.size() != layout.code_block_count ||
        code_group_offsets.size() != layout.code_group_count ||
        code_supergroup_offsets.size() != layout.code_supergroup_count) {
      throw std::runtime_error("invalid compressed shard router code count");
    }
    const RouterStorageLayout finalized_layout = router_storage_layout(
        data.entry_count, code_payload_written, packed_size);
    if (finalized_layout.code_block_count != layout.code_block_count ||
        finalized_layout.code_group_count != layout.code_group_count ||
        finalized_layout.code_supergroup_count !=
            layout.code_supergroup_count ||
        finalized_layout.payload_begin != layout.payload_begin) {
      throw std::runtime_error("invalid finalized shard router layout");
    }
    out.seekp(static_cast<std::streamoff>(layout.bases_begin),
              std::ios::beg);
    out.write(reinterpret_cast<const char*>(code_bases.data()),
              static_cast<std::streamsize>(
                  code_bases.size() * sizeof(code_bases[0])));
    if (!out) throw std::runtime_error("failed to write shard router bases");
    hash_bytes(&checksum, code_bases.data(),
               code_bases.size() * sizeof(code_bases[0]));
    out.write(reinterpret_cast<const char*>(code_widths.data()),
              static_cast<std::streamsize>(code_widths.size()));
    if (!out) throw std::runtime_error("failed to write shard router widths");
    hash_bytes(&checksum, code_widths.data(), code_widths.size());
    out.seekp(static_cast<std::streamoff>(layout.group_offsets_begin),
              std::ios::beg);
    out.write(reinterpret_cast<const char*>(code_group_offsets.data()),
              static_cast<std::streamsize>(
                  code_group_offsets.size() * sizeof(code_group_offsets[0])));
    if (!out) {
      throw std::runtime_error("failed to write shard router group offsets");
    }
    hash_bytes(&checksum, code_group_offsets.data(),
               code_group_offsets.size() * sizeof(code_group_offsets[0]));
    out.seekp(static_cast<std::streamoff>(layout.supergroup_offsets_begin),
              std::ios::beg);
    out.write(reinterpret_cast<const char*>(code_supergroup_offsets.data()),
              static_cast<std::streamsize>(
                  code_supergroup_offsets.size() *
                  sizeof(code_supergroup_offsets[0])));
    if (!out) {
      throw std::runtime_error(
          "failed to write shard router supergroup offsets");
    }
    hash_bytes(&checksum, code_supergroup_offsets.data(),
               code_supergroup_offsets.size() *
                   sizeof(code_supergroup_offsets[0]));

    std::ifstream packed_in(packed_temporary, std::ios::binary);
    if (!packed_in) {
      throw std::runtime_error("unable to reopen packed shard router IDs");
    }
    out.seekp(static_cast<std::streamoff>(finalized_layout.shard_ids_begin),
              std::ios::beg);
    size_t packed_copied = 0;
    while (packed_in) {
      packed_in.read(
          reinterpret_cast<char*>(shard_buffer.data()),
          static_cast<std::streamsize>(shard_buffer.size()));
      const size_t count = static_cast<size_t>(packed_in.gcount());
      if (count != 0) {
        out.write(
            reinterpret_cast<const char*>(shard_buffer.data()),
            static_cast<std::streamsize>(count));
        hash_bytes(&checksum, shard_buffer.data(), count);
        packed_copied += count;
      }
    }
    if (!packed_in.eof() || packed_copied != packed_size) {
      throw std::runtime_error("failed to read packed shard router IDs");
    }
    packed_in.close();
    if (checksum == 0) {
      throw std::runtime_error("invalid zero shard router checksum");
    }
    out.seekp(kRouterCodePayloadSizeOffset, std::ios::beg);
    write_pod<uint64_t>(out, code_payload_written);
    out.seekp(kRouterChecksumOffset, std::ios::beg);
    write_pod<uint64_t>(out, checksum);
    out.close();
    if (!out) {
      throw std::runtime_error("failed to finalize shard router output");
    }
    std::error_code remove_error;
    std::filesystem::remove(packed_temporary, remove_error);
    if (remove_error) {
      throw std::runtime_error(
          "unable to remove temporary shard router IDs: " +
          remove_error.message());
    }
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    ignored.clear();
    std::filesystem::remove(packed_temporary, ignored);
    throw;
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error(
        "unable to install shard router: " + error.message());
  }
  return checksum;
}

template <typename SliceLoader>
RouterBuildData build_router_data(
    const std::vector<IndexShardDescriptor>& descriptors,
    uint32_t k, uint32_t window,
    const std::filesystem::path& spool_path,
    SliceLoader&& load_slice) {
  if (descriptors.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("too many shards for seed router");
  }
  RouterBuildData data(spool_path);
  data.shard_counts.reserve(descriptors.size());
  std::ofstream spool(data.spool_path, std::ios::binary);
  if (!spool) {
    throw std::runtime_error("unable to create shard router code spool");
  }
  size_t spool_position = 0;
  std::vector<uint32_t> minimizer_scratch;
  for (const auto& descriptor : descriptors) {
    // Keep lists contiguous. Their offsets are implicit prefix sums of the
    // counts, avoiding one 64-bit build record per shard.
    const auto slice = load_slice(descriptor);
    reference_minimizers(slice, k, window, &minimizer_scratch);
    if (minimizer_scratch.size() > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(
          "one shard has too many router minimizers");
    }
    if (minimizer_scratch.size() >
        std::numeric_limits<size_t>::max() - data.entry_count) {
      throw std::runtime_error("shard router entry count overflow");
    }
    if (minimizer_scratch.size() >
        (std::numeric_limits<size_t>::max() - spool_position) /
            sizeof(uint32_t)) {
      throw std::runtime_error("shard router spool size overflow");
    }
    data.shard_counts.push_back(
        static_cast<uint32_t>(minimizer_scratch.size()));
    if (!minimizer_scratch.empty()) {
      const size_t byte_count =
          minimizer_scratch.size() * sizeof(uint32_t);
      spool.write(
          reinterpret_cast<const char*>(minimizer_scratch.data()),
          static_cast<std::streamsize>(byte_count));
      spool_position += byte_count;
    }
    data.entry_count += minimizer_scratch.size();
  }
  spool.close();
  if (!spool) {
    throw std::runtime_error("failed to finalize shard router code spool");
  }
  data.spool_size = spool_position;
  return data;
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
  if (manifest.part_manifest.format_version != 64 ||
      manifest.part_manifest.signature.empty() ||
      manifest.part_manifest.sequence_count != 0 ||
      manifest.part_manifest.world_node_count != 0 ||
      manifest.part_manifest.edge_count != 0 ||
      manifest.part_manifest.leaf_link_count != 0 ||
      manifest.pack_paths.empty() || manifest.contig_ids.empty() ||
      manifest.shards.empty()) {
    throw std::runtime_error("sharded index contains no shards");
  }
  if (manifest.pack_paths.size() > manifest.shards.size()) {
    throw std::runtime_error("sharded index has too many pack files");
  }
  for (const auto& path : manifest.pack_paths) {
    if (path.empty()) {
      throw std::runtime_error("sharded index has an empty pack path");
    }
  }
  for (const auto& id : manifest.contig_ids) {
    if (id.empty()) {
      throw std::runtime_error("sharded index has an empty contig ID");
    }
  }
  if (manifest.shards.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("sharded index has too many router targets");
  }
  const bool has_router = manifest.router_entry_count != 0;
  if (has_router) {
    if (manifest.router_k == 0 || manifest.router_k > 16 ||
        manifest.router_window < manifest.router_k ||
        manifest.router_window > manifest.window_length ||
        manifest.router_checksum == 0) {
      throw std::runtime_error(
          "sharded index has invalid seed router metadata");
    }
  } else if (manifest.router_k != 0 || manifest.router_window != 0 ||
             manifest.router_checksum != 0) {
    throw std::runtime_error(
        "sharded index has inconsistent empty router metadata");
  }
  size_t total_windows = 0;
  size_t total_sequences = 0;
  size_t total_nodes = 0;
  const IndexShardDescriptor* previous = nullptr;
  std::vector<uint64_t> pack_ends(manifest.pack_paths.size(), 0);
  std::vector<uint8_t> pack_referenced(manifest.pack_paths.size(), 0);
  for (const auto& shard : manifest.shards) {
    if (shard.pack_id >= manifest.pack_paths.size() ||
        shard.contig_id >= manifest.contig_ids.size() ||
        shard.file_offset % kShardPackAlignment != 0 ||
        shard.file_size == 0 ||
        shard.window_count == 0) {
      throw std::runtime_error(
          "sharded index contains an invalid shard descriptor");
    }
    if (shard.file_offset < pack_ends[shard.pack_id] ||
        shard.file_size >
            std::numeric_limits<uint64_t>::max() - shard.file_offset) {
      throw std::runtime_error(
          "sharded index contains overlapping pack ranges");
    }
    pack_ends[shard.pack_id] = shard.file_offset + shard.file_size;
    pack_referenced[shard.pack_id] = 1;
    const uint64_t expected_source_end =
        static_cast<uint64_t>(shard.source_begin) +
        (static_cast<uint64_t>(shard.window_count) - 1) *
            manifest.stride +
        manifest.window_length;
    if (expected_source_end > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(
          "sharded index has inconsistent shard coordinates");
    }
    if (previous && previous->contig_id == shard.contig_id) {
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
  if (std::find(pack_referenced.begin(), pack_referenced.end(), 0) !=
      pack_referenced.end()) {
    throw std::runtime_error("sharded index contains an unused pack file");
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
    write_pod<uint32_t>(out, manifest.router_k);
    write_pod<uint32_t>(out, manifest.router_window);
    write_pod<uint64_t>(out, manifest.router_entry_count);
    write_pod<uint64_t>(out, manifest.router_checksum);
    write_index_build_manifest(out, manifest.part_manifest);
    write_pod<uint64_t>(out, manifest.pack_paths.size());
    for (const auto& pack_path : manifest.pack_paths) {
      write_string(out, pack_path);
    }
    write_pod<uint64_t>(out, manifest.contig_ids.size());
    for (const auto& contig_id : manifest.contig_ids) {
      write_string(out, contig_id);
    }
    write_pod<uint64_t>(out, manifest.shards.size());
    for (const auto& shard : manifest.shards) {
      write_pod<uint32_t>(out, shard.pack_id);
      write_pod<uint64_t>(out, shard.file_offset);
      write_pod<uint64_t>(out, shard.file_size);
      write_pod<uint32_t>(out, shard.contig_id);
      write_pod<uint32_t>(out, shard.source_begin);
      write_pod<uint32_t>(out, shard.window_count);
      write_pod<uint32_t>(out, shard.sequence_count);
      write_pod<uint32_t>(out, shard.world_node_count);
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
  manifest.router_k =
      read_pod<uint32_t>(in, "router_k");
  manifest.router_window =
      read_pod<uint32_t>(in, "router_window");
  manifest.router_entry_count =
      read_size(in, "router_entry_count");
  manifest.router_checksum =
      read_pod<uint64_t>(in, "router_checksum");
  manifest.part_manifest = read_index_build_manifest(in);
  const uint64_t pack_count =
      read_pod<uint64_t>(in, "pack_count");
  if (pack_count == 0 || pack_count > kMaxShardCount) {
    throw std::runtime_error("invalid sharded index pack count");
  }
  manifest.pack_paths.reserve(static_cast<size_t>(pack_count));
  for (uint64_t pack_idx = 0; pack_idx < pack_count; ++pack_idx) {
    manifest.pack_paths.push_back(
        read_string(in, "pack.path"));
  }
  const uint64_t contig_count =
      read_pod<uint64_t>(in, "contig_count");
  if (contig_count == 0 || contig_count > kMaxShardCount) {
    throw std::runtime_error("invalid sharded index contig count");
  }
  manifest.contig_ids.reserve(static_cast<size_t>(contig_count));
  for (uint64_t contig_idx = 0; contig_idx < contig_count; ++contig_idx) {
    manifest.contig_ids.push_back(
        read_string(in, "contig.id"));
  }
  const uint64_t shard_count =
      read_pod<uint64_t>(in, "shard_count");
  if (shard_count == 0 || shard_count > kMaxShardCount) {
    throw std::runtime_error("invalid sharded index shard count");
  }
  manifest.shards.reserve(static_cast<size_t>(shard_count));
  for (uint64_t shard_idx = 0; shard_idx < shard_count;
       ++shard_idx) {
    IndexShardDescriptor shard;
    shard.pack_id = read_pod<uint32_t>(in, "shard.pack_id");
    shard.file_offset = read_pod<uint64_t>(in, "shard.file_offset");
    shard.file_size = read_pod<uint64_t>(in, "shard.file_size");
    shard.contig_id = read_pod<uint32_t>(in, "shard.contig_id");
    shard.source_begin =
        read_pod<uint32_t>(in, "shard.source_begin");
    shard.window_count = read_pod<uint32_t>(in, "shard.window_count");
    shard.sequence_count =
        read_pod<uint32_t>(in, "shard.sequence_count");
    shard.world_node_count =
        read_pod<uint32_t>(in, "shard.world_node_count");
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

ShardedSeedRouter load_sharded_seed_router(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest) {
  validate_manifest(manifest);
  ShardedSeedRouter router;
  if (manifest.router_entry_count == 0) return router;
  const uint32_t expected_shard_id_bits =
      required_shard_id_bits(
          static_cast<uint32_t>(manifest.shards.size()));
  const size_t packed_size = packed_shard_byte_count(
      manifest.router_entry_count, expected_shard_id_bits);
  const auto path = router_output_path(manifest_path);

#if defined(__unix__) || defined(__APPLE__)
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("unable to open shard router sidecar");
  }
  struct stat status {};
  if (fstat(fd, &status) != 0 || status.st_size < 0 ||
      static_cast<uint64_t>(status.st_size) < kRouterHeaderBytes ||
      static_cast<uint64_t>(status.st_size) >
          std::numeric_limits<size_t>::max()) {
    close(fd);
    throw std::runtime_error("invalid shard router sidecar size");
  }
  const size_t mapped_size = static_cast<size_t>(status.st_size);
  void* address = mmap(
      nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (address == MAP_FAILED) {
    throw std::runtime_error("unable to map shard router sidecar");
  }
#if defined(MADV_RANDOM)
  (void)madvise(address, mapped_size, MADV_RANDOM);
#endif
  auto mapping = std::make_shared<MappedRouterFile>(
      address, mapped_size);
  const uint8_t* bytes = mapping->data();
  const auto read_field = [&bytes](auto* value) {
    std::memcpy(value, bytes, sizeof(*value));
    bytes += sizeof(*value);
  };
  std::array<char, 8> magic{};
  uint32_t version = 0;
  uint32_t stored_k = 0;
  uint32_t stored_window = 0;
  uint32_t stored_shard_count = 0;
  uint32_t stored_shard_id_bits = 0;
  uint32_t stored_code_block_size = 0;
  uint32_t reserved = 0;
  uint64_t stored_entry_count = 0;
  uint64_t stored_code_block_count = 0;
  uint64_t stored_code_group_count = 0;
  uint64_t stored_code_payload_size = 0;
  uint64_t stored_checksum = 0;
  uint32_t reserved_tail = 0;
  read_field(&magic);
  read_field(&version);
  read_field(&stored_k);
  read_field(&stored_window);
  read_field(&stored_shard_count);
  read_field(&stored_shard_id_bits);
  read_field(&stored_code_block_size);
  read_field(&reserved);
  read_field(&stored_entry_count);
  read_field(&stored_code_block_count);
  read_field(&stored_code_group_count);
  read_field(&stored_code_payload_size);
  read_field(&stored_checksum);
  read_field(&reserved_tail);
  if (stored_entry_count > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("shard router entry count overflow");
  }
  const RouterStorageLayout layout = router_storage_layout(
      static_cast<size_t>(stored_entry_count), stored_code_payload_size,
      packed_size);
  if (magic != kRouterMagic || version != kRouterFormatVersion ||
      stored_k != manifest.router_k ||
      stored_window != manifest.router_window ||
      stored_shard_count != manifest.shards.size() ||
      stored_shard_id_bits != expected_shard_id_bits ||
      stored_code_block_size != kRouterCodeBlockSize ||
      reserved != 0 ||
      reserved_tail != 0 ||
      stored_entry_count != manifest.router_entry_count ||
      stored_code_block_count != layout.code_block_count ||
      stored_code_group_count != layout.code_group_count ||
      layout.total_size != mapped_size ||
      stored_checksum != manifest.router_checksum) {
    throw std::runtime_error("shard router metadata mismatch");
  }
  router.k = stored_k;
  router.window = stored_window;
  router.shard_count = stored_shard_count;
  router.shard_id_bits = stored_shard_id_bits;
  router.code_block_size = stored_code_block_size;
  router.code_entry_count = static_cast<size_t>(stored_entry_count);
  router.minimizer_code_bases.set_mapped(
      mapping, reinterpret_cast<const uint32_t*>(
          mapping->data() + layout.bases_begin),
      layout.code_block_count);
  router.minimizer_code_widths.set_mapped(
      mapping, mapping->data() + layout.widths_begin,
      layout.code_block_count);
  router.minimizer_code_group_offsets.set_mapped(
      mapping, reinterpret_cast<const uint16_t*>(
          mapping->data() + layout.group_offsets_begin),
      layout.code_group_count);
  router.minimizer_code_supergroup_offsets.set_mapped(
      mapping, reinterpret_cast<const uint64_t*>(
          mapping->data() + layout.supergroup_offsets_begin),
      layout.code_supergroup_count);
  router.packed_minimizer_code_deltas.set_mapped(
      mapping, mapping->data() + layout.payload_begin,
      static_cast<size_t>(stored_code_payload_size));
  router.packed_shard_ids.set_mapped(
      mapping, mapping->data() + layout.shard_ids_begin, packed_size);
#else
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("unable to open shard router sidecar");
  }
  std::array<char, 8> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  const uint32_t version = read_pod<uint32_t>(in, "router.version");
  const uint32_t stored_k = read_pod<uint32_t>(in, "router.k");
  const uint32_t stored_window =
      read_pod<uint32_t>(in, "router.window");
  const uint32_t stored_shard_count =
      read_pod<uint32_t>(in, "router.shard_count");
  const uint32_t stored_shard_id_bits =
      read_pod<uint32_t>(in, "router.shard_id_bits");
  const uint32_t stored_code_block_size =
      read_pod<uint32_t>(in, "router.code_block_size");
  const uint32_t reserved =
      read_pod<uint32_t>(in, "router.reserved");
  const size_t stored_entry_count =
      read_size(in, "router.entry_count");
  const size_t stored_code_block_count =
      read_size(in, "router.code_block_count");
  const size_t stored_code_group_count =
      read_size(in, "router.code_group_count");
  const uint64_t stored_code_payload_size =
      read_pod<uint64_t>(in, "router.code_payload_size");
  const uint64_t stored_checksum =
      read_pod<uint64_t>(in, "router.checksum");
  const uint32_t reserved_tail =
      read_pod<uint32_t>(in, "router.reserved_tail");
  const RouterStorageLayout layout = router_storage_layout(
      stored_entry_count, stored_code_payload_size, packed_size);
  if (magic != kRouterMagic || version != kRouterFormatVersion ||
      stored_k != manifest.router_k ||
      stored_window != manifest.router_window ||
      stored_shard_count != manifest.shards.size() ||
      stored_shard_id_bits != expected_shard_id_bits ||
      stored_code_block_size != kRouterCodeBlockSize ||
      reserved != 0 ||
      reserved_tail != 0 ||
      stored_entry_count != manifest.router_entry_count ||
      stored_code_block_count != layout.code_block_count ||
      stored_code_group_count != layout.code_group_count ||
      stored_checksum != manifest.router_checksum) {
    throw std::runtime_error("shard router metadata mismatch");
  }
  std::vector<uint32_t> code_bases(layout.code_block_count);
  std::vector<uint8_t> code_widths(layout.code_block_count);
  std::vector<uint16_t> code_group_offsets(layout.code_group_count);
  std::vector<uint64_t> code_supergroup_offsets(
      layout.code_supergroup_count);
  std::vector<uint8_t> packed_minimizer_code_deltas(
      static_cast<size_t>(stored_code_payload_size));
  std::vector<uint8_t> packed_shard_ids(packed_size);
  in.read(reinterpret_cast<char*>(code_bases.data()),
          static_cast<std::streamsize>(
              code_bases.size() * sizeof(uint32_t)));
  in.read(reinterpret_cast<char*>(code_widths.data()),
          static_cast<std::streamsize>(code_widths.size()));
  const size_t descriptor_end =
      layout.widths_begin + code_widths.size();
  in.ignore(static_cast<std::streamsize>(
      layout.group_offsets_begin - descriptor_end));
  in.read(reinterpret_cast<char*>(code_group_offsets.data()),
          static_cast<std::streamsize>(
              code_group_offsets.size() * sizeof(code_group_offsets[0])));
  const size_t group_offsets_end = layout.group_offsets_begin +
      code_group_offsets.size() * sizeof(code_group_offsets[0]);
  in.ignore(static_cast<std::streamsize>(
      layout.supergroup_offsets_begin - group_offsets_end));
  in.read(reinterpret_cast<char*>(code_supergroup_offsets.data()),
          static_cast<std::streamsize>(
              code_supergroup_offsets.size() *
              sizeof(code_supergroup_offsets[0])));
  in.read(reinterpret_cast<char*>(packed_minimizer_code_deltas.data()),
          static_cast<std::streamsize>(
              packed_minimizer_code_deltas.size()));
  in.read(reinterpret_cast<char*>(packed_shard_ids.data()),
          static_cast<std::streamsize>(packed_shard_ids.size()));
  uint64_t checksum = begin_router_checksum(
      stored_k, stored_window, stored_shard_count,
      stored_shard_id_bits, stored_entry_count);
  hash_bytes(&checksum, packed_minimizer_code_deltas.data(),
             packed_minimizer_code_deltas.size());
  hash_bytes(&checksum, code_bases.data(),
             code_bases.size() * sizeof(uint32_t));
  hash_bytes(&checksum, code_widths.data(), code_widths.size());
  hash_bytes(&checksum, code_group_offsets.data(),
             code_group_offsets.size() * sizeof(code_group_offsets[0]));
  hash_bytes(&checksum, code_supergroup_offsets.data(),
             code_supergroup_offsets.size() *
                 sizeof(code_supergroup_offsets[0]));
  hash_bytes(&checksum, packed_shard_ids.data(), packed_shard_ids.size());
  if (!in || in.peek() != std::char_traits<char>::eof() ||
      checksum != stored_checksum) {
    throw std::runtime_error("invalid shard router contents");
  }
  router.k = stored_k;
  router.window = stored_window;
  router.shard_count = stored_shard_count;
  router.shard_id_bits = stored_shard_id_bits;
  router.code_block_size = stored_code_block_size;
  router.code_entry_count = stored_entry_count;
  router.minimizer_code_bases.set_owned(std::move(code_bases));
  router.minimizer_code_widths.set_owned(std::move(code_widths));
  router.minimizer_code_group_offsets.set_owned(
      std::move(code_group_offsets));
  router.minimizer_code_supergroup_offsets.set_owned(
      std::move(code_supergroup_offsets));
  router.packed_minimizer_code_deltas.set_owned(
      std::move(packed_minimizer_code_deltas));
  router.packed_shard_ids.set_owned(std::move(packed_shard_ids));
#endif
  return router;
}

static ShardedIndexManifest build_sharded_reference_index_impl(
    const std::string& bundle_path,
    const std::string& ref_input,
    const std::string& reference_id,
    size_t reference_size,
    const std::vector<ReferenceContig>& reference_contigs,
    const std::function<std::string(size_t, size_t)>& load_slice,
    const std::string* contiguous_reference,
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
  if (reference_size == 0 || reference_contigs.empty()) {
    throw std::invalid_argument(
        "sharded index reference must not be empty");
  }

  const std::filesystem::path bundle(bundle_path);
  const IndexBuildManifest part_manifest =
      make_reference_window_index_manifest(
          ref_input, reference_size,
          static_cast<int>(window_length),
          static_cast<int>(stride), hierarchy, range_config);

  // Reuse the final manifest descriptors as the build plan. Keeping a second
  // 16-byte plan beside 48-byte descriptors costs nearly 10 MiB for a
  // 3 Gb reference split into recommended 5k-window shards.
  std::vector<IndexShardDescriptor> descriptors;
  descriptors.reserve(
      reference_size / max_shard_windows + reference_contigs.size() + 1);
  std::vector<std::string> contig_ids;
  contig_ids.reserve(reference_contigs.size());
  uint32_t expected_contig_begin = 0;
  for (size_t contig_idx = 0; contig_idx < reference_contigs.size();
       ++contig_idx) {
    const auto& contig = reference_contigs[contig_idx];
    if (contig.id.empty() || contig.begin != expected_contig_begin ||
        contig.end < contig.begin ||
        contig.end > reference_size) {
      throw std::invalid_argument(
          "reference contigs must be contiguous and in bounds");
    }
    contig_ids.push_back(contig.id);
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

      IndexShardDescriptor descriptor;
      descriptor.contig_id = static_cast<uint32_t>(contig_idx);
      descriptor.source_begin = static_cast<uint32_t>(source_begin);
      descriptor.window_count = static_cast<uint32_t>(shard_window_count);
      descriptors.push_back(descriptor);
    }
  }
  if (expected_contig_begin != reference_size) {
    throw std::invalid_argument(
        "reference contigs do not cover the reference");
  }
  if (descriptors.empty()) {
    throw std::invalid_argument(
        "sharded index reference contains no complete windows");
  }
  if (descriptors.size() > kMaxShardCount) {
    throw std::runtime_error(
        "reference requires more than 2^32 logical shards");
  }

  const size_t available_threads = static_cast<size_t>(
      std::max(1, omp_get_max_threads()));
  // Small (the recommended 5k-window) parts spend much of their time in the
  // serial sketch and finalization phases. More independent parts therefore
  // use a fixed OpenMP budget more efficiently than a few wide teams. Keep
  // larger parts conservative so their peak working sets do not multiply.
  const bool small_shards = max_shard_windows <= 16384;
  // At the recommended 5k-window scale, 20 shallow teams improve aggregate
  // throughput over 16 on a 128-thread host while retaining a bounded peak.
  // Wider small shards keep the 16-team cap because their Phase-2 work grows
  // enough that deeper teams are more efficient.
  const size_t small_shard_job_cap =
      max_shard_windows <= 8192 ? 20 : 16;
  const size_t automatic_jobs =
      available_threads < 8
          ? 1
          : available_threads < 16
                ? 2
                : small_shards
                      ? std::max<size_t>(
                            2, std::min<size_t>(
                                   small_shard_job_cap,
                                   available_threads / 4))
                      : std::max<size_t>(
                            2, std::min<size_t>(
                                   4, static_cast<size_t>(std::sqrt(
                                          static_cast<double>(
                                              available_threads)))));
  const size_t requested_jobs =
      build_jobs == 0 ? automatic_jobs : build_jobs;
  const size_t job_count_size = std::min(
      {requested_jobs, descriptors.size(), available_threads});
  if (job_count_size >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "sharded index build job count is too large");
  }
  const int job_count = static_cast<int>(job_count_size);
  const int threads_per_job = static_cast<int>(
      std::max<size_t>(1, available_threads / job_count_size));
  const auto descriptor_slice_begin =
      [&](const IndexShardDescriptor& descriptor) {
        if (descriptor.contig_id >= reference_contigs.size()) {
          throw std::runtime_error("shard descriptor has invalid contig ID");
        }
        const auto& contig = reference_contigs[descriptor.contig_id];
        if (descriptor.source_begin < contig.source_begin) {
          throw std::runtime_error("shard descriptor precedes its contig");
        }
        const size_t slice_begin = static_cast<size_t>(contig.begin) +
            (static_cast<size_t>(descriptor.source_begin) -
             contig.source_begin);
        if (slice_begin > contig.end) {
          throw std::runtime_error("shard descriptor lies beyond its contig");
        }
        return slice_begin;
      };
  const auto descriptor_slice_end = [&](const IndexShardDescriptor& descriptor) {
    return descriptor_slice_begin(descriptor) +
           (static_cast<size_t>(descriptor.window_count) - 1) * stride +
           window_length;
  };

  const auto build_one = [&](size_t spec_idx,
                             const std::function<void(
                                 size_t,
                                 const BioGeometryIndexBuilder&)>&
                                 write_payload,
                             const std::function<void()>& signal_failure,
                             std::exception_ptr* error) {
    try {
      const auto& descriptor = descriptors[spec_idx];
      const size_t slice_begin = descriptor_slice_begin(descriptor);
      const size_t slice_end = descriptor_slice_end(descriptor);
      const std::string& ref_id = contig_ids[descriptor.contig_id];
      const std::filesystem::path part_path =
          shard_output_path(bundle, spec_idx);
      std::string slice =
          load_slice(slice_begin, slice_end);
      if (slice.size() != slice_end - slice_begin) {
        throw std::runtime_error(
            "reference slice loader returned the wrong number of bases");
      }
      const uint32_t slice_size = static_cast<uint32_t>(slice.size());
      ReferenceContig slice_contig{
          ref_id, 0, slice_size,
          descriptor.source_begin};

      LoadedIndex reusable;
      const bool reused = load_reusable_shard(
          part_path, part_manifest, reference_id, slice,
          slice_contig, window_length, &reusable);

      size_t sequence_count = 0;
      size_t world_node_count = 0;
      if (reused) {
        sequence_count = reusable.builder.num_sequences();
        world_node_count = reusable.builder.num_world_nodes();
        write_payload(spec_idx, reusable.builder);
      } else {
        BuildRangeConfig shard_range_config = range_config;
        if (job_count > 1) {
          shard_range_config.progress_interval_seconds = 0;
          shard_range_config.emit_build_output = false;
        }
        BioGeometryIndexBuilder builder(hierarchy, shard_range_config);
        builder.build_reference_windows(
            reference_id, std::move(slice), window_length, stride,
            {slice_contig});
        sequence_count = builder.num_sequences();
        world_node_count = builder.num_world_nodes();
        write_payload(spec_idx, builder);
      }
      if (sequence_count > std::numeric_limits<uint32_t>::max() ||
          world_node_count > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("logical shard count exceeds 32-bit storage");
      }

      auto& completed_descriptor = descriptors[spec_idx];
      completed_descriptor.sequence_count =
          static_cast<uint32_t>(sequence_count);
      completed_descriptor.world_node_count =
          static_cast<uint32_t>(world_node_count);
    } catch (...) {
      *error = std::current_exception();
      signal_failure();
    }
  };

  const size_t pack_count =
      (descriptors.size() + kShardsPerPack - 1) / kShardsPerPack;
  std::vector<std::string> pack_paths(pack_count);
  for (size_t pack_idx = 0; pack_idx < pack_count; ++pack_idx) {
    const std::filesystem::path pack_path =
        shard_pack_output_path(bundle, pack_idx);
    pack_paths[pack_idx] = pack_path.filename().string();
    const size_t group_begin = pack_idx * kShardsPerPack;
    const size_t group_end =
        std::min(descriptors.size(), group_begin + kShardsPerPack);

    bool reused_pack = false;
    try {
      const auto entries = read_shard_pack_directory(pack_path);
      if (entries.size() != group_end - group_begin) {
        throw std::runtime_error("shard pack count changed");
      }
      std::vector<IndexShardDescriptor> reused_descriptors;
      reused_descriptors.reserve(entries.size());
      for (size_t local_idx = 0; local_idx < entries.size(); ++local_idx) {
        const size_t spec_idx = group_begin + local_idx;
        const auto& planned_descriptor = descriptors[spec_idx];
        const size_t slice_begin =
            descriptor_slice_begin(planned_descriptor);
        const size_t slice_end =
            descriptor_slice_end(planned_descriptor);
        const std::string& ref_id =
            contig_ids[planned_descriptor.contig_id];
        std::string slice =
            load_slice(slice_begin, slice_end);
        if (slice.size() != slice_end - slice_begin) {
          throw std::runtime_error(
              "reference slice loader returned the wrong number of bases");
        }
        ReferenceContig slice_contig{
            ref_id, 0, static_cast<uint32_t>(slice.size()),
            planned_descriptor.source_begin};
        LoadedIndex candidate = load_index_payload_range(
            pack_path.string(), entries[local_idx].offset,
            entries[local_idx].size, part_manifest);
        if (!reusable_shard_matches(
                candidate, part_manifest.signature, reference_id,
                slice, slice_contig, window_length)) {
          throw std::runtime_error("shard pack input changed");
        }
        IndexShardDescriptor reused_descriptor = planned_descriptor;
        reused_descriptor.pack_id = static_cast<uint32_t>(pack_idx);
        reused_descriptor.file_offset = entries[local_idx].offset;
        reused_descriptor.file_size = entries[local_idx].size;
        reused_descriptor.sequence_count = static_cast<uint32_t>(
            candidate.builder.num_sequences());
        reused_descriptor.world_node_count = static_cast<uint32_t>(
            candidate.builder.num_world_nodes());
        reused_descriptors.push_back(std::move(reused_descriptor));
      }
      std::move(
          reused_descriptors.begin(), reused_descriptors.end(),
          descriptors.begin() + static_cast<std::ptrdiff_t>(group_begin));
      reused_pack = true;
    } catch (const std::exception&) {
      reused_pack = false;
    }

    if (!reused_pack) {
      const std::filesystem::path temporary = pack_path.string() + ".tmp";
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      std::ofstream out(temporary, std::ios::binary);
      if (!out) throw std::runtime_error("unable to create shard pack");

      const size_t part_count = group_end - group_begin;
      std::vector<ShardPackEntry> entries(part_count);
      out.write(kShardPackMagic.data(),
                static_cast<std::streamsize>(kShardPackMagic.size()));
      write_pod<uint32_t>(out, kShardPackFormatVersion);
      write_pod<uint32_t>(out, static_cast<uint32_t>(part_count));
      for (size_t local_idx = 0; local_idx < part_count; ++local_idx) {
        write_pod<uint64_t>(out, 0);
        write_pod<uint64_t>(out, 0);
      }
      const uint64_t payload_begin = align_shard_pack_offset(
          16 + static_cast<uint64_t>(part_count) * 16);
      out.seekp(static_cast<std::streamoff>(payload_begin));
      if (!out) throw std::runtime_error("unable to reserve shard pack directory");

      std::mutex pack_write_mutex;
      std::condition_variable pack_write_ready;
      bool pack_build_failed = false;
      size_t next_payload_spec = group_begin;
      const auto signal_failure = [&]() {
        {
          std::lock_guard<std::mutex> lock(pack_write_mutex);
          pack_build_failed = true;
        }
        pack_write_ready.notify_all();
      };
      const auto write_payload = [&](size_t spec_idx,
                                     const BioGeometryIndexBuilder& builder) {
        const size_t local_idx = spec_idx - group_begin;
        std::unique_lock<std::mutex> lock(pack_write_mutex);
        pack_write_ready.wait(lock, [&]() {
          return pack_build_failed || spec_idx == next_payload_spec;
        });
        if (pack_build_failed) {
          throw std::runtime_error("shard pack build was cancelled");
        }
        const std::streampos position = out.tellp();
        if (position < 0) {
          throw std::runtime_error("unable to determine shard pack offset");
        }
        const uint64_t offset = align_shard_pack_offset(
            static_cast<uint64_t>(position));
        out.seekp(static_cast<std::streamoff>(offset));
        write_index_payload(out, builder);
        const std::streampos end = out.tellp();
        if (end < 0 || static_cast<uint64_t>(end) <= offset) {
          throw std::runtime_error("failed to append shard payload");
        }
        entries[local_idx] = {
            offset, static_cast<uint64_t>(end) - offset};
        ++next_payload_spec;
        lock.unlock();
        pack_write_ready.notify_all();
      };

      try {
        std::vector<std::exception_ptr> errors(part_count);
        if (job_count == 1) {
          for (size_t spec_idx = group_begin; spec_idx < group_end;
               ++spec_idx) {
            build_one(spec_idx, write_payload, signal_failure,
                      &errors[spec_idx - group_begin]);
          }
        } else {
          const int previous_active_levels = omp_get_max_active_levels();
          omp_set_max_active_levels(std::max(2, previous_active_levels));
#pragma omp parallel num_threads(job_count)
          {
            // Bound concurrent shards times each nested team by the original
            // OpenMP thread budget.
            const int previous_nested_threads = omp_get_max_threads();
            omp_set_num_threads(threads_per_job);
#pragma omp for schedule(dynamic, 1)
            for (size_t spec_idx = group_begin;
                 spec_idx < group_end; ++spec_idx) {
              build_one(spec_idx, write_payload, signal_failure,
                        &errors[spec_idx - group_begin]);
            }
            omp_set_num_threads(previous_nested_threads);
          }
          omp_set_max_active_levels(previous_active_levels);
        }
        for (const auto& error : errors) {
          if (error) std::rethrow_exception(error);
        }

        out.seekp(static_cast<std::streamoff>(16));
        for (const auto& entry : entries) {
          write_pod<uint64_t>(out, entry.offset);
          write_pod<uint64_t>(out, entry.size);
        }
        out.close();
        if (!out) throw std::runtime_error("failed to finalize shard pack");
        std::error_code error;
        std::filesystem::rename(temporary, pack_path, error);
        if (error) {
          throw std::runtime_error(
              "unable to install shard pack: " + error.message());
        }
      } catch (...) {
        out.close();
        std::filesystem::remove(temporary, ignored);
        throw;
      }
      for (size_t local_idx = 0; local_idx < entries.size(); ++local_idx) {
        auto& descriptor = descriptors[group_begin + local_idx];
        descriptor.pack_id = static_cast<uint32_t>(pack_idx);
        descriptor.file_offset = entries[local_idx].offset;
        descriptor.file_size = entries[local_idx].size;
      }
    }

    for (size_t spec_idx = group_begin; spec_idx < group_end;
         ++spec_idx) {
      const auto part_path = shard_output_path(bundle, spec_idx);
      std::error_code error;
      if (std::filesystem::exists(part_path, error) &&
          !std::filesystem::remove(part_path, error)) {
        throw std::runtime_error(
            "unable to remove packed shard: " + error.message());
      }
      if (error) {
        throw std::runtime_error(
            "unable to clean packed shard: " + error.message());
      }
    }
  }

  ShardedSeedRouter router_metadata;
  uint64_t router_checksum = 0;
  size_t router_entry_count = 0;
  if (descriptors.size() > 1 && window_length >= kRouterWindow) {
    const auto router_path = router_output_path(bundle);
    auto router_data = contiguous_reference
        ? build_router_data(
              descriptors, kRouterK, kRouterWindow,
              router_path.string() + ".codes.tmp",
              [&](const IndexShardDescriptor& descriptor) {
                const size_t slice_begin =
                    descriptor_slice_begin(descriptor);
                return std::string_view(
                    contiguous_reference->data() + slice_begin,
                    descriptor_slice_end(descriptor) - slice_begin);
              })
        : build_router_data(
              descriptors, kRouterK, kRouterWindow,
              router_path.string() + ".codes.tmp",
              [&](const IndexShardDescriptor& descriptor) {
                return load_slice(descriptor_slice_begin(descriptor),
                                  descriptor_slice_end(descriptor));
              });
    if (router_data.entry_count != 0) {
      router_metadata.k = kRouterK;
      router_metadata.window = kRouterWindow;
      router_entry_count = router_data.entry_count;
      router_checksum = save_router_sidecar(
          router_path, router_metadata.k,
          router_metadata.window,
          static_cast<uint32_t>(descriptors.size()),
          router_data);
    }
  }

  ShardedIndexManifest manifest;
  manifest.window_length = window_length;
  manifest.stride = stride;
  manifest.part_manifest = part_manifest;
  manifest.pack_paths = std::move(pack_paths);
  manifest.contig_ids = std::move(contig_ids);
  for (const auto& descriptor : descriptors) {
    manifest.total_window_count += descriptor.window_count;
    manifest.total_sequence_count += descriptor.sequence_count;
    manifest.total_world_node_count += descriptor.world_node_count;
  }
  manifest.shards = std::move(descriptors);
  if (router_entry_count != 0) {
    manifest.router_k = router_metadata.k;
    manifest.router_window = router_metadata.window;
    manifest.router_entry_count = router_entry_count;
    manifest.router_checksum = router_checksum;
  }
  save_sharded_index_manifest(bundle_path, manifest);
  return manifest;
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
  return build_sharded_reference_index_impl(
      bundle_path, ref_input, reference_id,
      reference_sequence.size(), reference_contigs,
      [&](size_t begin, size_t end) {
        return reference_sequence.substr(begin, end - begin);
      },
      &reference_sequence, window_length, stride,
      max_shard_windows, hierarchy, range_config, build_jobs);
}

ShardedIndexManifest build_sharded_reference_index(
    const std::string& bundle_path,
    const std::string& ref_input,
    const IndexedReferenceFile& reference,
    size_t window_length,
    size_t stride,
    size_t max_shard_windows,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config,
    size_t build_jobs) {
  return build_sharded_reference_index_impl(
      bundle_path, ref_input, reference.id,
      reference.sequence_size, reference.contigs,
      [&](size_t begin, size_t end) {
        return reference.slice(begin, end);
      },
      nullptr, window_length, stride, max_shard_windows,
      hierarchy, range_config, build_jobs);
}

LoadedIndex load_sharded_index_part(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest,
    uint32_t shard_id) {
  if (shard_id >= manifest.shards.size()) {
    throw std::out_of_range("sharded index part ID is out of range");
  }
  const auto& descriptor = manifest.shards[shard_id];
  if (descriptor.pack_id >= manifest.pack_paths.size() ||
      descriptor.contig_id >= manifest.contig_ids.size()) {
    throw std::runtime_error("sharded index part has invalid interned ID");
  }
  LoadedIndex loaded = load_index_payload_range(
      resolve_index_shard_path(
          manifest_path, manifest.pack_paths[descriptor.pack_id]),
      descriptor.file_offset, descriptor.file_size,
      manifest.part_manifest,
      IndexLoadValidation::Structural);
  const auto& store = loaded.builder.sequence_store();
  if (!store.reference_backed ||
      loaded.manifest.signature != manifest.part_manifest.signature ||
      store.fixed_sequence_length != manifest.window_length ||
      store.reference_contigs.size() != 1) {
    throw std::runtime_error(
        "sharded index part has incompatible sequence storage");
  }
  const auto& contig = store.reference_contigs.front();
  const size_t source_end =
      static_cast<size_t>(contig.source_begin) +
      contig.end - contig.begin;
  const uint64_t descriptor_source_end =
      static_cast<uint64_t>(descriptor.source_begin) +
      (static_cast<uint64_t>(descriptor.window_count) - 1) *
          manifest.stride + manifest.window_length;
  if (contig.id != manifest.contig_ids[descriptor.contig_id] ||
      contig.source_begin != descriptor.source_begin ||
      source_end != descriptor_source_end ||
      store.size() != descriptor.sequence_count ||
      loaded.builder.num_world_nodes() !=
          descriptor.world_node_count) {
    throw std::runtime_error(
        "sharded index part does not match its descriptor");
  }
  return loaded;
}

std::vector<LoadedIndex> load_sharded_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest,
    const std::vector<uint32_t>& shard_ids) {
  validate_manifest(manifest);
  std::vector<LoadedIndex> loaded;
  loaded.reserve(shard_ids.size());
  for (uint32_t shard_id : shard_ids) {
    loaded.push_back(load_sharded_index_part(
        manifest_path, manifest, shard_id));
  }
  return loaded;
}

std::vector<LoadedIndex> load_sharded_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest) {
  std::vector<uint32_t> shard_ids(manifest.shards.size());
  std::iota(shard_ids.begin(), shard_ids.end(), uint32_t{0});
  return load_sharded_index(manifest_path, manifest, shard_ids);
}

}  // namespace navigamer
