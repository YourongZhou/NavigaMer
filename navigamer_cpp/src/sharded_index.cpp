#include "sharded_index.hpp"
#include "io_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
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
    'N', 'G', 'S', 'H', 'R', 'D', '0', '3'};
constexpr std::array<char, 8> kRouterMagic = {
    'N', 'G', 'R', 'O', 'U', 'T', '0', '2'};
constexpr uint32_t kShardFormatVersion = 3;
constexpr uint32_t kRouterFormatVersion = 2;
constexpr size_t kRouterHeaderBytes = 48;
constexpr std::streamoff kRouterChecksumOffset = 40;
constexpr uint32_t kRouterK = 16;
constexpr uint32_t kRouterWindow = 32;
constexpr uint64_t kMaxShardCount = uint64_t{1} << 20;
constexpr uint64_t kMaxStringLength = uint64_t{1} << 30;
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

[[maybe_unused]] uint64_t router_storage_checksum(
    uint32_t k, uint32_t window, uint32_t shard_count,
    uint32_t shard_id_bits, size_t entry_count,
    const uint32_t* minimizer_codes,
    const uint8_t* packed_shard_ids, size_t packed_size) {
  uint64_t hash = begin_router_checksum(
      k, window, shard_count, shard_id_bits, entry_count);
  hash_bytes(
      &hash, minimizer_codes, entry_count * sizeof(uint32_t));
  hash_bytes(&hash, packed_shard_ids, packed_size);
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

std::vector<uint32_t> reference_minimizers(
    std::string_view sequence, uint32_t k, uint32_t window) {
  std::vector<uint32_t> minimizers;
  if (k == 0 || k > 16 || window < k ||
      sequence.size() < window) {
    return minimizers;
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
        (minimizers.empty() ||
         minimizers.back() != minimum_queue.front().second)) {
      minimizers.push_back(minimum_queue.front().second);
    }
  }
  std::sort(minimizers.begin(), minimizers.end());
  minimizers.erase(
      std::unique(minimizers.begin(), minimizers.end()),
      minimizers.end());
  return minimizers;
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

struct RouterBuildData {
  explicit RouterBuildData(std::filesystem::path path)
      : spool_path(std::move(path)) {}
  RouterBuildData(const RouterBuildData&) = delete;
  RouterBuildData& operator=(const RouterBuildData&) = delete;
  RouterBuildData(RouterBuildData&& other) noexcept
      : spool_path(std::move(other.spool_path)),
        shard_offsets(std::move(other.shard_offsets)),
        shard_counts(std::move(other.shard_counts)),
        spool_size(other.spool_size),
        page_size(other.page_size),
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
  std::vector<size_t> shard_offsets;
  std::vector<size_t> shard_counts;
  size_t spool_size = 0;
  size_t page_size = 4096;
  size_t entry_count = 0;
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
        store.reference_view() != reference_slice ||
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

uint64_t save_router_sidecar(
    const std::filesystem::path& path,
    uint32_t k, uint32_t window, uint32_t shard_count,
    const RouterBuildData& data) {
  if (data.shard_offsets.size() != shard_count ||
      data.shard_counts.size() != shard_count ||
      data.entry_count == 0 || data.spool_size == 0 ||
      data.page_size == 0) {
    throw std::runtime_error("invalid shard router build data");
  }
  const std::filesystem::path temporary = path.string() + ".tmp";
  const std::filesystem::path packed_temporary =
      path.string() + ".packed.tmp";
  const uint32_t shard_id_bits =
      required_shard_id_bits(shard_count);
  const size_t packed_size =
      packed_shard_byte_count(data.entry_count, shard_id_bits);
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
    for (uint32_t shard_id = 0; shard_id < shard_count; ++shard_id) {
      const size_t offset = data.shard_offsets[shard_id];
      const size_t count = data.shard_counts[shard_id];
      if (offset % data.page_size != 0 ||
          count >
              (std::numeric_limits<size_t>::max() - offset) /
                  sizeof(uint32_t) ||
          offset + count * sizeof(uint32_t) > data.spool_size) {
        throw std::runtime_error("invalid shard router code range");
      }
    }
    const auto code_at = [&](uint32_t shard_id, size_t offset) {
      uint32_t code = 0;
      std::memcpy(
          &code,
          spool_bytes + data.shard_offsets[shard_id] +
              offset * sizeof(uint32_t),
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
    write_pod<uint32_t>(out, 0);
    write_pod<uint64_t>(out, data.entry_count);
    write_pod<uint64_t>(out, 0);

    struct Cursor {
      uint32_t code = 0;
      uint32_t shard_id = 0;
      size_t offset = 0;
    };
    struct CursorGreater {
      bool operator()(const Cursor& left, const Cursor& right) const {
        if (left.code != right.code) return left.code > right.code;
        return left.shard_id > right.shard_id;
      }
    };
    std::priority_queue<
        Cursor, std::vector<Cursor>, CursorGreater> queue;
    for (uint32_t shard_id = 0; shard_id < shard_count; ++shard_id) {
      if (data.shard_counts[shard_id] != 0) {
        queue.push({code_at(shard_id, 0), shard_id, 0});
      }
    }

    std::vector<uint32_t> code_buffer(
        std::min<size_t>(16384, data.entry_count));
    size_t code_buffered = 0;
    const auto flush_codes = [&]() {
      if (code_buffered == 0) return;
      const size_t byte_count =
          code_buffered * sizeof(uint32_t);
      out.write(
          reinterpret_cast<const char*>(code_buffer.data()),
          static_cast<std::streamsize>(byte_count));
      hash_bytes(&checksum, code_buffer.data(), byte_count);
      code_buffered = 0;
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
    while (!queue.empty()) {
      const Cursor cursor = queue.top();
      queue.pop();
      code_buffer[code_buffered++] = cursor.code;
      if (code_buffered == code_buffer.size()) flush_codes();

      pending_shard_bits |=
          static_cast<uint64_t>(cursor.shard_id) << pending_bit_count;
      pending_bit_count += shard_id_bits;
      while (pending_bit_count >= 8) {
        emit_packed_byte(static_cast<uint8_t>(pending_shard_bits));
        pending_shard_bits >>= 8;
        pending_bit_count -= 8;
      }
      ++emitted;

      const size_t next_offset = cursor.offset + 1;
      if (next_offset < data.shard_counts[cursor.shard_id]) {
        queue.push({
            code_at(cursor.shard_id, next_offset),
            cursor.shard_id, next_offset});
      }
#if defined(__unix__) || defined(__APPLE__)
#if defined(MADV_DONTNEED)
      const size_t consumed_bytes =
          next_offset * sizeof(uint32_t);
      if (consumed_bytes % data.page_size == 0 ||
          next_offset == data.shard_counts[cursor.shard_id]) {
        const size_t consumed_page =
            consumed_bytes == 0
                ? 0
                : (consumed_bytes - 1) / data.page_size;
        void* page_address = const_cast<uint8_t*>(
            spool_bytes + data.shard_offsets[cursor.shard_id] +
            consumed_page * data.page_size);
        (void)madvise(page_address, data.page_size, MADV_DONTNEED);
      }
#endif
#endif
    }
    flush_codes();
    if (pending_bit_count != 0) {
      emit_packed_byte(static_cast<uint8_t>(pending_shard_bits));
    }
    flush_shards();
    packed_out.close();
    if (!packed_out || emitted != data.entry_count ||
        packed_written != packed_size) {
      throw std::runtime_error("failed to pack shard router IDs");
    }

    std::ifstream packed_in(packed_temporary, std::ios::binary);
    if (!packed_in) {
      throw std::runtime_error("unable to reopen packed shard router IDs");
    }
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

size_t router_spool_page_size() {
#if defined(__unix__) || defined(__APPLE__)
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size > 0) return static_cast<size_t>(page_size);
#endif
  return 4096;
}

template <typename SliceLoader>
RouterBuildData build_router_data(
    const std::vector<ShardBuildSpec>& specs,
    uint32_t k, uint32_t window,
    const std::filesystem::path& spool_path,
    SliceLoader&& load_slice) {
  if (specs.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("too many shards for seed router");
  }
  RouterBuildData data(spool_path);
  data.page_size = router_spool_page_size();
  data.shard_offsets.reserve(specs.size());
  data.shard_counts.reserve(specs.size());
  std::ofstream spool(data.spool_path, std::ios::binary);
  if (!spool) {
    throw std::runtime_error("unable to create shard router code spool");
  }
  std::vector<uint8_t> zero_page(data.page_size, 0);
  size_t spool_position = 0;
  const auto align_spool = [&]() {
    const size_t remainder = spool_position % data.page_size;
    if (remainder == 0) return;
    const size_t padding = data.page_size - remainder;
    spool.write(
        reinterpret_cast<const char*>(zero_page.data()),
        static_cast<std::streamsize>(padding));
    spool_position += padding;
  };
  for (const auto& spec : specs) {
    align_spool();
    data.shard_offsets.push_back(spool_position);
    const auto slice = load_slice(spec);
    const auto minimizers = reference_minimizers(slice, k, window);
    if (minimizers.size() >
        std::numeric_limits<size_t>::max() - data.entry_count) {
      throw std::runtime_error("shard router entry count overflow");
    }
    if (minimizers.size() >
        (std::numeric_limits<size_t>::max() - spool_position) /
            sizeof(uint32_t)) {
      throw std::runtime_error("shard router spool size overflow");
    }
    data.shard_counts.push_back(minimizers.size());
    if (!minimizers.empty()) {
      const size_t byte_count =
          minimizers.size() * sizeof(uint32_t);
      spool.write(
          reinterpret_cast<const char*>(minimizers.data()),
          static_cast<std::streamsize>(byte_count));
      spool_position += byte_count;
    }
    data.entry_count += minimizers.size();
  }
  align_spool();
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
  if (manifest.part_signature.empty() ||
      manifest.shards.empty()) {
    throw std::runtime_error("sharded index contains no shards");
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
    write_pod<uint32_t>(out, manifest.router_k);
    write_pod<uint32_t>(out, manifest.router_window);
    write_pod<uint64_t>(out, manifest.router_entry_count);
    write_pod<uint64_t>(out, manifest.router_checksum);
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
  manifest.router_k =
      read_pod<uint32_t>(in, "router_k");
  manifest.router_window =
      read_pod<uint32_t>(in, "router_window");
  manifest.router_entry_count =
      read_size(in, "router_entry_count");
  manifest.router_checksum =
      read_pod<uint64_t>(in, "router_checksum");
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
  if (packed_size >
          std::numeric_limits<size_t>::max() -
              kRouterHeaderBytes ||
      manifest.router_entry_count >
      (std::numeric_limits<size_t>::max() -
       kRouterHeaderBytes - packed_size) /
          sizeof(uint32_t)) {
    throw std::runtime_error("shard router size overflow");
  }
  const size_t expected_size =
      kRouterHeaderBytes +
      manifest.router_entry_count * sizeof(uint32_t) +
      packed_size;
  const auto path = router_output_path(manifest_path);

#if defined(__unix__) || defined(__APPLE__)
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("unable to open shard router sidecar");
  }
  struct stat status {};
  if (fstat(fd, &status) != 0 || status.st_size < 0 ||
      static_cast<uint64_t>(status.st_size) != expected_size) {
    close(fd);
    throw std::runtime_error("invalid shard router sidecar size");
  }
  void* address = mmap(
      nullptr, expected_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (address == MAP_FAILED) {
    throw std::runtime_error("unable to map shard router sidecar");
  }
#if defined(MADV_RANDOM)
  (void)madvise(address, expected_size, MADV_RANDOM);
#endif
  auto mapping = std::make_shared<MappedRouterFile>(
      address, expected_size);
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
  uint32_t reserved = 0;
  uint64_t stored_entry_count = 0;
  uint64_t stored_checksum = 0;
  read_field(&magic);
  read_field(&version);
  read_field(&stored_k);
  read_field(&stored_window);
  read_field(&stored_shard_count);
  read_field(&stored_shard_id_bits);
  read_field(&reserved);
  read_field(&stored_entry_count);
  read_field(&stored_checksum);
  if (magic != kRouterMagic || version != kRouterFormatVersion ||
      stored_k != manifest.router_k ||
      stored_window != manifest.router_window ||
      stored_shard_count != manifest.shards.size() ||
      stored_shard_id_bits != expected_shard_id_bits ||
      reserved != 0 ||
      stored_entry_count != manifest.router_entry_count ||
      stored_checksum != manifest.router_checksum) {
    throw std::runtime_error("shard router metadata mismatch");
  }
  router.k = stored_k;
  router.window = stored_window;
  router.shard_count = stored_shard_count;
  router.shard_id_bits = stored_shard_id_bits;
  router.minimizer_codes.set_mapped(
      mapping, reinterpret_cast<const uint32_t*>(bytes),
      manifest.router_entry_count);
  bytes += manifest.router_entry_count * sizeof(uint32_t);
  router.packed_shard_ids.set_mapped(
      mapping, bytes, packed_size);
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
  const uint32_t reserved =
      read_pod<uint32_t>(in, "router.reserved");
  const size_t stored_entry_count =
      read_size(in, "router.entry_count");
  const uint64_t stored_checksum =
      read_pod<uint64_t>(in, "router.checksum");
  if (magic != kRouterMagic || version != kRouterFormatVersion ||
      stored_k != manifest.router_k ||
      stored_window != manifest.router_window ||
      stored_shard_count != manifest.shards.size() ||
      stored_shard_id_bits != expected_shard_id_bits ||
      reserved != 0 ||
      stored_entry_count != manifest.router_entry_count ||
      stored_checksum != manifest.router_checksum) {
    throw std::runtime_error("shard router metadata mismatch");
  }
  std::vector<uint32_t> minimizer_codes(stored_entry_count);
  std::vector<uint8_t> packed_shard_ids(packed_size);
  in.read(reinterpret_cast<char*>(minimizer_codes.data()),
          static_cast<std::streamsize>(
              minimizer_codes.size() * sizeof(uint32_t)));
  in.read(reinterpret_cast<char*>(packed_shard_ids.data()),
          static_cast<std::streamsize>(packed_shard_ids.size()));
  if (!in || in.peek() != std::char_traits<char>::eof() ||
      router_storage_checksum(
          stored_k, stored_window, stored_shard_count,
          stored_shard_id_bits, stored_entry_count,
          minimizer_codes.data(), packed_shard_ids.data(),
          packed_shard_ids.size()) != stored_checksum) {
    throw std::runtime_error("invalid shard router contents");
  }
  router.k = stored_k;
  router.window = stored_window;
  router.shard_count = stored_shard_count;
  router.shard_id_bits = stored_shard_id_bits;
  router.minimizer_codes.set_owned(std::move(minimizer_codes));
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

  std::vector<ShardBuildSpec> specs;
  size_t shard_ordinal = 0;
  uint32_t expected_contig_begin = 0;
  for (const auto& contig : reference_contigs) {
    if (contig.id.empty() || contig.begin != expected_contig_begin ||
        contig.end < contig.begin ||
        contig.end > reference_size) {
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
  if (expected_contig_begin != reference_size) {
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
  const size_t automatic_jobs =
      available_threads < 16
          ? 1
          : std::max<size_t>(
                2, std::min<size_t>(
                       4, static_cast<size_t>(std::sqrt(
                              static_cast<double>(available_threads)))));
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
      std::string slice =
          load_slice(spec.slice_begin, spec.slice_end);
      if (slice.size() != spec.slice_end - spec.slice_begin) {
        throw std::runtime_error(
            "reference slice loader returned the wrong number of bases");
      }
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
  if (specs.size() > 1 && window_length >= kRouterWindow) {
    const auto router_path = router_output_path(bundle);
    auto router_data = contiguous_reference
        ? build_router_data(
              specs, kRouterK, kRouterWindow,
              router_path.string() + ".codes.tmp",
              [&](const ShardBuildSpec& spec) {
                return std::string_view(
                    contiguous_reference->data() + spec.slice_begin,
                    spec.slice_end - spec.slice_begin);
              })
        : build_router_data(
              specs, kRouterK, kRouterWindow,
              router_path.string() + ".codes.tmp",
              [&](const ShardBuildSpec& spec) {
                return load_slice(spec.slice_begin, spec.slice_end);
              });
    if (router_data.entry_count != 0) {
      manifest.router_k = kRouterK;
      manifest.router_window = kRouterWindow;
      manifest.router_entry_count = router_data.entry_count;
      manifest.router_checksum = save_router_sidecar(
          router_path, manifest.router_k,
          manifest.router_window,
          static_cast<uint32_t>(manifest.shards.size()),
          router_data);
    }
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
  LoadedIndex loaded = load_index(
      resolve_index_shard_path(
          manifest_path, descriptor.path),
      IndexLoadValidation::Structural);
  const auto& store = loaded.builder.sequence_store();
  if (!store.reference_backed ||
      loaded.manifest.signature != manifest.part_signature ||
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
