#include "sharded_index.hpp"
#include "tools.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <omp.h>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace navigamer {

class SampledQgramFileMapping {
 public:
  SampledQgramFileMapping(void* address, size_t size)
      : address_(address), size_(size) {}
  explicit SampledQgramFileMapping(std::vector<uint8_t> bytes)
      : owned_(std::move(bytes)), size_(owned_.size()) {}
  ~SampledQgramFileMapping() {
#if defined(__unix__) || defined(__APPLE__)
    if (address_ && size_ != 0) munmap(address_, size_);
#endif
  }

  const uint8_t* data() const {
    return address_ ? static_cast<const uint8_t*>(address_)
                    : owned_.data();
  }
  size_t size() const { return size_; }

  void initialize_bucket_validation(size_t bucket_count) {
    validated_bucket_count_ = bucket_count;
    validated_buckets_ = std::make_unique<std::atomic<uint8_t>[]>(
        bucket_count);
    for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
      validated_buckets_[bucket].store(0, std::memory_order_relaxed);
    }
  }
  bool begin_bucket_validation(size_t bucket) const {
    if (bucket >= validated_bucket_count_) return true;
    auto& state = validated_buckets_[bucket];
    for (;;) {
      uint8_t expected = 0;
      if (state.compare_exchange_weak(
              expected, 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        return true;
      }
      if (expected == 2) return false;
      std::this_thread::yield();
    }
  }
  void finish_bucket_validation(size_t bucket) const {
    if (bucket < validated_bucket_count_) {
      validated_buckets_[bucket].store(2, std::memory_order_release);
    }
  }
  void cancel_bucket_validation(size_t bucket) const {
    if (bucket < validated_bucket_count_) {
      validated_buckets_[bucket].store(0, std::memory_order_release);
    }
  }

 private:
  void* address_ = nullptr;
  std::vector<uint8_t> owned_;
  size_t size_ = 0;
  std::unique_ptr<std::atomic<uint8_t>[]> validated_buckets_;
  size_t validated_bucket_count_ = 0;
};

namespace {

constexpr std::array<char, 8> kSampledQgramMagic = {
    'N', 'G', 'Q', 'P', 'O', 'S', '0', '6'};
constexpr uint32_t kSampledQgramFormatVersion = 6;
constexpr uint32_t kSampledQgramK = 13;
constexpr uint32_t kSampledQgramPrefixK = 10;
constexpr uint32_t kSampledQgramPeriod = 12;
constexpr uint32_t kSampledQgramPositionWidth = 4;
constexpr uint32_t kSampledQgramSuffixBits =
    2 * (kSampledQgramK - kSampledQgramPrefixK);
constexpr size_t kSampledQgramHeaderBytes = 136;
constexpr size_t kBuildSliceBases = size_t{4} << 20;
constexpr size_t kSpoolGroupCount = 256;
constexpr size_t kSpoolBufferRecords = 4096;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

template <typename T>
void hash_pod(uint64_t* hash, T value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  for (size_t idx = 0; idx < sizeof(T); ++idx) {
    *hash ^= bytes[idx];
    *hash *= kFnvPrime;
  }
}

void hash_bytes(uint64_t* hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t idx = 0; idx < size; ++idx) {
    *hash ^= bytes[idx];
    *hash *= kFnvPrime;
  }
}

void hash_string(uint64_t* hash, const std::string& value) {
  hash_pod<uint64_t>(hash, value.size());
  hash_bytes(hash, value.data(), value.size());
}

uint64_t fingerprint_checksum(const std::string& fingerprint) {
  uint64_t checksum = kFnvOffset;
  hash_string(&checksum, fingerprint);
  return checksum;
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

bool encode_qgram(std::string_view sequence, uint32_t* code) {
  if (!code || sequence.size() != kSampledQgramK) return false;
  uint32_t value = 0;
  for (char base : sequence) {
    const int encoded = dna_code(base);
    if (encoded < 0) return false;
    value = (value << 2) | static_cast<uint32_t>(encoded);
  }
  *code = value;
  return true;
}

bool pack_acgt_word(std::string_view sequence, uint64_t* packed) {
  if (!packed || sequence.size() > 32) return false;
  uint64_t value = 0;
  for (size_t idx = 0; idx < sequence.size(); ++idx) {
    const int encoded = dna_code(sequence[idx]);
    if (encoded < 0) return false;
    value |= static_cast<uint64_t>(encoded) << (2 * idx);
  }
  *packed = value;
  return true;
}

size_t checked_add(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::runtime_error("sampled q-gram index size overflow");
  }
  return left + right;
}

size_t checked_multiply(size_t left, size_t right) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error("sampled q-gram index size overflow");
  }
  return left * right;
}

size_t align_offset(size_t offset, size_t alignment) {
  const size_t remainder = offset % alignment;
  return remainder == 0 ? offset
                        : checked_add(offset, alignment - remainder);
}

struct SampledQgramLayout {
  size_t contigs_begin = kSampledQgramHeaderBytes;
  size_t offsets_begin = 0;
  size_t checksums_begin = 0;
  size_t suffixes_begin = 0;
  size_t positions_begin = 0;
  size_t total_size = 0;
};

SampledQgramLayout sampled_qgram_layout(
    size_t contig_count, size_t prefix_bucket_count,
    size_t position_count) {
  SampledQgramLayout layout;
  layout.offsets_begin = align_offset(
      checked_add(
          layout.contigs_begin,
          checked_multiply(contig_count, size_t{3} * sizeof(uint32_t))),
      alignof(uint32_t));
  layout.checksums_begin = align_offset(
      checked_add(
          layout.offsets_begin,
          checked_multiply(
              prefix_bucket_count + 1, sizeof(uint32_t))),
      alignof(uint64_t));
  layout.suffixes_begin = align_offset(
      checked_add(
          layout.checksums_begin,
          checked_multiply(prefix_bucket_count, sizeof(uint64_t))),
      64);
  const size_t suffix_bytes = checked_add(
      checked_multiply(position_count, kSampledQgramSuffixBits), 7) / 8;
  layout.positions_begin = align_offset(
      checked_add(layout.suffixes_begin, suffix_bytes), 64);
  layout.total_size = checked_add(
      layout.positions_begin,
      checked_multiply(position_count, sizeof(uint32_t)));
  return layout;
}

uint16_t packed_suffix_at(const uint8_t* suffixes, size_t entry) {
  const size_t bit_offset = entry * kSampledQgramSuffixBits;
  const size_t byte_offset = bit_offset >> 3;
  const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
  const size_t byte_count =
      (shift + kSampledQgramSuffixBits + 7) / 8;
  uint32_t word = 0;
  for (size_t byte = 0; byte < byte_count; ++byte) {
    word |= static_cast<uint32_t>(suffixes[byte_offset + byte]) <<
        (8 * byte);
  }
  return static_cast<uint16_t>(
      (word >> shift) & ((uint32_t{1} << kSampledQgramSuffixBits) - 1));
}

void set_packed_suffix(
    uint8_t* suffixes, size_t entry, uint16_t suffix) {
  const size_t bit_offset = entry * kSampledQgramSuffixBits;
  const size_t byte_offset = bit_offset >> 3;
  const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
  const size_t byte_count =
      (shift + kSampledQgramSuffixBits + 7) / 8;
  const uint32_t mask =
      ((uint32_t{1} << kSampledQgramSuffixBits) - 1) << shift;
  uint32_t word = 0;
  for (size_t byte = 0; byte < byte_count; ++byte) {
    word |= static_cast<uint32_t>(suffixes[byte_offset + byte]) <<
        (8 * byte);
  }
  word = (word & ~mask) | (static_cast<uint32_t>(suffix) << shift);
  for (size_t byte = 0; byte < byte_count; ++byte) {
    suffixes[byte_offset + byte] = static_cast<uint8_t>(
        word >> (8 * byte));
  }
}

uint64_t prefix_bucket_checksum(
    uint32_t prefix, const uint8_t* suffixes,
    const uint32_t* positions, size_t begin, size_t end) {
  uint64_t checksum = kFnvOffset;
  hash_pod<uint32_t>(&checksum, prefix);
  hash_pod<uint64_t>(&checksum, end - begin);
  for (size_t entry = begin; entry < end; ++entry) {
    hash_pod<uint16_t>(&checksum, packed_suffix_at(suffixes, entry));
    hash_pod<uint32_t>(&checksum, positions[entry]);
  }
  return checksum;
}

uint64_t metadata_checksum(
    size_t sequence_size,
    const std::vector<ReferenceContig>& contigs,
    size_t bucket_count,
    size_t prefix_bucket_count,
    size_t position_count,
    uint64_t reference_checksum,
    const SampledQgramLayout& layout,
    const uint32_t* offsets,
    const uint64_t* checksums) {
  uint64_t checksum = kFnvOffset;
  hash_pod<uint32_t>(&checksum, kSampledQgramFormatVersion);
  hash_pod<uint32_t>(&checksum, kSampledQgramK);
  hash_pod<uint32_t>(&checksum, kSampledQgramPrefixK);
  hash_pod<uint32_t>(&checksum, kSampledQgramPeriod);
  hash_pod<uint32_t>(&checksum, kSampledQgramPositionWidth);
  hash_pod<uint32_t>(&checksum, kSampledQgramSuffixBits);
  hash_pod<uint64_t>(&checksum, sequence_size);
  hash_pod<uint64_t>(&checksum, contigs.size());
  hash_pod<uint64_t>(&checksum, bucket_count);
  hash_pod<uint64_t>(&checksum, prefix_bucket_count);
  hash_pod<uint64_t>(&checksum, position_count);
  hash_pod<uint64_t>(&checksum, reference_checksum);
  hash_pod<uint64_t>(&checksum, layout.offsets_begin);
  hash_pod<uint64_t>(&checksum, layout.checksums_begin);
  hash_pod<uint64_t>(&checksum, layout.suffixes_begin);
  hash_pod<uint64_t>(&checksum, layout.positions_begin);
  hash_pod<uint64_t>(&checksum, layout.total_size);
  for (const auto& contig : contigs) {
    hash_pod<uint32_t>(&checksum, contig.begin);
    hash_pod<uint32_t>(&checksum, contig.end);
    hash_pod<uint32_t>(&checksum, contig.source_begin);
  }
  hash_bytes(
      &checksum, offsets,
      (prefix_bucket_count + 1) * sizeof(uint32_t));
  hash_bytes(
      &checksum, checksums,
      prefix_bucket_count * sizeof(uint64_t));
  return checksum;
}

std::filesystem::path sampled_qgram_output_path(
    const std::filesystem::path& manifest_path) {
  return manifest_path.string() + ".qpos";
}

struct SampledQgramSpoolRecord {
  uint32_t code = 0;
  uint32_t position = 0;
};
static_assert(sizeof(SampledQgramSpoolRecord) == 8,
              "sampled q-gram spool records must remain compact");

class TemporarySpoolDirectory {
 public:
  explicit TemporarySpoolDirectory(std::filesystem::path path)
      : path_(std::move(path)) {
    std::filesystem::create_directories(path_);
  }
  ~TemporarySpoolDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

template <typename Callback>
void for_each_sampled_qgram(
    const PackedReferenceFile& reference, Callback&& callback) {
  std::string slice;
  for (const auto& contig : reference.contigs) {
    const size_t contig_size =
        static_cast<size_t>(contig.end) - contig.begin;
    if (contig_size < kSampledQgramK) continue;
    for (size_t chunk_begin = 0;
         chunk_begin + kSampledQgramK <= contig_size;
         chunk_begin += kBuildSliceBases) {
      const size_t sample_end = std::min(
          contig_size - kSampledQgramK + 1,
          chunk_begin + kBuildSliceBases);
      const size_t slice_end = std::min(
          contig_size, sample_end + kSampledQgramK - 1);
      reference.slice(
          static_cast<size_t>(contig.begin) + chunk_begin,
          static_cast<size_t>(contig.begin) + slice_end, &slice);
      size_t sample = chunk_begin;
      const size_t remainder = sample % kSampledQgramPeriod;
      if (remainder != 0) {
        sample += kSampledQgramPeriod - remainder;
      }
      for (; sample < sample_end; sample += kSampledQgramPeriod) {
        uint32_t code = 0;
        if (encode_qgram(
                std::string_view(slice).substr(
                    sample - chunk_begin, kSampledQgramK),
                &code)) {
          callback(code, static_cast<uint32_t>(
                             static_cast<size_t>(contig.begin) + sample));
        }
      }
    }
  }
}

void build_sampled_qgram_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest,
    const PackedReferenceFile& reference) {
  if (reference.sequence_size == 0 ||
      reference.sequence_size > std::numeric_limits<uint32_t>::max() ||
      reference.contigs.size() != manifest.contig_ids.size()) {
    throw std::runtime_error(
        "reference is incompatible with sampled q-gram positions");
  }
  constexpr size_t bucket_count =
      size_t{1} << (2 * kSampledQgramK);
  constexpr size_t prefix_bucket_count =
      size_t{1} << (2 * kSampledQgramPrefixK);
  std::vector<uint32_t> prefix_counts(
      prefix_bucket_count, uint32_t{0});
  const auto output_path = sampled_qgram_output_path(manifest_path);
#if defined(__unix__) || defined(__APPLE__)
  TemporarySpoolDirectory spool_directory(
      std::filesystem::temp_directory_path() /
      ("navigamer-qpos-" + std::to_string(
           static_cast<unsigned long long>(getpid()))));
  std::vector<std::ofstream> spool_outputs(kSpoolGroupCount);
  std::vector<std::vector<SampledQgramSpoolRecord>> spool_buffers(
      kSpoolGroupCount);
  for (size_t group = 0; group < kSpoolGroupCount; ++group) {
    const auto path = spool_directory.path() /
        std::to_string(group);
    spool_outputs[group].open(
        path, std::ios::binary | std::ios::trunc);
    if (!spool_outputs[group]) {
      throw std::runtime_error(
          "unable to create sampled q-gram construction spool");
    }
    spool_buffers[group].reserve(kSpoolBufferRecords);
  }
  const auto flush_spool = [&](size_t group) {
    auto& buffer = spool_buffers[group];
    if (buffer.empty()) return;
    spool_outputs[group].write(
        reinterpret_cast<const char*>(buffer.data()),
        static_cast<std::streamsize>(
            buffer.size() * sizeof(SampledQgramSpoolRecord)));
    if (!spool_outputs[group]) {
      throw std::runtime_error(
          "unable to write sampled q-gram construction spool");
    }
    buffer.clear();
  };
#endif
  size_t position_count = 0;
  for_each_sampled_qgram(
      reference, [&](uint32_t code, uint32_t position) {
        constexpr uint32_t suffix_bits =
            2 * (kSampledQgramK - kSampledQgramPrefixK);
        const size_t prefix = code >> suffix_bits;
        if (prefix_counts[prefix] ==
            std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error(
              "sampled q-gram prefix exceeds 32-bit count");
        }
        ++prefix_counts[prefix];
        ++position_count;
#if defined(__unix__) || defined(__APPLE__)
        constexpr uint32_t group_shift =
            2 * kSampledQgramK - 8;
        const size_t group = code >> group_shift;
        auto& buffer = spool_buffers[group];
        buffer.push_back({code, position});
        if (buffer.size() == kSpoolBufferRecords) {
          flush_spool(group);
        }
#endif
      });
#if defined(__unix__) || defined(__APPLE__)
  for (size_t group = 0; group < kSpoolGroupCount; ++group) {
    flush_spool(group);
    spool_outputs[group].close();
    if (!spool_outputs[group]) {
      throw std::runtime_error(
          "unable to finalize sampled q-gram construction spool");
    }
  }
  spool_outputs.clear();
  spool_buffers.clear();
#endif
  if (position_count > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "sampled q-gram position array exceeds 32-bit indexing");
  }
  const auto layout = sampled_qgram_layout(
      reference.contigs.size(), prefix_bucket_count, position_count);
  const auto temporary_path = output_path.string() + ".tmp";
  std::error_code ignored;
  std::filesystem::remove(temporary_path, ignored);

#if defined(__unix__) || defined(__APPLE__)
  int fd = open(
      temporary_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) {
    throw std::runtime_error(
        "unable to create sampled q-gram sidecar: " +
        std::string(std::strerror(errno)));
  }
  void* address = MAP_FAILED;
  try {
    if (layout.total_size >
            static_cast<size_t>(std::numeric_limits<off_t>::max()) ||
        ftruncate(fd, static_cast<off_t>(layout.total_size)) != 0) {
      throw std::runtime_error(
          "unable to size sampled q-gram sidecar");
    }
    address = mmap(
        nullptr, layout.total_size, PROT_READ | PROT_WRITE,
        MAP_SHARED, fd, 0);
    if (address == MAP_FAILED) {
      throw std::runtime_error(
          "unable to map sampled q-gram sidecar for construction");
    }
    auto* bytes = static_cast<uint8_t*>(address);
    auto* offsets = reinterpret_cast<uint32_t*>(
        bytes + layout.offsets_begin);
    auto* checksums = reinterpret_cast<uint64_t*>(
        bytes + layout.checksums_begin);
    auto* suffixes = bytes + layout.suffixes_begin;
    auto* positions = reinterpret_cast<uint32_t*>(
        bytes + layout.positions_begin);
    static const size_t page_size = [] {
      const long raw_page_size = sysconf(_SC_PAGESIZE);
      return raw_page_size > 0
          ? static_cast<size_t>(raw_page_size)
          : size_t{4096};
    }();
    const auto flush_and_discard_full_pages = [&](size_t begin,
                                                   size_t end) {
      if (begin >= end || end > layout.total_size) return;
      const size_t aligned_begin =
          begin + (page_size - begin % page_size) % page_size;
      const size_t aligned_end = end - end % page_size;
      if (aligned_begin >= aligned_end) return;
      if (msync(
              bytes + aligned_begin, aligned_end - aligned_begin,
              MS_SYNC) != 0) {
        throw std::runtime_error(
            "unable to flush sampled q-gram construction range");
      }
#if defined(MADV_DONTNEED)
      (void)madvise(
          bytes + aligned_begin, aligned_end - aligned_begin,
          MADV_DONTNEED);
#endif
    };
    offsets[0] = 0;
    constexpr size_t suffix_count =
        size_t{1} << kSampledQgramSuffixBits;
    for (size_t prefix = 0; prefix < prefix_bucket_count; ++prefix) {
      offsets[prefix + 1] =
          offsets[prefix] + prefix_counts[prefix];
    }
    if (offsets[prefix_bucket_count] != position_count) {
      throw std::runtime_error(
          "sampled q-gram position count mismatch");
    }
    constexpr size_t codes_per_spool_group =
        bucket_count / kSpoolGroupCount;
    constexpr size_t prefixes_per_spool_group =
        prefix_bucket_count / kSpoolGroupCount;
    std::vector<SampledQgramSpoolRecord> spool_records(
        size_t{1} << 16);
    const auto for_each_spool_record = [&spool_directory,
                                        &spool_records](
        size_t group, const auto& callback) {
      std::ifstream spool(
          spool_directory.path() / std::to_string(group),
          std::ios::binary);
      if (!spool) {
        throw std::runtime_error(
            "unable to read sampled q-gram construction spool");
      }
      while (spool) {
        spool.read(
            reinterpret_cast<char*>(spool_records.data()),
            static_cast<std::streamsize>(
                spool_records.size() *
                sizeof(SampledQgramSpoolRecord)));
        const std::streamsize bytes_read = spool.gcount();
        if (bytes_read %
                static_cast<std::streamsize>(
                    sizeof(SampledQgramSpoolRecord)) !=
            0) {
          throw std::runtime_error(
              "truncated sampled q-gram construction spool");
        }
        const size_t record_count = static_cast<size_t>(bytes_read) /
            sizeof(SampledQgramSpoolRecord);
        for (size_t record_idx = 0;
             record_idx < record_count; ++record_idx) {
          callback(spool_records[record_idx]);
        }
      }
      if (!spool.eof()) {
        throw std::runtime_error(
            "unable to finish sampled q-gram construction spool");
      }
    };
    for (size_t group = 0; group < kSpoolGroupCount; ++group) {
      const size_t group_code_begin = group * codes_per_spool_group;
      std::vector<uint32_t> local_counts(
          codes_per_spool_group, uint32_t{0});
      for_each_spool_record(group, [&](const auto& record) {
        const size_t expected_group =
            record.code / codes_per_spool_group;
        if (expected_group != group || record.code >= bucket_count) {
          throw std::runtime_error(
              "invalid sampled q-gram construction spool record");
        }
        auto& count = local_counts[record.code - group_code_begin];
        if (count == std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error(
              "sampled q-gram posting list exceeds 32-bit count");
        }
        ++count;
      });
      const size_t group_prefix_begin =
          group * prefixes_per_spool_group;
      const size_t group_prefix_end =
          group_prefix_begin + prefixes_per_spool_group;
      uint32_t running = static_cast<uint32_t>(
          offsets[group_prefix_begin]);
      for (size_t local_code = 0;
           local_code < local_counts.size(); ++local_code) {
        const uint32_t count = local_counts[local_code];
        local_counts[local_code] = running;
        running += count;
      }
      if (running != offsets[group_prefix_end]) {
        throw std::runtime_error(
            "sampled q-gram group count mismatch");
      }
      std::vector<uint32_t> cursors = local_counts;
      for_each_spool_record(group, [&](const auto& record) {
        const size_t local_code = record.code - group_code_begin;
        const size_t entry = cursors[local_code]++;
        set_packed_suffix(
            suffixes, entry,
            static_cast<uint16_t>(
                record.code & (suffix_count - 1)));
        positions[entry] = record.position;
      });
      for (size_t local_code = 0;
           local_code < local_counts.size(); ++local_code) {
        const uint32_t expected_end =
            local_code + 1 == local_counts.size()
                ? running
                : local_counts[local_code + 1];
        if (cursors[local_code] != expected_end) {
          throw std::runtime_error(
              "sampled q-gram fill count mismatch");
        }
      }
      for (size_t prefix = group_prefix_begin;
           prefix < group_prefix_end; ++prefix) {
        checksums[prefix] = prefix_bucket_checksum(
            static_cast<uint32_t>(prefix), suffixes, positions,
            static_cast<size_t>(offsets[prefix]),
            static_cast<size_t>(offsets[prefix + 1]));
      }
      const size_t entry_begin = static_cast<size_t>(
          offsets[group_prefix_begin]);
      const size_t entry_end = static_cast<size_t>(
          offsets[group_prefix_end]);
      flush_and_discard_full_pages(
          layout.suffixes_begin +
              entry_begin * kSampledQgramSuffixBits / 8,
          layout.suffixes_begin +
              (entry_end * kSampledQgramSuffixBits + 7) / 8);
      flush_and_discard_full_pages(
          layout.positions_begin + entry_begin * sizeof(uint32_t),
          layout.positions_begin + entry_end * sizeof(uint32_t));
    }

    uint8_t* contig_bytes = bytes + layout.contigs_begin;
    for (const auto& contig : reference.contigs) {
      std::memcpy(contig_bytes, &contig.begin, sizeof(contig.begin));
      contig_bytes += sizeof(contig.begin);
      std::memcpy(contig_bytes, &contig.end, sizeof(contig.end));
      contig_bytes += sizeof(contig.end);
      std::memcpy(
          contig_bytes, &contig.source_begin,
          sizeof(contig.source_begin));
      contig_bytes += sizeof(contig.source_begin);
    }
    const uint64_t reference_checksum =
        fingerprint_checksum(manifest.part_manifest.ref_fingerprint);
    const uint64_t stored_metadata_checksum = metadata_checksum(
        reference.sequence_size, reference.contigs, bucket_count,
        prefix_bucket_count, position_count, reference_checksum,
        layout, offsets, checksums);
    uint8_t* cursor = bytes;
    const auto write_field = [&cursor](const auto& value) {
      std::memcpy(cursor, &value, sizeof(value));
      cursor += sizeof(value);
    };
    write_field(kSampledQgramMagic);
    write_field(kSampledQgramFormatVersion);
    write_field(kSampledQgramK);
    write_field(kSampledQgramPrefixK);
    write_field(kSampledQgramPeriod);
    write_field(kSampledQgramPositionWidth);
    write_field(kSampledQgramSuffixBits);
    write_field(static_cast<uint64_t>(reference.sequence_size));
    write_field(static_cast<uint64_t>(reference.contigs.size()));
    write_field(static_cast<uint64_t>(bucket_count));
    write_field(static_cast<uint64_t>(prefix_bucket_count));
    write_field(static_cast<uint64_t>(position_count));
    write_field(reference_checksum);
    write_field(stored_metadata_checksum);
    write_field(static_cast<uint64_t>(layout.offsets_begin));
    write_field(static_cast<uint64_t>(layout.checksums_begin));
    write_field(static_cast<uint64_t>(layout.suffixes_begin));
    write_field(static_cast<uint64_t>(layout.positions_begin));
    write_field(static_cast<uint64_t>(layout.total_size));
    write_field(uint64_t{0});
    if (static_cast<size_t>(cursor - bytes) !=
        kSampledQgramHeaderBytes) {
      throw std::runtime_error(
          "sampled q-gram header layout mismatch");
    }
    if (msync(address, layout.total_size, MS_SYNC) != 0) {
      throw std::runtime_error(
          "unable to flush sampled q-gram sidecar");
    }
    munmap(address, layout.total_size);
    address = MAP_FAILED;
    if (close(fd) != 0) {
      fd = -1;
      throw std::runtime_error(
          "unable to close sampled q-gram sidecar");
    }
    fd = -1;
    std::filesystem::rename(temporary_path, output_path);
  } catch (...) {
    if (address != MAP_FAILED) munmap(address, layout.total_size);
    if (fd >= 0) close(fd);
    std::filesystem::remove(temporary_path, ignored);
    throw;
  }
#else
  (void)prefix_counts;
  (void)layout;
  (void)output_path;
  (void)temporary_path;
  throw std::runtime_error(
      "sampled q-gram index construction requires memory mapping");
#endif
}

size_t containing_contig(
    const std::vector<ReferenceContig>& contigs,
    uint32_t global_position) {
  const auto found = std::upper_bound(
      contigs.begin(), contigs.end(), global_position,
      [](uint32_t position, const ReferenceContig& contig) {
        return position < contig.end;
      });
  if (found == contigs.end() || global_position < found->begin) {
    return contigs.size();
  }
  return static_cast<size_t>(found - contigs.begin());
}

void sort_unique_uint32(std::vector<uint32_t>* values) {
  if (!values || values->size() < 2) return;
  constexpr size_t kRadixThreshold = size_t{32} << 10;
  if (values->size() < kRadixThreshold) {
    std::sort(values->begin(), values->end());
  } else {
    constexpr size_t kBucketCount = size_t{1} << 16;
    std::vector<size_t> offsets(kBucketCount);
    std::vector<uint32_t> scratch(values->size());
    // Two stable 16-bit passes produce the same unsigned order as std::sort,
    // but avoid comparison-sort growth on repeat-heavy posting lists.
    const auto pass = [&](const std::vector<uint32_t>& source,
                          std::vector<uint32_t>* destination,
                          uint32_t shift) {
      std::fill(offsets.begin(), offsets.end(), size_t{0});
      for (uint32_t value : source) {
        ++offsets[(value >> shift) & 0xffffU];
      }
      size_t begin = 0;
      for (size_t bucket = 0; bucket < offsets.size(); ++bucket) {
        const size_t count = offsets[bucket];
        offsets[bucket] = begin;
        begin += count;
      }
      for (uint32_t value : source) {
        (*destination)[offsets[(value >> shift) & 0xffffU]++] = value;
      }
    };
    pass(*values, &scratch, 0);
    pass(scratch, values, 16);
  }
  values->erase(
      std::unique(values->begin(), values->end()), values->end());
}

}  // namespace

bool SampledQgramIndex::enabled() const {
  return mapping && k != 0 && prefix_k != 0 && prefix_k < k &&
         sample_period != 0 && sequence_size != 0 &&
         bucket_count != 0 && prefix_bucket_count != 0 &&
         bucket_offsets && bucket_checksums && packed_suffixes &&
         positions;
}

bool SampledQgramIndex::supports(
    std::string_view query, int tolerance) const {
  if (!enabled() || tolerance < 0 || query.empty()) return false;
  const size_t partition_count = static_cast<size_t>(tolerance) + 1;
  if (partition_count == 0 || partition_count > query.size() ||
      query.size() / partition_count <
          static_cast<size_t>(k) + sample_period - 1) {
    return false;
  }
  return std::all_of(query.begin(), query.end(), [](char base) {
    const char normalized = static_cast<char>(
        std::toupper(static_cast<unsigned char>(base)));
    return normalized == 'A' || normalized == 'C' ||
           normalized == 'G' || normalized == 'T';
  });
}

std::pair<const uint32_t*, const uint32_t*>
SampledQgramIndex::posting_list(uint32_t code) const {
  if (!enabled() || code >= bucket_count) {
    throw std::out_of_range("sampled q-gram code");
  }
  const uint32_t suffix_bits = 2 * (k - prefix_k);
  const uint32_t suffix_mask =
      (uint32_t{1} << suffix_bits) - 1;
  const uint32_t prefix = code >> suffix_bits;
  const uint16_t suffix = static_cast<uint16_t>(code & suffix_mask);
  if (prefix >= prefix_bucket_count) {
    throw std::runtime_error(
        "invalid sampled q-gram prefix");
  }
  const size_t begin = static_cast<size_t>(bucket_offsets[prefix]);
  const size_t end = static_cast<size_t>(bucket_offsets[prefix + 1]);
  if (begin > end || end > position_count) {
    throw std::runtime_error(
        "invalid sampled q-gram posting range");
  }
  if (mapping->begin_bucket_validation(prefix)) {
    const uint64_t actual = prefix_bucket_checksum(
        prefix, packed_suffixes, positions, begin, end);
    if (actual != bucket_checksums[prefix]) {
      mapping->cancel_bucket_validation(prefix);
      throw std::runtime_error(
          "sampled q-gram posting checksum mismatch");
    }
    mapping->finish_bucket_validation(prefix);
  }
  const auto lower_suffix = [&](size_t first, size_t last,
                                bool upper) {
    while (first < last) {
      const size_t middle = first + (last - first) / 2;
      const uint16_t value = packed_suffix_at(
          packed_suffixes, middle);
      if (value < suffix || (upper && value == suffix)) {
        first = middle + 1;
      } else {
        last = middle;
      }
    }
    return first;
  };
  const size_t exact_begin = lower_suffix(begin, end, false);
  const size_t exact_end = lower_suffix(exact_begin, end, true);
  return {positions + exact_begin, positions + exact_end};
}

SampledQgramIndex load_sampled_qgram_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest,
    const PackedReferenceFile& reference) {
  const auto path = sampled_qgram_output_path(manifest_path);
  std::shared_ptr<SampledQgramFileMapping> mapping;
#if defined(__unix__) || defined(__APPLE__)
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error(
        "unable to open sampled q-gram sidecar");
  }
  struct stat status {};
  if (fstat(fd, &status) != 0 || status.st_size < 0 ||
      static_cast<uint64_t>(status.st_size) >
          std::numeric_limits<size_t>::max()) {
    close(fd);
    throw std::runtime_error(
        "invalid sampled q-gram sidecar size");
  }
  const size_t mapped_size = static_cast<size_t>(status.st_size);
  void* address = mapped_size == 0
      ? MAP_FAILED
      : mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (address == MAP_FAILED) {
    throw std::runtime_error(
        "unable to map sampled q-gram sidecar");
  }
#if defined(MADV_RANDOM)
  (void)madvise(address, mapped_size, MADV_RANDOM);
#endif
  mapping = std::make_shared<SampledQgramFileMapping>(
      address, mapped_size);
#else
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || in.tellg() < 0 ||
      static_cast<uint64_t>(in.tellg()) >
          std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(
        "unable to open sampled q-gram sidecar");
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(in.tellg()));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  if (!in) {
    throw std::runtime_error(
        "unable to read sampled q-gram sidecar");
  }
  mapping = std::make_shared<SampledQgramFileMapping>(
      std::move(bytes));
#endif
  if (mapping->size() < kSampledQgramHeaderBytes) {
    throw std::runtime_error("truncated sampled q-gram header");
  }
  const uint8_t* cursor = mapping->data();
  const auto read_field = [&cursor](auto* value) {
    std::memcpy(value, cursor, sizeof(*value));
    cursor += sizeof(*value);
  };
  std::array<char, 8> magic{};
  uint32_t version = 0;
  uint32_t k = 0;
  uint32_t prefix_k = 0;
  uint32_t sample_period = 0;
  uint32_t position_width = 0;
  uint32_t suffix_bits = 0;
  uint64_t sequence_size = 0;
  uint64_t contig_count = 0;
  uint64_t bucket_count = 0;
  uint64_t prefix_bucket_count = 0;
  uint64_t position_count = 0;
  uint64_t reference_checksum = 0;
  uint64_t stored_metadata_checksum = 0;
  uint64_t offsets_begin = 0;
  uint64_t checksums_begin = 0;
  uint64_t suffixes_begin = 0;
  uint64_t positions_begin = 0;
  uint64_t stored_total_size = 0;
  uint64_t reserved = 0;
  read_field(&magic);
  read_field(&version);
  read_field(&k);
  read_field(&prefix_k);
  read_field(&sample_period);
  read_field(&position_width);
  read_field(&suffix_bits);
  read_field(&sequence_size);
  read_field(&contig_count);
  read_field(&bucket_count);
  read_field(&prefix_bucket_count);
  read_field(&position_count);
  read_field(&reference_checksum);
  read_field(&stored_metadata_checksum);
  read_field(&offsets_begin);
  read_field(&checksums_begin);
  read_field(&suffixes_begin);
  read_field(&positions_begin);
  read_field(&stored_total_size);
  read_field(&reserved);
  if (sequence_size > std::numeric_limits<size_t>::max() ||
      contig_count > std::numeric_limits<size_t>::max() ||
      bucket_count > std::numeric_limits<size_t>::max() ||
      prefix_bucket_count > std::numeric_limits<size_t>::max() ||
      position_count > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(
        "sampled q-gram sidecar exceeds this platform");
  }
  const auto layout = sampled_qgram_layout(
      static_cast<size_t>(contig_count),
      static_cast<size_t>(prefix_bucket_count),
      static_cast<size_t>(position_count));
  const size_t expected_bucket_count =
      size_t{1} << (2 * kSampledQgramK);
  const size_t expected_prefix_bucket_count =
      size_t{1} << (2 * kSampledQgramPrefixK);
  if (magic != kSampledQgramMagic ||
      version != kSampledQgramFormatVersion ||
      k != kSampledQgramK ||
      prefix_k != kSampledQgramPrefixK ||
      sample_period != kSampledQgramPeriod ||
      position_width != kSampledQgramPositionWidth ||
      suffix_bits != kSampledQgramSuffixBits ||
      sequence_size != reference.sequence_size ||
      contig_count != reference.contigs.size() ||
      contig_count != manifest.contig_ids.size() ||
      bucket_count != expected_bucket_count ||
      prefix_bucket_count != expected_prefix_bucket_count ||
      reference_checksum != fingerprint_checksum(
          manifest.part_manifest.ref_fingerprint) ||
      offsets_begin != layout.offsets_begin ||
      checksums_begin != layout.checksums_begin ||
      suffixes_begin != layout.suffixes_begin ||
      positions_begin != layout.positions_begin ||
      stored_total_size != layout.total_size ||
      stored_total_size != mapping->size() || reserved != 0) {
    throw std::runtime_error(
        "sampled q-gram sidecar metadata mismatch");
  }
  std::vector<ReferenceContig> stored_contigs;
  stored_contigs.reserve(reference.contigs.size());
  cursor = mapping->data() + layout.contigs_begin;
  for (size_t contig_idx = 0;
       contig_idx < reference.contigs.size(); ++contig_idx) {
    ReferenceContig contig;
    read_field(&contig.begin);
    read_field(&contig.end);
    read_field(&contig.source_begin);
    contig.id = manifest.contig_ids[contig_idx];
    const auto& expected = reference.contigs[contig_idx];
    if (contig.begin != expected.begin || contig.end != expected.end ||
        contig.source_begin != expected.source_begin ||
        contig.id != expected.id) {
      throw std::runtime_error(
          "sampled q-gram contig metadata mismatch");
    }
    stored_contigs.push_back(std::move(contig));
  }
  if (position_count > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "sampled q-gram position count exceeds offset width");
  }
  const auto* offsets = reinterpret_cast<const uint32_t*>(
      mapping->data() + layout.offsets_begin);
  const auto* checksums = reinterpret_cast<const uint64_t*>(
      mapping->data() + layout.checksums_begin);
  const auto* suffixes = mapping->data() + layout.suffixes_begin;
  const auto* positions = reinterpret_cast<const uint32_t*>(
      mapping->data() + layout.positions_begin);
  if (offsets[0] != 0 ||
      offsets[prefix_bucket_count] != position_count) {
    throw std::runtime_error(
        "sampled q-gram offsets are inconsistent");
  }
  for (size_t prefix = 0; prefix < prefix_bucket_count; ++prefix) {
    if (offsets[prefix] > offsets[prefix + 1]) {
      throw std::runtime_error(
          "sampled q-gram offsets are not monotonic");
    }
  }
  const uint64_t actual_metadata_checksum = metadata_checksum(
      static_cast<size_t>(sequence_size), stored_contigs,
      static_cast<size_t>(bucket_count),
      static_cast<size_t>(prefix_bucket_count),
      static_cast<size_t>(position_count), reference_checksum,
      layout, offsets, checksums);
  if (actual_metadata_checksum != stored_metadata_checksum) {
    throw std::runtime_error(
        "sampled q-gram metadata checksum mismatch");
  }
  mapping->initialize_bucket_validation(
      static_cast<size_t>(prefix_bucket_count));
  SampledQgramIndex index;
  index.path = path.string();
  index.k = k;
  index.prefix_k = prefix_k;
  index.sample_period = sample_period;
  index.sequence_size = static_cast<size_t>(sequence_size);
  index.bucket_count = static_cast<size_t>(bucket_count);
  index.prefix_bucket_count =
      static_cast<size_t>(prefix_bucket_count);
  index.position_count = static_cast<size_t>(position_count);
  index.mapping = std::move(mapping);
  index.bucket_offsets = offsets;
  index.bucket_checksums = checksums;
  index.packed_suffixes = suffixes;
  index.positions = positions;
  return index;
}

void ensure_sampled_qgram_index(
    const std::string& manifest_path,
    const ShardedIndexManifest& manifest,
    const PackedReferenceFile& reference) {
  try {
    (void)load_sampled_qgram_index(
        manifest_path, manifest, reference);
    return;
  } catch (const std::exception&) {
  }
  build_sampled_qgram_index(manifest_path, manifest, reference);
  (void)load_sampled_qgram_index(
      manifest_path, manifest, reference);
}

std::vector<ExactBlockVerificationResult>
verify_by_sampled_qgram_positions_batch(
    int tolerance,
    const ShardedIndexManifest& manifest,
    const PackedReferenceFile& reference,
    const SampledQgramIndex& occurrence_index,
    const std::vector<std::string_view>& queries) {
  std::vector<ExactBlockVerificationResult> results(queries.size());
  if (queries.empty() || tolerance < 0 || manifest.stride == 0 ||
      reference.sequence_size != occurrence_index.sequence_size ||
      reference.contigs.size() != manifest.contig_ids.size()) {
    return results;
  }
  struct QueryPlan {
    std::string query;
    PreparedMyersPattern prepared;
    PreparedMyersDnaPattern batch_prepared;
    std::vector<uint32_t> candidate_starts;
  };
  std::vector<QueryPlan> plans(queries.size());
  std::vector<std::exception_ptr> errors(queries.size());
#pragma omp parallel for schedule(dynamic, 1) if(queries.size() > 1)
  for (size_t query_idx = 0; query_idx < queries.size(); ++query_idx) {
    try {
      if (!occurrence_index.supports(queries[query_idx], tolerance)) {
        continue;
      }
      auto& plan = plans[query_idx];
      plan.query.assign(queries[query_idx]);
      std::transform(
          plan.query.begin(), plan.query.end(), plan.query.begin(),
          [](unsigned char base) {
            return static_cast<char>(std::toupper(base));
          });
      const size_t partition_count =
          static_cast<size_t>(tolerance) + 1;
      // The edit-distance pigeonhole proof holds for any d+1 disjoint query
      // blocks. Choose their boundaries by posting cost, while keeping every
      // block long enough for the sampled-qgram coverage proof below.
      const size_t exact_seed_span =
          static_cast<size_t>(occurrence_index.k) +
          occurrence_index.sample_period - 1;
      const size_t qgram_start_count =
          plan.query.size() - occurrence_index.k + 1;
      std::vector<std::pair<const uint32_t*, const uint32_t*>>
          query_postings(qgram_start_count);
      std::vector<uint64_t> posting_counts(qgram_start_count);
      uint32_t rolling_qgram = 0;
      if (!encode_qgram(
              std::string_view(plan.query).substr(
                  0, occurrence_index.k),
              &rolling_qgram)) {
        throw std::runtime_error(
            "invalid exact block in sampled q-gram verifier");
      }
      const uint32_t qgram_mask =
          (uint32_t{1} << (2 * occurrence_index.k)) - 1;
      for (size_t qgram_start = 0;
           qgram_start < qgram_start_count; ++qgram_start) {
        if (qgram_start != 0) {
          const int encoded = dna_code(
              plan.query[qgram_start + occurrence_index.k - 1]);
          if (encoded < 0) {
            throw std::runtime_error(
                "invalid exact block in sampled q-gram verifier");
          }
          rolling_qgram =
              ((rolling_qgram << 2) & qgram_mask) |
              static_cast<uint32_t>(encoded);
        }
        query_postings[qgram_start] =
            occurrence_index.posting_list(rolling_qgram);
        posting_counts[qgram_start] = static_cast<uint64_t>(
            query_postings[qgram_start].second -
            query_postings[qgram_start].first);
      }
      std::vector<uint64_t> anchor_cost(
          plan.query.size() - exact_seed_span + 1);
      // Every exact span of k + period - 1 bases contains `period`
      // consecutive q-gram starts, exactly one of which has sampled phase.
      // Any such span inside an exact block is therefore a no-FN anchor.
      uint64_t sliding_cost = 0;
      for (size_t qgram_start = 0;
           qgram_start < occurrence_index.sample_period;
           ++qgram_start) {
        sliding_cost += posting_counts[qgram_start];
      }
      for (size_t anchor = 0; anchor < anchor_cost.size(); ++anchor) {
        anchor_cost[anchor] = sliding_cost;
        if (anchor + occurrence_index.sample_period <
            posting_counts.size()) {
          sliding_cost -= posting_counts[anchor];
          sliding_cost += posting_counts[
              anchor + occurrence_index.sample_period];
        }
      }
      struct ExactBlockPlan {
        size_t begin = 0;
        size_t end = 0;
        size_t anchor = 0;
      };
      const size_t state_width = plan.query.size() + 1;
      const uint64_t unreachable =
          std::numeric_limits<uint64_t>::max();
      std::vector<uint64_t> partition_cost(
          (partition_count + 1) * state_width, unreachable);
      std::vector<size_t> previous_begin(
          partition_cost.size(), state_width);
      std::vector<size_t> selected_anchor(
          partition_cost.size(), state_width);
      partition_cost[0] = 0;
      for (size_t partition = 1; partition <= partition_count;
           ++partition) {
        const size_t minimum_end = partition * exact_seed_span;
        const size_t maximum_end = plan.query.size() -
            (partition_count - partition) * exact_seed_span;
        for (size_t block_end = minimum_end;
             block_end <= maximum_end; ++block_end) {
          const size_t minimum_begin =
              (partition - 1) * exact_seed_span;
          const size_t maximum_begin = block_end - exact_seed_span;
          for (size_t block_begin = minimum_begin;
               block_begin <= maximum_begin; ++block_begin) {
            const uint64_t prefix_cost = partition_cost[
                (partition - 1) * state_width + block_begin];
            if (prefix_cost == unreachable) continue;
            size_t best_anchor = block_begin;
            uint64_t best_anchor_cost = anchor_cost[block_begin];
            for (size_t anchor = block_begin + 1;
                 anchor <= block_end - exact_seed_span; ++anchor) {
              if (anchor_cost[anchor] < best_anchor_cost) {
                best_anchor = anchor;
                best_anchor_cost = anchor_cost[anchor];
              }
            }
            if (best_anchor_cost > unreachable - prefix_cost) {
              throw std::runtime_error(
                  "sampled q-gram posting cost overflow");
            }
            const uint64_t candidate_cost =
                prefix_cost + best_anchor_cost;
            const size_t state =
                partition * state_width + block_end;
            if (candidate_cost < partition_cost[state]) {
              partition_cost[state] = candidate_cost;
              previous_begin[state] = block_begin;
              selected_anchor[state] = best_anchor;
            }
          }
        }
      }
      std::vector<ExactBlockPlan> exact_blocks(partition_count);
      size_t reconstructed_end = plan.query.size();
      for (size_t partition = partition_count; partition != 0;
           --partition) {
        const size_t state =
            partition * state_width + reconstructed_end;
        if (partition_cost[state] == unreachable ||
            previous_begin[state] == state_width) {
          throw std::runtime_error(
              "unable to plan exact sampled q-gram blocks");
        }
        exact_blocks[partition - 1] = {
            previous_begin[state], reconstructed_end,
            selected_anchor[state]};
        reconstructed_end = previous_begin[state];
      }
      std::vector<uint32_t> block_occurrences;
      for (size_t partition = 0; partition < partition_count;
           ++partition) {
        const size_t block_begin = exact_blocks[partition].begin;
        const size_t block_end = exact_blocks[partition].end;
        const size_t anchor_begin = exact_blocks[partition].anchor;
        const size_t block_length = block_end - block_begin;
        block_occurrences.clear();
        for (size_t shift = 0;
             shift < occurrence_index.sample_period; ++shift) {
          const size_t seed_offset =
              anchor_begin + shift - block_begin;
          const auto postings = query_postings[anchor_begin + shift];
          size_t posting_contig_idx = postings.first == postings.second
              ? reference.contigs.size()
              : containing_contig(
                    reference.contigs, *postings.first);
          for (const uint32_t* posting = postings.first;
               posting != postings.second; ++posting) {
            while (posting_contig_idx < reference.contigs.size() &&
                   *posting >=
                       reference.contigs[posting_contig_idx].end) {
              ++posting_contig_idx;
            }
            if (posting_contig_idx == reference.contigs.size() ||
                *posting <
                    reference.contigs[posting_contig_idx].begin) {
              continue;
            }
            const auto& contig =
                reference.contigs[posting_contig_idx];
            const size_t local_position =
                static_cast<size_t>(*posting) - contig.begin;
            if (local_position < seed_offset) continue;
            const size_t occurrence =
                static_cast<size_t>(*posting) - seed_offset;
            if (occurrence < contig.begin ||
                block_length >
                    static_cast<size_t>(contig.end) - occurrence) {
              continue;
            }
            block_occurrences.push_back(
                static_cast<uint32_t>(occurrence));
          }
        }
        sort_unique_uint32(&block_occurrences);
        const std::string_view query_block(
            plan.query.data() + block_begin, block_length);
        uint64_t packed_query_block = 0;
        const bool packed_query_block_supported =
            pack_acgt_word(query_block, &packed_query_block);
        const auto verify_occurrence_range = [
            &block_occurrences, &reference, &manifest, query_block,
            packed_query_block, packed_query_block_supported,
            block_begin, block_length, tolerance](
            size_t range_begin, size_t range_end,
            std::vector<uint32_t>* candidates) {
          std::string reference_block;
          size_t occurrence_contig_idx = range_begin == range_end
              ? reference.contigs.size()
              : containing_contig(
                    reference.contigs,
                    block_occurrences[range_begin]);
          for (size_t occurrence_idx = range_begin;
               occurrence_idx < range_end; ++occurrence_idx) {
            const uint32_t occurrence =
                block_occurrences[occurrence_idx];
            while (occurrence_contig_idx < reference.contigs.size() &&
                   occurrence >=
                       reference.contigs[occurrence_contig_idx].end) {
              ++occurrence_contig_idx;
            }
            if (occurrence_contig_idx == reference.contigs.size() ||
                occurrence <
                    reference.contigs[occurrence_contig_idx].begin) {
              continue;
            }
            if (packed_query_block_supported) {
              if (!reference.matches_packed_acgt(
                      occurrence_contig_idx, occurrence,
                      packed_query_block, block_length)) {
                continue;
              }
            } else {
              reference.slice(
                  occurrence,
                  static_cast<size_t>(occurrence) + block_length,
                  &reference_block);
              if (reference_block != query_block) continue;
            }
            const auto& contig =
                reference.contigs[occurrence_contig_idx];
            const int64_t nominal_start =
                static_cast<int64_t>(occurrence) -
                static_cast<int64_t>(block_begin);
            for (int edit_shift = -tolerance;
                 edit_shift <= tolerance; ++edit_shift) {
              const int64_t signed_start = nominal_start + edit_shift;
              if (signed_start < static_cast<int64_t>(contig.begin)) {
                continue;
              }
              const size_t start = static_cast<size_t>(signed_start);
              if (manifest.window_length >
                      static_cast<size_t>(contig.end) - start ||
                  (start - contig.begin) % manifest.stride != 0) {
                continue;
              }
              candidates->push_back(static_cast<uint32_t>(start));
            }
          }
        };
        constexpr size_t kParallelVerifyGrain = size_t{2} << 10;
        if (omp_in_parallel() &&
            block_occurrences.size() > 2 * kParallelVerifyGrain) {
          const size_t chunk_count =
              (block_occurrences.size() + kParallelVerifyGrain - 1) /
              kParallelVerifyGrain;
          std::vector<std::vector<uint32_t>> chunk_candidates(
              chunk_count);
          std::vector<std::exception_ptr> chunk_errors(chunk_count);
#pragma omp taskgroup
          {
            for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
              const size_t range_begin = chunk * kParallelVerifyGrain;
              const size_t range_end = std::min(
                  block_occurrences.size(),
                  range_begin + kParallelVerifyGrain);
#pragma omp task firstprivate(chunk, range_begin, range_end) shared(chunk_candidates, chunk_errors, verify_occurrence_range)
              {
                try {
                  verify_occurrence_range(
                      range_begin, range_end,
                      &chunk_candidates[chunk]);
                } catch (...) {
                  chunk_errors[chunk] = std::current_exception();
                }
              }
            }
          }
          for (const auto& chunk_error : chunk_errors) {
            if (chunk_error) std::rethrow_exception(chunk_error);
          }
          for (const auto& candidates : chunk_candidates) {
            plan.candidate_starts.insert(
                plan.candidate_starts.end(), candidates.begin(),
                candidates.end());
          }
        } else {
          verify_occurrence_range(
              0, block_occurrences.size(),
              &plan.candidate_starts);
        }
      }
      sort_unique_uint32(&plan.candidate_starts);
      auto& result = results[query_idx];
      result.enabled = true;
      result.candidate_window_count =
          plan.candidate_starts.size();
      plan.prepared = prepare_myers_pattern(plan.query);
      plan.batch_prepared = prepare_myers_dna_pattern(plan.query);
    } catch (...) {
      errors[query_idx] = std::current_exception();
    }
  }
  for (const auto& error : errors) {
    if (error) std::rethrow_exception(error);
  }

  struct CandidateTask {
    uint32_t start = 0;
    uint32_t query_idx = 0;
  };
  size_t total_candidates = 0;
  for (size_t query_idx = 0; query_idx < plans.size(); ++query_idx) {
    if (plans[query_idx].candidate_starts.size() >
        std::numeric_limits<size_t>::max() -
            total_candidates) {
      throw std::runtime_error(
          "sampled q-gram candidate count overflow");
    }
    total_candidates += plans[query_idx].candidate_starts.size();
  }
  std::vector<CandidateTask> candidate_tasks;
  candidate_tasks.reserve(total_candidates);
  for (size_t query_idx = 0; query_idx < plans.size(); ++query_idx) {
    for (uint32_t start : plans[query_idx].candidate_starts) {
      candidate_tasks.push_back(
          {start, static_cast<uint32_t>(query_idx)});
    }
  }
  struct DistanceThreadResult {
    std::vector<size_t> distance_counts;
    std::vector<std::vector<ExactBlockVerifiedOccurrence>> occurrences;
    std::exception_ptr error;
  };
  const int thread_count = std::max(1, omp_get_max_threads());
  std::vector<DistanceThreadResult> thread_results(
      static_cast<size_t>(thread_count));
  for (auto& thread_result : thread_results) {
    thread_result.distance_counts.assign(queries.size(), 0);
    thread_result.occurrences.resize(queries.size());
  }
#pragma omp parallel if(total_candidates > 1)
  {
    const int thread_id = omp_get_thread_num();
    auto& thread_result =
        thread_results[static_cast<size_t>(thread_id)];
    const size_t active_thread_count = static_cast<size_t>(
        omp_get_num_threads());
    const size_t task_begin =
        total_candidates * static_cast<size_t>(thread_id) /
        active_thread_count;
    const size_t task_end =
        total_candidates * (static_cast<size_t>(thread_id) + 1) /
        active_thread_count;
    std::array<std::string, 4> reference_windows;
    std::array<bool, 4> reference_windows_acgt{};
    std::string combined_reference;
    try {
      size_t task = task_begin;
      size_t hinted_query_idx = plans.size();
      size_t contig_hint = reference.contigs.size();
      while (task < task_end) {
        const size_t query_idx = candidate_tasks[task].query_idx;
        if (query_idx >= plans.size()) {
          throw std::runtime_error(
              "sampled q-gram candidate query mismatch");
        }
        std::array<uint32_t, 4> starts{};
        std::array<size_t, 4> contig_indices{};
        size_t lane_count = 0;
        while (task < task_end && lane_count < starts.size() &&
               candidate_tasks[task].query_idx == query_idx) {
          const uint32_t start = candidate_tasks[task++].start;
          // Candidate starts are sorted within each query, so only the first
          // start needs a binary contig lookup.
          if (hinted_query_idx != query_idx) {
            hinted_query_idx = query_idx;
            contig_hint = containing_contig(reference.contigs, start);
          } else {
            while (contig_hint < reference.contigs.size() &&
                   start >= reference.contigs[contig_hint].end) {
              ++contig_hint;
            }
            if (contig_hint == reference.contigs.size() ||
                start < reference.contigs[contig_hint].begin) {
              contig_hint = containing_contig(reference.contigs, start);
            }
          }
          const size_t contig_idx = contig_hint;
          if (contig_idx == reference.contigs.size()) continue;
          const auto& contig = reference.contigs[contig_idx];
          if (manifest.window_length >
              static_cast<size_t>(contig.end) - start) {
            continue;
          }
          starts[lane_count] = start;
          contig_indices[lane_count] = contig_idx;
          ++lane_count;
        }
        if (lane_count == 0) continue;
        std::array<std::string_view, 4> texts{};
        // Nearby sorted windows overlap heavily. Decode their union once
        // whenever it is no larger than the separate slices.
        bool coalesced = true;
        for (size_t lane = 1; lane < lane_count; ++lane) {
          coalesced = coalesced &&
              contig_indices[lane] == contig_indices[0];
        }
        const size_t combined_size =
            static_cast<size_t>(starts[lane_count - 1]) - starts[0] +
            manifest.window_length;
        coalesced = coalesced &&
            combined_size <= lane_count * manifest.window_length;
        bool combined_reference_acgt = false;
        if (coalesced) {
          combined_reference_acgt = reference.slice_acgt(
              starts[0], static_cast<size_t>(starts[0]) + combined_size,
              &combined_reference);
          for (size_t lane = 0; lane < lane_count; ++lane) {
            texts[lane] = std::string_view(combined_reference).substr(
                static_cast<size_t>(starts[lane]) - starts[0],
                manifest.window_length);
          }
        } else {
          for (size_t lane = 0; lane < lane_count; ++lane) {
            reference_windows_acgt[lane] = reference.slice_acgt(
                starts[lane],
                static_cast<size_t>(starts[lane]) +
                    manifest.window_length,
                &reference_windows[lane]);
            texts[lane] = reference_windows[lane];
          }
        }
        size_t valid_lane_count = 0;
        for (size_t lane = 0; lane < lane_count; ++lane) {
          if (!(coalesced ? combined_reference_acgt
                          : reference_windows_acgt[lane]) &&
              !std::all_of(
                  texts[lane].begin(), texts[lane].end(),
                  [](char base) {
                    return base == 'A' || base == 'C' ||
                           base == 'G' || base == 'T';
                  })) {
            continue;
          }
          starts[valid_lane_count] = starts[lane];
          contig_indices[valid_lane_count] = contig_indices[lane];
          texts[valid_lane_count] = texts[lane];
          ++valid_lane_count;
        }
        lane_count = valid_lane_count;
        if (lane_count == 0) continue;
        thread_result.distance_counts[query_idx] += lane_count;

        std::array<int, 4> distances{};
        bool computed_batch = false;
        if (lane_count == texts.size()) {
          // This is the exact same Myers recurrence as the scalar fallback,
          // evaluated for four independent texts in AVX2 lanes.
          computed_batch =
              compute_distance_bounded_myers_prepared_batch4_trusted_acgt(
                  plans[query_idx].batch_prepared, texts, tolerance,
                  distances);
        }
        for (size_t lane = 0; lane < lane_count; ++lane) {
          const int distance = computed_batch
              ? distances[lane]
              : compute_distance_bounded_myers_prepared(
                    plans[query_idx].prepared,
                    texts[lane], tolerance);
          if (distance > tolerance) continue;
          const auto& contig = reference.contigs[contig_indices[lane]];
          thread_result.occurrences[query_idx].push_back(
              {static_cast<uint32_t>(contig_indices[lane]),
               static_cast<uint32_t>(
                   static_cast<size_t>(contig.source_begin) +
                   starts[lane] - contig.begin),
               distance, std::string(texts[lane])});
        }
      }
    } catch (...) {
      thread_result.error = std::current_exception();
    }
  }
  for (auto& thread_result : thread_results) {
    if (thread_result.error) {
      std::rethrow_exception(thread_result.error);
    }
    for (size_t query_idx = 0; query_idx < queries.size(); ++query_idx) {
      results[query_idx].distance_call_count +=
          thread_result.distance_counts[query_idx];
      auto& partial_occurrences =
          thread_result.occurrences[query_idx];
      results[query_idx].occurrences.insert(
          results[query_idx].occurrences.end(),
          std::make_move_iterator(partial_occurrences.begin()),
          std::make_move_iterator(partial_occurrences.end()));
    }
  }
  return results;
}

ExactBlockVerificationResult verify_by_sampled_qgram_positions(
    std::string_view query,
    int tolerance,
    const ShardedIndexManifest& manifest,
    const PackedReferenceFile& reference,
    const SampledQgramIndex& occurrence_index) {
  auto results = verify_by_sampled_qgram_positions_batch(
      tolerance, manifest, reference, occurrence_index, {query});
  return results.empty() ? ExactBlockVerificationResult{}
                         : std::move(results.front());
}

}  // namespace navigamer
