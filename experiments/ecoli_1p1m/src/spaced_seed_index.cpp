#include "candidate_indexes.hpp"

#include "reference_windows.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "index_persistence.hpp"

namespace {

constexpr std::array<char, 8> kIndexMagic = {'E', 'C', 'O', 'L', 'I', 'B', 'L',
                                             '1'};
constexpr uint32_t kIndexFormatVersion = 1;
constexpr uint64_t kMaskIdShift = 62U;
constexpr uint64_t kMaskValueBits = (uint64_t{1} << kMaskIdShift) - 1U;

struct Posting {
  uint64_t key = 0;
  uint32_t position = 0;
  uint32_t span = 0;
};

struct DirectoryEntry {
  uint64_t key = 0;
  uint32_t begin = 0;
  uint32_t end = 0;
};

struct LoadedIndexFile {
  IndexManifest manifest;
  std::vector<uint8_t> payload;
};

void append_u32(std::vector<uint8_t>& bytes, uint32_t value);
void append_u64(std::vector<uint8_t>& bytes, uint64_t value);
uint32_t read_u32(const std::vector<uint8_t>& bytes, std::size_t& offset);
uint64_t read_u64(const std::vector<uint8_t>& bytes, std::size_t& offset);

std::string current_utc_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buffer[32] = {};
  if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) {
    throw std::runtime_error("unable to format timestamp");
  }
  return buffer;
}

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += "'";
  return quoted;
}

std::string git_commit_for_root(const std::filesystem::path& root) {
  const std::string command = "git -C " + shell_quote(root.string()) +
                              " rev-parse --short=12 HEAD 2>/dev/null";
  std::array<char, 128> buffer{};
  std::string output;
  if (FILE* pipe = popen(command.c_str(), "r")) {
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) !=
           nullptr) {
      output += buffer.data();
    }
    const int status = pclose(pipe);
    if (status != 0) {
      return {};
    }
  }
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

std::filesystem::path navigamer_repo_root() {
  return std::filesystem::path(NAVIGAMER_REPO_ROOT);
}

std::string sha256_hex_of_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input && !input.eof()) {
    throw std::runtime_error("unable to read file: " + path.string());
  }
  return sha256_hex(buffer.str());
}

uint8_t encode_base(char base) {
  switch (base) {
    case 'A':
    case 'a':
      return 0;
    case 'C':
    case 'c':
      return 1;
    case 'G':
    case 'g':
      return 2;
    case 'T':
    case 't':
      return 3;
    default:
      return 255;
  }
}

uint64_t encode_spaced_key(std::string_view sequence,
                           const std::vector<uint8_t>& bits) {
  if (sequence.size() != bits.size()) {
    throw std::invalid_argument("spaced key length mismatch");
  }
  uint64_t key = 0;
  for (std::size_t index = 0; index < bits.size(); ++index) {
    if (!bits[index]) {
      continue;
    }
    const uint8_t code = encode_base(sequence[index]);
    if (code > 3U) {
      return std::numeric_limits<uint64_t>::max();
    }
    key = (key << 2U) | code;
  }
  return key;
}

uint64_t prefix_mask_id(uint8_t mask_id, uint64_t key) {
  return (static_cast<uint64_t>(mask_id) << kMaskIdShift) |
         (key & kMaskValueBits);
}

uint8_t mask_id_from_key(uint64_t key) {
  return static_cast<uint8_t>(key >> kMaskIdShift);
}

std::vector<uint8_t> build_mask_bits(uint32_t span, uint32_t weight) {
  if (weight == 0 || weight > span) {
    throw std::invalid_argument("spaced seed weight must not exceed the span");
  }
  std::vector<uint8_t> bits(span, 0);
  for (uint32_t index = 0; index < span; ++index) {
    if (((index + 1U) * weight) / span != (index * weight) / span) {
      bits[index] = 1;
    }
  }
  const uint32_t observed =
      static_cast<uint32_t>(std::count(bits.begin(), bits.end(), uint8_t{1}));
  if (observed != weight) {
    throw std::runtime_error("unable to construct spaced mask of requested weight");
  }
  return bits;
}

std::string bits_to_string(const std::vector<uint8_t>& bits) {
  std::string value;
  value.reserve(bits.size());
  for (uint8_t bit : bits) {
    value.push_back(bit ? '1' : '0');
  }
  return value;
}

std::vector<uint8_t> serialize_spaced_payload(
    uint32_t weight, const std::vector<uint8_t>& packed_index) {
  std::vector<uint8_t> bytes;
  append_u32(bytes, weight);
  append_u32(bytes, 4);
  bytes.insert(bytes.end(), packed_index.begin(), packed_index.end());
  return bytes;
}

std::pair<uint32_t, std::vector<uint8_t>> parse_spaced_payload(
    const std::vector<uint8_t>& payload) {
  std::size_t offset = 0;
  const uint32_t weight = read_u32(payload, offset);
  const uint32_t mask_count = read_u32(payload, offset);
  if (mask_count != 4) {
    throw std::runtime_error("spaced payload mask count mismatch");
  }
  return {weight, std::vector<uint8_t>(payload.begin() +
                                           static_cast<std::ptrdiff_t>(offset),
                                       payload.end())};
}

std::vector<Posting> build_postings(std::string_view sequence,
                                    const std::vector<SpacedMask>& masks) {
  std::vector<Posting> postings;
  for (uint8_t mask_id = 0; mask_id < masks.size(); ++mask_id) {
    const SpacedMask& mask = masks[mask_id];
    if (mask.bits.size() != mask.span) {
      throw std::runtime_error("spaced mask bits do not match span");
    }
    if (sequence.size() < mask.span) {
      continue;
    }
    for (uint32_t position = 0; position + mask.span <= sequence.size();
         ++position) {
      const uint64_t key =
          encode_spaced_key(sequence.substr(position, mask.span), mask.bits);
      if (key == std::numeric_limits<uint64_t>::max()) {
        continue;
      }
      postings.push_back(
          Posting{prefix_mask_id(mask_id, key), position, mask.span});
    }
  }
  return postings;
}

std::vector<uint64_t> query_keys(std::string_view sequence,
                                 const std::vector<SpacedMask>& masks) {
  std::vector<uint64_t> keys;
  for (uint8_t mask_id = 0; mask_id < masks.size(); ++mask_id) {
    const SpacedMask& mask = masks[mask_id];
    if (sequence.size() < mask.span) {
      continue;
    }
    for (uint32_t position = 0; position + mask.span <= sequence.size();
         ++position) {
      const uint64_t key =
          encode_spaced_key(sequence.substr(position, mask.span), mask.bits);
      if (key == std::numeric_limits<uint64_t>::max()) {
        continue;
      }
      keys.push_back(prefix_mask_id(mask_id, key));
    }
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

std::vector<uint32_t> covering_window_ids_from_manifest(
    const IndexManifest& manifest, uint32_t occurrence_start, uint32_t span) {
  if (span == 0) {
    throw std::invalid_argument("occurrence span must be greater than zero");
  }
  if (manifest.window_length == 0 || manifest.stride == 0 ||
      manifest.number_of_windows == 0 || manifest.reference_length == 0) {
    throw std::runtime_error("spaced index manifest is incomplete");
  }
  if (span > manifest.window_length) {
    throw std::invalid_argument(
        "occurrence span must not exceed the window length");
  }

  const uint64_t occurrence_end = static_cast<uint64_t>(occurrence_start) + span;
  if (occurrence_start >= manifest.reference_length ||
      occurrence_end > manifest.reference_length) {
    return {};
  }

  const uint64_t minimum_start =
      occurrence_end > manifest.window_length
          ? occurrence_end - manifest.window_length
          : 0;
  const uint64_t maximum_window_start =
      static_cast<uint64_t>(manifest.number_of_windows - 1) * manifest.stride;
  const uint64_t maximum_start =
      std::min<uint64_t>(occurrence_start, maximum_window_start);
  if (minimum_start > maximum_start) {
    return {};
  }

  const uint64_t first_id = (minimum_start + manifest.stride - 1) / manifest.stride;
  const uint64_t last_id = maximum_start / manifest.stride;
  if (first_id > last_id) {
    return {};
  }

  std::vector<uint32_t> ids;
  ids.reserve(static_cast<std::size_t>(last_id - first_id + 1));
  for (uint64_t id = first_id; id <= last_id; ++id) {
    ids.push_back(static_cast<uint32_t>(id));
  }
  return ids;
}

void read_exact(std::istream& input, void* destination, std::size_t size) {
  input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
  if (!input) {
    throw std::runtime_error("truncated index file");
  }
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
  }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
  }
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    throw std::runtime_error("truncated spaced occurrence index payload");
  }
  uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[offset++]) << shift;
  }
  return value;
}

uint64_t read_u64(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint64_t) > bytes.size()) {
    throw std::runtime_error("truncated spaced occurrence index payload");
  }
  uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(bytes[offset++]) << shift;
  }
  return value;
}

class PackedOccurrenceIndex {
 public:
  static PackedOccurrenceIndex from_postings(std::vector<Posting> postings) {
    std::sort(postings.begin(), postings.end(),
              [](const Posting& lhs, const Posting& rhs) {
                if (lhs.key != rhs.key) {
                  return lhs.key < rhs.key;
                }
                if (lhs.position != rhs.position) {
                  return lhs.position < rhs.position;
                }
                return lhs.span < rhs.span;
              });

    PackedOccurrenceIndex index;
    index.postings_ = std::move(postings);
    for (std::size_t position = 0; position < index.postings_.size();) {
      const uint64_t key = index.postings_[position].key;
      const std::size_t begin = position;
      do {
        ++position;
      } while (position < index.postings_.size() &&
               index.postings_[position].key == key);
      index.directory_.push_back(
          DirectoryEntry{key, static_cast<uint32_t>(begin),
                         static_cast<uint32_t>(position)});
    }
    return index;
  }

  static PackedOccurrenceIndex deserialize(const std::vector<uint8_t>& payload) {
    PackedOccurrenceIndex index;
    std::size_t offset = 0;
    const uint64_t posting_count = read_u64(payload, offset);
    index.postings_.resize(static_cast<std::size_t>(posting_count));
    for (std::size_t position = 0; position < index.postings_.size(); ++position) {
      index.postings_[position].key = read_u64(payload, offset);
      index.postings_[position].position = read_u32(payload, offset);
      index.postings_[position].span = read_u32(payload, offset);
    }

    const uint64_t directory_count = read_u64(payload, offset);
    index.directory_.resize(static_cast<std::size_t>(directory_count));
    for (std::size_t position = 0; position < index.directory_.size(); ++position) {
      index.directory_[position].key = read_u64(payload, offset);
      index.directory_[position].begin = read_u32(payload, offset);
      index.directory_[position].end = read_u32(payload, offset);
      if (index.directory_[position].begin > index.directory_[position].end ||
          index.directory_[position].end > index.postings_.size()) {
        throw std::runtime_error("invalid spaced occurrence index directory range");
      }
    }

    if (offset != payload.size()) {
      throw std::runtime_error("unexpected trailing bytes in spaced occurrence index");
    }
    return index;
  }

  std::vector<uint8_t> serialize() const {
    std::vector<uint8_t> bytes;
    bytes.reserve(8 + postings_.size() * (8 + 4 + 4) + 8 +
                  directory_.size() * (8 + 4 + 4));
    append_u64(bytes, postings_.size());
    for (const Posting& posting : postings_) {
      append_u64(bytes, posting.key);
      append_u32(bytes, posting.position);
      append_u32(bytes, posting.span);
    }
    append_u64(bytes, directory_.size());
    for (const DirectoryEntry& entry : directory_) {
      append_u64(bytes, entry.key);
      append_u32(bytes, entry.begin);
      append_u32(bytes, entry.end);
    }
    return bytes;
  }

  std::vector<Posting> records_for_key(uint64_t key) const {
    const auto iterator = std::lower_bound(
        directory_.begin(), directory_.end(), key,
        [](const DirectoryEntry& entry, uint64_t key_value) {
          return entry.key < key_value;
        });
    if (iterator == directory_.end() || iterator->key != key) {
      return {};
    }
    std::vector<Posting> records;
    records.reserve(iterator->end - iterator->begin);
    for (uint32_t index = iterator->begin; index < iterator->end; ++index) {
      records.push_back(postings_[index]);
    }
    return records;
  }

 private:
  std::vector<Posting> postings_;
  std::vector<DirectoryEntry> directory_;
};

LoadedIndexFile read_index_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open index file: " + path.string());
  }

  std::array<char, 8> magic{};
  read_exact(input, magic.data(), magic.size());
  if (magic != kIndexMagic) {
    throw std::runtime_error("invalid index magic");
  }

  auto read_u32_stream = [&]() {
    std::array<uint8_t, 4> bytes{};
    read_exact(input, bytes.data(), bytes.size());
    uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
      value |= static_cast<uint32_t>(bytes[shift / 8U]) << shift;
    }
    return value;
  };
  auto read_u64_stream = [&]() {
    std::array<uint8_t, 8> bytes{};
    read_exact(input, bytes.data(), bytes.size());
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
      value |= static_cast<uint64_t>(bytes[shift / 8U]) << shift;
    }
    return value;
  };
  auto read_double_stream = [&]() {
    const uint64_t bits = read_u64_stream();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  auto read_string_stream = [&]() {
    const uint64_t size = read_u64_stream();
    if (size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("string field too large");
    }
    std::string value(static_cast<std::size_t>(size), '\0');
    read_exact(input, value.data(), static_cast<std::size_t>(size));
    return value;
  };

  if (read_u32_stream() != kIndexFormatVersion) {
    throw std::runtime_error("unsupported index format version");
  }

  LoadedIndexFile loaded;
  loaded.manifest.method = read_string_stream();
  const uint64_t parameter_count = read_u64_stream();
  loaded.manifest.parameters.reserve(static_cast<std::size_t>(parameter_count));
  for (uint64_t index = 0; index < parameter_count; ++index) {
    loaded.manifest.parameters.emplace_back(read_string_stream(),
                                            read_string_stream());
  }
  loaded.manifest.reference_path = read_string_stream();
  loaded.manifest.reference_sha256 = read_string_stream();
  loaded.manifest.reference_length = read_u64_stream();
  loaded.manifest.window_length = read_u32_stream();
  loaded.manifest.stride = read_u32_stream();
  loaded.manifest.number_of_windows = read_u64_stream();
  loaded.manifest.build_command = read_string_stream();
  loaded.manifest.build_seconds = read_double_stream();
  loaded.manifest.index_bytes = read_u64_stream();
  loaded.manifest.created_at = read_string_stream();
  loaded.manifest.git_commit = read_string_stream();
  loaded.manifest.format_version = read_u32_stream();
  loaded.manifest.tool_version = read_string_stream();

  const uint64_t payload_size = read_u64_stream();
  std::array<uint8_t, 32> digest{};
  read_exact(input, digest.data(), digest.size());
  loaded.payload.resize(static_cast<std::size_t>(payload_size));
  read_exact(input, loaded.payload.data(), loaded.payload.size());
  std::array<uint8_t, 32> actual_digest = sha256(loaded.payload);
  if (!std::equal(digest.begin(), digest.end(), actual_digest.begin())) {
    throw std::runtime_error("payload checksum mismatch");
  }
  if (loaded.manifest.index_bytes != payload_size) {
    throw std::runtime_error("manifest index bytes do not match payload length");
  }
  if (input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("unexpected trailing bytes after index payload");
  }
  return loaded;
}

}  // namespace

std::vector<SpacedMask> make_spaced_masks(uint32_t weight) {
  if (weight == 0 || weight > 24) {
    throw std::invalid_argument(
        "spaced seed weight must be between 1 and 24 for the fixed mask family");
  }

  std::vector<SpacedMask> masks;
  masks.reserve(4);
  for (uint32_t span : {24U, 26U, 29U, 32U}) {
    SpacedMask mask;
    mask.span = span;
    mask.bits = build_mask_bits(span, weight);
    masks.push_back(std::move(mask));
  }
  return masks;
}

SpacedSeedIndex SpacedSeedIndex::build(const SpacedSeedIndexConfig& config) {
  if (config.reference_path.empty()) {
    throw std::invalid_argument("reference path must not be empty");
  }
  if (config.window_length < 32) {
    throw std::invalid_argument("window length must be at least 32 bases");
  }
  if (config.stride == 0) {
    throw std::invalid_argument("stride must be greater than zero");
  }
  if (config.weight == 0) {
    throw std::invalid_argument("weight must be greater than zero");
  }

  const auto started = std::chrono::steady_clock::now();
  const ReferenceWindows reference =
      ReferenceWindows::from_fasta(config.reference_path.string(),
                                   config.window_length, config.stride);

  SpacedSeedIndex index;
  index.masks_ = make_spaced_masks(config.weight);
  index.weight_ = config.weight;
  const std::vector<Posting> postings = build_postings(reference.sequence(), index.masks_);
  const PackedOccurrenceIndex packed_index =
      PackedOccurrenceIndex::from_postings(postings);
  index.payload_ = serialize_spaced_payload(config.weight, packed_index.serialize());

  index.manifest_.method = "spaced";
  index.manifest_.parameters = {
      {"method", "spaced"},
      {"weight", std::to_string(config.weight)},
      {"mask_count", "4"},
  };
  for (std::size_t mask_id = 0; mask_id < index.masks_.size(); ++mask_id) {
    index.manifest_.parameters.emplace_back(
        "mask_" + std::to_string(mask_id) + "_span",
        std::to_string(index.masks_[mask_id].span));
    index.manifest_.parameters.emplace_back(
        "mask_" + std::to_string(mask_id) + "_bits",
        bits_to_string(index.masks_[mask_id].bits));
  }
  index.manifest_.reference_path = config.reference_path.string();
  index.manifest_.reference_sha256 = sha256_hex_of_file(config.reference_path);
  index.manifest_.reference_length = reference.sequence().size();
  index.manifest_.window_length = config.window_length;
  index.manifest_.stride = config.stride;
  index.manifest_.number_of_windows = reference.size();
  index.manifest_.build_command =
      "candidate_tool build --method spaced --weight " +
      std::to_string(config.weight);
  const auto finished = std::chrono::steady_clock::now();
  index.manifest_.build_seconds =
      std::chrono::duration<double>(finished - started).count();
  index.manifest_.created_at = current_utc_timestamp();
  index.manifest_.git_commit = git_commit_for_root(navigamer_repo_root());
  index.manifest_.format_version = 1;
  index.manifest_.tool_version = "spaced-index/1";
  return index;
}

SpacedSeedIndex SpacedSeedIndex::load(const std::filesystem::path& index_path) {
  const LoadedIndexFile loaded = read_index_file(index_path);
  if (loaded.manifest.method != "spaced") {
    throw std::runtime_error("unsupported spaced index method");
  }

  SpacedSeedIndex index;
  index.manifest_ = loaded.manifest;
  const auto [weight, packed_payload] = parse_spaced_payload(loaded.payload);
  (void)packed_payload;
  index.weight_ = weight;
  index.masks_ = make_spaced_masks(weight);
  index.payload_ = std::move(loaded.payload);
  return index;
}

void SpacedSeedIndex::save(const std::filesystem::path& out_dir) const {
  if (out_dir.empty()) {
    throw std::invalid_argument("output directory must not be empty");
  }
  std::filesystem::create_directories(out_dir);
  const std::filesystem::path index_path = out_dir / "index.bin";
  IndexManifest manifest = manifest_;
  manifest.index_bytes = payload_.size();
  write_index_atomic(index_path, manifest, payload_);
  write_manifest_json(out_dir / "manifest.json", manifest);
}

std::vector<uint32_t> SpacedSeedIndex::query(
    std::string_view query_sequence) const {
  if (masks_.empty() || payload_.empty()) {
    return {};
  }

  const auto [weight, packed_payload] = parse_spaced_payload(payload_);
  if (weight != weight_) {
    throw std::runtime_error("spaced payload weight mismatch");
  }
  const PackedOccurrenceIndex packed_index =
      PackedOccurrenceIndex::deserialize(packed_payload);
  const std::vector<uint64_t> keys = query_keys(query_sequence, masks_);
  if (keys.empty()) {
    return {};
  }

  std::vector<uint8_t> marked(
      static_cast<std::size_t>(manifest_.number_of_windows), 0);
  for (uint64_t key : keys) {
    const uint8_t mask_id = mask_id_from_key(key);
    if (mask_id >= masks_.size()) {
      throw std::runtime_error("spaced query encountered an invalid mask ID");
    }
    const uint32_t span = masks_[mask_id].span;
    const std::vector<Posting> records = packed_index.records_for_key(key);
    for (const Posting& record : records) {
      const std::vector<uint32_t> window_ids =
          covering_window_ids_from_manifest(manifest_, record.position, record.span);
      if (record.span != span) {
        throw std::runtime_error("spaced occurrence span does not match mask");
      }
      for (uint32_t window_id : window_ids) {
        marked[window_id] = 1;
      }
    }
  }

  std::vector<uint32_t> window_ids;
  for (uint32_t window_id = 0; window_id < marked.size(); ++window_id) {
    if (marked[window_id]) {
      window_ids.push_back(window_id);
    }
  }
  return window_ids;
}
