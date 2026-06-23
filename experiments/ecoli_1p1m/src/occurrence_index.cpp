#include "occurrence_index.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

constexpr std::array<char, 8> kMagic = {'O', 'C', 'C', 'I', 'N', 'D', 'X', '1'};
constexpr uint32_t kFormatVersion = 1;

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
    throw std::runtime_error("truncated occurrence index payload");
  }
  uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[offset++]) << shift;
  }
  return value;
}

uint64_t read_u64(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (offset + sizeof(uint64_t) > bytes.size()) {
    throw std::runtime_error("truncated occurrence index payload");
  }
  uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(bytes[offset++]) << shift;
  }
  return value;
}

void read_magic(const std::vector<uint8_t>& bytes, std::size_t& offset) {
  if (bytes.size() < kMagic.size()) {
    throw std::runtime_error("truncated occurrence index payload");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    throw std::runtime_error("invalid occurrence index magic");
  }
  offset = kMagic.size();
}

}  // namespace

OccurrenceIndex OccurrenceIndex::build(std::string_view sequence, uint32_t k) {
  if (k == 0) {
    throw std::invalid_argument("occurrence index k must be greater than zero");
  }
  if (k > 32) {
    throw std::invalid_argument(
        "occurrence index k must not exceed 32 bases for 2-bit packing");
  }

  OccurrenceIndex index;
  index.k_ = k;

  std::vector<Posting> postings;
  postings.reserve(sequence.size() >= k ? sequence.size() - k + 1 : 0);

  uint64_t key = 0;
  uint32_t run_length = 0;
  const uint64_t mask = k == 32 ? std::numeric_limits<uint64_t>::max()
                                : ((uint64_t{1} << (2U * k)) - 1U);
  for (uint32_t position = 0; position < sequence.size(); ++position) {
    const uint8_t code = encode_base(sequence[position]);
    if (code > 3U) {
      key = 0;
      run_length = 0;
      continue;
    }
    key = (key << 2U) | code;
    if (k < 32) {
      key &= mask;
    }
    if (run_length < k) {
      ++run_length;
    }
    if (run_length >= k) {
      postings.push_back(Posting{key, position + 1U - k});
    }
  }

  std::sort(postings.begin(), postings.end(),
            [](const Posting& lhs, const Posting& rhs) {
              if (lhs.key != rhs.key) {
                return lhs.key < rhs.key;
              }
              return lhs.position < rhs.position;
            });

  std::vector<DirectoryEntry> directory;
  directory.reserve(postings.size());
  for (std::size_t index_position = 0; index_position < postings.size();) {
    const uint64_t key_value = postings[index_position].key;
    const std::size_t begin = index_position;
    do {
      ++index_position;
    } while (index_position < postings.size() &&
             postings[index_position].key == key_value);
    directory.push_back(DirectoryEntry{
        key_value, static_cast<uint32_t>(begin),
        static_cast<uint32_t>(index_position)});
  }

  index.postings_ = std::move(postings);
  index.directory_ = std::move(directory);
  return index;
}

OccurrenceIndex OccurrenceIndex::deserialize(const std::vector<uint8_t>& payload) {
  OccurrenceIndex index;
  std::size_t offset = 0;
  read_magic(payload, offset);
  const uint32_t format_version = read_u32(payload, offset);
  if (format_version != kFormatVersion) {
    throw std::runtime_error("unsupported occurrence index format version");
  }

  index.k_ = read_u32(payload, offset);
  if (index.k_ == 0 || index.k_ > 32) {
    throw std::runtime_error("invalid occurrence index k");
  }
  const uint64_t posting_count = read_u64(payload, offset);
  if (posting_count > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("occurrence index posting count exceeds size");
  }
  index.postings_.resize(static_cast<std::size_t>(posting_count));
  for (std::size_t position = 0; position < index.postings_.size(); ++position) {
    index.postings_[position].key = read_u64(payload, offset);
    index.postings_[position].position = read_u32(payload, offset);
  }

  const uint64_t directory_count = read_u64(payload, offset);
  if (directory_count > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("occurrence index directory count exceeds size");
  }
  index.directory_.resize(static_cast<std::size_t>(directory_count));
  for (std::size_t position = 0; position < index.directory_.size(); ++position) {
    index.directory_[position].key = read_u64(payload, offset);
    index.directory_[position].begin = read_u32(payload, offset);
    index.directory_[position].end = read_u32(payload, offset);
    if (index.directory_[position].begin > index.directory_[position].end ||
        index.directory_[position].end > index.postings_.size()) {
      throw std::runtime_error("invalid occurrence index directory range");
    }
  }

  if (offset != payload.size()) {
    throw std::runtime_error("unexpected trailing bytes in occurrence index");
  }
  return index;
}

uint32_t OccurrenceIndex::k() const { return k_; }

const std::vector<OccurrenceIndex::Posting>& OccurrenceIndex::postings() const {
  return postings_;
}

const std::vector<OccurrenceIndex::DirectoryEntry>&
OccurrenceIndex::directory() const {
  return directory_;
}

std::vector<uint32_t> OccurrenceIndex::positions_for_key(uint64_t key) const {
  const auto iterator = std::lower_bound(
      directory_.begin(), directory_.end(), key,
      [](const DirectoryEntry& entry, uint64_t key_value) {
        return entry.key < key_value;
      });
  if (iterator == directory_.end() || iterator->key != key) {
    return {};
  }
  std::vector<uint32_t> positions;
  positions.reserve(iterator->end - iterator->begin);
  for (uint32_t index = iterator->begin; index < iterator->end; ++index) {
    positions.push_back(postings_[index].position);
  }
  return positions;
}

std::vector<uint8_t> OccurrenceIndex::serialize() const {
  std::vector<uint8_t> bytes;
  bytes.reserve(8 + 4 + 4 + 8 + postings_.size() * (8 + 4) +
                8 + directory_.size() * (8 + 4 + 4));
  bytes.insert(bytes.end(), kMagic.begin(), kMagic.end());
  append_u32(bytes, kFormatVersion);
  append_u32(bytes, k_);
  append_u64(bytes, postings_.size());
  for (const Posting& posting : postings_) {
    append_u64(bytes, posting.key);
    append_u32(bytes, posting.position);
  }
  append_u64(bytes, directory_.size());
  for (const DirectoryEntry& entry : directory_) {
    append_u64(bytes, entry.key);
    append_u32(bytes, entry.begin);
    append_u32(bytes, entry.end);
  }
  return bytes;
}
