#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "phase1_seed_index.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace navigamer {

namespace phase1_detail {

#if defined(__linux__)
namespace {

size_t page_rounded_bytes(size_t bytes) {
  const long page_size_value = ::sysconf(_SC_PAGESIZE);
  if (page_size_value <= 0) {
    throw std::runtime_error("could not determine system page size");
  }
  const size_t page_size = static_cast<size_t>(page_size_value);
  if (bytes > std::numeric_limits<size_t>::max() - (page_size - 1)) {
    throw std::length_error("phase1 posting mapping is too large");
  }
  return (bytes + page_size - 1) / page_size * page_size;
}

}  // namespace

void* resize_anonymous_mapping(void* address, size_t old_bytes,
                               size_t new_bytes) {
  const size_t rounded_new_bytes = page_rounded_bytes(new_bytes);
  if (address == nullptr) {
    void* mapped = ::mmap(nullptr, rounded_new_bytes,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED) throw std::bad_alloc();
    return mapped;
  }
  const size_t rounded_old_bytes = page_rounded_bytes(old_bytes);
  void* resized = ::mremap(address, rounded_old_bytes,
                           rounded_new_bytes, MREMAP_MAYMOVE);
  if (resized == MAP_FAILED) throw std::bad_alloc();
  return resized;
}

void release_anonymous_mapping(void* address, size_t bytes) noexcept {
  if (address == nullptr || bytes == 0) return;
  const long page_size_value = ::sysconf(_SC_PAGESIZE);
  if (page_size_value <= 0) return;
  const size_t page_size = static_cast<size_t>(page_size_value);
  const size_t rounded_bytes = (bytes + page_size - 1) / page_size * page_size;
  (void)::munmap(address, rounded_bytes);
}
#endif

}  // namespace phase1_detail

namespace {

constexpr uint32_t kInvalidPostingIndex =
    std::numeric_limits<uint32_t>::max();
constexpr uint16_t kInvalidCompactPostingIndex =
    std::numeric_limits<uint16_t>::max();
constexpr uint32_t kInvalidCompact24PostingIndex = UINT32_C(0x00ffffff);
constexpr uint32_t kMaxPackedItemIndex = UINT32_C(0x00ffffff);

bool encode_seed(std::string_view sequence, size_t start, int seed_len,
                 uint64_t& code) {
  if (seed_len <= 0 || seed_len > 32 ||
      start + static_cast<size_t>(seed_len) > sequence.size()) {
    return false;
  }

  code = 0;
  for (int offset = 0; offset < seed_len; ++offset) {
    uint64_t value = 0;
    switch (sequence[start + static_cast<size_t>(offset)]) {
      case 'A': value = 0; break;
      case 'C': value = 1; break;
      case 'G': value = 2; break;
      case 'T': value = 3; break;
      default: return false;
    }
    code = (code << 2) | value;
  }
  return true;
}

bool is_acgt(std::string_view sequence) {
  for (char base : sequence) {
    if (base != 'A' && base != 'C' && base != 'G' && base != 'T') return false;
  }
  return true;
}

uint64_t dna_base_bits(char base) {
  switch (base) {
    case 'A': return 0;
    case 'C': return 1;
    case 'G': return 2;
    case 'T': return 3;
    default: return 0;
  }
}

bool length_compatible(size_t lhs, size_t rhs, int tau) {
  return std::llabs(static_cast<long long>(lhs) -
                    static_cast<long long>(rhs)) <= tau;
}

}  // namespace

IncrementalPigeonholeIndex::IncrementalPigeonholeIndex(
    Phase1SeedIndexConfig config)
    : config_(config) {
  if (config_.min_seed_len <= 0) {
    throw std::invalid_argument("phase1 seed minimum length must be positive");
  }
  if (config_.max_seed_len < config_.min_seed_len ||
      config_.max_seed_len > 32) {
    throw std::invalid_argument(
        "phase1 seed maximum length must be in [min_seed_len, 32]");
  }
  if (config_.fixed_query_length != 0 && config_.max_tau < 0) {
    throw std::invalid_argument(
        "fixed phase1 seed layout requires a non-negative maximum threshold");
  }
}

size_t IncrementalPigeonholeIndex::posting_entry_count() const {
  size_t count = 0;
  for (const auto& state_entry : states_) {
    const auto& state = state_entry.second;
    count += state.compact_entries.size();
    count += state.compact24_entries.size();
    count += state.packed_entries.size();
    count += state.wide_entries.size();
  }
  return count;
}

size_t IncrementalPigeonholeIndex::full_posting_entry_count() const {
  size_t count = 0;
  for (const auto& state_entry : states_) {
    count += state_entry.second.full_posting_entry_count;
  }
  return count;
}

size_t IncrementalPigeonholeIndex::posting_bytes() const {
  size_t bytes = 0;
  for (const auto& state_entry : states_) {
    const auto& state = state_entry.second;
    bytes += state.compact_entries.capacity() * sizeof(uint32_t);
    bytes += state.compact24_entries.size() *
             sizeof(SeedState::Compact24PostingEntry);
    bytes += state.packed_entries.capacity() * sizeof(uint64_t);
    bytes += state.wide_entries.capacity() *
             sizeof(SeedState::WidePostingEntry);
  }
  return bytes;
}

uint8_t IncrementalPigeonholeIndex::position_stride_for(
    int seed_len) const {
  if (config_.fixed_query_length == 0 || config_.max_tau < 0 ||
      config_.fixed_query_length >
          static_cast<size_t>(std::numeric_limits<uint8_t>::max())) {
    return 1;
  }
  size_t stride = std::numeric_limits<size_t>::max();
  for (int tau = 0; tau <= config_.max_tau; ++tau) {
    const size_t block_count = static_cast<size_t>(tau) + 1;
    if (block_count > config_.fixed_query_length) continue;
    const size_t block_len = config_.fixed_query_length / block_count;
    const int layout_seed_len = std::min(
        config_.max_seed_len, static_cast<int>(block_len));
    if (layout_seed_len != seed_len ||
        layout_seed_len < config_.min_seed_len) {
      continue;
    }
    stride = std::min(
        stride, block_len - static_cast<size_t>(seed_len) + 1);
  }
  if (stride == std::numeric_limits<size_t>::max() || stride == 0 ||
      stride > std::numeric_limits<uint8_t>::max()) {
    return 1;
  }
  return static_cast<uint8_t>(stride);
}

void IncrementalPigeonholeIndex::append(size_t item_id,
                                        std::string_view sequence) {
  if (items_.size() >= std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("phase1 seed index exceeds uint32 item capacity");
  }
  const uint32_t item_idx = static_cast<uint32_t>(items_.size());
  items_.push_back({item_id, sequence});
  seen_epoch_.push_back(0);
  for (auto& entry : states_) {
    index_item(entry.second, entry.first, item_idx);
    entry.second.indexed_count = items_.size();
  }
}

IncrementalPigeonholeIndex::SeedState&
IncrementalPigeonholeIndex::ensure_state(int seed_len) {
  auto [it, inserted] = states_.try_emplace(seed_len);
  SeedState& state = it->second;
  if (inserted) state.position_stride = position_stride_for(seed_len);
  while (state.indexed_count < items_.size()) {
    index_item(state, seed_len, static_cast<uint32_t>(state.indexed_count));
    state.indexed_count++;
  }
  return state;
}

void IncrementalPigeonholeIndex::promote_to_compact24_postings(
    SeedState& state) {
  if (state.posting_storage !=
      SeedState::PostingStorage::Compact16) {
    return;
  }
  state.compact24_heads.reserve(state.compact_heads.size());
  state.compact24_entries.reserve(state.compact_entries.size());
  state.compact_heads.for_each([&](uint64_t code, uint16_t head) {
    uint32_t compact24_head = kInvalidCompact24PostingIndex;
    uint16_t compact_idx = head;
    while (compact_idx != kInvalidCompactPostingIndex) {
      const uint32_t entry = state.compact_entries[compact_idx];
      const uint16_t compact = static_cast<uint16_t>(entry >> 16);
      const uint32_t new_idx =
          static_cast<uint32_t>(state.compact24_entries.size());
      SeedState::Compact24PostingEntry compact24_entry;
      compact24_entry.item_idx = static_cast<uint16_t>(compact >> 8);
      compact24_entry.position = static_cast<uint8_t>(compact);
      compact24_entry.set_next(compact24_head);
      state.compact24_entries.push_back(compact24_entry);
      compact24_head = new_idx;
      compact_idx = static_cast<uint16_t>(entry);
    }
    state.compact24_heads.set(code, compact24_head);
  });
  state.compact_heads.clear_and_release();
  state.compact_entries.clear();
  state.compact_entries.shrink_to_fit();
  state.posting_storage = SeedState::PostingStorage::Compact24;
}

void IncrementalPigeonholeIndex::promote_to_packed_postings(
    SeedState& state) {
  if (state.posting_storage ==
      SeedState::PostingStorage::Compact16) {
    promote_to_compact24_postings(state);
  }
  if (state.posting_storage !=
      SeedState::PostingStorage::Compact24) {
    return;
  }
  state.packed_heads.reserve(state.compact24_heads.size());
  state.packed_entries.reserve(state.compact24_entries.size());
  state.compact24_heads.for_each([&](uint64_t code, uint32_t head) {
    uint32_t packed_head = kInvalidPostingIndex;
    uint32_t compact24_idx = head;
    while (compact24_idx != kInvalidCompact24PostingIndex) {
      const auto& entry = state.compact24_entries[compact24_idx];
      const uint32_t packed =
          (static_cast<uint32_t>(entry.item_idx) << 8) |
          entry.position;
      const uint32_t new_idx =
          static_cast<uint32_t>(state.packed_entries.size());
      state.packed_entries.push_back(
          (static_cast<uint64_t>(packed) << 32) | packed_head);
      packed_head = new_idx;
      compact24_idx = entry.next();
    }
    state.packed_heads.get_or_insert(code) = packed_head;
  });
  state.compact24_heads.clear_and_release();
  state.compact24_entries.clear();
  state.compact24_entries.shrink_to_fit();
  state.posting_storage = SeedState::PostingStorage::Packed32;
}

void IncrementalPigeonholeIndex::promote_to_wide_postings(
    SeedState& state) {
  if (state.posting_storage == SeedState::PostingStorage::Wide) return;
  if (state.posting_storage ==
      SeedState::PostingStorage::Compact16) {
    promote_to_packed_postings(state);
  }
  state.wide_heads.reserve(state.packed_heads.size());
  state.wide_entries.reserve(state.packed_entries.size());
  state.packed_heads.for_each([&](uint64_t code, uint32_t head) {
    uint32_t wide_head = kInvalidPostingIndex;
    uint32_t packed_idx = head;
    while (packed_idx != kInvalidPostingIndex) {
      const uint64_t entry = state.packed_entries[packed_idx];
      const uint32_t packed = static_cast<uint32_t>(entry >> 32);
      if (state.wide_entries.size() >= kInvalidPostingIndex) {
        throw std::overflow_error("phase1 seed posting arena exceeds uint32 capacity");
      }
      const uint32_t new_idx =
          static_cast<uint32_t>(state.wide_entries.size());
      state.wide_entries.push_back({
          packed >> 8,
          packed & 0xff,
          wide_head});
      wide_head = new_idx;
      packed_idx = static_cast<uint32_t>(entry);
    }
    state.wide_heads.get_or_insert(code) = wide_head;
  });
  state.packed_heads.clear_and_release();
  state.packed_entries.clear();
  state.packed_entries.shrink_to_fit();
  state.posting_storage = SeedState::PostingStorage::Wide;
}

void IncrementalPigeonholeIndex::index_item(SeedState& state, int seed_len,
                                            uint32_t item_idx) {
  const std::string_view sequence = items_[item_idx].sequence;
  if (!is_acgt(sequence) ||
      sequence.size() < static_cast<size_t>(seed_len)) {
    state.unindexable_items.push_back(item_idx);
    return;
  }

  const size_t seed_count =
      sequence.size() - static_cast<size_t>(seed_len) + 1;
  state.full_posting_entry_count += seed_count;
  const size_t indexed_seed_count =
      (seed_count + state.position_stride - 1) /
      state.position_stride;
  if (seed_count - 1 > std::numeric_limits<uint32_t>::max()) {
    state.unindexable_items.push_back(item_idx);
    return;
  }
  if (state.posting_storage ==
          SeedState::PostingStorage::Compact16 &&
      (item_idx > std::numeric_limits<uint8_t>::max() ||
       seed_count - 1 > std::numeric_limits<uint8_t>::max() ||
       indexed_seed_count > kInvalidCompactPostingIndex -
                        state.compact_entries.size())) {
    promote_to_compact24_postings(state);
  }
  if (state.posting_storage ==
          SeedState::PostingStorage::Compact24 &&
      (item_idx > std::numeric_limits<uint16_t>::max() ||
       seed_count - 1 > std::numeric_limits<uint8_t>::max() ||
       indexed_seed_count > kInvalidCompact24PostingIndex -
                        state.compact24_entries.size())) {
    promote_to_packed_postings(state);
  }
  if (state.posting_storage == SeedState::PostingStorage::Packed32 &&
      (item_idx > kMaxPackedItemIndex ||
       seed_count - 1 > std::numeric_limits<uint8_t>::max())) {
    promote_to_wide_postings(state);
  }
  const uint64_t mask =
      seed_len == 32
          ? std::numeric_limits<uint64_t>::max()
          : (uint64_t{1} << (2 * seed_len)) - 1;
  uint64_t code = 0;
  for (int offset = 0; offset < seed_len; ++offset) {
    code = (code << 2) |
           dna_base_bits(sequence[static_cast<size_t>(offset)]);
  }
  for (size_t start = 0; start < seed_count; ++start) {
    if (start > 0) {
      const size_t next = start + static_cast<size_t>(seed_len) - 1;
      code = ((code << 2) | dna_base_bits(sequence[next])) & mask;
    }
    if (start > std::numeric_limits<uint32_t>::max()) {
      state.unindexable_items.push_back(item_idx);
      return;
    }
    if (start % state.position_stride != 0) {
      continue;
    }
    if (state.posting_storage ==
        SeedState::PostingStorage::Compact16) {
      uint16_t& head = state.compact_heads.get_or_insert(code);
      const uint16_t entry_idx =
          static_cast<uint16_t>(state.compact_entries.size());
      const uint16_t packed = static_cast<uint16_t>(
          (item_idx << 8) | static_cast<uint32_t>(start));
      state.compact_entries.push_back(
          (static_cast<uint32_t>(packed) << 16) | head);
      head = entry_idx;
    } else if (state.posting_storage ==
               SeedState::PostingStorage::Compact24) {
      const uint32_t entry_idx =
          static_cast<uint32_t>(state.compact24_entries.size());
      SeedState::Compact24PostingEntry entry;
      entry.item_idx = static_cast<uint16_t>(item_idx);
      entry.position = static_cast<uint8_t>(start);
      entry.set_next(state.compact24_heads.exchange(code, entry_idx));
      state.compact24_entries.push_back(entry);
    } else if (state.posting_storage ==
               SeedState::PostingStorage::Packed32) {
      if (state.packed_entries.size() >= kInvalidPostingIndex) {
        throw std::overflow_error("phase1 seed posting arena exceeds uint32 capacity");
      }
      uint32_t& head = state.packed_heads.get_or_insert(code);
      const uint32_t entry_idx =
          static_cast<uint32_t>(state.packed_entries.size());
      const uint32_t packed =
          (item_idx << 8) | static_cast<uint32_t>(start);
      state.packed_entries.push_back(
          (static_cast<uint64_t>(packed) << 32) | head);
      head = entry_idx;
    } else {
      if (state.wide_entries.size() >= kInvalidPostingIndex) {
        throw std::overflow_error("phase1 seed posting arena exceeds uint32 capacity");
      }
      uint32_t& head = state.wide_heads.get_or_insert(code);
      const uint32_t entry_idx =
          static_cast<uint32_t>(state.wide_entries.size());
      state.wide_entries.push_back(
          {item_idx, static_cast<uint32_t>(start), head});
      head = entry_idx;
    }
  }
}

void IncrementalPigeonholeIndex::begin_query() {
  if (epoch_ == std::numeric_limits<uint32_t>::max()) {
    std::fill(seen_epoch_.begin(), seen_epoch_.end(), 0);
    epoch_ = 1;
  } else {
    epoch_++;
  }
}

Phase1SeedQueryResult IncrementalPigeonholeIndex::query(
    std::string_view sequence, int tau) {
  if (tau < 0) {
    throw std::invalid_argument("phase1 seed threshold must be non-negative");
  }

  Phase1SeedQueryResult result;
  if (config_.fixed_query_length != 0 &&
      (sequence.size() != config_.fixed_query_length ||
       tau > config_.max_tau)) {
    return result;
  }
  const size_t block_count = static_cast<size_t>(tau) + 1;
  if (block_count == 0 || block_count > sequence.size()) return result;

  const size_t block_len = sequence.size() / block_count;
  result.block_len = block_len > static_cast<size_t>(INT_MAX)
                         ? INT_MAX
                         : static_cast<int>(block_len);
  result.seed_len = std::min(config_.max_seed_len, result.block_len);
  if (result.seed_len < config_.min_seed_len || result.seed_len > 32 ||
      !is_acgt(sequence)) {
    return result;
  }

  int indexed_seed_len = result.seed_len;
  if (config_.fixed_query_length != 0 && config_.max_tau >= 0) {
    const size_t maximum_block_count =
        static_cast<size_t>(config_.max_tau) + 1;
    if (maximum_block_count > config_.fixed_query_length) return result;
    const size_t maximum_radius_block_len =
        config_.fixed_query_length / maximum_block_count;
    indexed_seed_len = std::min(
        config_.max_seed_len,
        static_cast<int>(maximum_radius_block_len));
    if (indexed_seed_len < config_.min_seed_len) return result;
  }

  SeedState& state = ensure_state(indexed_seed_len);
  begin_query();

  auto add_item = [&](uint32_t item_idx) {
    if (item_idx >= items_.size() || seen_epoch_[item_idx] == epoch_) return;
    if (!length_compatible(sequence.size(), items_[item_idx].sequence.size(),
                           tau)) {
      return;
    }
    seen_epoch_[item_idx] = epoch_;
    result.candidate_indices.push_back(items_[item_idx].item_id);
  };
  // The indexed occurrence may be shifted within an edit-free pigeonhole
  // block to land on the state's modulo grid. Verify the original logical
  // seed, not the shifted prefix. This preserves exactly the candidate set of
  // the dense logical-seed index while storing only one short-prefix state.
  const auto logical_seed_matches = [&](uint32_t item_idx,
                                        size_t indexed_item_position,
                                        size_t block_start,
                                        size_t seed_offset) {
    if (item_idx >= items_.size() || indexed_item_position < seed_offset) {
      return false;
    }
    const std::string_view item_sequence = items_[item_idx].sequence;
    const size_t item_position = indexed_item_position - seed_offset;
    const size_t logical_seed_len = static_cast<size_t>(result.seed_len);
    if (item_position + logical_seed_len > item_sequence.size() ||
        block_start + logical_seed_len > sequence.size()) {
      return false;
    }
    return item_sequence.substr(item_position, logical_seed_len) ==
           sequence.substr(block_start, logical_seed_len);
  };
  const auto posting_matches = [&](uint32_t item_idx, size_t item_position,
                                   size_t block_start, size_t seed_offset) {
    if (item_position < seed_offset) return false;
    const size_t logical_item_position = item_position - seed_offset;
    return std::llabs(
               static_cast<long long>(logical_item_position) -
               static_cast<long long>(block_start)) <= tau &&
           logical_seed_matches(
               item_idx, item_position, block_start, seed_offset);
  };

  for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
    const size_t block_start = block_idx * block_len;
    for (size_t seed_offset = 0;
         seed_offset < state.position_stride; ++seed_offset) {
      const size_t seed_start = block_start + seed_offset;
      uint64_t code = 0;
      if (!encode_seed(sequence, seed_start, indexed_seed_len, code)) return {};
      if (state.posting_storage ==
          SeedState::PostingStorage::Compact16) {
        const uint16_t posting = state.compact_heads.find(code);
        if (posting == kInvalidCompactPostingIndex) continue;
        for (uint16_t entry_idx = posting;
             entry_idx != kInvalidCompactPostingIndex;) {
          const uint32_t entry = state.compact_entries[entry_idx];
          const uint16_t packed = static_cast<uint16_t>(entry >> 16);
          entry_idx = static_cast<uint16_t>(entry);
          result.posting_entries_visited++;
          const uint32_t item_idx = packed >> 8;
          const uint32_t position = packed & 0xff;
          if (posting_matches(
                  item_idx, position, block_start, seed_offset)) {
            add_item(item_idx);
          }
        }
      } else if (state.posting_storage ==
                 SeedState::PostingStorage::Compact24) {
        const uint32_t posting = state.compact24_heads.find(code);
        if (posting == std::numeric_limits<uint32_t>::max()) continue;
        for (uint32_t entry_idx = posting;
             entry_idx != kInvalidCompact24PostingIndex;) {
          const auto& entry = state.compact24_entries[entry_idx];
          entry_idx = entry.next();
          result.posting_entries_visited++;
          if (posting_matches(
                  entry.item_idx, entry.position, block_start, seed_offset)) {
            add_item(entry.item_idx);
          }
        }
      } else if (state.posting_storage ==
                 SeedState::PostingStorage::Packed32) {
        const uint32_t posting = state.packed_heads.find(code);
        if (posting == kInvalidPostingIndex) continue;
        for (uint32_t entry_idx = posting;
             entry_idx != kInvalidPostingIndex;) {
          const uint64_t entry = state.packed_entries[entry_idx];
          const uint32_t packed = static_cast<uint32_t>(entry >> 32);
          entry_idx = static_cast<uint32_t>(entry);
          result.posting_entries_visited++;
          const uint32_t item_idx = packed >> 8;
          const uint32_t position = packed & 0xff;
          if (posting_matches(
                  item_idx, position, block_start, seed_offset)) {
            add_item(item_idx);
          }
        }
      } else {
        const uint32_t posting = state.wide_heads.find(code);
        if (posting == kInvalidPostingIndex) continue;
        for (uint32_t entry_idx = posting;
             entry_idx != kInvalidPostingIndex;) {
          const auto& seed_posting = state.wide_entries[entry_idx];
          entry_idx = seed_posting.next;
          result.posting_entries_visited++;
          if (posting_matches(
                  seed_posting.item_idx, seed_posting.position,
                  block_start, seed_offset)) {
            add_item(seed_posting.item_idx);
          }
        }
      }
    }
  }

  for (uint32_t item_idx : state.unindexable_items) add_item(item_idx);
  std::sort(result.candidate_indices.begin(), result.candidate_indices.end());
  result.candidate_indices.erase(
      std::unique(result.candidate_indices.begin(),
                  result.candidate_indices.end()),
      result.candidate_indices.end());
  result.safe = true;
  return result;
}

}  // namespace navigamer
