#include "sharded_index.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace navigamer {

namespace {

int dna_code(char base) {
  switch (base) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default: return -1;
  }
}

bool exact_minimizer_code(
    std::string_view sequence, uint32_t k, uint32_t window,
    uint32_t* minimizer) {
  if (!minimizer || k == 0 || k > 16 || window < k ||
      sequence.size() != window) {
    return false;
  }
  const uint64_t mask =
      k == 16 ? UINT32_MAX : ((uint64_t{1} << (2 * k)) - 1);
  uint64_t code = 0;
  size_t valid = 0;
  uint32_t best = UINT32_MAX;
  for (char base : sequence) {
    const int value = dna_code(base);
    if (value < 0) return false;
    code = ((code << 2) | static_cast<uint64_t>(value)) & mask;
    if (++valid >= k) {
      best = std::min(best, static_cast<uint32_t>(code));
    }
  }
  *minimizer = best;
  return true;
}

uint32_t unpack_shard_id(
    const FinalArray<uint8_t>& packed_shard_ids,
    size_t entry_index, uint32_t shard_id_bits) {
  const size_t bit_offset = entry_index * shard_id_bits;
  const size_t byte_offset = bit_offset / 8;
  const uint32_t shift = static_cast<uint32_t>(bit_offset % 8);
  uint64_t word = 0;
  const size_t bytes_needed =
      (shift + shard_id_bits + 7) / 8;
  for (size_t byte = 0; byte < bytes_needed; ++byte) {
    word |= static_cast<uint64_t>(
                packed_shard_ids[byte_offset + byte])
            << (8 * byte);
  }
  const uint64_t mask =
      shard_id_bits == 32
          ? UINT32_MAX
          : ((uint64_t{1} << shard_id_bits) - 1);
  return static_cast<uint32_t>((word >> shift) & mask);
}

size_t code_block_entry_count(
    size_t total_entry_count, uint32_t block_size, size_t block_index) {
  const size_t begin = block_index * static_cast<size_t>(block_size);
  if (begin >= total_entry_count) {
    throw std::out_of_range("router minimizer code block");
  }
  return std::min(
      static_cast<size_t>(block_size), total_entry_count - begin);
}

size_t packed_delta_byte_count(size_t code_count, uint8_t width) {
  if (code_count == 0 || width == 0 || width > 32 ||
      code_count - 1 >
          (std::numeric_limits<size_t>::max() - 7) / width) {
    throw std::runtime_error("invalid router minimizer code block");
  }
  return ((code_count - 1) * static_cast<size_t>(width) + 7) / 8;
}

}  // namespace

size_t ShardedSeedRouter::minimizer_code_count() const {
  return code_entry_count != 0 ? code_entry_count : minimizer_codes.size();
}

uint32_t ShardedSeedRouter::minimizer_code_at(size_t entry_index) const {
  const size_t entry_count = minimizer_code_count();
  if (entry_index >= entry_count) {
    throw std::out_of_range("router minimizer code");
  }
  if (code_entry_count == 0) return minimizer_codes[entry_index];
  if (code_block_size == 0 || minimizer_code_bases.empty() ||
      minimizer_code_widths.size() != minimizer_code_bases.size() ||
      minimizer_code_group_offsets.empty() ||
      minimizer_code_supergroup_offsets.empty()) {
    throw std::runtime_error("invalid compressed shard router codes");
  }

  const size_t block_index = entry_index / code_block_size;
  const size_t group_index = block_index / 4;
  const size_t supergroup_index =
      group_index / kRouterCodeGroupsPerSupergroup;
  if (block_index >= minimizer_code_bases.size() ||
      group_index >= minimizer_code_group_offsets.size() ||
      supergroup_index >= minimizer_code_supergroup_offsets.size()) {
    throw std::runtime_error("invalid shard router code block index");
  }
  size_t payload_offset = static_cast<size_t>(
      minimizer_code_supergroup_offsets[supergroup_index]) +
      minimizer_code_group_offsets[group_index];
  const size_t first_block = group_index * 4;
  for (size_t previous = first_block; previous < block_index; ++previous) {
    const uint8_t previous_width = minimizer_code_widths[previous];
    payload_offset += packed_delta_byte_count(
        code_block_entry_count(entry_count, code_block_size, previous),
        previous_width);
  }

  const size_t local_index = entry_index % code_block_size;
  const size_t block_count = code_block_entry_count(
      entry_count, code_block_size, block_index);
  const uint32_t base = minimizer_code_bases[block_index];
  if (local_index == 0) return base;
  const uint8_t width = minimizer_code_widths[block_index];
  const size_t payload_size = packed_delta_byte_count(block_count, width);
  if (payload_offset > packed_minimizer_code_deltas.size() ||
      payload_size > packed_minimizer_code_deltas.size() - payload_offset) {
    throw std::runtime_error("invalid shard router code payload range");
  }
  const size_t bit_offset = (local_index - 1) * static_cast<size_t>(width);
  const size_t byte_offset = payload_offset + bit_offset / 8;
  const uint32_t shift = static_cast<uint32_t>(bit_offset % 8);
  const size_t bytes_needed = (shift + width + 7) / 8;
  if (bytes_needed > sizeof(uint64_t) ||
      bytes_needed > packed_minimizer_code_deltas.size() ||
      byte_offset > packed_minimizer_code_deltas.size() - bytes_needed) {
    throw std::runtime_error("invalid shard router code payload");
  }
  uint64_t packed_delta = 0;
  for (size_t byte = 0; byte < bytes_needed; ++byte) {
    packed_delta |= static_cast<uint64_t>(
        packed_minimizer_code_deltas[byte_offset + byte]) << (byte * 8);
  }
  const uint64_t mask = width == 32
      ? UINT32_MAX
      : ((uint64_t{1} << width) - 1);
  return base + static_cast<uint32_t>((packed_delta >> shift) & mask);
}

size_t ShardedSeedRouter::lower_bound_minimizer_code(uint32_t code) const {
  size_t begin = 0;
  size_t end = minimizer_code_count();
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (minimizer_code_at(middle) < code) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

size_t ShardedSeedRouter::upper_bound_minimizer_code(uint32_t code) const {
  size_t begin = 0;
  size_t end = minimizer_code_count();
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (code < minimizer_code_at(middle)) {
      end = middle;
    } else {
      begin = middle + 1;
    }
  }
  return begin;
}

bool ShardedSeedRouter::append_selected_shards(
    std::string_view query, int tolerance,
    std::vector<uint32_t>* shard_ids) const {
  if (!shard_ids) {
    throw std::invalid_argument("router shard output must not be null");
  }
  const size_t initial_size = shard_ids->size();
  if (!enabled() || tolerance < 0 || query.empty()) {
    return false;
  }
  const size_t partition_count =
      static_cast<size_t>(tolerance) + 1;
  if (partition_count == 0 || partition_count > query.size() ||
      query.size() / partition_count < window) {
    return false;
  }

  constexpr size_t kInlineMinimizerCount = 16;
  std::array<uint32_t, kInlineMinimizerCount> inline_minimizers{};
  std::vector<uint32_t> overflow_minimizers;
  uint32_t* minimizers = inline_minimizers.data();
  if (partition_count > inline_minimizers.size()) {
    overflow_minimizers.resize(partition_count);
    minimizers = overflow_minimizers.data();
  }
  for (size_t partition = 0; partition < partition_count;
       ++partition) {
    const size_t begin = partition * query.size() / partition_count;
    const size_t end =
        (partition + 1) * query.size() / partition_count;
    // A minimizer of a longer exact seed is also the minimizer of at least
    // one `window`-base subwindow containing it, so it is present in the
    // reference router. Retain the more selective 64-base query seed when
    // the partition is long enough while keeping the router window as the
    // safe floor.
    const size_t seed_length =
        std::min<size_t>(64, end - begin);
    const size_t seed_begin =
        begin + (end - begin - seed_length) / 2;
    uint32_t minimizer = 0;
    if (!exact_minimizer_code(
            query.substr(seed_begin, seed_length), k,
            static_cast<uint32_t>(seed_length),
            &minimizer)) {
      return false;
    }
    minimizers[partition] = minimizer;
  }
  std::sort(minimizers, minimizers + partition_count);
  const size_t minimizer_count = static_cast<size_t>(
      std::unique(minimizers, minimizers + partition_count) - minimizers);

  struct CodeRange {
    size_t first = 0;
    size_t last = 0;
  };
  std::array<CodeRange, kInlineMinimizerCount> inline_ranges{};
  std::vector<CodeRange> overflow_ranges;
  CodeRange* ranges = inline_ranges.data();
  if (minimizer_count > inline_ranges.size()) {
    overflow_ranges.resize(minimizer_count);
    ranges = overflow_ranges.data();
  }
  size_t routed_entry_count = 0;
  for (size_t minimizer_idx = 0; minimizer_idx < minimizer_count;
       ++minimizer_idx) {
    CodeRange& range = ranges[minimizer_idx];
    range.first = lower_bound_minimizer_code(minimizers[minimizer_idx]);
    range.last = upper_bound_minimizer_code(minimizers[minimizer_idx]);
    routed_entry_count += range.last - range.first;
  }
  if (routed_entry_count >
      std::numeric_limits<size_t>::max() - initial_size) {
    throw std::length_error("router shard selection is too large");
  }
  const size_t required_size = initial_size + routed_entry_count;
  if (required_size > shard_ids->capacity()) {
    size_t new_capacity = required_size;
    const size_t current_capacity = shard_ids->capacity();
    if (current_capacity <=
        std::numeric_limits<size_t>::max() - current_capacity / 2) {
      new_capacity = std::max(
          new_capacity, current_capacity + current_capacity / 2);
    }
    shard_ids->reserve(new_capacity);
  }
  for (size_t minimizer_idx = 0; minimizer_idx < minimizer_count;
       ++minimizer_idx) {
    const CodeRange range = ranges[minimizer_idx];
    const size_t first = range.first;
    const size_t last = range.last;
    for (size_t entry_index = first; entry_index < last; ++entry_index) {
      const uint32_t shard_id = unpack_shard_id(
          packed_shard_ids, entry_index, shard_id_bits);
      if (shard_id >= shard_count) {
        shard_ids->resize(initial_size);
        return false;
      }
      shard_ids->push_back(shard_id);
    }
  }
  auto selected_begin = shard_ids->begin() + initial_size;
  std::sort(selected_begin, shard_ids->end());
  shard_ids->erase(
      std::unique(selected_begin, shard_ids->end()), shard_ids->end());
  return true;
}

ShardRouteSelection ShardedSeedRouter::select(
    std::string_view query, int tolerance) const {
  ShardRouteSelection selection;
  selection.enabled = append_selected_shards(
      query, tolerance, &selection.shard_ids);
  return selection;
}

}  // namespace navigamer
