#include "sharded_index.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <queue>
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

size_t code_block_payload_offset(
    const ShardedSeedRouter& router, size_t block_index) {
  const size_t entry_count = router.minimizer_code_count();
  if (router.code_block_size == 0 ||
      block_index >= router.minimizer_code_bases.size()) {
    throw std::out_of_range("router minimizer code block payload");
  }
  const size_t group_index =
      block_index / kRouterCodeBlocksPerGroup;
  const size_t supergroup_index =
      group_index / kRouterCodeGroupsPerSupergroup;
  if (group_index >= router.minimizer_code_group_offsets.size() ||
      supergroup_index >=
          router.minimizer_code_supergroup_offsets.size()) {
    throw std::runtime_error("invalid shard router code block index");
  }
  size_t payload_offset = static_cast<size_t>(
      router.minimizer_code_supergroup_offsets[supergroup_index]) +
      router.minimizer_code_group_offsets[group_index];
  const size_t first_block =
      group_index * kRouterCodeBlocksPerGroup;
  for (size_t previous = first_block; previous < block_index; ++previous) {
    payload_offset += packed_delta_byte_count(
        code_block_entry_count(
            entry_count, router.code_block_size, previous),
        router.minimizer_code_widths[previous]);
  }
  return payload_offset;
}

class PackedCodeDeltaReader {
 public:
  PackedCodeDeltaReader(
      const ShardedSeedRouter& router, size_t payload_offset,
      size_t payload_size, uint8_t width)
      : mask_(width == 32
                  ? UINT32_MAX
                  : ((uint64_t{1} << width) - 1)),
        width_(width) {
    if (width == 0 || width > 32 ||
        payload_offset > router.packed_minimizer_code_deltas.size() ||
        payload_size >
            router.packed_minimizer_code_deltas.size() - payload_offset) {
      throw std::runtime_error("invalid shard router code payload");
    }
    data_ = router.packed_minimizer_code_deltas.data() + payload_offset;
    size_ = payload_size;
  }

  uint32_t next() {
    while (pending_bit_count_ < width_) {
      if (position_ == size_) {
        throw std::runtime_error("truncated shard router code payload");
      }
      pending_bits_ |=
          static_cast<uint64_t>(data_[position_++]) << pending_bit_count_;
      pending_bit_count_ += 8;
    }
    const uint32_t delta =
        static_cast<uint32_t>(pending_bits_ & mask_);
    pending_bits_ >>= width_;
    pending_bit_count_ -= width_;
    return delta;
  }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  size_t position_ = 0;
  uint64_t pending_bits_ = 0;
  uint64_t mask_ = 0;
  uint32_t pending_bit_count_ = 0;
  uint8_t width_ = 0;
};

uint32_t add_code_delta(uint32_t code, uint32_t delta) {
  if (delta > std::numeric_limits<uint32_t>::max() - code) {
    throw std::runtime_error("shard router minimizer code overflow");
  }
  return code + delta;
}

template <typename Predicate>
size_t find_in_code_block(
    const ShardedSeedRouter& router, size_t block_index,
    Predicate&& matches) {
  const size_t entry_count = router.minimizer_code_count();
  const size_t entry_begin = block_index * router.code_block_size;
  const size_t block_count = code_block_entry_count(
      entry_count, router.code_block_size, block_index);
  uint32_t code = router.minimizer_code_bases[block_index];
  if (matches(code)) return entry_begin;
  if (block_count == 1) return entry_begin + 1;
  const uint8_t width = router.minimizer_code_widths[block_index];
  const size_t payload_size = packed_delta_byte_count(block_count, width);
  const size_t payload_offset =
      code_block_payload_offset(router, block_index);
  PackedCodeDeltaReader deltas(
      router, payload_offset, payload_size, width);
  for (size_t local = 1; local < block_count; ++local) {
    code = add_code_delta(code, deltas.next());
    if (matches(code)) return entry_begin + local;
  }
  return entry_begin + block_count;
}

std::pair<size_t, size_t> equal_range_in_code_block(
    const ShardedSeedRouter& router, size_t block_index,
    uint32_t target) {
  const size_t entry_count = router.minimizer_code_count();
  const size_t entry_begin = block_index * router.code_block_size;
  const size_t block_count = code_block_entry_count(
      entry_count, router.code_block_size, block_index);
  uint32_t code = router.minimizer_code_bases[block_index];
  size_t lower = code >= target ? entry_begin : entry_begin + block_count;
  if (code > target) return {lower, entry_begin};
  if (block_count == 1) return {lower, entry_begin + 1};
  const uint8_t width = router.minimizer_code_widths[block_index];
  const size_t payload_size = packed_delta_byte_count(block_count, width);
  const size_t payload_offset =
      code_block_payload_offset(router, block_index);
  PackedCodeDeltaReader deltas(
      router, payload_offset, payload_size, width);
  for (size_t local = 1; local < block_count; ++local) {
    code = add_code_delta(code, deltas.next());
    const size_t entry = entry_begin + local;
    if (lower == entry_begin + block_count && code >= target) {
      lower = entry;
    }
    if (code > target) return {lower, entry};
  }
  return {lower, entry_begin + block_count};
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
  if (block_index >= minimizer_code_bases.size() ||
      block_index >= minimizer_code_widths.size()) {
    throw std::runtime_error("invalid shard router code block index");
  }
  const size_t local_index = entry_index % code_block_size;
  const size_t block_count = code_block_entry_count(
      entry_count, code_block_size, block_index);
  uint32_t code = minimizer_code_bases[block_index];
  if (local_index == 0) return code;
  const uint8_t width = minimizer_code_widths[block_index];
  const size_t payload_size = packed_delta_byte_count(block_count, width);
  const size_t payload_offset =
      code_block_payload_offset(*this, block_index);
  PackedCodeDeltaReader deltas(
      *this, payload_offset, payload_size, width);
  for (size_t delta_index = 0; delta_index < local_index; ++delta_index) {
    code = add_code_delta(code, deltas.next());
  }
  return code;
}

size_t ShardedSeedRouter::lower_bound_minimizer_code(uint32_t code) const {
  return equal_range_minimizer_code(code).first;
}

size_t ShardedSeedRouter::upper_bound_minimizer_code(uint32_t code) const {
  return equal_range_minimizer_code(code).second;
}

std::pair<size_t, size_t>
ShardedSeedRouter::equal_range_minimizer_code(uint32_t code) const {
  if (code_entry_count == 0) {
    const auto range = std::equal_range(
        minimizer_codes.begin(), minimizer_codes.end(), code);
    return {
        static_cast<size_t>(range.first - minimizer_codes.begin()),
        static_cast<size_t>(range.second - minimizer_codes.begin())};
  }
  if (minimizer_code_bases.empty() || code_block_size == 0) {
    throw std::runtime_error("invalid compressed shard router codes");
  }

  const auto lower_base = std::lower_bound(
      minimizer_code_bases.begin(), minimizer_code_bases.end(), code);
  size_t lower_block = static_cast<size_t>(
      lower_base - minimizer_code_bases.begin());
  if (lower_block == minimizer_code_bases.size()) {
    --lower_block;
  } else if (*lower_base > code) {
    if (lower_block != 0) --lower_block;
  } else if (lower_block != 0) {
    const size_t previous_last = lower_block * code_block_size - 1;
    if (minimizer_code_at(previous_last) == code) --lower_block;
  }

  const auto upper_base = std::upper_bound(
      minimizer_code_bases.begin(), minimizer_code_bases.end(), code);
  size_t upper_block = static_cast<size_t>(
      upper_base - minimizer_code_bases.begin());
  if (upper_block != 0) --upper_block;
  if (lower_block == upper_block) {
    return equal_range_in_code_block(*this, lower_block, code);
  }
  return {
      find_in_code_block(
          *this, lower_block,
          [code](uint32_t candidate) { return candidate >= code; }),
      find_in_code_block(
          *this, upper_block,
          [code](uint32_t candidate) { return candidate > code; })};
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
    size_t next = 0;
    size_t last = 0;
    uint32_t shard_id = 0;
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
    const auto bounds =
        equal_range_minimizer_code(minimizers[minimizer_idx]);
    range.next = bounds.first;
    range.last = bounds.second;
    routed_entry_count += range.last - range.next;
  }
  constexpr size_t kDirectSortEntryLimit = 4096;
  const bool use_direct_sort =
      routed_entry_count <= kDirectSortEntryLimit;
  const size_t maximum_unique_count = std::min(
      routed_entry_count, static_cast<size_t>(shard_count));
  const size_t reserve_count =
      use_direct_sort ? routed_entry_count : maximum_unique_count;
  if (reserve_count >
      std::numeric_limits<size_t>::max() - initial_size) {
    throw std::length_error("router shard selection is too large");
  }
  const size_t required_size = initial_size + reserve_count;
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
  if (use_direct_sort) {
    for (size_t minimizer_idx = 0; minimizer_idx < minimizer_count;
         ++minimizer_idx) {
      const CodeRange& range = ranges[minimizer_idx];
      for (size_t entry_index = range.next;
           entry_index < range.last; ++entry_index) {
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
        std::unique(selected_begin, shard_ids->end()),
        shard_ids->end());
    return true;
  }
  size_t active_range_count = 0;
  for (size_t minimizer_idx = 0; minimizer_idx < minimizer_count;
       ++minimizer_idx) {
    CodeRange& range = ranges[minimizer_idx];
    if (range.next != range.last) {
      range.shard_id = unpack_shard_id(
          packed_shard_ids, range.next, shard_id_bits);
      if (range.shard_id >= shard_count) {
        return false;
      }
      ++active_range_count;
    }
  }
  const auto append_unique = [&](uint32_t shard_id) {
    if (shard_ids->size() == initial_size ||
        shard_ids->back() != shard_id) {
      shard_ids->push_back(shard_id);
    }
  };
  const auto advance_range = [&](CodeRange* range) {
    const uint32_t previous_shard_id = range->shard_id;
    ++range->next;
    if (range->next == range->last) return 0;
    range->shard_id = unpack_shard_id(
        packed_shard_ids, range->next, shard_id_bits);
    return range->shard_id < shard_count &&
                   range->shard_id >= previous_shard_id
               ? 1
               : -1;
  };

  // Entries with one minimizer code are emitted in shard-ID order. Merge the
  // already sorted code ranges directly instead of materializing every
  // duplicate ID and sorting the expanded list. Fixed-length mapping queries
  // stay in the allocation-free path; unusually many partitions use a heap
  // so long-query complexity remains O(entries log partitions).
  if (minimizer_count <= kInlineMinimizerCount) {
    while (active_range_count != 0) {
      uint32_t next_shard_id = UINT32_MAX;
      for (size_t minimizer_idx = 0; minimizer_idx < minimizer_count;
           ++minimizer_idx) {
        const CodeRange& range = ranges[minimizer_idx];
        if (range.next != range.last) {
          next_shard_id = std::min(next_shard_id, range.shard_id);
        }
      }
      append_unique(next_shard_id);
      for (size_t minimizer_idx = 0; minimizer_idx < minimizer_count;
           ++minimizer_idx) {
        CodeRange& range = ranges[minimizer_idx];
        if (range.next == range.last ||
            range.shard_id != next_shard_id) {
          continue;
        }
        const int state = advance_range(&range);
        if (state < 0) {
          shard_ids->resize(initial_size);
          return false;
        }
        if (state == 0) --active_range_count;
      }
    }
  } else {
    struct RangeHead {
      uint32_t shard_id = 0;
      size_t range_index = 0;
    };
    struct RangeHeadGreater {
      bool operator()(const RangeHead& left,
                      const RangeHead& right) const {
        return left.shard_id > right.shard_id;
      }
    };
    std::vector<RangeHead> heads;
    heads.reserve(active_range_count);
    for (size_t minimizer_idx = 0; minimizer_idx < minimizer_count;
         ++minimizer_idx) {
      if (ranges[minimizer_idx].next != ranges[minimizer_idx].last) {
        heads.push_back(
            {ranges[minimizer_idx].shard_id, minimizer_idx});
      }
    }
    std::priority_queue<
        RangeHead, std::vector<RangeHead>, RangeHeadGreater> queue(
            RangeHeadGreater{}, std::move(heads));
    while (!queue.empty()) {
      const uint32_t next_shard_id = queue.top().shard_id;
      append_unique(next_shard_id);
      do {
        const size_t range_index = queue.top().range_index;
        queue.pop();
        CodeRange& range = ranges[range_index];
        const int state = advance_range(&range);
        if (state < 0) {
          shard_ids->resize(initial_size);
          return false;
        }
        if (state > 0) {
          queue.push({range.shard_id, range_index});
        }
      } while (!queue.empty() &&
               queue.top().shard_id == next_shard_id);
    }
  }
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
