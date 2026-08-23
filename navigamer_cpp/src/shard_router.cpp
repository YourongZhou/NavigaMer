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

  struct CodeRange {
    size_t begin = 0;
    size_t end = 0;
  };

  std::vector<uint32_t> selected;
  std::vector<uint32_t> partition_candidates;
  std::vector<uint32_t> intersection;
  std::vector<uint32_t> merged;
  std::vector<uint32_t> minimizers;
  std::vector<CodeRange> ranges;
  for (size_t partition = 0; partition < partition_count;
       ++partition) {
    const size_t begin = partition * query.size() / partition_count;
    const size_t end =
        (partition + 1) * query.size() / partition_count;

    minimizers.clear();
    minimizers.reserve(end - begin - window + 1);
    for (size_t seed_begin = begin;
         seed_begin + window <= end; ++seed_begin) {
      uint32_t minimizer = 0;
      if (!exact_minimizer_code(
              query.substr(seed_begin, window), k, window,
              &minimizer)) {
        return false;
      }
      minimizers.push_back(minimizer);
    }
    std::sort(minimizers.begin(), minimizers.end());
    minimizers.erase(
        std::unique(minimizers.begin(), minimizers.end()),
        minimizers.end());

    ranges.clear();
    ranges.reserve(minimizers.size());
    for (uint32_t minimizer : minimizers) {
      const auto bounds = equal_range_minimizer_code(minimizer);
      ranges.push_back({bounds.first, bounds.second});
    }
    std::sort(
        ranges.begin(), ranges.end(),
        [](const CodeRange& left, const CodeRange& right) {
          return left.end - left.begin < right.end - right.begin;
        });

    partition_candidates.clear();
    if (!ranges.empty()) {
      const CodeRange& first = ranges.front();
      partition_candidates.reserve(first.end - first.begin);
      uint32_t previous = 0;
      bool has_previous = false;
      for (size_t entry = first.begin; entry < first.end; ++entry) {
        const uint32_t shard_id = unpack_shard_id(
            packed_shard_ids, entry, shard_id_bits);
        if (shard_id >= shard_count ||
            (has_previous && shard_id < previous)) {
          return false;
        }
        if (!has_previous || shard_id != previous) {
          partition_candidates.push_back(shard_id);
        }
        previous = shard_id;
        has_previous = true;
      }
    }

    for (size_t range_index = 1;
         range_index < ranges.size() &&
         !partition_candidates.empty(); ++range_index) {
      intersection.clear();
      intersection.reserve(std::min(
          partition_candidates.size(),
          ranges[range_index].end - ranges[range_index].begin));
      size_t candidate_index = 0;
      uint32_t previous = 0;
      bool has_previous = false;
      for (size_t entry = ranges[range_index].begin;
           entry < ranges[range_index].end &&
           candidate_index < partition_candidates.size(); ++entry) {
        const uint32_t shard_id = unpack_shard_id(
            packed_shard_ids, entry, shard_id_bits);
        if (shard_id >= shard_count ||
            (has_previous && shard_id < previous)) {
          return false;
        }
        if (has_previous && shard_id == previous) continue;
        while (candidate_index < partition_candidates.size() &&
               partition_candidates[candidate_index] < shard_id) {
          ++candidate_index;
        }
        if (candidate_index < partition_candidates.size() &&
            partition_candidates[candidate_index] == shard_id) {
          intersection.push_back(shard_id);
          ++candidate_index;
        }
        previous = shard_id;
        has_previous = true;
      }
      partition_candidates.swap(intersection);
    }

    if (selected.empty()) {
      selected = partition_candidates;
    } else if (!partition_candidates.empty()) {
      merged.clear();
      merged.reserve(std::min(
          static_cast<size_t>(shard_count),
          selected.size() + partition_candidates.size()));
      std::set_union(
          selected.begin(), selected.end(),
          partition_candidates.begin(), partition_candidates.end(),
          std::back_inserter(merged));
      selected.swap(merged);
    }
  }

  if (selected.size() >
      std::numeric_limits<size_t>::max() - initial_size) {
    throw std::length_error("router shard selection is too large");
  }
  shard_ids->insert(
      shard_ids->end(), selected.begin(), selected.end());
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
