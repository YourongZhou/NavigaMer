#include "sharded_index.hpp"

#include <algorithm>
#include <cstdint>
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

}  // namespace

ShardRouteSelection ShardedSeedRouter::select(
    std::string_view query, int tolerance) const {
  ShardRouteSelection selection;
  if (!enabled() || tolerance < 0 || query.empty()) {
    return selection;
  }
  const size_t partition_count =
      static_cast<size_t>(tolerance) + 1;
  if (partition_count == 0 || partition_count > query.size() ||
      query.size() / partition_count < window) {
    return selection;
  }

  std::vector<uint32_t> minimizers;
  minimizers.reserve(partition_count);
  for (size_t partition = 0; partition < partition_count;
       ++partition) {
    const size_t begin = partition * query.size() / partition_count;
    const size_t end =
        (partition + 1) * query.size() / partition_count;
    // A minimizer of a longer exact seed is also the minimizer of at least
    // one `window`-base subwindow containing it, so it is present in the
    // reference router. Retain the more selective 64-base query seed when
    // the partition is long enough while keeping 32 bases as the safe floor.
    const size_t seed_length =
        std::min<size_t>(64, end - begin);
    const size_t seed_begin =
        begin + (end - begin - seed_length) / 2;
    uint32_t minimizer = 0;
    if (!exact_minimizer_code(
            query.substr(seed_begin, seed_length), k,
            static_cast<uint32_t>(seed_length),
            &minimizer)) {
      return ShardRouteSelection{};
    }
    minimizers.push_back(minimizer);
  }
  std::sort(minimizers.begin(), minimizers.end());
  minimizers.erase(
      std::unique(minimizers.begin(), minimizers.end()),
      minimizers.end());

  for (uint32_t minimizer : minimizers) {
    const auto first =
        std::lower_bound(
            minimizer_codes.begin(), minimizer_codes.end(), minimizer);
    const auto last =
        std::upper_bound(first, minimizer_codes.end(), minimizer);
    for (auto code = first; code != last; ++code) {
      const size_t entry_index = static_cast<size_t>(
          code - minimizer_codes.begin());
      const uint32_t shard_id = unpack_shard_id(
          packed_shard_ids, entry_index, shard_id_bits);
      if (shard_id >= shard_count) {
        return ShardRouteSelection{};
      }
      selection.shard_ids.push_back(shard_id);
    }
  }
  std::sort(selection.shard_ids.begin(), selection.shard_ids.end());
  selection.shard_ids.erase(
      std::unique(selection.shard_ids.begin(), selection.shard_ids.end()),
      selection.shard_ids.end());
  selection.enabled = true;
  return selection;
}

}  // namespace navigamer
