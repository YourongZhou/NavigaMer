#include "phase1_seed_index.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace navigamer {

namespace {

bool encode_seed(const std::string& sequence, size_t start, int seed_len,
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

bool is_acgt(const std::string& sequence) {
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
}

void IncrementalPigeonholeIndex::append(size_t item_id,
                                        const std::string& sequence) {
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
  while (state.indexed_count < items_.size()) {
    index_item(state, seed_len, static_cast<uint32_t>(state.indexed_count));
    state.indexed_count++;
  }
  return state;
}

void IncrementalPigeonholeIndex::index_item(SeedState& state, int seed_len,
                                            uint32_t item_idx) {
  const std::string& sequence = items_[item_idx].sequence;
  if (!is_acgt(sequence) ||
      sequence.size() < static_cast<size_t>(seed_len)) {
    state.unindexable_items.push_back(item_idx);
    return;
  }

  const size_t seed_count =
      sequence.size() - static_cast<size_t>(seed_len) + 1;
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
    state.postings[code].push_back(
        {item_idx, static_cast<uint32_t>(start)});
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
    const std::string& sequence, int tau) {
  if (tau < 0) {
    throw std::invalid_argument("phase1 seed threshold must be non-negative");
  }

  Phase1SeedQueryResult result;
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

  SeedState& state = ensure_state(result.seed_len);
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

  for (size_t block_idx = 0; block_idx < block_count; ++block_idx) {
    const size_t block_start = block_idx * block_len;
    uint64_t code = 0;
    if (!encode_seed(sequence, block_start, result.seed_len, code)) return {};
    auto posting = state.postings.find(code);
    if (posting == state.postings.end()) continue;
    result.posting_entries_visited += posting->second.size();
    for (const auto& seed_posting : posting->second) {
      if (std::llabs(static_cast<long long>(seed_posting.position) -
                     static_cast<long long>(block_start)) <= tau) {
        add_item(seed_posting.item_idx);
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
