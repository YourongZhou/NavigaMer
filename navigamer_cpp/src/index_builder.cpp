#include "index_builder.hpp"
#include "build_progress.hpp"
#include "phase1_seed_index.hpp"
#include "phase2_distance_verifier.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>

namespace navigamer {

namespace {

using Clock = std::chrono::steady_clock;

bool is_acgt(char base) {
  return base == 'A' || base == 'C' || base == 'G' || base == 'T' ||
         base == 'a' || base == 'c' || base == 'g' || base == 't';
}

double elapsed_ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class ScopedTimer {
 public:
  explicit ScopedTimer(double* target_ms)
      : target_ms_(target_ms), start_(Clock::now()) {}

  ~ScopedTimer() {
    if (target_ms_) *target_ms_ += elapsed_ms_since(start_);
  }

 private:
  double* target_ms_;
  Clock::time_point start_;
};

int build_distance_bounded(std::string_view a, std::string_view b, int tau,
                           BuildDistanceMode mode);
int build_distance_bounded_prepared(
    std::string_view a, std::string_view b, int tau,
    BuildDistanceMode mode,
    const PreparedEdlibDnaPattern* prepared_query);
int build_distance(std::string_view a, std::string_view b,
                   BuildDistanceMode mode);
DistanceMode to_distance_mode(BuildDistanceMode mode);

constexpr size_t kPhase1ParallelScanMinFanout = 512;
constexpr size_t kPhase2DistanceBatchFlushPairs = 65536;
// Any beacon subset preserves the triangle-inequality lower bound. Capping the
// subset bounds the otherwise quadratic child-by-beacon matrix and also bounds
// the number of query-to-beacon distances evaluated at each world.
constexpr size_t kMaxBuildBeaconsPerNode = 12;
constexpr size_t kMaxCompactSequenceLength =
    std::numeric_limits<uint8_t>::max();

class ReferencePositionEncoder {
 public:
  LeafId append(uint32_t position) {
    if (count_ != 0 && position <= previous_) {
      throw std::runtime_error(
          "reference representative positions are not strictly increasing");
    }
    if (count_ >= static_cast<size_t>(INVALID_LEAF_ID - 1)) {
      throw std::runtime_error("too many reference sequences for 32-bit LeafId");
    }
    const LeafId id = static_cast<LeafId>(count_++);
    pending_[pending_size_++] = position;
    previous_ = position;
    if (pending_size_ == pending_.size()) flush();
    return id;
  }

  uint32_t position(LeafId id) const {
    if (id >= count_) {
      throw std::out_of_range("reference sequence id");
    }
    const size_t block_idx =
        static_cast<size_t>(id) / kReferencePositionBlockSize;
    const size_t in_block =
        static_cast<size_t>(id) % kReferencePositionBlockSize;
    if (block_idx == blocks_.size()) {
      if (in_block >= pending_size_) {
        throw std::runtime_error(
            "reference position encoder has too few entries");
      }
      return pending_[in_block];
    }
    const auto& block = blocks_.at(block_idx);
    if (in_block == 0) return block.base;
    if (block.encoding == ReferencePositionEncoding::Linear) {
      return block.base +
             static_cast<uint32_t>(in_block * block.payload_size);
    }
    const uint8_t* payload = payload_.data() + block.payload_begin;
    const size_t encoded_idx = in_block - 1;
    switch (block.encoding) {
      case ReferencePositionEncoding::Linear:
        break;
      case ReferencePositionEncoding::Bitset: {
        size_t remaining = encoded_idx;
        for (size_t byte_idx = 0;
             byte_idx < block.payload_size; ++byte_idx) {
          uint8_t bits = payload[byte_idx];
          const size_t bit_count = static_cast<size_t>(
              __builtin_popcount(static_cast<unsigned int>(bits)));
          if (remaining >= bit_count) {
            remaining -= bit_count;
            continue;
          }
          while (remaining != 0) {
            bits = static_cast<uint8_t>(bits & (bits - 1));
            --remaining;
          }
          const unsigned int bit = static_cast<unsigned int>(
              __builtin_ctz(static_cast<unsigned int>(bits)));
          return block.base +
                 static_cast<uint32_t>(byte_idx * 8 + bit + 1);
        }
        throw std::runtime_error(
            "reference position bitset has too few entries");
      }
      case ReferencePositionEncoding::Delta8:
        return block.base + payload[encoded_idx];
      case ReferencePositionEncoding::Delta16: {
        uint16_t delta = 0;
        std::memcpy(&delta, payload + encoded_idx * sizeof(delta),
                    sizeof(delta));
        return block.base + delta;
      }
      case ReferencePositionEncoding::Absolute32: {
        uint32_t position = 0;
        std::memcpy(&position,
                    payload + encoded_idx * sizeof(position),
                    sizeof(position));
        return position;
      }
    }
    throw std::runtime_error("invalid reference position encoding");
  }

  void finish(SequenceStore& store) {
    flush();
    store.reference_sequence_count = static_cast<uint32_t>(count_);
    store.reference_position_blocks.set_owned(std::move(blocks_));
    store.reference_position_payload.set_owned(std::move(payload_));
  }

 private:
  template <typename T>
  void append_pod(T value) {
    const size_t begin = payload_.size();
    payload_.resize(begin + sizeof(T));
    std::memcpy(payload_.data() + begin, &value, sizeof(T));
  }

  void flush() {
    if (pending_size_ == 0) return;
    ReferencePositionBlock block;
    block.payload_begin = payload_.size();
    block.base = pending_[0];

    uint32_t linear_step = 1;
    bool linear = true;
    if (pending_size_ > 1) {
      linear_step = pending_[1] - pending_[0];
      linear = linear_step <= std::numeric_limits<uint16_t>::max();
      for (size_t idx = 2; linear && idx < pending_size_; ++idx) {
        const uint64_t expected =
            static_cast<uint64_t>(pending_[0]) +
            static_cast<uint64_t>(idx) * linear_step;
        linear = expected == pending_[idx];
      }
    }
    if (linear) {
      block.encoding = ReferencePositionEncoding::Linear;
      block.payload_size = static_cast<uint16_t>(linear_step);
      blocks_.push_back(block);
      pending_size_ = 0;
      return;
    }

    const uint32_t span = pending_[pending_size_ - 1] - pending_[0];
    const size_t encoded_count = pending_size_ - 1;
    ReferencePositionEncoding encoding;
    size_t encoded_bytes = 0;
    if (span <= std::numeric_limits<uint8_t>::max()) {
      encoding = ReferencePositionEncoding::Delta8;
      encoded_bytes = encoded_count;
    } else if (span <= std::numeric_limits<uint16_t>::max()) {
      encoding = ReferencePositionEncoding::Delta16;
      encoded_bytes = encoded_count * sizeof(uint16_t);
    } else {
      encoding = ReferencePositionEncoding::Absolute32;
      encoded_bytes = encoded_count * sizeof(uint32_t);
    }
    const uint64_t bitset_bytes_64 =
        (static_cast<uint64_t>(span) + 7) / 8;
    if (bitset_bytes_64 <= std::numeric_limits<uint16_t>::max() &&
        bitset_bytes_64 < encoded_bytes) {
      encoding = ReferencePositionEncoding::Bitset;
      encoded_bytes = static_cast<size_t>(bitset_bytes_64);
    }
    if (encoded_bytes > std::numeric_limits<uint16_t>::max()) {
      throw std::runtime_error("reference position block payload is too large");
    }
    block.encoding = encoding;
    block.payload_size = static_cast<uint16_t>(encoded_bytes);

    if (encoding == ReferencePositionEncoding::Bitset) {
      const size_t payload_begin = payload_.size();
      payload_.resize(payload_begin + encoded_bytes, 0);
      for (size_t idx = 1; idx < pending_size_; ++idx) {
        const uint32_t bit = pending_[idx] - pending_[0] - 1;
        payload_[payload_begin + bit / 8] |=
            static_cast<uint8_t>(uint8_t{1} << (bit % 8));
      }
    } else if (encoding == ReferencePositionEncoding::Delta8) {
      for (size_t idx = 1; idx < pending_size_; ++idx) {
        payload_.push_back(
            static_cast<uint8_t>(pending_[idx] - pending_[0]));
      }
    } else if (encoding == ReferencePositionEncoding::Delta16) {
      for (size_t idx = 1; idx < pending_size_; ++idx) {
        append_pod<uint16_t>(
            static_cast<uint16_t>(pending_[idx] - pending_[0]));
      }
    } else {
      for (size_t idx = 1; idx < pending_size_; ++idx) {
        append_pod<uint32_t>(pending_[idx]);
      }
    }
    blocks_.push_back(block);
    pending_size_ = 0;
  }

  std::array<uint32_t, kReferencePositionBlockSize> pending_{};
  size_t pending_size_ = 0;
  size_t count_ = 0;
  uint32_t previous_ = 0;
  std::vector<ReferencePositionBlock> blocks_;
  std::vector<uint8_t> payload_;
};

uint32_t reference_sequence_hash(std::string_view sequence) {
  return static_cast<uint32_t>(
      std::hash<std::string_view>{}(sequence));
}

// Build-only exact dictionary for fixed-length reference windows. The hash is
// only a probe hint: equal hashes are confirmed against the shared reference,
// so collisions cannot merge distinct sequences or introduce false negatives.
class ReferenceSequenceTable {
 public:
  struct Lookup {
    LeafId id = INVALID_LEAF_ID;
    size_t slot = 0;
  };

  explicit ReferenceSequenceTable(size_t maximum_items)
      : capacity_(capacity_for(maximum_items)),
        slots_(new uint64_t[capacity_]()) {}

  template <typename PositionOf>
  Lookup find(uint32_t hash, std::string_view sequence,
              std::string_view reference,
              PositionOf&& position_of) const {
    size_t slot = static_cast<size_t>(hash % capacity_);
    for (size_t probe = 0; probe < capacity_; ++probe) {
      const uint64_t packed = slots_[slot];
      if (packed == 0) return {INVALID_LEAF_ID, slot};
      const LeafId id =
          static_cast<LeafId>(static_cast<uint32_t>(packed) - 1);
      if (static_cast<uint32_t>(packed >> 32) == hash) {
        const size_t source_pos = position_of(id);
        if (source_pos <= reference.size() &&
            sequence.size() <= reference.size() - source_pos &&
            sequence == reference.substr(source_pos, sequence.size())) {
          return {id, slot};
        }
      }
      if (++slot == capacity_) slot = 0;
    }
    throw std::runtime_error("reference sequence hash table is full");
  }

  void insert(const Lookup& lookup, uint32_t hash, LeafId id) {
    if (lookup.id != INVALID_LEAF_ID ||
        slots_[lookup.slot] != 0 ||
        size_ >= capacity_) {
      throw std::runtime_error("invalid reference sequence table insertion");
    }
    slots_[lookup.slot] =
        (static_cast<uint64_t>(hash) << 32) |
        (static_cast<uint64_t>(id) + 1);
    ++size_;
  }

 private:
  static size_t capacity_for(size_t maximum_items) {
    if (maximum_items == 0) return 1;
    const size_t extra =
        maximum_items / 7 + (maximum_items % 7 != 0 ? 1 : 0);
    if (maximum_items >
        std::numeric_limits<size_t>::max() - extra) {
      throw std::runtime_error("reference sequence table size overflow");
    }
    return maximum_items + extra;
  }

  size_t capacity_ = 0;
  size_t size_ = 0;
  std::unique_ptr<uint64_t[]> slots_;
};

struct Phase1CoverScanResult {
  NodeId best = INVALID_NODE_ID;
  int best_dist = INT_MAX;
  size_t best_idx = std::numeric_limits<size_t>::max();
  size_t candidate_scans = 0;
  size_t length_pruned = 0;
  size_t lower_bound_pruned = 0;
  size_t exact_distance_reused = 0;
  size_t exact_rejection_reused = 0;
  size_t cross_layer_distance_reused = 0;
  size_t exact_distance_calls = 0;
};

class Phase1DistanceCache {
 public:
  explicit Phase1DistanceCache(size_t sequence_count)
      : entries_(sequence_count) {}

  void begin_query() {
    if (epoch_ == kMaxEpoch) {
      std::fill(entries_.begin(), entries_.end(), Entry{});
      epoch_ = 1;
    } else {
      epoch_++;
    }
  }

  bool lookup(LeafId sequence_id, int tau, int* result) const {
    if (!result || sequence_id >= entries_.size()) return false;
    const Entry& entry = entries_[sequence_id];
    if (entry.epoch != epoch_) return false;
    const int value = static_cast<int>(entry.value);
    if (entry.exact != 0) {
      *result = value <= tau
                    ? value
                    : tau + 1;
      return true;
    }
    if (value >= tau) {
      if (tau == std::numeric_limits<int>::max()) return false;
      *result = tau + 1;
      return true;
    }
    return false;
  }

  void store(LeafId sequence_id, int tau, int result) {
    if (sequence_id >= entries_.size()) return;
    Entry& entry = entries_[sequence_id];
    if (result <= tau) {
      if (result >= 0 && result <= kMaxValue) {
        entry = {epoch_, static_cast<uint8_t>(result), 1};
      }
      return;
    }
    if (tau < 0 || tau > kMaxValue) return;
    uint8_t rejected_tau = static_cast<uint8_t>(tau);
    if (entry.epoch == epoch_ && entry.exact == 0) {
      rejected_tau = std::max(rejected_tau, entry.value);
    }
    entry = {epoch_, rejected_tau, 0};
  }

 private:
  struct Entry {
    uint16_t epoch = 0;
    uint8_t value = 0;
    uint8_t exact = 0;
  };
  static constexpr int kMaxValue = std::numeric_limits<uint8_t>::max();
  static constexpr uint16_t kMaxEpoch =
      std::numeric_limits<uint16_t>::max();
  static_assert(sizeof(Entry) == 4,
                "phase1 distance-cache entry must remain 4 bytes");

  std::vector<Entry> entries_;
  uint16_t epoch_ = 0;
};

struct Phase1BaseCountSignature {
  uint32_t packed_counts = 0;

  bool safe() const {
    return packed_counts != std::numeric_limits<uint32_t>::max();
  }
};
static_assert(sizeof(Phase1BaseCountSignature) == 4,
              "compact base-count signature must remain 4 bytes");

Phase1BaseCountSignature phase1_base_count_signature(
    std::string_view sequence) {
  uint32_t packed = 0;
  for (char base : sequence) {
    uint32_t shift = 0;
    switch (base) {
      case 'A': break;
      case 'C': shift = 8; break;
      case 'G': shift = 16; break;
      case 'T': shift = 24; break;
      default:
        return {std::numeric_limits<uint32_t>::max()};
    }
    if (((packed >> shift) & 0xff) ==
        std::numeric_limits<uint8_t>::max()) {
      return {std::numeric_limits<uint32_t>::max()};
    }
    packed += uint32_t{1} << shift;
  }
  // Four counts of 255 collide with the reserved unsafe code. Such a
  // sequence is longer than the compact reference-window path, so disabling
  // this lower bound remains lossless.
  return {packed};
}

int phase1_base_count_lower_bound(
    const Phase1BaseCountSignature& lhs,
    const Phase1BaseCountSignature& rhs) {
  if (!lhs.safe() || !rhs.safe()) return 0;
  uint32_t l1 = 0;
  for (size_t base = 0; base < 4; ++base) {
    const uint32_t shift = static_cast<uint32_t>(base * 8);
    const uint32_t left = (lhs.packed_counts >> shift) & 0xff;
    const uint32_t right = (rhs.packed_counts >> shift) & 0xff;
    l1 += left > right ? left - right : right - left;
  }
  return static_cast<int>((l1 + 1) / 2);
}

struct Phase4QGramSignature {
  int q = 0;
  uint8_t count_bits = 0;
  bool safe_for_pruning = false;
  bool compact = false;
  std::vector<uint16_t> entries;
  QGramSignature fallback;
};

Phase4QGramSignature phase4_qgram_signature(
    std::string_view sequence, int q) {
  Phase4QGramSignature result;
  QGramSignature signature = compute_qgram_signature(sequence, q);
  result.q = signature.q;
  result.safe_for_pruning = signature.safe_for_pruning;
  if (!signature.safe_for_pruning || q <= 0 || q > 7) {
    result.fallback = std::move(signature);
    return result;
  }

  result.count_bits = static_cast<uint8_t>(16 - 2 * q);
  const uint32_t max_count =
      (uint32_t{1} << result.count_bits) - 1;
  result.entries.reserve(signature.entries.size());
  for (const auto& entry : signature.entries) {
    if (entry.code > std::numeric_limits<uint16_t>::max() ||
        entry.count > max_count) {
      result.entries.clear();
      result.fallback = std::move(signature);
      return result;
    }
    result.entries.push_back(static_cast<uint16_t>(
        (entry.code << result.count_bits) | entry.count));
  }
  result.compact = true;
  return result;
}

bool phase4_qgram_can_prune_edit_distance(
    const Phase4QGramSignature& lhs,
    const Phase4QGramSignature& rhs,
    int tau) {
  if (!lhs.safe_for_pruning || !rhs.safe_for_pruning ||
      lhs.q <= 0 || lhs.q != rhs.q || tau < 0) {
    return false;
  }
  if (!lhs.compact || !rhs.compact) {
    if (!lhs.compact && !rhs.compact) {
      return qgram_can_prune_edit_distance(lhs.fallback, rhs.fallback, tau);
    }
    return false;
  }

  const size_t q = static_cast<size_t>(lhs.q);
  const size_t threshold_tau = static_cast<size_t>(tau);
  if (threshold_tau > std::numeric_limits<size_t>::max() / q / 2) {
    return false;
  }
  const size_t max_l1 = 2 * q * threshold_tau;
  size_t l1 = 0;
  size_t lhs_idx = 0;
  size_t rhs_idx = 0;
  const uint16_t count_mask = static_cast<uint16_t>(
      (uint32_t{1} << lhs.count_bits) - 1);
  const auto entry_code = [&](uint16_t entry) {
    return static_cast<uint16_t>(entry >> lhs.count_bits);
  };
  const auto entry_count = [&](uint16_t entry) {
    return static_cast<uint16_t>(entry & count_mask);
  };
  while (lhs_idx < lhs.entries.size() || rhs_idx < rhs.entries.size()) {
    size_t delta = 0;
    if (rhs_idx == rhs.entries.size() ||
        (lhs_idx < lhs.entries.size() &&
         entry_code(lhs.entries[lhs_idx]) <
             entry_code(rhs.entries[rhs_idx]))) {
      delta = entry_count(lhs.entries[lhs_idx++]);
    } else if (lhs_idx == lhs.entries.size() ||
               entry_code(rhs.entries[rhs_idx]) <
                   entry_code(lhs.entries[lhs_idx])) {
      delta = entry_count(rhs.entries[rhs_idx++]);
    } else {
      const uint16_t left = entry_count(lhs.entries[lhs_idx++]);
      const uint16_t right = entry_count(rhs.entries[rhs_idx++]);
      delta = left > right ? left - right : right - left;
    }
    if (delta > max_l1 - std::min(l1, max_l1)) {
      return true;
    }
    l1 += delta;
  }
  return l1 > max_l1;
}

bool phase1_better_cover(size_t idx, int dist,
                         const Phase1CoverScanResult& current) {
  if (dist < current.best_dist) return true;
  return dist == current.best_dist && idx < current.best_idx;
}

Phase1CoverScanResult find_best_phase1_cover(
    const std::vector<NodeId>& candidates,
    const std::vector<BuildWorldNodeRecord>& nodes,
    const SequenceStore& sequences,
    const std::vector<Phase1BaseCountSignature>& sequence_signatures,
    const Phase1BaseCountSignature& query_signature,
    Phase1DistanceCache* distance_cache,
    const PreparedEdlibDnaPattern* prepared_query,
    std::string_view sequence,
    int radius,
    BuildDistanceMode distance_mode,
    const Phase1CoverScanResult& initial,
    size_t known_rejected_idx) {
  Phase1CoverScanResult result = initial;
  if (candidates.empty()) return result;
  if (candidates.size() >= kPhase1ParallelScanMinFanout) {
    distance_cache = nullptr;
  }

  auto scan_one = [&](size_t idx, Phase1CoverScanResult& local) {
    const NodeId node_id = candidates[idx];
    if (node_id >= nodes.size()) return;
    const auto& node = nodes[node_id];
    if (node.center_sequence_id >= sequences.size()) return;
    local.candidate_scans++;
    if (std::llabs(static_cast<long long>(sequence.size()) -
                   static_cast<long long>(sequences.sequence(
                       node.center_sequence_id).size())) >
        radius) {
      local.length_pruned++;
      return;
    }
    if (local.best == node_id && local.best_idx == idx) {
      local.exact_distance_reused++;
      return;
    }
    if (idx == known_rejected_idx) {
      local.exact_rejection_reused++;
      return;
    }
    if (node.center_sequence_id < sequence_signatures.size()) {
      const int lower_bound = phase1_base_count_lower_bound(
          query_signature, sequence_signatures[node.center_sequence_id]);
      if (lower_bound > radius ||
          (local.best != INVALID_NODE_ID &&
           (lower_bound > local.best_dist ||
            (lower_bound == local.best_dist && idx >= local.best_idx)))) {
        local.lower_bound_pruned++;
        return;
      }
    }
    int exact_tau = radius;
    if (local.best != INVALID_NODE_ID) {
      exact_tau = std::min(
          exact_tau,
          local.best_dist - (idx >= local.best_idx ? 1 : 0));
      if (exact_tau < 0) {
        local.lower_bound_pruned++;
        return;
      }
    }
    int dist = 0;
    if (distance_cache &&
        distance_cache->lookup(node.center_sequence_id, exact_tau, &dist)) {
      local.cross_layer_distance_reused++;
    } else {
      local.exact_distance_calls++;
      dist = build_distance_bounded_prepared(
          sequence, sequences.sequence(node.center_sequence_id), exact_tau,
          distance_mode, prepared_query);
      if (distance_cache) {
        distance_cache->store(node.center_sequence_id, exact_tau, dist);
      }
    }
    if (dist <= exact_tau && phase1_better_cover(idx, dist, local)) {
      local.best = node_id;
      local.best_dist = dist;
      local.best_idx = idx;
    }
  };

  if (candidates.size() < kPhase1ParallelScanMinFanout) {
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
      scan_one(idx, result);
    }
    return result;
  }

  const int thread_count = std::max(1, omp_get_max_threads());
  std::vector<Phase1CoverScanResult> local_results(
      static_cast<size_t>(thread_count), initial);
  #pragma omp parallel for schedule(static)
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    const int tid = omp_get_thread_num();
    Phase1CoverScanResult& local =
        local_results[static_cast<size_t>(std::min(tid, thread_count - 1))];
    scan_one(idx, local);
  }

  for (const auto& local : local_results) {
    result.candidate_scans += local.candidate_scans;
    result.length_pruned += local.length_pruned;
    result.lower_bound_pruned += local.lower_bound_pruned;
    result.exact_distance_reused += local.exact_distance_reused;
    result.exact_rejection_reused += local.exact_rejection_reused;
    result.cross_layer_distance_reused += local.cross_layer_distance_reused;
    result.exact_distance_calls += local.exact_distance_calls;
    if (local.best != INVALID_NODE_ID &&
        phase1_better_cover(local.best_idx, local.best_dist, result)) {
      result.best = local.best;
      result.best_dist = local.best_dist;
      result.best_idx = local.best_idx;
    }
  }
  return result;
}

Phase1CoverScanResult find_best_phase1_cover_by_indices(
    const std::vector<NodeId>& candidates,
    const std::vector<BuildWorldNodeRecord>& nodes,
    const SequenceStore& sequences,
    const std::vector<Phase1BaseCountSignature>& sequence_signatures,
    const Phase1BaseCountSignature& query_signature,
    Phase1DistanceCache* distance_cache,
    const PreparedEdlibDnaPattern* prepared_query,
    const std::vector<size_t>& candidate_indices,
    std::string_view sequence,
    int radius,
    BuildDistanceMode distance_mode,
    const Phase1CoverScanResult& initial,
    size_t known_rejected_idx) {
  Phase1CoverScanResult result = initial;
  if (candidate_indices.empty()) return result;
  if (candidate_indices.size() >= kPhase1ParallelScanMinFanout) {
    distance_cache = nullptr;
  }

  auto scan_one = [&](size_t pos, Phase1CoverScanResult& local) {
    const size_t idx = candidate_indices[pos];
    if (idx >= candidates.size()) return;
    const NodeId node_id = candidates[idx];
    if (node_id >= nodes.size()) return;
    const auto& node = nodes[node_id];
    if (node.center_sequence_id >= sequences.size()) return;
    local.candidate_scans++;
    if (std::llabs(static_cast<long long>(sequence.size()) -
                   static_cast<long long>(sequences.sequence(
                       node.center_sequence_id).size())) >
        radius) {
      local.length_pruned++;
      return;
    }
    if (local.best == node_id && local.best_idx == idx) {
      local.exact_distance_reused++;
      return;
    }
    if (idx == known_rejected_idx) {
      local.exact_rejection_reused++;
      return;
    }
    if (node.center_sequence_id < sequence_signatures.size()) {
      const int lower_bound = phase1_base_count_lower_bound(
          query_signature, sequence_signatures[node.center_sequence_id]);
      if (lower_bound > radius ||
          (local.best != INVALID_NODE_ID &&
           (lower_bound > local.best_dist ||
            (lower_bound == local.best_dist && idx >= local.best_idx)))) {
        local.lower_bound_pruned++;
        return;
      }
    }
    int exact_tau = radius;
    if (local.best != INVALID_NODE_ID) {
      exact_tau = std::min(
          exact_tau,
          local.best_dist - (idx >= local.best_idx ? 1 : 0));
      if (exact_tau < 0) {
        local.lower_bound_pruned++;
        return;
      }
    }
    int dist = 0;
    if (distance_cache &&
        distance_cache->lookup(node.center_sequence_id, exact_tau, &dist)) {
      local.cross_layer_distance_reused++;
    } else {
      local.exact_distance_calls++;
      dist = build_distance_bounded_prepared(
          sequence, sequences.sequence(node.center_sequence_id), exact_tau,
          distance_mode, prepared_query);
      if (distance_cache) {
        distance_cache->store(node.center_sequence_id, exact_tau, dist);
      }
    }
    if (dist <= exact_tau && phase1_better_cover(idx, dist, local)) {
      local.best = node_id;
      local.best_dist = dist;
      local.best_idx = idx;
    }
  };

  if (candidate_indices.size() < kPhase1ParallelScanMinFanout) {
    for (size_t pos = 0; pos < candidate_indices.size(); ++pos) {
      scan_one(pos, result);
    }
    return result;
  }

  const int thread_count = std::max(1, omp_get_max_threads());
  std::vector<Phase1CoverScanResult> local_results(
      static_cast<size_t>(thread_count), initial);
  #pragma omp parallel for schedule(static)
  for (size_t pos = 0; pos < candidate_indices.size(); ++pos) {
    const int tid = omp_get_thread_num();
    Phase1CoverScanResult& local =
        local_results[static_cast<size_t>(std::min(tid, thread_count - 1))];
    scan_one(pos, local);
  }

  for (const auto& local : local_results) {
    result.candidate_scans += local.candidate_scans;
    result.length_pruned += local.length_pruned;
    result.lower_bound_pruned += local.lower_bound_pruned;
    result.exact_distance_reused += local.exact_distance_reused;
    result.exact_rejection_reused += local.exact_rejection_reused;
    result.cross_layer_distance_reused += local.cross_layer_distance_reused;
    result.exact_distance_calls += local.exact_distance_calls;
    if (local.best != INVALID_NODE_ID &&
        phase1_better_cover(local.best_idx, local.best_dist, result)) {
      result.best = local.best;
      result.best_dist = local.best_dist;
      result.best_idx = local.best_idx;
    }
  }
  return result;
}

enum class Phase1CoverSource {
  Scan,
  Metric,
  Pigeonhole,
  QGram,
  FallbackScan,
};

struct Phase1CandidateQueryResult {
  Phase1CoverSource source = Phase1CoverSource::Scan;
  bool fallback_scan = false;
  std::vector<size_t> candidate_indices;
  size_t total_possible = 0;
  size_t metric_distance_calls = 0;
  size_t metric_build_distance_calls = 0;
  size_t pigeonhole_queries = 0;
  size_t seed_posting_entries_visited = 0;
  size_t pigeonhole_candidates = 0;
  size_t pigeonhole_fallbacks = 0;
  size_t qgram_touched_candidates = 0;
  size_t qgram_pruned_candidates = 0;
};

bool phase1_length_compatible(size_t lhs_len, size_t rhs_len, int radius) {
  return std::llabs(static_cast<long long>(lhs_len) -
                    static_cast<long long>(rhs_len)) <= radius;
}

class Phase1CoverGroupIndex {
 public:
  Phase1CandidateQueryResult query(
      const std::vector<NodeId>& candidates,
      const std::vector<BuildWorldNodeRecord>& nodes,
      const SequenceStore& sequences,
      std::string_view sequence,
      int radius,
      const BuildRangeConfig& config) {
    nodes_ = &nodes;
    sequences_ = &sequences;
    Phase1CandidateQueryResult result;
    result.total_possible = candidates.size();
    if (candidates.empty() ||
        config.phase1_candidate_mode == Phase1CandidateMode::Scan ||
        candidates.size() < config.phase1_metric_min_fanout) {
      result.source = Phase1CoverSource::Scan;
      return result;
    }

    const size_t build_calls_before = metric_build_distance_calls_;
    sync_items(candidates, config);
    result.metric_build_distance_calls =
        metric_build_distance_calls_ - build_calls_before;

    if (candidates.size() >= config.phase1_qgram_min_fanout) {
      auto seed = query_pigeonhole(candidates, sequence, radius, config);
      seed.metric_build_distance_calls = result.metric_build_distance_calls;
      if (seed.source == Phase1CoverSource::Pigeonhole) return seed;

      auto qgram = query_qgram(candidates, sequence, radius, config);
      qgram.metric_build_distance_calls = result.metric_build_distance_calls;
      qgram.pigeonhole_queries = seed.pigeonhole_queries;
      qgram.seed_posting_entries_visited =
          seed.seed_posting_entries_visited;
      qgram.pigeonhole_candidates = seed.pigeonhole_candidates;
      qgram.pigeonhole_fallbacks = seed.pigeonhole_fallbacks;
      return qgram;
    }

    auto metric = query_metric(candidates, sequence, radius, config);
    metric.metric_build_distance_calls = result.metric_build_distance_calls;
    return metric;
  }

 private:
  struct ItemInfo {
    uint8_t sequence_length = 0;
    uint8_t total_qgrams = 0;
    bool qgram_safe = false;
  };
  static_assert(sizeof(ItemInfo) <= 4,
                "phase1 q-gram item metadata must remain compact");

  struct MetricTreeNode {
    size_t item_idx = 0;
    std::unordered_map<int, size_t> children;
  };

  // Indexed sequences are at most 255 bases, so one byte is exact for the
  // multiplicity. Oversized groups fall back before a 24-bit item id wraps.
  using QGramPosting = uint32_t;
  static constexpr uint32_t kQGramPostingMaxItem = 0x00ffffffU;

  static QGramPosting pack_qgram_posting(uint32_t item_idx, uint8_t count) {
    return (item_idx << 8) | count;
  }

  static uint32_t qgram_posting_item(QGramPosting posting) {
    return posting >> 8;
  }

  static uint8_t qgram_posting_count(QGramPosting posting) {
    return static_cast<uint8_t>(posting & 0xffU);
  }

  std::string_view center_sequence(
      const std::vector<NodeId>& candidates, size_t idx) const {
    static constexpr std::string_view empty;
    if (!nodes_ || !sequences_ || idx >= candidates.size() ||
        candidates[idx] >= nodes_->size()) {
      return empty;
    }
    const LeafId center_id = (*nodes_)[candidates[idx]].center_sequence_id;
    if (center_id >= sequences_->size()) return empty;
    return sequences_->sequence(center_id);
  }

  static Phase1CandidateQueryResult fallback_scan_result(size_t total_possible) {
    Phase1CandidateQueryResult result;
    result.source = Phase1CoverSource::FallbackScan;
    result.fallback_scan = true;
    result.total_possible = total_possible;
    return result;
  }

  void sync_items(
      const std::vector<NodeId>& candidates,
      const BuildRangeConfig& config) {
    while (items_.size() < candidates.size()) {
      const size_t idx = items_.size();
      ItemInfo info;
      const size_t sequence_length = center_sequence(candidates, idx).size();
      if (sequence_length > std::numeric_limits<uint8_t>::max()) {
        throw std::runtime_error(
            "phase1 sequence length exceeds compact 8-bit storage");
      }
      info.sequence_length = static_cast<uint8_t>(sequence_length);
      items_.push_back(info);
      if (metric_built_ && items_.size() < config.phase1_qgram_min_fanout) {
        insert_metric_item(candidates, idx, config.distance_mode);
      }
      if (qgram_built_) {
        add_qgram_item(candidates, idx, config.range_join.qgram_q);
      }
    }
  }

  void ensure_metric(
      const std::vector<NodeId>& candidates,
      BuildDistanceMode distance_mode) {
    if (metric_built_) return;
    metric_nodes_.clear();
    metric_nodes_.reserve(items_.size());
    metric_root_ = std::numeric_limits<size_t>::max();
    metric_built_ = true;
    for (size_t idx = 0; idx < items_.size(); ++idx) {
      insert_metric_item(candidates, idx, distance_mode);
    }
  }

  void insert_metric_item(
      const std::vector<NodeId>& candidates,
      size_t item_idx,
      BuildDistanceMode distance_mode) {
    if (!nodes_ || !sequences_ || item_idx >= candidates.size() ||
        candidates[item_idx] >= nodes_->size() ||
        (*nodes_)[candidates[item_idx]].center_sequence_id >=
            sequences_->size()) {
      return;
    }
    if (metric_root_ == std::numeric_limits<size_t>::max()) {
      metric_root_ = metric_nodes_.size();
      metric_nodes_.push_back({item_idx, {}});
      return;
    }

    const std::string_view sequence =
        center_sequence(candidates, item_idx);
    size_t current = metric_root_;
    while (true) {
      MetricTreeNode& node = metric_nodes_[current];
      const int dist = build_distance(
          sequence, center_sequence(candidates, node.item_idx), distance_mode);
      metric_build_distance_calls_++;
      auto child_it = node.children.find(dist);
      if (child_it == node.children.end()) {
        const size_t new_node_idx = metric_nodes_.size();
        node.children.emplace(dist, new_node_idx);
        metric_nodes_.push_back({item_idx, {}});
        return;
      }
      current = child_it->second;
    }
  }

  Phase1CandidateQueryResult query_metric(
      const std::vector<NodeId>& candidates,
      std::string_view sequence,
      int radius,
      const BuildRangeConfig& config) {
    Phase1CandidateQueryResult result;
    result.source = Phase1CoverSource::Metric;
    result.total_possible = candidates.size();
    ensure_metric(candidates, config.distance_mode);
    if (metric_root_ == std::numeric_limits<size_t>::max()) return result;

    std::vector<size_t> stack;
    stack.push_back(metric_root_);
    while (!stack.empty()) {
      const size_t node_idx = stack.back();
      stack.pop_back();
      const MetricTreeNode& node = metric_nodes_[node_idx];
      const int dist = build_distance(
          sequence, center_sequence(candidates, node.item_idx),
          config.distance_mode);
      result.metric_distance_calls++;
      if (dist <= radius) result.candidate_indices.push_back(node.item_idx);

      const int min_edge = std::max(0, dist - radius);
      const int max_edge = dist + radius;
      for (const auto& child : node.children) {
        if (child.first >= min_edge && child.first <= max_edge) {
          stack.push_back(child.second);
        }
      }
    }

    std::sort(result.candidate_indices.begin(), result.candidate_indices.end());
    result.candidate_indices.erase(
        std::unique(result.candidate_indices.begin(),
                    result.candidate_indices.end()),
        result.candidate_indices.end());
    return result;
  }

  void ensure_qgram(
      const std::vector<NodeId>& candidates,
      int q) {
    if (qgram_built_ && qgram_q_ == q) return;
    qgram_postings_.clear();
    qgram_unsafe_indices_.clear();
    qgram_zero_total_indices_.clear();
    qgram_index_available_ = true;
    min_safe_total_qgrams_ = std::numeric_limits<size_t>::max();
    qgram_q_ = q;
    qgram_built_ = true;
    for (size_t idx = 0; idx < items_.size(); ++idx) {
      add_qgram_item(candidates, idx, q);
    }
  }

  Phase1CandidateQueryResult query_pigeonhole(
      const std::vector<NodeId>& candidates,
      std::string_view sequence,
      int radius,
      const BuildRangeConfig& config) {
    Phase1CandidateQueryResult result;
    result.total_possible = candidates.size();
    result.pigeonhole_queries = 1;

    const int min_seed_len = config.range_join.min_seed_len;
    const int max_seed_len = config.range_join.max_seed_len;
    if (!seed_index_ || seed_min_len_ != min_seed_len ||
        seed_max_len_ != max_seed_len) {
      seed_index_ = std::make_unique<IncrementalPigeonholeIndex>(
          Phase1SeedIndexConfig{min_seed_len, max_seed_len});
      seed_min_len_ = min_seed_len;
      seed_max_len_ = max_seed_len;
      seed_synced_items_ = 0;
    }
    while (seed_synced_items_ < candidates.size()) {
      seed_index_->append(
          seed_synced_items_,
          center_sequence(candidates, seed_synced_items_));
      seed_synced_items_++;
    }

    auto seed_result = seed_index_->query(sequence, radius);
    result.seed_posting_entries_visited =
        seed_result.posting_entries_visited;
    result.pigeonhole_candidates = seed_result.candidate_indices.size();
    if (!seed_result.safe) {
      result.source = Phase1CoverSource::FallbackScan;
      result.fallback_scan = true;
      result.pigeonhole_fallbacks = 1;
      return result;
    }

    result.source = Phase1CoverSource::Pigeonhole;
    result.candidate_indices = std::move(seed_result.candidate_indices);
    return result;
  }

  void add_qgram_item(
      const std::vector<NodeId>& candidates,
      size_t idx,
      int q) {
    if (idx >= items_.size()) return;
    const std::string_view sequence = center_sequence(candidates, idx);
    if (sequence.size() > std::numeric_limits<uint8_t>::max()) {
      qgram_index_available_ = false;
      qgram_postings_.clear();
      return;
    }
    auto signature = compute_qgram_signature(sequence, q);
    if (signature.total_qgrams > std::numeric_limits<uint8_t>::max() ||
        idx > kQGramPostingMaxItem) {
      qgram_index_available_ = false;
      qgram_postings_.clear();
      return;
    }
    items_[idx].sequence_length = static_cast<uint8_t>(sequence.size());
    items_[idx].qgram_safe = signature.safe_for_pruning;
    items_[idx].total_qgrams =
        static_cast<uint8_t>(signature.total_qgrams);
    if (!qgram_index_available_) return;
    if (!signature.safe_for_pruning) {
      qgram_unsafe_indices_.push_back(static_cast<uint32_t>(idx));
      return;
    }
    min_safe_total_qgrams_ =
        std::min(min_safe_total_qgrams_, signature.total_qgrams);
    if (signature.total_qgrams == 0) {
      qgram_zero_total_indices_.push_back(static_cast<uint32_t>(idx));
      return;
    }
    for (const auto& entry : signature.entries) {
      if (entry.count > std::numeric_limits<uint8_t>::max()) {
        qgram_index_available_ = false;
        qgram_postings_.clear();
        return;
      }
      qgram_postings_[entry.code].push_back(
          pack_qgram_posting(
              static_cast<uint32_t>(idx),
              static_cast<uint8_t>(entry.count)));
    }
  }

  void reset_qgram_workspace(size_t item_count) {
    if (qgram_shared_.size() < item_count) qgram_shared_.resize(item_count, 0);
    if (qgram_seen_epoch_.size() < item_count) {
      qgram_seen_epoch_.resize(item_count, 0);
    }
    qgram_touched_.clear();
    if (qgram_epoch_ == std::numeric_limits<uint16_t>::max()) {
      std::fill(qgram_seen_epoch_.begin(), qgram_seen_epoch_.end(), 0);
      qgram_epoch_ = 1;
    } else {
      qgram_epoch_++;
    }
  }

  Phase1CandidateQueryResult query_qgram(
      const std::vector<NodeId>& candidates,
      std::string_view sequence,
      int radius,
      const BuildRangeConfig& config) {
    Phase1CandidateQueryResult result;
    result.source = Phase1CoverSource::QGram;
    result.total_possible = candidates.size();
    ensure_qgram(candidates, config.range_join.qgram_q);
    if (!qgram_index_available_) {
      return fallback_scan_result(candidates.size());
    }

    const int q = config.range_join.qgram_q;
    const auto query_signature = compute_qgram_signature(sequence, q);
    if (!query_signature.safe_for_pruning ||
        query_signature.total_qgrams == 0 ||
        min_safe_total_qgrams_ == std::numeric_limits<size_t>::max()) {
      return fallback_scan_result(candidates.size());
    }

    const size_t tau = static_cast<size_t>(radius);
    const size_t q_size = static_cast<size_t>(q);
    if (tau > std::numeric_limits<size_t>::max() / q_size / 2) {
      return fallback_scan_result(candidates.size());
    }
    const size_t max_l1 = 2 * q_size * tau;
    if (query_signature.total_qgrams >
        std::numeric_limits<size_t>::max() - min_safe_total_qgrams_) {
      return fallback_scan_result(candidates.size());
    }
    if (query_signature.total_qgrams + min_safe_total_qgrams_ <= max_l1) {
      return fallback_scan_result(candidates.size());
    }

    reset_qgram_workspace(items_.size());
    for (const auto& query_entry : query_signature.entries) {
      auto posting_it = qgram_postings_.find(query_entry.code);
      if (posting_it == qgram_postings_.end()) continue;
      for (QGramPosting posting : posting_it->second) {
        const size_t idx = qgram_posting_item(posting);
        if (idx >= items_.size()) continue;
        if (qgram_seen_epoch_[idx] != qgram_epoch_) {
          qgram_seen_epoch_[idx] = qgram_epoch_;
          qgram_shared_[idx] = 0;
          qgram_touched_.push_back(idx);
          if (qgram_touched_.size() > config.phase1_qgram_max_touched) {
            return fallback_scan_result(candidates.size());
          }
        }
        const uint32_t shared = std::min<uint32_t>(
            query_entry.count, qgram_posting_count(posting));
        const uint32_t remaining =
            static_cast<uint32_t>(std::numeric_limits<uint8_t>::max()) -
            qgram_shared_[idx];
        if (shared > remaining) {
          return fallback_scan_result(candidates.size());
        }
        qgram_shared_[idx] = static_cast<uint8_t>(
            qgram_shared_[idx] + shared);
      }
    }
    result.qgram_touched_candidates = qgram_touched_.size();

    for (uint32_t idx : qgram_touched_) {
      const auto& item = items_[idx];
      if (!phase1_length_compatible(sequence.size(),
                                    item.sequence_length, radius)) {
        continue;
      }
      const size_t total_sum =
          query_signature.total_qgrams + item.total_qgrams;
      if (total_sum <= max_l1) {
        return fallback_scan_result(candidates.size());
      }
      const size_t required_shared = (total_sum - max_l1 + 1) / 2;
      if (qgram_shared_[idx] >= required_shared) {
        result.candidate_indices.push_back(idx);
        if (result.candidate_indices.size() >
            config.phase1_qgram_max_touched) {
          return fallback_scan_result(candidates.size());
        }
      }
    }

    auto append_unprunable = [&](const std::vector<uint32_t>& indices) {
      for (uint32_t idx : indices) {
        if (idx >= items_.size()) continue;
        if (phase1_length_compatible(sequence.size(),
                                     items_[idx].sequence_length, radius)) {
          result.candidate_indices.push_back(idx);
        }
      }
    };
    append_unprunable(qgram_unsafe_indices_);
    append_unprunable(qgram_zero_total_indices_);
    if (result.candidate_indices.size() > config.phase1_qgram_max_touched) {
      return fallback_scan_result(candidates.size());
    }

    std::sort(result.candidate_indices.begin(), result.candidate_indices.end());
    result.candidate_indices.erase(
        std::unique(result.candidate_indices.begin(),
                    result.candidate_indices.end()),
        result.candidate_indices.end());
    result.qgram_pruned_candidates =
        result.total_possible > result.candidate_indices.size()
            ? result.total_possible - result.candidate_indices.size()
            : 0;
    return result;
  }

  std::vector<ItemInfo> items_;
  bool metric_built_ = false;
  size_t metric_root_ = std::numeric_limits<size_t>::max();
  std::vector<MetricTreeNode> metric_nodes_;
  size_t metric_build_distance_calls_ = 0;

  bool qgram_built_ = false;
  bool qgram_index_available_ = true;
  int qgram_q_ = 0;
  size_t min_safe_total_qgrams_ = std::numeric_limits<size_t>::max();
  std::unordered_map<uint64_t, std::vector<QGramPosting>> qgram_postings_;
  std::vector<uint32_t> qgram_unsafe_indices_;
  std::vector<uint32_t> qgram_zero_total_indices_;
  std::vector<uint8_t> qgram_shared_;
  std::vector<uint16_t> qgram_seen_epoch_;
  std::vector<uint32_t> qgram_touched_;
  uint16_t qgram_epoch_ = 1;

  std::unique_ptr<IncrementalPigeonholeIndex> seed_index_;
  int seed_min_len_ = 0;
  int seed_max_len_ = 0;
  size_t seed_synced_items_ = 0;
  const std::vector<BuildWorldNodeRecord>* nodes_ = nullptr;
  const SequenceStore* sequences_ = nullptr;
};

std::vector<int> make_auxiliary_radii(const std::vector<int>& primary_radii) {
  std::vector<int> out;
  if (primary_radii.size() < 2) return out;
  out.reserve(primary_radii.size() - 1);
  for (size_t i = 0; i + 1 < primary_radii.size(); ++i) {
    out.push_back(std::max(1, (primary_radii[i] + primary_radii[i + 1]) / 2));
  }
  return out;
}

int build_distance(std::string_view a, std::string_view b,
                   BuildDistanceMode mode) {
  return mode == BuildDistanceMode::Edlib ? compute_distance_edlib(a, b)
                                          : compute_distance(a, b);
}

int build_distance_bounded(std::string_view a, std::string_view b, int tau,
                           BuildDistanceMode mode) {
  return mode == BuildDistanceMode::Edlib
             ? compute_distance_bounded_edlib(a, b, tau)
             : compute_distance_bounded_dp(a, b, tau);
}

int build_distance_bounded_prepared(
    std::string_view a, std::string_view b, int tau,
    BuildDistanceMode mode,
    const PreparedEdlibDnaPattern* prepared_query) {
  if (mode == BuildDistanceMode::Edlib && prepared_query) {
    return compute_distance_bounded_edlib_prepared(
        *prepared_query, b, tau);
  }
  return build_distance_bounded(a, b, tau, mode);
}

DistanceMode to_distance_mode(BuildDistanceMode mode) {
  switch (mode) {
    case BuildDistanceMode::DP:
      return DistanceMode::DP;
    case BuildDistanceMode::Edlib:
      return DistanceMode::Edlib;
    case BuildDistanceMode::Auto:
      return DistanceMode::Auto;
  }
  return DistanceMode::DP;
}

std::vector<int> build_expanded_radii(const HierarchyConfig& config) {
  std::vector<int> expanded;
  expanded.reserve(static_cast<size_t>(config.num_expanded_layers()));
  for (int i = 0; i < config.num_primary_layers(); ++i) {
    expanded.push_back(config.primary_radii[static_cast<size_t>(i)]);
    if (i < config.num_auxiliary_layers()) {
      expanded.push_back(config.auxiliary_radii[static_cast<size_t>(i)]);
    }
  }
  return expanded;
}

bool expanded_layer_is_primary(int expanded_layer_idx) {
  return expanded_layer_idx % 2 == 0;
}

int expanded_to_primary_index(int expanded_layer_idx) {
  return expanded_layer_idx / 2;
}

double reduction_ratio(size_t before, size_t after) {
  if (before == 0) return 0.0;
  return 1.0 - static_cast<double>(after) / static_cast<double>(before);
}

std::string format_ms(double value) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(3) << value;
  return os.str();
}

void accumulate_range_timing(BioGeometryIndexBuilder::Statistics& stats,
                             const RangeJoinQueryResult& result) {
  stats.range_posting_lookup_ms += result.range_posting_lookup_ms;
  stats.range_seed_union_ms += result.range_seed_union_ms;
  stats.range_length_filter_ms += result.range_length_filter_ms;
  stats.range_qgram_query_ms += result.range_qgram_query_ms;
  stats.range_hybrid_intersection_ms += result.range_hybrid_intersection_ms;
  stats.range_full_scan_ms += result.range_full_scan_ms;
}

void accumulate_phase2_candidate_stats(
    BioGeometryIndexBuilder::Statistics& stats,
    const RangeJoinQueryResult& candidates) {
  accumulate_range_timing(stats, candidates);
  stats.phase2_candidate_pairs += candidates.candidate_item_ids.size();
  if (candidates.used_full_scan) stats.phase2_full_scan_fallback_count++;
  if (candidates.mode_used == RangeCandidateMode::PigeonholeOnly) {
    stats.phase2_pigeonhole_queries++;
  } else if (candidates.mode_used == RangeCandidateMode::QGramOnly) {
    stats.phase2_qgram_queries++;
  } else if (candidates.mode_used == RangeCandidateMode::Hybrid) {
    stats.phase2_hybrid_queries++;
  }
  stats.phase2_qgram_candidate_pairs += candidates.qgram_candidate_count;
  stats.phase2_qgram_pruned_by_l1 += candidates.qgram_pruned_by_l1;
  stats.phase2_length_pruned_pairs += candidates.length_filtered_items;
  stats.phase2_seed_candidate_pairs_before_length_filter +=
      candidates.seed_candidate_pairs_before_length_filter;
  stats.phase2_seed_length_pruned_candidates +=
      candidates.seed_length_pruned_candidates;
  stats.phase2_pigeonhole_early_abort_count +=
      candidates.pigeonhole_early_abort_count;
  stats.phase2_range_final_candidate_pairs += candidates.final_candidate_pairs;
  stats.phase2_required_shared_nonpositive_count +=
      candidates.required_shared_nonpositive;
  stats.phase2_auto_pigeonhole_accepted +=
      candidates.auto_pigeonhole_accepted;
  stats.phase2_auto_pigeonhole_rejected_large_candidates +=
      candidates.auto_pigeonhole_rejected_large_candidates;
  stats.phase2_auto_qgram_invoked += candidates.auto_qgram_invoked;
  stats.phase2_auto_hybrid_invoked += candidates.auto_hybrid_invoked;
  stats.phase2_auto_final_candidate_pairs +=
      candidates.auto_final_candidate_pairs;
  stats.phase2_auto_candidate_ratio_sum +=
      candidates.auto_candidate_ratio_sum;
}

void accumulate_leaf_candidate_stats(
    BioGeometryIndexBuilder::Statistics& stats,
    const RangeJoinQueryResult& candidates) {
  accumulate_range_timing(stats, candidates);
  stats.leaf_candidate_pairs += candidates.candidate_item_ids.size();
  if (candidates.used_full_scan) stats.leaf_full_scan_fallback_count++;
  if (candidates.mode_used == RangeCandidateMode::PigeonholeOnly) {
    stats.leaf_pigeonhole_queries++;
  } else if (candidates.mode_used == RangeCandidateMode::QGramOnly) {
    stats.leaf_qgram_queries++;
  } else if (candidates.mode_used == RangeCandidateMode::Hybrid) {
    stats.leaf_hybrid_queries++;
  }
  stats.leaf_qgram_candidate_pairs += candidates.qgram_candidate_count;
  stats.leaf_qgram_pruned_by_l1 += candidates.qgram_pruned_by_l1;
  stats.leaf_length_pruned_pairs += candidates.length_filtered_items;
  stats.leaf_seed_candidate_pairs_before_length_filter +=
      candidates.seed_candidate_pairs_before_length_filter;
  stats.leaf_seed_length_pruned_candidates +=
      candidates.seed_length_pruned_candidates;
  stats.leaf_pigeonhole_early_abort_count +=
      candidates.pigeonhole_early_abort_count;
  stats.leaf_range_final_candidate_pairs += candidates.final_candidate_pairs;
  stats.leaf_required_shared_nonpositive_count +=
      candidates.required_shared_nonpositive;
  stats.leaf_auto_pigeonhole_accepted +=
      candidates.auto_pigeonhole_accepted;
  stats.leaf_auto_pigeonhole_rejected_large_candidates +=
      candidates.auto_pigeonhole_rejected_large_candidates;
  stats.leaf_auto_qgram_invoked += candidates.auto_qgram_invoked;
  stats.leaf_auto_hybrid_invoked += candidates.auto_hybrid_invoked;
  stats.leaf_auto_final_candidate_pairs +=
      candidates.auto_final_candidate_pairs;
  stats.leaf_auto_candidate_ratio_sum +=
      candidates.auto_candidate_ratio_sum;
}

void merge_leaf_local_stats(
    BioGeometryIndexBuilder::Statistics& target,
    const BioGeometryIndexBuilder::Statistics& local) {
  target.leaf_candidate_pairs += local.leaf_candidate_pairs;
  target.leaf_exact_distance_calls += local.leaf_exact_distance_calls;
  target.leaf_full_scan_fallback_count +=
      local.leaf_full_scan_fallback_count;
  target.leaf_pigeonhole_queries += local.leaf_pigeonhole_queries;
  target.leaf_qgram_queries += local.leaf_qgram_queries;
  target.leaf_hybrid_queries += local.leaf_hybrid_queries;
  target.leaf_qgram_candidate_pairs += local.leaf_qgram_candidate_pairs;
  target.leaf_qgram_pruned_by_l1 += local.leaf_qgram_pruned_by_l1;
  target.leaf_base_count_pruned_pairs +=
      local.leaf_base_count_pruned_pairs;
  target.leaf_length_pruned_pairs += local.leaf_length_pruned_pairs;
  target.leaf_seed_candidate_pairs_before_length_filter +=
      local.leaf_seed_candidate_pairs_before_length_filter;
  target.leaf_seed_length_pruned_candidates +=
      local.leaf_seed_length_pruned_candidates;
  target.leaf_pigeonhole_early_abort_count +=
      local.leaf_pigeonhole_early_abort_count;
  target.leaf_range_final_candidate_pairs +=
      local.leaf_range_final_candidate_pairs;
  target.leaf_required_shared_nonpositive_count +=
      local.leaf_required_shared_nonpositive_count;
  target.leaf_auto_pigeonhole_accepted +=
      local.leaf_auto_pigeonhole_accepted;
  target.leaf_auto_pigeonhole_rejected_large_candidates +=
      local.leaf_auto_pigeonhole_rejected_large_candidates;
  target.leaf_auto_qgram_invoked += local.leaf_auto_qgram_invoked;
  target.leaf_auto_hybrid_invoked += local.leaf_auto_hybrid_invoked;
  target.leaf_auto_final_candidate_pairs +=
      local.leaf_auto_final_candidate_pairs;
  target.leaf_auto_candidate_ratio_sum +=
      local.leaf_auto_candidate_ratio_sum;
  target.range_posting_lookup_ms += local.range_posting_lookup_ms;
  target.range_seed_union_ms += local.range_seed_union_ms;
  target.range_length_filter_ms += local.range_length_filter_ms;
  target.range_qgram_query_ms += local.range_qgram_query_ms;
  target.range_hybrid_intersection_ms +=
      local.range_hybrid_intersection_ms;
  target.range_full_scan_ms += local.range_full_scan_ms;
  target.leaf_candidate_query_ms += local.leaf_candidate_query_ms;
  target.leaf_exact_verify_ms += local.leaf_exact_verify_ms;
  target.leaf_beacon_distance_ms += local.leaf_beacon_distance_ms;
  target.leaf_tuple_emit_ms += local.leaf_tuple_emit_ms;
}

void merge_phase2_local_stats(BioGeometryIndexBuilder::Statistics& target,
                              const BioGeometryIndexBuilder::Statistics& local) {
  target.phase2_candidate_pairs += local.phase2_candidate_pairs;
  target.phase2_exact_distance_calls += local.phase2_exact_distance_calls;
  target.phase2_edges_added += local.phase2_edges_added;
  target.phase2_full_scan_fallback_count += local.phase2_full_scan_fallback_count;
  target.phase2_pigeonhole_queries += local.phase2_pigeonhole_queries;
  target.phase2_qgram_queries += local.phase2_qgram_queries;
  target.phase2_hybrid_queries += local.phase2_hybrid_queries;
  target.phase2_qgram_candidate_pairs += local.phase2_qgram_candidate_pairs;
  target.phase2_qgram_pruned_by_l1 += local.phase2_qgram_pruned_by_l1;
  target.phase2_base_count_pruned_pairs +=
      local.phase2_base_count_pruned_pairs;
  target.phase2_length_pruned_pairs += local.phase2_length_pruned_pairs;
  target.phase2_seed_candidate_pairs_before_length_filter +=
      local.phase2_seed_candidate_pairs_before_length_filter;
  target.phase2_seed_length_pruned_candidates +=
      local.phase2_seed_length_pruned_candidates;
  target.phase2_pigeonhole_early_abort_count +=
      local.phase2_pigeonhole_early_abort_count;
  target.phase2_range_final_candidate_pairs +=
      local.phase2_range_final_candidate_pairs;
  target.phase2_required_shared_nonpositive_count +=
      local.phase2_required_shared_nonpositive_count;
  target.phase2_auto_pigeonhole_accepted +=
      local.phase2_auto_pigeonhole_accepted;
  target.phase2_auto_pigeonhole_rejected_large_candidates +=
      local.phase2_auto_pigeonhole_rejected_large_candidates;
  target.phase2_auto_qgram_invoked += local.phase2_auto_qgram_invoked;
  target.phase2_auto_hybrid_invoked += local.phase2_auto_hybrid_invoked;
  target.phase2_auto_final_candidate_pairs +=
      local.phase2_auto_final_candidate_pairs;
  target.phase2_auto_candidate_ratio_sum +=
      local.phase2_auto_candidate_ratio_sum;
  target.range_posting_lookup_ms += local.range_posting_lookup_ms;
  target.range_seed_union_ms += local.range_seed_union_ms;
  target.range_length_filter_ms += local.range_length_filter_ms;
  target.range_qgram_query_ms += local.range_qgram_query_ms;
  target.range_hybrid_intersection_ms += local.range_hybrid_intersection_ms;
  target.range_full_scan_ms += local.range_full_scan_ms;
  target.phase2_distance_batches += local.phase2_distance_batches;
}

void validate_range_config(const BuildRangeConfig& config) {
  if (config.range_join.min_seed_len <= 0) {
    throw std::invalid_argument("range-join min seed length must be positive");
  }
  if (config.range_join.max_seed_len < config.range_join.min_seed_len) {
    throw std::invalid_argument(
        "range-join max seed length must be at least min seed length");
  }
  if (config.range_join.qgram_q <= 0) {
    throw std::invalid_argument("range-join q-gram length must be positive");
  }
  if (!std::isfinite(config.range_join.auto_pigeonhole_max_ratio) ||
      config.range_join.auto_pigeonhole_max_ratio < 0.0 ||
      config.range_join.auto_pigeonhole_max_ratio > 1.0) {
    throw std::invalid_argument(
        "auto pigeonhole max ratio must be finite and in [0, 1]");
  }
  if (config.min_rect_index_fanout == 0) {
    throw std::invalid_argument("minimum rectangle-index fanout must be positive");
  }
  if (config.phase1_metric_min_fanout == 0) {
    throw std::invalid_argument("phase1 metric fanout threshold must be positive");
  }
  if (config.phase1_qgram_min_fanout < config.phase1_metric_min_fanout) {
    throw std::invalid_argument(
        "phase1 q-gram fanout threshold must be at least metric threshold");
  }
  if (config.phase1_qgram_max_touched == 0) {
    throw std::invalid_argument("phase1 q-gram touched limit must be positive");
  }
}

}  // namespace

HierarchyConfig::HierarchyConfig(std::vector<int> primary_radii_in)
    : primary_radii(std::move(primary_radii_in)),
      auxiliary_radii(make_auxiliary_radii(primary_radii)) {
  validate();
}

HierarchyConfig::HierarchyConfig(std::vector<int> primary_radii_in,
                                 std::vector<int> auxiliary_radii_in)
    : primary_radii(std::move(primary_radii_in)),
      auxiliary_radii(std::move(auxiliary_radii_in)) {
  validate();
}

int HierarchyConfig::num_primary_layers() const {
  return static_cast<int>(primary_radii.size());
}

int HierarchyConfig::num_auxiliary_layers() const {
  return static_cast<int>(auxiliary_radii.size());
}

int HierarchyConfig::num_expanded_layers() const {
  if (primary_radii.empty()) return 0;
  return static_cast<int>(primary_radii.size() * 2 - 1);
}

void HierarchyConfig::validate() const {
  if (primary_radii.size() < 2) {
    throw std::invalid_argument("HierarchyConfig requires at least two primary radii");
  }
  if (auxiliary_radii.size() != primary_radii.size() - 1) {
    throw std::invalid_argument("HierarchyConfig auxiliary_radii must have size K-1");
  }
  if (primary_radii.front() >
      static_cast<int>(std::numeric_limits<uint16_t>::max())) {
    throw std::invalid_argument(
        "HierarchyConfig radii exceed 16-bit stored distance range");
  }
  for (size_t i = 0; i + 1 < primary_radii.size(); ++i) {
    if (primary_radii[i] <= primary_radii[i + 1]) {
      throw std::invalid_argument("HierarchyConfig primary_radii must be strictly decreasing");
    }
    int aux = auxiliary_radii[i];
    if (aux <= 0) {
      throw std::invalid_argument("HierarchyConfig auxiliary radii must be positive");
    }
    if (aux > primary_radii[i] || aux < primary_radii[i + 1]) {
      throw std::invalid_argument("HierarchyConfig auxiliary radius must lie between adjacent primary radii");
    }
  }
}

const char* build_range_mode_name(BuildRangeMode mode) {
  return mode == BuildRangeMode::Full ? "full" : "indexed";
}

BuildRangeMode parse_build_range_mode(const std::string& value) {
  if (value == "full") return BuildRangeMode::Full;
  if (value == "indexed") return BuildRangeMode::Indexed;
  throw std::invalid_argument("build range mode must be full or indexed");
}

const char* leaf_attach_direction_name(LeafAttachDirection direction) {
  switch (direction) {
    case LeafAttachDirection::SeqToWorld:
      return "seq_to_world";
    case LeafAttachDirection::WorldToSeq:
      return "world_to_seq";
    case LeafAttachDirection::Auto:
      return "auto";
  }
  return "auto";
}

LeafAttachDirection parse_leaf_attach_direction(const std::string& value) {
  if (value == "seq-to-world" || value == "seq_to_world") {
    return LeafAttachDirection::SeqToWorld;
  }
  if (value == "world-to-seq" || value == "world_to_seq") {
    return LeafAttachDirection::WorldToSeq;
  }
  if (value == "auto") return LeafAttachDirection::Auto;
  throw std::invalid_argument(
      "leaf attach direction must be auto, seq-to-world, or world-to-seq");
}

const char* build_distance_mode_name(BuildDistanceMode mode) {
  switch (mode) {
    case BuildDistanceMode::DP:
      return "dp";
    case BuildDistanceMode::Edlib:
      return "edlib";
    case BuildDistanceMode::Auto:
      return "auto";
  }
  return "dp";
}

BuildDistanceMode parse_build_distance_mode(const std::string& value) {
  if (value == "dp") return BuildDistanceMode::DP;
  if (value == "edlib") return BuildDistanceMode::Edlib;
  if (value == "auto") return BuildDistanceMode::Auto;
  throw std::invalid_argument("build distance mode must be dp, edlib, or auto");
}

const char* phase1_candidate_mode_name(Phase1CandidateMode mode) {
  switch (mode) {
    case Phase1CandidateMode::Scan:
      return "scan";
    case Phase1CandidateMode::Hybrid:
      return "hybrid";
  }
  return "hybrid";
}

Phase1CandidateMode parse_phase1_candidate_mode(const std::string& value) {
  if (value == "scan") return Phase1CandidateMode::Scan;
  if (value == "hybrid") return Phase1CandidateMode::Hybrid;
  throw std::invalid_argument("phase1 candidate mode must be scan or hybrid");
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder()
    : stats_{},
      hierarchy_(HierarchyConfig({R_LW, R_MW, R_SW})),
      range_config_{},
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())) {
  stats_.created_primary_nodes.assign(static_cast<size_t>(hierarchy_.num_primary_layers()), 0);
}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw)
    : BioGeometryIndexBuilder(HierarchyConfig({r_lw, r_mw, r_sw})) {}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(const HierarchyConfig& config)
    : BioGeometryIndexBuilder(config, BuildRangeConfig{}) {}

BioGeometryIndexBuilder::BioGeometryIndexBuilder(
    const HierarchyConfig& config, const BuildRangeConfig& range_config)
    : stats_{},
      hierarchy_(config),
      range_config_(range_config),
      expanded_radii_(build_expanded_radii(hierarchy_)),
      primary_layers_(static_cast<size_t>(hierarchy_.num_primary_layers())) {
  validate_range_config(range_config_);
  stats_.created_primary_nodes.assign(static_cast<size_t>(hierarchy_.num_primary_layers()), 0);
}

size_t BioGeometryIndexBuilder::primary_layer_size(int idx) const {
  const size_t layer = static_cast<size_t>(idx);
  if (layer >= search_graph_view_.layer_begin.size() ||
      layer >= search_graph_view_.layer_end.size()) {
    throw std::out_of_range("primary layer index is outside array index");
  }
  return static_cast<size_t>(search_graph_view_.layer_end[layer] -
                             search_graph_view_.layer_begin[layer]);
}

bool BioGeometryIndexBuilder::validate_integer_ids() const {
  const auto& view = search_graph_view_;
  if (view.node_records.size() != world_node_count_ ||
      view.sequences.size() != sequence_count_ ||
      view.child_mbb_bits_by_layer.size() != view.layer_begin.size()) {
    return false;
  }
  if (view.sequences.reference_backed) {
    const std::string_view reference =
        view.sequences.reference_view();
    if (!view.sequences.records.empty() ||
        view.sequences.reference_sequence_count != sequence_count_ ||
        view.sequences.reference_position_blocks.size() !=
            (sequence_count_ + kReferencePositionBlockSize - 1) /
                kReferencePositionBlockSize) {
      return false;
    }
    uint32_t previous_contig_end = 0;
    for (const auto& contig : view.sequences.reference_contigs) {
      if (contig.id.empty() || contig.begin != previous_contig_end ||
          contig.end < contig.begin ||
          contig.end > reference.size()) {
        return false;
      }
      previous_contig_end = contig.end;
    }
    if ((!reference.empty() &&
         view.sequences.reference_contigs.empty()) ||
        previous_contig_end != reference.size()) {
      return false;
    }
    const auto is_valid_reference_window =
        [&](uint32_t source_pos) {
          if (source_pos >=
                  reference.size() ||
              view.sequences.fixed_sequence_length >
                  reference.size() - source_pos) {
            return false;
          }
          const auto contig = std::upper_bound(
              view.sequences.reference_contigs.begin(),
              view.sequences.reference_contigs.end(), source_pos,
              [](uint32_t position, const ReferenceContig& candidate) {
                return position < candidate.begin;
              });
          if (contig == view.sequences.reference_contigs.begin()) {
            return false;
          }
          const auto& containing_contig = *std::prev(contig);
          return source_pos >= containing_contig.begin &&
                 static_cast<size_t>(source_pos) +
                         view.sequences.fixed_sequence_length <=
                     containing_contig.end;
        };
    uint32_t previous_representative = 0;
    for (size_t sequence_idx = 0; sequence_idx < sequence_count_;
         ++sequence_idx) {
      const uint32_t representative = static_cast<uint32_t>(
          view.sequences.source_position(
              static_cast<LeafId>(sequence_idx)));
      if (!is_valid_reference_window(representative) ||
          (sequence_idx != 0 &&
           representative <= previous_representative)) {
        return false;
      }
      previous_representative = representative;
    }
    LeafId previous_sequence_id = 0;
    bool first_occurrence = true;
    for (const auto& occurrence :
         view.sequences.singleton_occurrences) {
      if (occurrence.sequence_id >= sequence_count_ ||
          occurrence.source_pos ==
              view.sequences.source_position(occurrence.sequence_id) ||
          !is_valid_reference_window(occurrence.source_pos) ||
          (!first_occurrence &&
           occurrence.sequence_id <= previous_sequence_id)) {
        return false;
      }
      previous_sequence_id = occurrence.sequence_id;
      first_occurrence = false;
    }
    size_t expected_position_begin = 0;
    previous_sequence_id = 0;
    first_occurrence = true;
    for (size_t group_idx = 0;
         group_idx < view.sequences.occurrence_groups.size(); ++group_idx) {
      const auto& group = view.sequences.occurrence_groups[group_idx];
      const size_t position_end =
          group_idx + 1 < view.sequences.occurrence_groups.size()
              ? view.sequences.occurrence_groups[group_idx + 1].position_begin
              : view.sequences.grouped_occurrence_positions.size();
      const auto singleton = std::lower_bound(
          view.sequences.singleton_occurrences.begin(),
          view.sequences.singleton_occurrences.end(), group.sequence_id,
          [](const ReferenceOccurrence& occurrence, LeafId sequence_id) {
            return occurrence.sequence_id < sequence_id;
          });
      if (group.sequence_id >= sequence_count_ ||
          group.position_begin != expected_position_begin ||
          position_end < group.position_begin ||
          position_end >
              view.sequences.grouped_occurrence_positions.size() ||
          position_end - group.position_begin < 2 ||
          (!first_occurrence &&
           group.sequence_id <= previous_sequence_id) ||
          (singleton != view.sequences.singleton_occurrences.end() &&
           singleton->sequence_id == group.sequence_id)) {
        return false;
      }
      uint32_t previous_position = 0;
      bool first_position = true;
      for (size_t position_idx = group.position_begin;
           position_idx < position_end; ++position_idx) {
        const uint32_t position =
            view.sequences.grouped_occurrence_positions[position_idx];
        if (position ==
                view.sequences.source_position(group.sequence_id) ||
            !is_valid_reference_window(position) ||
            (!first_position && position <= previous_position)) {
          return false;
        }
        previous_position = position;
        first_position = false;
      }
      expected_position_begin = position_end;
      previous_sequence_id = group.sequence_id;
      first_occurrence = false;
    }
    if (expected_position_begin !=
        view.sequences.grouped_occurrence_positions.size()) {
      return false;
    }
  } else {
    if (view.sequences.reference_sequence_count != 0 ||
        !view.sequences.reference_position_blocks.empty() ||
        !view.sequences.reference_position_payload.empty() ||
        !view.sequences.singleton_occurrences.empty() ||
        !view.sequences.occurrence_groups.empty() ||
        !view.sequences.grouped_occurrence_positions.empty() ||
        !view.sequences.mapped_reference_sequence.empty() ||
        !view.sequences.reference_contigs.empty()) {
      return false;
    }
    for (size_t sequence_id = 0; sequence_id < view.sequences.size();
         ++sequence_id) {
      if (view.sequences.records[sequence_id].sequence_id != sequence_id) {
        return false;
      }
    }
  }
  const NodeId finest_begin =
      view.layer_begin.empty()
          ? static_cast<NodeId>(view.node_records.size())
          : view.layer_begin.back();
  for (NodeId node_id = 0; node_id < view.node_records.size(); ++node_id) {
    const auto& node = view.node_records[node_id];
    if (node.center_sequence_id >= sequence_count_ ||
        (node.counts_overflow() &&
         node.inline_link_count_or_overflow_index() >=
             view.node_count_overflows.size())) {
      return false;
    }
    const uint32_t link_count = view.link_count(node_id);
    if (node_id < finest_begin) {
      if (node.link_storage() !=
              WorldNodeRecord::LinkStorage::Absolute32 &&
          node.link_storage() !=
              WorldNodeRecord::LinkStorage::Delta16) {
        return false;
      }
      const size_t child_array_size =
          view.child_ids_are_delta16(node_id)
              ? view.child_id_deltas16.size()
              : view.child_ids.size();
      if (static_cast<size_t>(node.child_begin()) +
              link_count >
          child_array_size) {
        return false;
      }
      for (uint32_t offset = 0; offset < link_count; ++offset) {
        if (view.child_id(node_id, offset) >= world_node_count_) {
          return false;
        }
      }
    } else {
      size_t leaf_array_size = 0;
      switch (node.link_storage()) {
        case WorldNodeRecord::LinkStorage::Delta8:
          leaf_array_size = view.leaf_id_deltas8.size();
          break;
        case WorldNodeRecord::LinkStorage::Delta16:
          leaf_array_size = view.leaf_id_deltas16.size();
          break;
        case WorldNodeRecord::LinkStorage::Absolute32:
          leaf_array_size = view.leaf_ids.size();
          break;
        default:
          return false;
      }
      if (static_cast<size_t>(node.leaf_begin()) +
              link_count >
          leaf_array_size) {
        return false;
      }
      for (uint32_t offset = 0; offset < link_count; ++offset) {
        if (view.leaf_id(node_id, offset) >= sequence_count_) {
          return false;
        }
      }
    }
  }
  return true;
}

bool BioGeometryIndexBuilder::validate_search_graph_view() const {
  const auto& view = search_graph_view_;
  if (view.node_records.size() != world_node_count_ ||
      view.sequences.size() != sequence_count_ ||
      view.layer_begin.size() != static_cast<size_t>(num_primary_layers()) ||
      view.layer_end.size() != static_cast<size_t>(num_primary_layers()) ||
      view.child_mbb_bits_by_layer.size() !=
          static_cast<size_t>(num_primary_layers())) {
    return false;
  }
  uint32_t expected_layer_begin = 0;
  for (size_t layer = 0; layer < view.layer_begin.size(); ++layer) {
    if (view.layer_begin[layer] != expected_layer_begin ||
        view.layer_end[layer] < view.layer_begin[layer] ||
        view.layer_end[layer] > view.node_records.size()) {
      return false;
    }
    for (uint32_t node_id = view.layer_begin[layer];
         node_id < view.layer_end[layer]; ++node_id) {
      const auto& node = view.node_records[node_id];
      if (node.counts_overflow() &&
          node.inline_link_count_or_overflow_index() >=
              view.node_count_overflows.size()) {
        return false;
      }
      const uint32_t link_count = view.link_count(node_id);
      const uint32_t beacon_count = view.beacon_count(node_id);
      if (layer + 1 == view.layer_begin.size()) {
        size_t leaf_array_size = 0;
        switch (node.link_storage()) {
          case WorldNodeRecord::LinkStorage::Delta8:
            leaf_array_size = view.leaf_id_deltas8.size();
            break;
          case WorldNodeRecord::LinkStorage::Delta16:
            leaf_array_size = view.leaf_id_deltas16.size();
            break;
          case WorldNodeRecord::LinkStorage::Absolute32:
            leaf_array_size = view.leaf_ids.size();
            break;
          default:
            return false;
        }
        if (static_cast<size_t>(node.leaf_begin()) +
                    link_count >
                leaf_array_size ||
            static_cast<size_t>(node.mbb_begin) + link_count >
                view.leaf_beacon_dists.size() ||
            beacon_count != 1 ||
            node.beacon_storage() !=
                WorldNodeRecord::BeaconStorage::ImplicitCenter) {
          return false;
        }
      } else {
        if ((node.link_storage() !=
                 WorldNodeRecord::LinkStorage::Absolute32 &&
             node.link_storage() !=
                 WorldNodeRecord::LinkStorage::Delta16) ||
            static_cast<size_t>(node.child_begin()) +
                    link_count >
                (view.child_ids_are_delta16(node_id)
                     ? view.child_id_deltas16.size()
                     : view.child_ids.size()) ||
            beacon_count > link_count ||
            !view.child_mbb_range_valid(node_id) ||
            node.beacon_storage() ==
                WorldNodeRecord::BeaconStorage::ImplicitCenter) {
          return false;
        }
        for (uint32_t child_offset = 0;
             child_offset < link_count; ++child_offset) {
          const NodeId child_id =
              view.child_id(node_id, child_offset);
          if (child_id < view.layer_begin[layer + 1] ||
              child_id >= view.layer_end[layer + 1]) {
            return false;
          }
        }
      }
    }
    expected_layer_begin = view.layer_end[layer];
  }
  if (expected_layer_begin != world_node_count_) return false;

  size_t expected_child_delta16_begin = 0;
  size_t expected_child_id32_begin = 0;
  size_t expected_leaf_delta8_begin = 0;
  size_t expected_leaf_delta16_begin = 0;
  size_t expected_leaf_id32_begin = 0;
  size_t expected_leaf_beacon_begin = 0;
  size_t expected_mbb_begin = 0;
  size_t expected_beacon_delta8_begin = 0;
  size_t expected_beacon_delta16_begin = 0;
  size_t expected_beacon_id32_begin = 0;
  size_t expected_count_overflow = 0;
  if (view.layer_begin.empty() ||
      view.child_mbb_bits_by_layer.size() != view.layer_begin.size() ||
      view.beacon_begins.size() != view.layer_begin.back()) {
    return false;
  }
  for (NodeId node_id = 0; node_id < view.node_records.size();
       ++node_id) {
    const auto& node = view.node_records[node_id];
    const bool is_finest = node_id >= view.layer_begin.back();
    if (node.counts_overflow()) {
      if (node.inline_link_count_or_overflow_index() !=
              expected_count_overflow ||
          expected_count_overflow >=
              view.node_count_overflows.size()) {
        return false;
      }
      ++expected_count_overflow;
    }
    const uint32_t link_count = view.link_count(node_id);
    const uint32_t beacon_count = view.beacon_count(node_id);
    if (node.center_sequence_id >= sequence_count_) {
      return false;
    }
    if (is_finest) {
      size_t* expected_leaf_begin = nullptr;
      switch (node.link_storage()) {
        case WorldNodeRecord::LinkStorage::Delta8:
          expected_leaf_begin = &expected_leaf_delta8_begin;
          break;
        case WorldNodeRecord::LinkStorage::Delta16:
          expected_leaf_begin = &expected_leaf_delta16_begin;
          break;
        case WorldNodeRecord::LinkStorage::Absolute32:
          expected_leaf_begin = &expected_leaf_id32_begin;
          break;
        default:
          return false;
      }
      if (!expected_leaf_begin ||
          node.leaf_begin() != *expected_leaf_begin ||
          node.mbb_begin != expected_leaf_beacon_begin) {
        return false;
      }
      *expected_leaf_begin += link_count;
      expected_leaf_beacon_begin += link_count;
    } else {
      if ((node.link_storage() !=
               WorldNodeRecord::LinkStorage::Absolute32 &&
           node.link_storage() !=
               WorldNodeRecord::LinkStorage::Delta16) ||
          node.mbb_begin != expected_mbb_begin ||
          !view.child_mbb_range_valid(node_id)) {
        return false;
      }
      size_t& expected_child_begin =
          view.child_ids_are_delta16(node_id)
              ? expected_child_delta16_begin
              : expected_child_id32_begin;
      if (node.child_begin() != expected_child_begin) return false;
      expected_child_begin += link_count;
      expected_mbb_begin += view.child_mbb_byte_count(node_id);
    }
    const uint32_t beacon_begin =
        is_finest ? 0 : view.beacon_begins[node_id];
    switch (node.beacon_storage()) {
      case WorldNodeRecord::BeaconStorage::Delta8:
        if (beacon_begin != expected_beacon_delta8_begin) return false;
        expected_beacon_delta8_begin += beacon_count;
        break;
      case WorldNodeRecord::BeaconStorage::Delta16:
        if (beacon_begin != expected_beacon_delta16_begin) return false;
        expected_beacon_delta16_begin += beacon_count;
        break;
      case WorldNodeRecord::BeaconStorage::Absolute32:
        if (beacon_begin != expected_beacon_id32_begin) return false;
        expected_beacon_id32_begin += beacon_count;
        break;
      case WorldNodeRecord::BeaconStorage::ImplicitCenter:
        if (!is_finest || beacon_count != 1) return false;
        break;
    }
    if (expected_child_delta16_begin >
            view.child_id_deltas16.size() ||
        expected_child_id32_begin > view.child_ids.size() ||
        expected_leaf_delta8_begin > view.leaf_id_deltas8.size() ||
        expected_leaf_delta16_begin > view.leaf_id_deltas16.size() ||
        expected_leaf_id32_begin > view.leaf_ids.size() ||
        expected_mbb_begin > view.child_beacon_dists.size() ||
        expected_leaf_beacon_begin > view.leaf_beacon_dists.size() ||
        expected_beacon_delta8_begin > view.beacon_deltas8.size() ||
        expected_beacon_delta16_begin > view.beacon_deltas16.size() ||
        expected_beacon_id32_begin > view.beacon_ids32.size()) {
      return false;
    }
    for (uint32_t beacon_offset = 0;
         beacon_offset < beacon_count; ++beacon_offset) {
      if (view.beacon_sequence_id(node_id, beacon_offset) >=
          sequence_count_) {
        return false;
      }
    }
  }
  if (expected_child_delta16_begin !=
          view.child_id_deltas16.size() ||
      expected_child_id32_begin != view.child_ids.size() ||
      expected_leaf_delta8_begin != view.leaf_id_deltas8.size() ||
      expected_leaf_delta16_begin != view.leaf_id_deltas16.size() ||
      expected_leaf_id32_begin != view.leaf_ids.size() ||
      expected_count_overflow != view.node_count_overflows.size() ||
      expected_mbb_begin != view.child_beacon_dists.size() ||
      expected_leaf_beacon_begin != view.leaf_beacon_dists.size() ||
      expected_beacon_delta8_begin != view.beacon_deltas8.size() ||
      expected_beacon_delta16_begin != view.beacon_deltas16.size() ||
      expected_beacon_id32_begin != view.beacon_ids32.size()) {
    return false;
  }
  return validate_integer_ids();
}

std::vector<std::shared_ptr<BioSequence>> BioGeometryIndexBuilder::deduplicate(
    std::vector<std::shared_ptr<BioSequence>> raw) {
  if (!range_config_.phase1_preserve_input_order) {
    std::unordered_map<std::string_view, std::shared_ptr<BioSequence>>
        sequence_map;
    sequence_map.reserve(raw.size());
    for (const auto& sequence : raw) {
      if (!sequence) {
        throw std::invalid_argument("null BioSequence in input");
      }
      stats_.added_sequences++;
      auto it = sequence_map.find(sequence->seq);
      if (it == sequence_map.end()) {
        sequence_map.emplace(sequence->seq, sequence);
        continue;
      }
      auto& existing = it->second;
      for (const auto& occurrence : sequence->ref_positions) {
        existing->add_occurrence(occurrence.ref_id, occurrence.start,
                                 occurrence.end, occurrence.strand);
      }
      if (sequence->ref_positions.empty() && existing->ref_positions.empty()) {
        existing->add_occurrence(sequence->id, 0,
                                 static_cast<int>(sequence->seq.size()), "+");
      }
      if (!existing->bwt_interval.valid() && sequence->bwt_interval.valid()) {
        existing->set_bwt_interval(sequence->bwt_interval.start,
                                   sequence->bwt_interval.end);
      }
      stats_.deduplicated++;
    }

    std::vector<std::shared_ptr<BioSequence>> unordered;
    unordered.reserve(sequence_map.size());
    for (const auto& entry : sequence_map) {
      unordered.push_back(entry.second);
    }
    stats_.unique_sequences = unordered.size();
    return unordered;
  }

  std::unordered_map<std::string_view, size_t> sequence_indices;
  sequence_indices.reserve(raw.size());
  std::vector<std::shared_ptr<BioSequence>> out;
  out.reserve(raw.size());
  for (const auto& seq : raw) {
    if (!seq) {
      throw std::invalid_argument("null BioSequence in input");
    }
    stats_.added_sequences++;
    auto it = sequence_indices.find(seq->seq);
    if (it != sequence_indices.end()) {
      auto& existing = out[it->second];
      for (const auto& occ : seq->ref_positions) {
        existing->add_occurrence(occ.ref_id, occ.start, occ.end, occ.strand);
      }
      if (seq->ref_positions.empty() && existing->ref_positions.empty()) {
        existing->add_occurrence(
            seq->id, 0, static_cast<int>(seq->seq.size()), "+");
      }
      if (!existing->bwt_interval.valid() && seq->bwt_interval.valid()) {
        existing->set_bwt_interval(
            seq->bwt_interval.start, seq->bwt_interval.end);
      }
      stats_.deduplicated++;
    } else {
      sequence_indices.emplace(seq->seq, out.size());
      out.push_back(seq);
    }
  }

  stats_.unique_sequences = out.size();
  return out;
}

void BioGeometryIndexBuilder::initialize_sequence_store(
    const std::vector<std::shared_ptr<BioSequence>>& unique_seqs,
    bool consume_records) {
  sequence_count_ = unique_seqs.size();
  auto& store = search_graph_view_.sequences;
  store.reference_backed = false;
  store.reference_sequence_count = 0;
  store.reference_position_blocks.clear();
  store.reference_position_payload.clear();
  store.singleton_occurrences.clear();
  store.occurrence_groups.clear();
  store.grouped_occurrence_positions.clear();
  store.reference_contigs.clear();
  store.reference_id.clear();
  store.reference_sequence.clear();
  store.mapped_reference_sequence.clear();
  store.fixed_sequence_length = 0;
  store.records.resize(sequence_count_);
  for (size_t sequence_idx = 0; sequence_idx < unique_seqs.size();
       ++sequence_idx) {
    if (!unique_seqs[sequence_idx]) {
      throw std::invalid_argument("null BioSequence in deduplicated input");
    }
    if (unique_seqs[sequence_idx]->seq.size() >
        kMaxCompactSequenceLength) {
      throw std::invalid_argument(
          "indexed sequence length exceeds compact 8-bit distance range");
    }
    if (sequence_idx > static_cast<size_t>(INVALID_LEAF_ID - 1)) {
      throw std::runtime_error("too many sequences for 32-bit LeafId");
    }
    const LeafId sequence_id = static_cast<LeafId>(sequence_idx);
    unique_seqs[sequence_idx]->sequence_id = sequence_id;
    if (consume_records) {
      store.records[sequence_id] = std::move(*unique_seqs[sequence_idx]);
      store.records[sequence_id].sequence_id = sequence_id;
    } else {
      store.records[sequence_id] = *unique_seqs[sequence_idx];
    }
  }
}

void BioGeometryIndexBuilder::initialize_reference_sequence_store(
    std::string reference_id,
    std::string reference_sequence,
    size_t window_length,
    size_t stride,
    std::vector<ReferenceContig> reference_contigs,
    BuildProgressReporter* progress) {
  if (window_length == 0) {
    throw std::invalid_argument("reference window length must be positive");
  }
  if (window_length > kMaxCompactSequenceLength) {
    throw std::invalid_argument(
        "reference window length exceeds compact 8-bit distance range");
  }
  if (stride == 0) {
    throw std::invalid_argument("reference window stride must be positive");
  }

  auto& store = search_graph_view_.sequences;
  store.reference_id = std::move(reference_id);
  store.reference_sequence = std::move(reference_sequence);
  store.mapped_reference_sequence.clear();
  store.fixed_sequence_length = window_length;
  store.reference_backed = true;
  store.records.clear();
  store.reference_sequence_count = 0;
  store.reference_position_blocks.clear();
  store.reference_position_payload.clear();
  store.singleton_occurrences.clear();
  store.occurrence_groups.clear();
  store.grouped_occurrence_positions.clear();

  if (store.reference_sequence.size() >=
      static_cast<size_t>(UINT32_MAX)) {
    throw std::runtime_error(
        "reference-backed index requires reference coordinates below 2^32");
  }

  if (reference_contigs.empty() && !store.reference_sequence.empty()) {
    reference_contigs.push_back(
        {store.reference_id, 0,
         static_cast<uint32_t>(store.reference_sequence.size())});
  }
  uint32_t expected_begin = 0;
  size_t window_count = 0;
  for (const auto& contig : reference_contigs) {
    if (contig.id.empty() || contig.begin != expected_begin ||
        contig.end < contig.begin ||
        contig.end > store.reference_sequence.size()) {
      throw std::invalid_argument(
          "reference contigs must be ordered, contiguous, and in bounds");
    }
    const size_t contig_length =
        static_cast<size_t>(contig.end - contig.begin);
    if (contig.source_begin >
        std::numeric_limits<uint32_t>::max() - contig_length) {
      throw std::invalid_argument(
          "reference contig source coordinates exceed 32-bit storage");
    }
    if (contig_length >= window_length) {
      window_count +=
          1 + (contig_length - window_length) / stride;
    }
    expected_begin = contig.end;
  }
  if (expected_begin != store.reference_sequence.size()) {
    throw std::invalid_argument(
        "reference contigs do not cover the flattened reference");
  }
  store.reference_contigs = std::move(reference_contigs);
  if (window_count > static_cast<size_t>(INVALID_LEAF_ID - 1)) {
    throw std::runtime_error(
        "too many reference windows for 32-bit LeafId");
  }

  size_t valid_window_count = 0;
  for (const auto& contig : store.reference_contigs) {
    size_t run_begin = contig.begin;
    while (run_begin < contig.end) {
      while (run_begin < contig.end &&
             !is_acgt(store.reference_sequence[run_begin])) {
        ++run_begin;
      }
      size_t run_end = run_begin;
      while (run_end < contig.end &&
             is_acgt(store.reference_sequence[run_end])) {
        ++run_end;
      }
      if (run_end - run_begin >= window_length) {
        const size_t offset = run_begin - contig.begin;
        size_t first_start = run_begin;
        const size_t remainder = offset % stride;
        if (remainder != 0) {
          const size_t adjustment = stride - remainder;
          if (adjustment > run_end - run_begin) {
            run_begin = run_end;
            continue;
          }
          first_start += adjustment;
        }
        const size_t last_start = run_end - window_length;
        if (first_start <= last_start) {
          valid_window_count +=
              1 + (last_start - first_start) / stride;
        }
      }
      run_begin = run_end;
    }
  }

  ReferencePositionEncoder position_encoder;
  std::vector<ReferenceOccurrence> additional_occurrences;
  {
    ReferenceSequenceTable sequence_ids(valid_window_count);
    const std::string_view reference(store.reference_sequence);
    const auto position_of = [&](LeafId id) {
      return position_encoder.position(id);
    };
    size_t processed_windows = 0;
    for (const auto& contig : store.reference_contigs) {
      const size_t contig_begin = contig.begin;
      const size_t contig_end = contig.end;
      if (contig_end - contig_begin < window_length) continue;
      size_t next_invalid = contig_begin;
      for (size_t start = contig_begin;
           start + window_length <= contig_end;
           start += stride) {
        stats_.added_sequences++;
        processed_windows++;
        if (next_invalid < start) next_invalid = start;
        while (next_invalid < contig_end &&
               is_acgt(store.reference_sequence[next_invalid])) {
          next_invalid++;
        }
        if (next_invalid < start + window_length) {
          stats_.invalid_reference_windows++;
        } else {
          const std::string_view sequence(
              store.reference_sequence.data() + start, window_length);
          const uint32_t hash = reference_sequence_hash(sequence);
          const auto lookup = sequence_ids.find(
              hash, sequence, reference, position_of);
          if (lookup.id != INVALID_LEAF_ID) {
            stats_.deduplicated++;
            if (stride == 1) {
              additional_occurrences.push_back(
                  {lookup.id, static_cast<uint32_t>(start)});
            }
          } else {
            const LeafId sequence_id =
                position_encoder.append(static_cast<uint32_t>(start));
            sequence_ids.insert(lookup, hash, sequence_id);
          }
        }
        if (progress &&
            (processed_windows % 65536 == 0 ||
             processed_windows == window_count)) {
          progress->set_completed(processed_windows);
        }
      }
    }
    if (processed_windows != window_count ||
        stats_.invalid_reference_windows > processed_windows ||
        valid_window_count !=
            processed_windows - stats_.invalid_reference_windows) {
      throw std::runtime_error(
          "reference valid-window count is inconsistent");
    }
    // A sampled leaf represents its sequence at every valid reference
    // occurrence, including positions that are not themselves stride-selected.
    // For stride 1 the first pass already collected exactly this set.
    if (stride != 1) {
      for (const auto& contig : store.reference_contigs) {
        const size_t contig_begin = contig.begin;
        const size_t contig_end = contig.end;
        if (contig_end - contig_begin < window_length) continue;
        size_t next_invalid = contig_begin;
        for (size_t start = contig_begin;
             start + window_length <= contig_end; ++start) {
          if (next_invalid < start) next_invalid = start;
          while (next_invalid < contig_end &&
                 is_acgt(store.reference_sequence[next_invalid])) {
            next_invalid++;
          }
          if (next_invalid < start + window_length) continue;
          const std::string_view sequence(
              store.reference_sequence.data() + start, window_length);
          const uint32_t hash = reference_sequence_hash(sequence);
          const auto lookup = sequence_ids.find(
              hash, sequence, reference, position_of);
          if (lookup.id == INVALID_LEAF_ID) continue;
          if (position_encoder.position(lookup.id) != start) {
            additional_occurrences.push_back(
                {lookup.id, static_cast<uint32_t>(start)});
          }
        }
      }
    }
  }
  position_encoder.finish(store);
  std::sort(
      additional_occurrences.begin(),
      additional_occurrences.end(),
      [](const ReferenceOccurrence& left,
         const ReferenceOccurrence& right) {
        return std::tie(left.sequence_id, left.source_pos) <
               std::tie(right.sequence_id, right.source_pos);
      });
  for (size_t occurrence_begin = 0;
       occurrence_begin < additional_occurrences.size();) {
    size_t occurrence_end = occurrence_begin + 1;
    while (occurrence_end < additional_occurrences.size() &&
           additional_occurrences[occurrence_end].sequence_id ==
               additional_occurrences[occurrence_begin].sequence_id) {
      ++occurrence_end;
    }
    const size_t occurrence_count = occurrence_end - occurrence_begin;
    if (occurrence_count == 1) {
      store.singleton_occurrences.push_back(
          additional_occurrences[occurrence_begin]);
    } else {
      store.occurrence_groups.push_back(
          {additional_occurrences[occurrence_begin].sequence_id,
           static_cast<uint32_t>(
               store.grouped_occurrence_positions.size())});
      for (size_t occurrence_idx = occurrence_begin;
           occurrence_idx < occurrence_end; ++occurrence_idx) {
        store.grouped_occurrence_positions.push_back(
            additional_occurrences[occurrence_idx].source_pos);
      }
    }
    occurrence_begin = occurrence_end;
  }
  sequence_count_ = store.size();
  stats_.unique_sequences = sequence_count_;
}

void BioGeometryIndexBuilder::phase1_build_extended_sketch(
    BuildProgressReporter* progress) {
  const auto& sequences = search_graph_view_.sequences;
  if (progress) progress->begin_phase("phase1_sketch", sequences.size());
  extended_layers_.assign(static_cast<size_t>(hierarchy_.num_expanded_layers()),
                          std::vector<NodeId>());
  std::unordered_map<NodeId, Phase1CoverGroupIndex> cover_group_indexes;
  struct CoverHint {
    NodeId candidate_group = INVALID_NODE_ID;
    NodeId node = INVALID_NODE_ID;
    size_t candidate_idx = std::numeric_limits<size_t>::max();
  };
  std::vector<CoverHint> hints(
      static_cast<size_t>(hierarchy_.num_expanded_layers()));
  std::vector<Phase1BaseCountSignature> sequence_signatures(
      search_graph_view_.sequences.size());
  for (size_t sequence_id = 0;
       sequence_id < search_graph_view_.sequences.size(); ++sequence_id) {
    sequence_signatures[sequence_id] = phase1_base_count_signature(
        search_graph_view_.sequences.sequence(sequence_id));
  }
  const auto phase1_start = Clock::now();
  Phase1DistanceCache distance_cache(search_graph_view_.sequences.size());

  auto add_phase1_scan_stats = [&](const Phase1CoverScanResult& scan) {
    stats_.phase1_cover_candidate_scans += scan.candidate_scans;
    stats_.phase1_length_pruned_candidates += scan.length_pruned;
    stats_.phase1_lower_bound_pruned_candidates += scan.lower_bound_pruned;
    stats_.phase1_exact_distance_reused += scan.exact_distance_reused;
    stats_.phase1_exact_rejection_reused += scan.exact_rejection_reused;
    stats_.phase1_cross_layer_distance_reused +=
        scan.cross_layer_distance_reused;
    stats_.phase1_exact_distance_calls += scan.exact_distance_calls;
    stats_.phase1_candidate_pairs += scan.exact_distance_calls;
  };

  for (size_t sequence_idx = 0; sequence_idx < sequences.size();
       ++sequence_idx) {
    const LeafId sequence_id = static_cast<LeafId>(sequence_idx);
    const std::string_view sequence = sequences.sequence(sequence_id);
    distance_cache.begin_query();
    PreparedEdlibDnaPattern prepared_query;
    const PreparedEdlibDnaPattern* prepared_query_ptr = nullptr;
    if (range_config_.distance_mode == BuildDistanceMode::Edlib) {
      prepared_query = prepare_edlib_dna_pattern(sequence);
      prepared_query_ptr = &prepared_query;
    }
    const Phase1BaseCountSignature query_signature =
        sequence_signatures[sequence_id];
    NodeId parent = INVALID_NODE_ID;
    for (int layer_idx = 0; layer_idx < hierarchy_.num_expanded_layers(); ++layer_idx) {
      const int radius = expanded_radii_[static_cast<size_t>(layer_idx)];
      const std::vector<NodeId>* candidates = nullptr;
      NodeId candidate_group = INVALID_NODE_ID;
      if (layer_idx == 0) {
        candidates = &extended_layers_[0];
      } else if (parent != INVALID_NODE_ID) {
        candidates = &build_nodes_[parent].child_or_leaf_ids;
        candidate_group = parent;
      }

      Phase1CoverScanResult scan;
      Phase1CoverScanResult initial;
      size_t known_rejected_idx = std::numeric_limits<size_t>::max();
      if (candidates && !candidates->empty()) {
        stats_.phase1_total_possible_pairs += candidates->size();
        int candidate_radius = radius;
        if (range_config_.phase1_candidate_mode != Phase1CandidateMode::Scan) {
          const CoverHint& hint = hints[static_cast<size_t>(layer_idx)];
          if (hint.candidate_group == candidate_group &&
              hint.node != INVALID_NODE_ID &&
              hint.node < build_nodes_.size() &&
              hint.candidate_idx < candidates->size() &&
              (*candidates)[hint.candidate_idx] == hint.node &&
              build_nodes_[hint.node].center_sequence_id <
                  search_graph_view_.sequences.size() &&
              phase1_length_compatible(sequence.size(),
                                       search_graph_view_.sequences.sequence(
                                           build_nodes_[hint.node]
                                               .center_sequence_id)
                                           .size(),
                                       radius)) {
            stats_.phase1_hint_checks++;
            const LeafId hint_center_id =
                build_nodes_[hint.node].center_sequence_id;
            int hint_distance = 0;
            if (distance_cache.lookup(
                    hint_center_id, radius, &hint_distance)) {
              stats_.phase1_cross_layer_distance_reused++;
            } else {
              hint_distance = build_distance_bounded_prepared(
                  sequence,
                  search_graph_view_.sequences.sequence(hint_center_id),
                  radius,
                  range_config_.distance_mode, prepared_query_ptr);
              distance_cache.store(hint_center_id, radius, hint_distance);
            }
            if (hint_distance <= radius) {
              stats_.phase1_hint_hits++;
              candidate_radius = hint_distance;
              initial.best = hint.node;
              initial.best_dist = hint_distance;
              initial.best_idx = hint.candidate_idx;
            } else {
              known_rejected_idx = hint.candidate_idx;
            }
          }
        }
        auto use_scan = [&]() {
          stats_.phase1_scan_queries++;
          scan = find_best_phase1_cover(
              *candidates, build_nodes_, search_graph_view_.sequences,
              sequence_signatures, query_signature, &distance_cache,
              prepared_query_ptr,
              sequence, radius,
              range_config_.distance_mode, initial, known_rejected_idx);
          add_phase1_scan_stats(scan);
        };

        if (range_config_.phase1_candidate_mode == Phase1CandidateMode::Scan ||
            candidates->size() <
                range_config_.phase1_metric_min_fanout) {
          use_scan();
        } else {
          auto& group_index =
              cover_group_indexes[candidate_group];
          auto candidate_query = group_index.query(
              *candidates, build_nodes_, search_graph_view_.sequences,
              sequence, candidate_radius, range_config_);
          stats_.phase1_metric_build_distance_calls +=
              candidate_query.metric_build_distance_calls;
          stats_.phase1_pigeonhole_queries +=
              candidate_query.pigeonhole_queries;
          stats_.phase1_seed_posting_entries_visited +=
              candidate_query.seed_posting_entries_visited;
          stats_.phase1_pigeonhole_candidates +=
              candidate_query.pigeonhole_candidates;
          stats_.phase1_pigeonhole_fallbacks +=
              candidate_query.pigeonhole_fallbacks;
          switch (candidate_query.source) {
            case Phase1CoverSource::Scan:
              use_scan();
              break;
            case Phase1CoverSource::FallbackScan:
              stats_.phase1_fallback_scan_queries++;
              scan = find_best_phase1_cover(
                  *candidates, build_nodes_, search_graph_view_.sequences,
                  sequence_signatures, query_signature, &distance_cache,
                  prepared_query_ptr,
                  sequence, radius,
                  range_config_.distance_mode, initial, known_rejected_idx);
              add_phase1_scan_stats(scan);
              break;
            case Phase1CoverSource::Metric:
              stats_.phase1_metric_index_queries++;
              stats_.phase1_metric_distance_calls +=
                  candidate_query.metric_distance_calls;
              scan = find_best_phase1_cover_by_indices(
                  *candidates, build_nodes_, search_graph_view_.sequences,
                  sequence_signatures, query_signature, &distance_cache,
                  prepared_query_ptr,
                  candidate_query.candidate_indices, sequence, radius,
                  range_config_.distance_mode, initial, known_rejected_idx);
              add_phase1_scan_stats(scan);
              break;
            case Phase1CoverSource::Pigeonhole:
              scan = find_best_phase1_cover_by_indices(
                  *candidates, build_nodes_, search_graph_view_.sequences,
                  sequence_signatures, query_signature, &distance_cache,
                  prepared_query_ptr,
                  candidate_query.candidate_indices, sequence, radius,
                  range_config_.distance_mode, initial, known_rejected_idx);
              add_phase1_scan_stats(scan);
              break;
            case Phase1CoverSource::QGram:
              stats_.phase1_qgram_index_queries++;
              stats_.phase1_qgram_touched_candidates +=
                  candidate_query.qgram_touched_candidates;
              stats_.phase1_qgram_pruned_candidates +=
                  candidate_query.qgram_pruned_candidates;
              scan = find_best_phase1_cover_by_indices(
                  *candidates, build_nodes_, search_graph_view_.sequences,
                  sequence_signatures, query_signature, &distance_cache,
                  prepared_query_ptr,
                  candidate_query.candidate_indices, sequence, radius,
                  range_config_.distance_mode, initial, known_rejected_idx);
              add_phase1_scan_stats(scan);
              break;
          }
        }
      }

      if (scan.best == INVALID_NODE_ID) {
        const size_t new_candidate_idx = candidates ? candidates->size() : 0;
        if (build_nodes_.size() >
            static_cast<size_t>(INVALID_NODE_ID - 1)) {
          throw std::runtime_error("too many build nodes for 32-bit NodeId");
        }
        const NodeId new_node_id =
            static_cast<NodeId>(build_nodes_.size());
        BuildWorldNodeRecord new_node;
        new_node.center_sequence_id = sequence_id;
        const bool is_primary = expanded_layer_is_primary(layer_idx);
        const int primary_idx = is_primary ? expanded_to_primary_index(layer_idx) : -1;
        build_nodes_.push_back(std::move(new_node));
        extended_layers_[static_cast<size_t>(layer_idx)].push_back(
            new_node_id);
        if (parent != INVALID_NODE_ID) {
          build_nodes_[parent].child_or_leaf_ids.push_back(new_node_id);
        }
        if (is_primary) {
          stats_.created_primary_nodes[static_cast<size_t>(primary_idx)]++;
        } else {
          stats_.created_auxiliary_nodes++;
        }
        stats_.phase1_cover_misses++;
        parent = new_node_id;
        hints[static_cast<size_t>(layer_idx)] = {
            candidate_group, parent, new_candidate_idx};
      } else {
        stats_.phase1_best_cover_hits++;
        parent = scan.best;
        hints[static_cast<size_t>(layer_idx)] = {
            candidate_group, parent, scan.best_idx};
      }
      if (parent == INVALID_NODE_ID) break;
    }

    const size_t processed = sequence_idx + 1;
    if (progress &&
        (processed % 1024 == 0 || processed == sequences.size())) {
      progress->set_completed(processed);
    }
    if (sequences.size() >= 100000 && processed % 100000 == 0) {
      const double percent =
          100.0 * static_cast<double>(processed) /
          static_cast<double>(sequences.size());
      std::cerr << "    Phase1 progress: processed=" << processed << "/"
                << sequences.size() << " (" << std::fixed
                << std::setprecision(1) << percent << "%)"
                << " elapsed_s=" << std::setprecision(1)
                << elapsed_ms_since(phase1_start) / 1000.0
                << " misses=" << stats_.phase1_cover_misses
                << " hint_hits=" << stats_.phase1_hint_hits
                << " seed_queries=" << stats_.phase1_pigeonhole_queries
                << " seed_postings="
                << stats_.phase1_seed_posting_entries_visited << "\n"
                << std::defaultfloat << std::setprecision(6);
    }
  }
  if (progress) progress->finish_phase();
}

void BioGeometryIndexBuilder::phase2_inter_tier_rebinding(
    BuildProgressReporter* progress) {
  struct Phase2EdgeTuple {
    uint32_t parent_idx = 0;
    uint32_t child_idx = 0;
  };
  static_assert(sizeof(Phase2EdgeTuple) == 8,
                "phase2 edge tuple must remain 8 bytes");

  struct Phase2LocalStats {
    BioGeometryIndexBuilder::Statistics stats;
    double candidate_query_worker_ms = 0.0;
    double exact_verify_worker_ms = 0.0;
  };

  uint64_t phase_total = 0;
  for (int layer_idx = 0;
       layer_idx + 1 < hierarchy_.num_expanded_layers(); ++layer_idx) {
    const auto& parents = extended_layers_[static_cast<size_t>(layer_idx)];
    const auto& children =
        extended_layers_[static_cast<size_t>(layer_idx + 1)];
    phase_total += range_config_.link_mode == BuildRangeMode::Full
                       ? parents.size()
                       : children.size();
  }
  if (progress) progress->begin_phase("phase2_rebinding", phase_total);
  uint64_t phase_completed = 0;

  for (int layer_idx = 0; layer_idx + 1 < hierarchy_.num_expanded_layers(); ++layer_idx) {
    const auto layer_start = Clock::now();
    const size_t candidate_pairs_before = stats_.phase2_candidate_pairs;
    const size_t exact_calls_before = stats_.phase2_exact_distance_calls;
    const size_t qgram_pruned_before = stats_.phase2_qgram_pruned_by_l1;
    const size_t edges_before = stats_.phase2_edges_added;
    auto& parents = extended_layers_[static_cast<size_t>(layer_idx)];
    auto& children = extended_layers_[static_cast<size_t>(layer_idx + 1)];
    const int parent_radius =
        expanded_radii_[static_cast<size_t>(layer_idx)];
    const int child_radius =
        expanded_radii_[static_cast<size_t>(layer_idx + 1)];
    const int link_tolerance = parent_radius + child_radius;
    stats_.phase2_total_possible_pairs += parents.size() * children.size();
    for (NodeId parent_id : parents) {
      build_nodes_[parent_id].child_or_leaf_ids.clear();
    }

    std::vector<std::string_view> parent_sequences(parents.size());
    std::vector<std::string_view> child_sequences(children.size());
    for (size_t parent_idx = 0; parent_idx < parents.size(); ++parent_idx) {
      const auto& parent = build_nodes_[parents[parent_idx]];
      parent_sequences[parent_idx] =
          search_graph_view_.sequences.sequence(parent.center_sequence_id);
    }
    for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
      const auto& child = build_nodes_[children[child_idx]];
      child_sequences[child_idx] =
          search_graph_view_.sequences.sequence(child.center_sequence_id);
    }
    const DistanceMode verifier_distance_mode =
        to_distance_mode(range_config_.distance_mode);

    if (range_config_.link_mode == BuildRangeMode::Full) {
      auto verifier = make_phase2_distance_verifier(verifier_distance_mode);
      for (size_t parent_idx = 0; parent_idx < parents.size(); ++parent_idx) {
        auto& parent = build_nodes_[parents[parent_idx]];
        std::vector<Phase2DistancePair> verify_pairs;
        verify_pairs.reserve(children.size());
        for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
          stats_.phase2_candidate_pairs++;
          stats_.phase2_exact_distance_calls++;
          verify_pairs.push_back(
              {static_cast<uint32_t>(parent_idx),
               static_cast<uint32_t>(child_idx)});
        }
        if (verify_pairs.empty()) continue;
        Phase2DistanceBatchResult result;
        {
          ScopedTimer verify_timer(&stats_.phase2_exact_verify_ms);
          result = verifier->verify(parent_sequences, child_sequences,
                                    verify_pairs, link_tolerance);
        }
        stats_.phase2_distance_batches++;
        for (size_t accepted_idx : result.accepted_pair_indices) {
          const auto& pair = verify_pairs[accepted_idx];
          {
            ScopedTimer timer(&stats_.phase2_edge_insert_ms);
            parent.child_or_leaf_ids.push_back(children[pair.child_idx]);
          }
          stats_.phase2_edges_added++;
        }
        if (progress) progress->advance(1);
      }
      phase_completed += parents.size();
      if (progress) progress->set_completed(phase_completed);
      if (stats_.unique_sequences >= 100000) {
        std::cerr << "    Phase2 layer " << layer_idx << "->"
                  << layer_idx + 1 << ": parents=" << parents.size()
                  << " children=" << children.size()
                  << " candidates="
                  << stats_.phase2_candidate_pairs - candidate_pairs_before
                  << " exact="
                  << stats_.phase2_exact_distance_calls - exact_calls_before
                  << " qgram_pruned="
                  << stats_.phase2_qgram_pruned_by_l1 - qgram_pruned_before
                  << " edges=" << stats_.phase2_edges_added - edges_before
                  << " elapsed_s=" << std::fixed << std::setprecision(1)
                  << elapsed_ms_since(layer_start) / 1000.0 << "\n"
                  << std::defaultfloat << std::setprecision(6);
      }
      continue;
    }

    std::vector<Phase1BaseCountSignature> parent_base_counts(parents.size());
    std::vector<Phase1BaseCountSignature> child_base_counts(children.size());
    for (size_t parent_idx = 0; parent_idx < parents.size(); ++parent_idx) {
      parent_base_counts[parent_idx] =
          phase1_base_count_signature(parent_sequences[parent_idx]);
    }
    for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
      child_base_counts[child_idx] =
          phase1_base_count_signature(child_sequences[child_idx]);
    }

    ExactRangeJoinIndex parent_index(
        range_config_.range_join, true, false, true);
    std::vector<int> seed_lengths;
    seed_lengths.reserve(children.size());
    for (NodeId child_id : children) {
      const auto& child = build_nodes_[child_id];
      const auto& child_sequence =
          search_graph_view_.sequences.sequence(child.center_sequence_id);
      const int block_count = link_tolerance + 1;
      const int block_len = static_cast<int>(
          child_sequence.size() / static_cast<size_t>(block_count));
      seed_lengths.push_back(std::min(
          range_config_.range_join.max_seed_len, block_len));
    }
    std::sort(seed_lengths.begin(), seed_lengths.end());
    seed_lengths.erase(std::unique(seed_lengths.begin(), seed_lengths.end()),
                       seed_lengths.end());
    {
      ScopedTimer timer(&stats_.phase2_index_build_ms);
      if (search_graph_view_.sequences.fixed_sequence_length != 0) {
        std::vector<const char*> parent_sequence_data;
        parent_sequence_data.reserve(parent_sequences.size());
        for (std::string_view sequence : parent_sequences) {
          parent_sequence_data.push_back(sequence.data());
        }
        parent_index.build_uniform_identity_views(
            std::move(parent_sequence_data),
            search_graph_view_.sequences.fixed_sequence_length);
      } else {
        std::vector<RangeJoinItemView> items;
        items.reserve(parent_sequences.size());
        for (size_t parent_idx = 0;
             parent_idx < parent_sequences.size(); ++parent_idx) {
          items.push_back({parent_idx, parent_sequences[parent_idx]});
        }
        parent_index.build_views(std::move(items));
      }
      parent_index.prepare_seed_lengths(seed_lengths);
    }

    std::vector<QGramSignature> parent_qgram_signatures;
    std::vector<QGramSignature> child_qgram_signatures;
    if (range_config_.phase2_qgram_postfilter) {
      const int q = range_config_.range_join.qgram_q;
      parent_qgram_signatures.resize(parents.size());
      child_qgram_signatures.resize(children.size());
      for (size_t parent_idx = 0; parent_idx < parents.size(); ++parent_idx) {
        const auto& parent = build_nodes_[parents[parent_idx]];
        parent_qgram_signatures[parent_idx] =
            compute_qgram_signature(
                search_graph_view_.sequences.sequence(
                    parent.center_sequence_id),
                q);
      }
      for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
        const auto& child = build_nodes_[children[child_idx]];
        child_qgram_signatures[child_idx] =
            compute_qgram_signature(
                search_graph_view_.sequences.sequence(
                    child.center_sequence_id),
                q);
      }
    }

    // Each child is one OpenMP task.  Do not create idle workers (or their
    // per-worker verification buffers) when a tier has less work than the
    // machine has hardware threads.
    const int thread_count = std::max(
        1, std::min(
               omp_get_max_threads(),
               static_cast<int>(std::min<size_t>(
                   children.size(),
                   static_cast<size_t>(std::numeric_limits<int>::max())))));
    std::vector<std::vector<Phase2EdgeTuple>> thread_edges(
        static_cast<size_t>(thread_count));
    std::vector<Phase2LocalStats> thread_stats(
        static_cast<size_t>(thread_count));
#pragma omp parallel if(thread_count > 1) num_threads(thread_count)
    {
      const int tid = omp_get_thread_num();
      const bool direct_edlib =
          verifier_distance_mode == DistanceMode::Edlib;
      RangeJoinQueryWorkspace workspace;
      std::unique_ptr<Phase2DistanceVerifier> verifier;
      if (!direct_edlib) {
        verifier = make_phase2_distance_verifier(verifier_distance_mode);
      }
      auto& local_edges = thread_edges[static_cast<size_t>(tid)];
      auto& local = thread_stats[static_cast<size_t>(tid)];
      std::vector<Phase2DistancePair> verify_batch;
      // The flush bound is useful for dense joins, but small tiers should only
      // reserve enough storage for the pairs they can possibly produce.
      size_t possible_pair_count = kPhase2DistanceBatchFlushPairs;
      if (parents.empty() || children.empty()) {
        possible_pair_count = 0;
      } else if (parents.size() <=
                 kPhase2DistanceBatchFlushPairs / children.size()) {
        possible_pair_count = parents.size() * children.size();
      }
      if (!direct_edlib) {
        verify_batch.reserve(
            std::min(kPhase2DistanceBatchFlushPairs, possible_pair_count));
      }
      auto flush_verify_batch = [&]() {
        if (verify_batch.empty()) return;
        const auto verify_start = Clock::now();
        const Phase2DistanceBatchResult result =
            verifier->verify(parent_sequences, child_sequences, verify_batch,
                             link_tolerance);
        local.exact_verify_worker_ms += elapsed_ms_since(verify_start);
        local.stats.phase2_distance_batches++;
        for (size_t accepted_idx : result.accepted_pair_indices) {
          const auto& pair = verify_batch[accepted_idx];
          local_edges.push_back({pair.parent_idx, pair.child_idx});
          local.stats.phase2_edges_added++;
        }
        verify_batch.clear();
      };

#pragma omp for schedule(dynamic, 8)
      for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
        auto& child = build_nodes_[children[child_idx]];
        const auto& child_sequence =
            search_graph_view_.sequences.sequence(child.center_sequence_id);

        const auto query_start = Clock::now();
        RangeJoinQueryResult candidates =
            parent_index.query(
                child_sequence, link_tolerance, &workspace);
        local.candidate_query_worker_ms += elapsed_ms_since(query_start);

        accumulate_phase2_candidate_stats(local.stats, candidates);

        size_t exact_candidate_count = 0;
        for (size_t parent_idx : candidates.candidate_item_ids) {
          auto& parent = build_nodes_[parents[parent_idx]];
          const auto& parent_sequence =
              search_graph_view_.sequences.sequence(
                  parent.center_sequence_id);
          if (std::llabs(
                  static_cast<long long>(parent_sequence.size()) -
                  static_cast<long long>(child_sequence.size())) >
              link_tolerance) {
            local.stats.phase2_length_pruned_pairs++;
            continue;
          }
          if (phase1_base_count_lower_bound(
                  child_base_counts[child_idx],
                  parent_base_counts[parent_idx]) > link_tolerance) {
            local.stats.phase2_base_count_pruned_pairs++;
            continue;
          }
          if (range_config_.phase2_qgram_postfilter &&
              qgram_can_prune_edit_distance(
                  child_qgram_signatures[child_idx],
                  parent_qgram_signatures[parent_idx],
                  link_tolerance)) {
            local.stats.phase2_qgram_pruned_by_l1++;
            continue;
          }
          local.stats.phase2_exact_distance_calls++;
          candidates.candidate_item_ids[exact_candidate_count++] =
              static_cast<RangeJoinItemId>(parent_idx);
        }
        candidates.candidate_item_ids.resize(exact_candidate_count);

        if (direct_edlib && !candidates.candidate_item_ids.empty()) {
          const auto verify_start = Clock::now();
          const PreparedEdlibDnaPattern prepared_child =
              prepare_edlib_dna_pattern(child_sequence);
          for (RangeJoinItemId parent_idx :
               candidates.candidate_item_ids) {
            if (compute_distance_bounded_edlib_prepared(
                    prepared_child, parent_sequences[parent_idx],
                    link_tolerance) <= link_tolerance) {
              local_edges.push_back(
                  {parent_idx, static_cast<uint32_t>(child_idx)});
              local.stats.phase2_edges_added++;
            }
          }
          local.exact_verify_worker_ms += elapsed_ms_since(verify_start);
          local.stats.phase2_distance_batches++;
        } else if (!direct_edlib) {
          for (RangeJoinItemId parent_idx :
               candidates.candidate_item_ids) {
            verify_batch.push_back(
                {parent_idx, static_cast<uint32_t>(child_idx)});
            if (verify_batch.size() >= kPhase2DistanceBatchFlushPairs) {
              flush_verify_batch();
            }
          }
        }
        if (progress && child_idx % 256 == 255) progress->advance(256);
      }
      flush_verify_batch();
    }

    phase_completed += children.size();
    if (progress) progress->set_completed(phase_completed);

    std::vector<Phase2EdgeTuple> edges;
    for (auto& local_edges : thread_edges) {
      edges.insert(edges.end(), local_edges.begin(), local_edges.end());
    }
    std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
      return std::tie(a.parent_idx, a.child_idx) <
             std::tie(b.parent_idx, b.child_idx);
    });
    edges.erase(std::unique(edges.begin(), edges.end(),
                            [](const auto& a, const auto& b) {
                              return a.parent_idx == b.parent_idx &&
                                     a.child_idx == b.child_idx;
                            }),
                edges.end());

    for (const auto& local : thread_stats) {
      merge_phase2_local_stats(stats_, local.stats);
      stats_.phase2_candidate_query_ms += local.candidate_query_worker_ms;
      stats_.phase2_exact_verify_ms += local.exact_verify_worker_ms;
      stats_.phase2_candidate_query_worker_ms +=
          local.candidate_query_worker_ms;
      stats_.phase2_exact_verify_worker_ms +=
          local.exact_verify_worker_ms;
    }

    {
      ScopedTimer timer(&stats_.phase2_edge_insert_ms);
      for (const auto& edge : edges) {
        build_nodes_[parents[edge.parent_idx]].child_or_leaf_ids.push_back(
            children[edge.child_idx]);
      }
    }

    if (stats_.unique_sequences >= 100000) {
      std::cerr << "    Phase2 layer " << layer_idx << "->"
                << layer_idx + 1 << ": parents=" << parents.size()
                << " children=" << children.size()
                << " query_tau=" << link_tolerance
                << " candidates="
                << stats_.phase2_candidate_pairs - candidate_pairs_before
                << " exact="
                << stats_.phase2_exact_distance_calls - exact_calls_before
                << " qgram_pruned="
                << stats_.phase2_qgram_pruned_by_l1 - qgram_pruned_before
                << " edges=" << stats_.phase2_edges_added - edges_before
                << " elapsed_s=" << std::fixed << std::setprecision(1)
                << elapsed_ms_since(layer_start) / 1000.0 << "\n"
                << std::defaultfloat << std::setprecision(6);
    }
  }
  if (progress) progress->finish_phase();
}

void BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb(
    BuildProgressReporter* progress) {
  uint64_t phase_total = 0;
  for (int primary_idx = 0;
       primary_idx + 1 < hierarchy_.num_primary_layers(); ++primary_idx) {
    phase_total += extended_layers_[static_cast<size_t>(primary_idx * 2)].size();
  }
  if (progress) progress->begin_phase("phase3_mbb", phase_total);
  uint64_t phase_completed = 0;
  primary_layers_.assign(static_cast<size_t>(hierarchy_.num_primary_layers()),
                         std::vector<NodeId>());

  size_t primary_node_count = 0;
  for (int primary_idx = 0;
       primary_idx < hierarchy_.num_primary_layers(); ++primary_idx) {
    primary_node_count +=
        extended_layers_[static_cast<size_t>(primary_idx * 2)].size();
  }
  build_node_geometry_.clear();
  build_node_geometry_.reserve(primary_node_count);

  for (int primary_idx = 0; primary_idx < hierarchy_.num_primary_layers(); ++primary_idx) {
    auto& target_layer = primary_layers_[static_cast<size_t>(primary_idx)];
    target_layer = extended_layers_[static_cast<size_t>(primary_idx * 2)];
    for (NodeId node_id : target_layer) {
      auto& node = build_nodes_[node_id];
      if (node.geometry_index != INVALID_NODE_ID) {
        throw std::runtime_error(
            "primary build node has duplicate geometry");
      }
      if (build_node_geometry_.size() >= INVALID_NODE_ID) {
        throw std::runtime_error(
            "too many primary build nodes for 32-bit geometry index");
      }
      node.geometry_index = static_cast<uint32_t>(
          build_node_geometry_.size());
      build_node_geometry_.emplace_back();
    }
  }

  for (int primary_idx = 0; primary_idx < hierarchy_.num_primary_layers();
       ++primary_idx) {
    auto& current_layer = primary_layers_[static_cast<size_t>(primary_idx)];
    const bool is_finest = (primary_idx == finest_primary_layer_index());
    if (is_finest) {
      for (NodeId node_id : current_layer) {
        build_nodes_[node_id].child_or_leaf_ids.clear();
      }
      continue;
    }

    const int thread_capacity = std::max(
        1, std::min(omp_get_max_threads(),
                    static_cast<int>(std::min<size_t>(
                        current_layer.size(),
                        static_cast<size_t>(std::numeric_limits<int>::max())))));
    std::vector<double> collect_ms(static_cast<size_t>(thread_capacity), 0.0);
    std::vector<double> collapse_ms(static_cast<size_t>(thread_capacity), 0.0);
    std::vector<double> distance_ms(static_cast<size_t>(thread_capacity), 0.0);
    int actual_threads = 1;

#pragma omp parallel if(thread_capacity > 1) num_threads(thread_capacity)
    {
      const int tid = omp_get_thread_num();
      std::vector<uint8_t> child_seen(build_nodes_.size(), uint8_t{0});

#pragma omp single
      actual_threads = omp_get_num_threads();

#pragma omp for schedule(dynamic, 1)
      for (size_t node_idx = 0; node_idx < current_layer.size(); ++node_idx) {
        auto& node = build_nodes_[current_layer[node_idx]];
        auto& geometry = build_node_geometry_[node.geometry_index];
        std::vector<NodeId> auxiliary_nodes = node.child_or_leaf_ids;
        {
          ScopedTimer timer(&collect_ms[static_cast<size_t>(tid)]);
          const size_t beacon_count = std::min(
              auxiliary_nodes.size(), kMaxBuildBeaconsPerNode);
          geometry.beacon_ids.reserve(beacon_count);
          for (size_t beacon_idx = 0; beacon_idx < beacon_count;
               ++beacon_idx) {
            const size_t auxiliary_idx =
                beacon_count <= 1
                    ? 0
                    : beacon_idx * (auxiliary_nodes.size() - 1) /
                          (beacon_count - 1);
            geometry.beacon_ids.push_back(
                build_nodes_[auxiliary_nodes[auxiliary_idx]]
                    .center_sequence_id);
          }
        }

        std::vector<NodeId> direct_children;
        {
          ScopedTimer timer(&collapse_ms[static_cast<size_t>(tid)]);
          for (NodeId aux_id : auxiliary_nodes) {
            for (NodeId child : build_nodes_[aux_id].child_or_leaf_ids) {
              if (!child_seen[child]) {
                child_seen[child] = 1;
                direct_children.push_back(child);
              }
            }
          }
          for (NodeId child : direct_children) child_seen[child] = 0;
          node.child_or_leaf_ids = std::move(direct_children);
        }

        {
          ScopedTimer timer(&distance_ms[static_cast<size_t>(tid)]);
          const size_t child_count = node.child_or_leaf_ids.size();
          const size_t beacon_count = geometry.beacon_ids.size();
          const size_t mbb_cell_count = child_count * beacon_count;
          geometry.link_beacon_dists.assign(mbb_cell_count, 0);
          for (size_t child_idx = 0; child_idx < child_count; ++child_idx) {
            const auto& child =
                build_nodes_[node.child_or_leaf_ids[child_idx]];
            const auto& child_sequence =
                search_graph_view_.sequences.sequence(
                    child.center_sequence_id);
            PreparedEdlibDnaPattern prepared_child;
            const PreparedEdlibDnaPattern* prepared_child_ptr = nullptr;
            if (range_config_.distance_mode == BuildDistanceMode::Edlib) {
              prepared_child = prepare_edlib_dna_pattern(child_sequence);
              prepared_child_ptr = &prepared_child;
            }
            for (size_t dim = 0; dim < beacon_count; ++dim) {
              const LeafId beacon_id = geometry.beacon_ids[dim];
              const auto& beacon =
                  search_graph_view_.sequences.sequence(beacon_id);
              int dist = prepared_child_ptr
                             ? compute_distance_edlib_prepared(
                                   *prepared_child_ptr, beacon)
                             : build_distance(
                                   child_sequence, beacon,
                                   range_config_.distance_mode);
              const size_t flat = dim * child_count + child_idx;
              geometry.link_beacon_dists[flat] =
                  static_cast<uint8_t>(dist);
            }
          }
        }

        if (progress && node_idx % 64 == 63) progress->advance(64);
      }
    }

    phase_completed += current_layer.size();
    if (progress) progress->set_completed(phase_completed);

    stats_.phase3_parallel_threads = std::max(
        stats_.phase3_parallel_threads, static_cast<size_t>(actual_threads));
    for (int tid = 0; tid < actual_threads; ++tid) {
      const size_t idx = static_cast<size_t>(tid);
      stats_.phase3_collect_beacons_ms += collect_ms[idx];
      stats_.phase3_collapse_children_ms += collapse_ms[idx];
      stats_.phase3_child_mbb_distance_ms += distance_ms[idx];
    }
  }
  if (progress) progress->finish_phase();
}

void BioGeometryIndexBuilder::attach_leaves(
    BuildProgressReporter* progress) {
  const auto& sequences = search_graph_view_.sequences;
  auto& finest_layer = primary_layers_[static_cast<size_t>(finest_primary_layer_index())];
  const int finest_radius = hierarchy_.primary_radii.back();
  stats_.total_possible_leaf_pairs = finest_layer.size() * sequences.size();
  for (NodeId node_id : finest_layer) {
    auto& node = build_nodes_[node_id];
    auto& geometry = build_node_geometry_[node.geometry_index];
    node.child_or_leaf_ids.clear();
    geometry.beacon_ids.clear();
    geometry.link_beacon_dists.clear();
  }

  LeafAttachDirection actual_direction = range_config_.leaf_attach_direction;
  if (range_config_.leaf_attach_mode == BuildRangeMode::Full) {
    actual_direction = LeafAttachDirection::WorldToSeq;
  } else if (actual_direction == LeafAttachDirection::Auto) {
    actual_direction =
        finest_layer.size() < sequences.size()
            ? LeafAttachDirection::WorldToSeq
            : LeafAttachDirection::SeqToWorld;
  }
  stats_.leaf_attach_direction_used = actual_direction;
  const uint64_t progress_total =
      actual_direction == LeafAttachDirection::SeqToWorld
          ? sequences.size()
          : finest_layer.size();
  if (progress) progress->begin_phase("phase4_attach", progress_total);

  if (range_config_.leaf_attach_mode == BuildRangeMode::Full) {
    const int thread_count = std::max(1, omp_get_max_threads());
    std::vector<double> thread_exact_ms(static_cast<size_t>(thread_count), 0.0);
    std::vector<double> thread_tuple_ms(static_cast<size_t>(thread_count), 0.0);
    std::vector<size_t> thread_exact_calls(
        static_cast<size_t>(thread_count), 0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t layer_idx = 0; layer_idx < finest_layer.size(); ++layer_idx) {
      const int tid = omp_get_thread_num();
      const size_t timer_idx =
          static_cast<size_t>(std::min(tid, thread_count - 1));
      auto& node = build_nodes_[finest_layer[layer_idx]];
      auto& geometry = build_node_geometry_[node.geometry_index];
      const std::string_view center =
          search_graph_view_.sequences.sequence(node.center_sequence_id);
      PreparedEdlibDnaPattern prepared_center;
      const PreparedEdlibDnaPattern* prepared_center_ptr = nullptr;
      if (range_config_.distance_mode == BuildDistanceMode::Edlib) {
        prepared_center = prepare_edlib_dna_pattern(center);
        prepared_center_ptr = &prepared_center;
      }
      {
        ScopedTimer verify_timer(&thread_exact_ms[timer_idx]);
        for (size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
          const LeafId sequence_id = static_cast<LeafId>(seq_idx);
          const std::string_view sequence = sequences.sequence(sequence_id);
          thread_exact_calls[timer_idx]++;
          int dist = build_distance_bounded_prepared(
              center, sequence, finest_radius,
              range_config_.distance_mode,
              prepared_center_ptr);
          if (dist <= finest_radius) {
            {
              ScopedTimer timer(&thread_tuple_ms[timer_idx]);
              node.child_or_leaf_ids.push_back(sequence_id);
              geometry.link_beacon_dists.push_back(
                  static_cast<uint8_t>(dist));
            }
          }
        }
      }
      if (progress && layer_idx % 256 == 255) progress->advance(256);
    }
    if (progress) progress->set_completed(finest_layer.size());
    for (double value : thread_exact_ms) stats_.leaf_exact_verify_ms += value;
    for (double value : thread_tuple_ms) stats_.leaf_tuple_emit_ms += value;
    for (size_t value : thread_exact_calls) {
      stats_.leaf_exact_distance_calls += value;
    }
    stats_.leaf_candidate_pairs = stats_.total_possible_leaf_pairs;
  } else {
    std::vector<Phase1BaseCountSignature> leaf_base_counts(
        sequences.size());
    for (size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
      leaf_base_counts[seq_idx] = phase1_base_count_signature(
          sequences.sequence(static_cast<LeafId>(seq_idx)));
    }
    std::vector<Phase1BaseCountSignature> world_base_counts(
        finest_layer.size());
    for (size_t world_idx = 0; world_idx < finest_layer.size();
         ++world_idx) {
      const auto& world = build_nodes_[finest_layer[world_idx]];
      world_base_counts[world_idx] = phase1_base_count_signature(
          search_graph_view_.sequences.sequence(world.center_sequence_id));
    }
    std::vector<Phase4QGramSignature> leaf_qgram_signatures;
    std::vector<Phase4QGramSignature> world_qgram_signatures;
    if (range_config_.leaf_qgram_postfilter) {
      const int q = range_config_.range_join.qgram_q;
      leaf_qgram_signatures.resize(sequences.size());
      for (size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
        leaf_qgram_signatures[seq_idx] = phase4_qgram_signature(
            sequences.sequence(static_cast<LeafId>(seq_idx)), q);
      }
      if (actual_direction == LeafAttachDirection::SeqToWorld) {
        world_qgram_signatures.resize(finest_layer.size());
        for (size_t world_idx = 0; world_idx < finest_layer.size();
             ++world_idx) {
          const auto& world = build_nodes_[finest_layer[world_idx]];
          world_qgram_signatures[world_idx] = phase4_qgram_signature(
              search_graph_view_.sequences.sequence(
                  world.center_sequence_id),
              q);
        }
      }
    }

    if (actual_direction == LeafAttachDirection::SeqToWorld) {
      std::vector<RangeJoinItemView> item_views;
      std::vector<const char*> uniform_sequence_data;
      if (sequences.fixed_sequence_length != 0) {
        uniform_sequence_data.reserve(finest_layer.size());
        for (NodeId node_id : finest_layer) {
          const auto& world = build_nodes_[node_id];
          uniform_sequence_data.push_back(
              sequences.sequence(world.center_sequence_id).data());
        }
      } else {
        item_views.reserve(finest_layer.size());
        for (size_t world_idx = 0;
             world_idx < finest_layer.size(); ++world_idx) {
          const NodeId node_id = finest_layer[world_idx];
          const auto& world = build_nodes_[node_id];
          item_views.push_back(
              {world_idx,
               sequences.sequence(world.center_sequence_id)});
        }
      }
      const int max_radius =
          finest_layer.empty() ? 0 : finest_radius;
      ExactRangeJoinIndex world_index(range_config_.range_join, true);
      {
        ScopedTimer timer(&stats_.leaf_index_build_ms);
        if (sequences.fixed_sequence_length != 0) {
          world_index.build_uniform_identity_views(
              std::move(uniform_sequence_data),
              sequences.fixed_sequence_length);
        } else {
          world_index.build_views(std::move(item_views));
        }
      }
      for (size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
        const LeafId sequence_id = static_cast<LeafId>(seq_idx);
        const std::string_view sequence = sequences.sequence(sequence_id);
        PreparedEdlibDnaPattern prepared_sequence;
        const PreparedEdlibDnaPattern* prepared_sequence_ptr = nullptr;
        if (range_config_.distance_mode == BuildDistanceMode::Edlib) {
          prepared_sequence = prepare_edlib_dna_pattern(sequence);
          prepared_sequence_ptr = &prepared_sequence;
        }
        RangeJoinQueryResult candidates;
        {
          ScopedTimer timer(&stats_.leaf_candidate_query_ms);
          candidates = world_index.query(sequence, max_radius);
        }
        accumulate_leaf_candidate_stats(stats_, candidates);
        {
          ScopedTimer verify_timer(&stats_.leaf_exact_verify_ms);
          for (size_t world_idx : candidates.candidate_item_ids) {
            auto& world = build_nodes_[finest_layer[world_idx]];
            auto& geometry =
                build_node_geometry_[world.geometry_index];
            const auto& center =
                search_graph_view_.sequences.sequence(
                    world.center_sequence_id);
            if (std::llabs(static_cast<long long>(sequence.size()) -
                           static_cast<long long>(center.size())) >
                finest_radius) {
              stats_.leaf_length_pruned_pairs++;
              continue;
            }
            if (phase1_base_count_lower_bound(
                    leaf_base_counts[seq_idx],
                    world_base_counts[world_idx]) > finest_radius) {
              stats_.leaf_base_count_pruned_pairs++;
              continue;
            }
            if (range_config_.leaf_qgram_postfilter &&
                phase4_qgram_can_prune_edit_distance(
                    leaf_qgram_signatures[seq_idx],
                    world_qgram_signatures[world_idx],
                    finest_radius)) {
              stats_.leaf_qgram_pruned_by_l1++;
              continue;
            }
            stats_.leaf_exact_distance_calls++;
            int dist = build_distance_bounded_prepared(
                sequence, center, finest_radius,
                range_config_.distance_mode, prepared_sequence_ptr);
            if (dist <= finest_radius) {
              {
                ScopedTimer timer(&stats_.leaf_tuple_emit_ms);
                world.child_or_leaf_ids.push_back(sequence_id);
                geometry.link_beacon_dists.push_back(
                    static_cast<uint8_t>(dist));
              }
            }
          }
        }
        if (progress && seq_idx % 256 == 255) progress->advance(256);
      }
      if (progress) progress->set_completed(sequences.size());
      {
        ScopedTimer timer(&stats_.leaf_tuple_merge_sort_ms);
      }
    } else {
      std::vector<RangeJoinItemView> item_views;
      std::vector<const char*> uniform_sequence_data;
      if (sequences.fixed_sequence_length != 0) {
        uniform_sequence_data.reserve(sequences.size());
        for (size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
          uniform_sequence_data.push_back(
              sequences.sequence(static_cast<LeafId>(seq_idx)).data());
        }
      } else {
        item_views.reserve(sequences.size());
        for (size_t seq_idx = 0; seq_idx < sequences.size(); ++seq_idx) {
          item_views.push_back(
              {seq_idx,
               sequences.sequence(static_cast<LeafId>(seq_idx))});
        }
      }
      ExactRangeJoinIndex sequence_index(
          range_config_.range_join, true, true, false);
      {
        ScopedTimer timer(&stats_.leaf_index_build_ms);
        if (sequences.fixed_sequence_length != 0) {
          sequence_index.build_uniform_identity_views(
              std::move(uniform_sequence_data),
              sequences.fixed_sequence_length);
        } else {
          sequence_index.build_views(std::move(item_views));
        }
        std::vector<int> seed_lengths;
        seed_lengths.reserve(finest_layer.size());
        for (NodeId node_id : finest_layer) {
          const auto& world = build_nodes_[node_id];
          const auto& center =
              search_graph_view_.sequences.sequence(
                  world.center_sequence_id);
          const int block_len = static_cast<int>(
              center.size() /
              static_cast<size_t>(finest_radius + 1));
          seed_lengths.push_back(std::min(
              range_config_.range_join.max_seed_len, block_len));
        }
        if (range_config_.range_join.candidate_mode ==
                RangeCandidateMode::Auto ||
            range_config_.range_join.candidate_mode ==
                RangeCandidateMode::PigeonholeOnly ||
            range_config_.range_join.candidate_mode ==
                RangeCandidateMode::Hybrid) {
          std::sort(seed_lengths.begin(), seed_lengths.end());
          seed_lengths.erase(
              std::unique(seed_lengths.begin(), seed_lengths.end()),
              seed_lengths.end());
          sequence_index.prepare_seed_lengths(seed_lengths);
        }
      }

      const int thread_count = std::max(
          1, std::min(
                 omp_get_max_threads(),
                 static_cast<int>(std::min<size_t>(
                     finest_layer.size(),
                     static_cast<size_t>(std::numeric_limits<int>::max())))));
      std::vector<BioGeometryIndexBuilder::Statistics> thread_stats(
          static_cast<size_t>(thread_count));

      // Every iteration is the sole writer for one world. Range-join results
      // follow the identity item IDs in ascending order here, so writing the
      // two final arrays directly preserves their canonical leaf order and
      // avoids a second, tuple-shaped copy of every attachment.
#pragma omp parallel if(thread_count > 1) num_threads(thread_count)
      {
        const int tid = omp_get_thread_num();
        auto& local_stats = thread_stats[static_cast<size_t>(tid)];
        RangeJoinQueryWorkspace workspace;

#pragma omp for schedule(dynamic, 8)
        for (size_t world_idx = 0; world_idx < finest_layer.size();
             ++world_idx) {
          auto& world = build_nodes_[finest_layer[world_idx]];
          auto& geometry =
              build_node_geometry_[world.geometry_index];
          const auto& center =
              search_graph_view_.sequences.sequence(
                  world.center_sequence_id);
          PreparedEdlibDnaPattern prepared_center;
          const PreparedEdlibDnaPattern* prepared_center_ptr = nullptr;
          if (range_config_.distance_mode == BuildDistanceMode::Edlib) {
            prepared_center = prepare_edlib_dna_pattern(center);
            prepared_center_ptr = &prepared_center;
          }
          Phase4QGramSignature world_qgram_signature;
          if (range_config_.leaf_qgram_postfilter) {
            world_qgram_signature = phase4_qgram_signature(
                center, range_config_.range_join.qgram_q);
          }
          RangeJoinQueryResult candidates;
          {
            ScopedTimer timer(&local_stats.leaf_candidate_query_ms);
            candidates =
                static_cast<const ExactRangeJoinIndex&>(sequence_index)
                    .query(center, finest_radius, &workspace);
          }
          accumulate_leaf_candidate_stats(local_stats, candidates);
          {
            ScopedTimer verify_timer(&local_stats.leaf_exact_verify_ms);
            for (size_t seq_idx : candidates.candidate_item_ids) {
              const std::string_view sequence =
                  sequences.sequence(static_cast<LeafId>(seq_idx));
              if (std::llabs(static_cast<long long>(sequence.size()) -
                             static_cast<long long>(center.size())) >
                  finest_radius) {
                local_stats.leaf_length_pruned_pairs++;
                continue;
              }
              if (phase1_base_count_lower_bound(
                      world_base_counts[world_idx],
                      leaf_base_counts[seq_idx]) > finest_radius) {
                local_stats.leaf_base_count_pruned_pairs++;
                continue;
              }
              if (range_config_.leaf_qgram_postfilter &&
                  phase4_qgram_can_prune_edit_distance(
                      world_qgram_signature,
                      leaf_qgram_signatures[seq_idx],
                      finest_radius)) {
                local_stats.leaf_qgram_pruned_by_l1++;
                continue;
              }
              local_stats.leaf_exact_distance_calls++;
              const int dist = build_distance_bounded_prepared(
                  center, sequence, finest_radius,
                  range_config_.distance_mode, prepared_center_ptr);
              if (dist <= finest_radius) {
                ScopedTimer timer(&local_stats.leaf_tuple_emit_ms);
                world.child_or_leaf_ids.push_back(
                    static_cast<LeafId>(seq_idx));
                geometry.link_beacon_dists.push_back(
                    static_cast<uint8_t>(dist));
              }
            }
          }
          if (progress && world_idx % 256 == 255) progress->advance(256);
        }
      }
      if (progress) progress->set_completed(finest_layer.size());

      for (const auto& local_stats : thread_stats) {
        merge_leaf_local_stats(stats_, local_stats);
      }
    }
  }

  size_t total_links = 0;
  for (NodeId node_id : finest_layer) {
    total_links += build_nodes_[node_id].child_or_leaf_ids.size();
  }
  stats_.leaf_attachments_added = total_links;
  double avg_links =
      finest_layer.empty() ? 0.0 : static_cast<double>(total_links) / finest_layer.size();
  std::cerr << "    Attached " << total_links << " leaf links to finest primary layer"
            << " (avg " << avg_links << " per node)\n";
  if (progress) progress->finish_phase();
}

void BioGeometryIndexBuilder::assign_integer_ids() {
  world_node_count_ = 0;
  final_node_ids_.assign(build_nodes_.size(), INVALID_NODE_ID);

  for (const auto& layer : primary_layers_) {
    for (NodeId build_node_id : layer) {
      if (build_node_id >= build_nodes_.size()) {
        throw std::runtime_error("primary layer contains invalid build node id");
      }
      if (final_node_ids_[build_node_id] != INVALID_NODE_ID) continue;
      if (world_node_count_ > static_cast<size_t>(INVALID_NODE_ID - 1)) {
        throw std::runtime_error("too many world nodes for 32-bit NodeId");
      }
      final_node_ids_[build_node_id] =
          static_cast<NodeId>(world_node_count_++);
    }
  }
}

void BioGeometryIndexBuilder::build_search_graph_view() {
  auto to_u32 = [](size_t value, const char* field) -> uint32_t {
    if (value > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(std::string(field) + " exceeds 32-bit view range");
    }
    return static_cast<uint32_t>(value);
  };
  auto beacon_storage_for =
      [](LeafId center, const std::vector<LeafId>& beacon_ids) {
        bool fits_delta8 = true;
        bool fits_delta16 = true;
        for (LeafId beacon_id : beacon_ids) {
          const int64_t delta = static_cast<int64_t>(beacon_id) - center;
          fits_delta8 =
              fits_delta8 &&
              delta >= std::numeric_limits<int8_t>::min() &&
              delta <= std::numeric_limits<int8_t>::max();
          fits_delta16 =
              fits_delta16 &&
              delta >= std::numeric_limits<int16_t>::min() &&
              delta <= std::numeric_limits<int16_t>::max();
        }
        if (fits_delta8) {
          return WorldNodeRecord::BeaconStorage::Delta8;
        }
        if (fits_delta16) {
          return WorldNodeRecord::BeaconStorage::Delta16;
        }
        return WorldNodeRecord::BeaconStorage::Absolute32;
      };
  const auto child_ids_fit_delta16 =
      [&](NodeId build_node_id) {
        const auto& node = build_nodes_[build_node_id];
        if (node.child_or_leaf_ids.empty()) return false;
        const NodeId parent_id = final_node_ids_[build_node_id];
        for (NodeId child_build_id : node.child_or_leaf_ids) {
          if (child_build_id >= final_node_ids_.size()) return false;
          const NodeId child_id = final_node_ids_[child_build_id];
          if (child_id <= parent_id ||
              static_cast<uint64_t>(child_id) - parent_id - 1 >
                  std::numeric_limits<uint16_t>::max()) {
            return false;
          }
        }
        return true;
      };
  const auto leaf_storage_for =
      [&](NodeId build_node_id) {
        const auto& node = build_nodes_[build_node_id];
        bool fits_delta8 = true;
        bool fits_delta16 = true;
        for (LeafId leaf_id : node.child_or_leaf_ids) {
          const int64_t delta =
              static_cast<int64_t>(leaf_id) -
              node.center_sequence_id;
          fits_delta8 =
              fits_delta8 &&
              delta >= std::numeric_limits<int8_t>::min() &&
              delta <= std::numeric_limits<int8_t>::max();
          fits_delta16 =
              fits_delta16 &&
              delta >= std::numeric_limits<int16_t>::min() &&
              delta <= std::numeric_limits<int16_t>::max();
        }
        if (fits_delta8) {
          return WorldNodeRecord::LinkStorage::Delta8;
        }
        if (fits_delta16) {
          return WorldNodeRecord::LinkStorage::Delta16;
        }
        return WorldNodeRecord::LinkStorage::Absolute32;
      };
  SearchGraphView view;
  view.sequences = std::move(search_graph_view_.sequences);
  view.node_records.assign(world_node_count_, WorldNodeRecord{});
  view.layer_begin.assign(static_cast<size_t>(num_primary_layers()), 0);
  view.layer_end.assign(static_cast<size_t>(num_primary_layers()), 0);
  view.child_mbb_bits_by_layer.assign(
      static_cast<size_t>(num_primary_layers()), 8);
  size_t total_child_deltas16 = 0;
  size_t total_child_ids32 = 0;
  size_t total_leaf_deltas8 = 0;
  size_t total_leaf_deltas16 = 0;
  size_t total_leaf_ids32 = 0;
  size_t total_leaf_beacon_cells = 0;
  size_t total_beacon_deltas8 = 0;
  size_t total_beacon_deltas16 = 0;
  size_t total_beacon_ids32 = 0;
  std::vector<uint8_t> layer_mbb_max(
      primary_layers_.size(), 0);
  for (size_t layer_idx = 0;
       layer_idx < primary_layers_.size(); ++layer_idx) {
    const bool is_finest =
        layer_idx + 1 == primary_layers_.size();
    for (NodeId build_node_id : primary_layers_[layer_idx]) {
      const auto& node = build_nodes_[build_node_id];
      const auto& geometry =
          build_node_geometry_[node.geometry_index];
      if (is_finest) {
        const auto storage = leaf_storage_for(build_node_id);
        if (storage == WorldNodeRecord::LinkStorage::Delta8) {
          total_leaf_deltas8 += node.child_or_leaf_ids.size();
        } else if (storage == WorldNodeRecord::LinkStorage::Delta16) {
          total_leaf_deltas16 += node.child_or_leaf_ids.size();
        } else {
          total_leaf_ids32 += node.child_or_leaf_ids.size();
        }
        total_leaf_beacon_cells +=
            geometry.link_beacon_dists.size();
      } else {
        if (child_ids_fit_delta16(build_node_id)) {
          total_child_deltas16 += node.child_or_leaf_ids.size();
        } else {
          total_child_ids32 += node.child_or_leaf_ids.size();
        }
        if (!geometry.link_beacon_dists.empty()) {
          layer_mbb_max[layer_idx] = std::max(
              layer_mbb_max[layer_idx],
              *std::max_element(
                  geometry.link_beacon_dists.begin(),
                  geometry.link_beacon_dists.end()));
        }
        const auto storage =
            beacon_storage_for(
                node.center_sequence_id, geometry.beacon_ids);
        if (storage == WorldNodeRecord::BeaconStorage::Delta8) {
          total_beacon_deltas8 += geometry.beacon_ids.size();
        } else if (storage ==
                   WorldNodeRecord::BeaconStorage::Delta16) {
          total_beacon_deltas16 += geometry.beacon_ids.size();
        } else {
          total_beacon_ids32 += geometry.beacon_ids.size();
        }
      }
    }
  }
  size_t total_mbb_bytes = 0;
  for (size_t layer_idx = 0;
       layer_idx + 1 < primary_layers_.size(); ++layer_idx) {
    uint32_t bits = 1;
    uint32_t maximum = layer_mbb_max[layer_idx];
    while (maximum >>= 1) ++bits;
    view.child_mbb_bits_by_layer[layer_idx] = bits;
    for (NodeId build_node_id : primary_layers_[layer_idx]) {
      const auto& node = build_nodes_[build_node_id];
      const auto& geometry =
          build_node_geometry_[node.geometry_index];
      const uint64_t bit_count =
          static_cast<uint64_t>(geometry.link_beacon_dists.size()) *
          bits;
      if (bit_count >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) * 8) {
        throw std::runtime_error("packed child MBB size overflow");
      }
      total_mbb_bytes += static_cast<size_t>((bit_count + 7) / 8);
    }
  }
  view.child_id_deltas16.reserve(total_child_deltas16);
  view.child_ids.reserve(total_child_ids32);
  view.leaf_id_deltas8.reserve(total_leaf_deltas8);
  view.leaf_id_deltas16.reserve(total_leaf_deltas16);
  view.leaf_ids.reserve(total_leaf_ids32);
  view.child_beacon_dists.reserve(total_mbb_bytes);
  view.beacon_begins.reserve(
      primary_layers_.empty()
          ? 0
          : world_node_count_ - primary_layers_.back().size());
  view.beacon_deltas8.reserve(total_beacon_deltas8);
  view.beacon_deltas16.reserve(total_beacon_deltas16);
  view.beacon_ids32.reserve(total_beacon_ids32);
  view.leaf_beacon_dists.reserve(total_leaf_beacon_cells);

  const auto append_packed_mbb =
      [&](const std::vector<uint8_t>& values, uint32_t bits) {
        if (bits == 0 || bits > 8) {
          throw std::runtime_error("invalid child MBB bit width");
        }
        if (bits == 8) {
          view.child_beacon_dists.append(values.begin(), values.end());
          return;
        }
        const size_t byte_count = static_cast<size_t>(
            (static_cast<uint64_t>(values.size()) * bits + 7) / 8);
        const size_t byte_begin = view.child_beacon_dists.size();
        view.child_beacon_dists.resize(byte_begin + byte_count);
        for (size_t value_idx = 0; value_idx < values.size(); ++value_idx) {
          const uint32_t value = values[value_idx];
          if (value >= (uint32_t{1} << bits)) {
            throw std::runtime_error(
                "child MBB value exceeds layer bit width");
          }
          const size_t bit_offset = value_idx * bits;
          const size_t byte_offset = bit_offset >> 3;
          const uint32_t shift =
              static_cast<uint32_t>(bit_offset & 7);
          view.child_beacon_dists[byte_begin + byte_offset] |=
              static_cast<uint8_t>(value << shift);
          if (shift + bits > 8) {
            view.child_beacon_dists[
                byte_begin + byte_offset + 1] |=
                static_cast<uint8_t>(value >> (8 - shift));
          }
        }
      };

  for (size_t layer_idx = 0; layer_idx < primary_layers_.size(); ++layer_idx) {
    const auto& layer = primary_layers_[layer_idx];
    const bool is_finest =
        layer_idx + 1 == primary_layers_.size();
    view.layer_begin[layer_idx] =
        layer.empty() ? to_u32(world_node_count_, "layer_begin")
                      : final_node_ids_[layer.front()];
    for (NodeId build_node_id : layer) {
      if (build_node_id >= build_nodes_.size() ||
          final_node_ids_[build_node_id] >= world_node_count_) {
        throw std::runtime_error("cannot build search graph view with invalid node id");
      }
      const auto& node = build_nodes_[build_node_id];
      const auto& geometry =
          build_node_geometry_[node.geometry_index];
      const NodeId node_id = final_node_ids_[build_node_id];
      WorldNodeRecord& record = view.node_records[node_id];
      if (node.center_sequence_id >= sequence_count_) {
        throw std::runtime_error(
            "cannot build array index with invalid center sequence id");
      }
      record.center_sequence_id = node.center_sequence_id;

      uint32_t link_count = 0;
      if (!is_finest) {
        const bool use_delta16 =
            child_ids_fit_delta16(build_node_id);
        if (use_delta16) {
          record.link_begin = to_u32(
              view.child_id_deltas16.size(),
              "child_id_deltas16");
          view.set_child_ids_delta16(node_id);
        } else {
          record.link_begin =
              to_u32(view.child_ids.size(), "child_ids");
        }
        for (NodeId child_id : node.child_or_leaf_ids) {
          if (child_id >= final_node_ids_.size() ||
              final_node_ids_[child_id] >= world_node_count_) {
            throw std::runtime_error(
                "cannot build search graph view with invalid child id");
          }
          const NodeId final_child_id = final_node_ids_[child_id];
          if (use_delta16) {
            view.child_id_deltas16.push_back(
                static_cast<uint16_t>(
                    final_child_id - node_id - 1));
          } else {
            view.child_ids.push_back(final_child_id);
          }
        }
        link_count = use_delta16
                         ? to_u32(
                               view.child_id_deltas16.size(),
                               "child_id_deltas16") -
                               record.link_begin
                         : to_u32(view.child_ids.size(), "child_ids") -
                               record.link_begin;
      } else {
        const auto link_storage = leaf_storage_for(build_node_id);
        record.set_link_storage(link_storage);
        if (link_storage == WorldNodeRecord::LinkStorage::Delta8) {
          record.link_begin = to_u32(
              view.leaf_id_deltas8.size(), "leaf_id_deltas8");
        } else if (
            link_storage == WorldNodeRecord::LinkStorage::Delta16) {
          record.link_begin = to_u32(
              view.leaf_id_deltas16.size(), "leaf_id_deltas16");
        } else {
          record.link_begin =
              to_u32(view.leaf_ids.size(), "leaf_ids");
        }
        for (LeafId leaf_id : node.child_or_leaf_ids) {
          if (leaf_id >= sequence_count_) {
            throw std::runtime_error(
                "cannot build search graph view with invalid leaf id");
          }
          const int64_t delta =
              static_cast<int64_t>(leaf_id) -
              node.center_sequence_id;
          if (link_storage == WorldNodeRecord::LinkStorage::Delta8) {
            view.leaf_id_deltas8.push_back(
                static_cast<int8_t>(delta));
          } else if (
              link_storage == WorldNodeRecord::LinkStorage::Delta16) {
            view.leaf_id_deltas16.push_back(
                static_cast<int16_t>(delta));
          } else {
            view.leaf_ids.push_back(leaf_id);
          }
        }
        link_count = to_u32(
            node.child_or_leaf_ids.size(), "leaf_count");
      }

      auto storage =
          WorldNodeRecord::BeaconStorage::ImplicitCenter;
      uint32_t beacon_count = 1;
      if (!is_finest) {
        storage = beacon_storage_for(
            node.center_sequence_id, geometry.beacon_ids);
        beacon_count =
            to_u32(geometry.beacon_ids.size(), "beacons");
        uint32_t beacon_begin = 0;
        if (storage == WorldNodeRecord::BeaconStorage::Delta8) {
          beacon_begin =
              to_u32(view.beacon_deltas8.size(), "beacon_deltas8");
          for (LeafId beacon_id : geometry.beacon_ids) {
            if (beacon_id >= sequence_count_) {
              throw std::runtime_error(
                  "cannot build search graph view with invalid beacon id");
            }
            view.beacon_deltas8.push_back(static_cast<int8_t>(
                static_cast<int64_t>(beacon_id) -
                node.center_sequence_id));
          }
        } else if (
            storage == WorldNodeRecord::BeaconStorage::Delta16) {
          beacon_begin =
              to_u32(view.beacon_deltas16.size(), "beacon_deltas16");
          for (LeafId beacon_id : geometry.beacon_ids) {
            if (beacon_id >= sequence_count_) {
              throw std::runtime_error(
                  "cannot build search graph view with invalid beacon id");
            }
            view.beacon_deltas16.push_back(static_cast<int16_t>(
                static_cast<int64_t>(beacon_id) -
                node.center_sequence_id));
          }
        } else {
          beacon_begin =
              to_u32(view.beacon_ids32.size(), "beacon_ids32");
          for (LeafId beacon_id : geometry.beacon_ids) {
            if (beacon_id >= sequence_count_) {
              throw std::runtime_error(
                  "cannot build search graph view with invalid beacon id");
            }
            view.beacon_ids32.push_back(beacon_id);
          }
        }
        if (view.beacon_begins.size() != node_id) {
          throw std::runtime_error(
              "non-finest beacon offsets are not a dense NodeId prefix");
        }
        view.beacon_begins.push_back(beacon_begin);
      }
      view.set_node_counts(
          node_id, link_count, beacon_count, storage);

      if (is_finest) {
        record.mbb_begin =
            to_u32(view.leaf_beacon_dists.size(),
                   "leaf_beacon_dists");
        if (geometry.link_beacon_dists.size() !=
            node.child_or_leaf_ids.size()) {
          throw std::runtime_error(
              "flat leaf beacon array does not match leaf dimensions");
        }
        view.leaf_beacon_dists.append(
            geometry.link_beacon_dists.begin(),
            geometry.link_beacon_dists.end());
      } else {
        record.mbb_begin =
            to_u32(view.child_beacon_dists.size(),
                   "child_beacon_dists");
        const size_t mbb_cells =
            node.child_or_leaf_ids.size() * geometry.beacon_ids.size();
        if (geometry.link_beacon_dists.size() != mbb_cells) {
          throw std::runtime_error(
              "child distance array does not match child/beacon dimensions");
        }
        append_packed_mbb(
            geometry.link_beacon_dists,
            view.child_mbb_bits_by_layer[layer_idx]);
      }
    }
    view.layer_end[layer_idx] =
        layer.empty() ? view.layer_begin[layer_idx]
                      : to_u32(
                            static_cast<size_t>(
                                final_node_ids_[layer.back()]) +
                                1,
                               "layer_end");
  }

  search_graph_view_ = std::move(view);
}

void BioGeometryIndexBuilder::release_build_arrays() {
  build_nodes_.clear();
  build_nodes_.shrink_to_fit();
  build_node_geometry_.clear();
  build_node_geometry_.shrink_to_fit();
  final_node_ids_.clear();
  final_node_ids_.shrink_to_fit();
  extended_layers_.clear();
  extended_layers_.shrink_to_fit();
  primary_layers_.clear();
  primary_layers_.shrink_to_fit();
}

void BioGeometryIndexBuilder::print_summary() const {
  Statistics stats = get_statistics();
  std::cerr << "  Build timing:\n"
            << "    total=" << format_ms(stats.total_build_ms) << " ms\n"
            << "    phase0_dedup=" << format_ms(stats.phase0_dedup_ms) << " ms\n"
            << "    phase1_sketch=" << format_ms(stats.phase1_sketch_ms) << " ms\n"
            << "      possible_pairs=" << stats.phase1_total_possible_pairs
            << " candidates=" << stats.phase1_candidate_pairs
            << " cover_scans=" << stats.phase1_cover_candidate_scans
            << " exact_calls=" << stats.phase1_exact_distance_calls
            << " length_pruned=" << stats.phase1_length_pruned_candidates
            << " lower_bound_pruned="
            << stats.phase1_lower_bound_pruned_candidates
            << " exact_reused=" << stats.phase1_exact_distance_reused
            << " rejection_reused="
            << stats.phase1_exact_rejection_reused
            << " cross_layer_reused="
            << stats.phase1_cross_layer_distance_reused
            << " hits=" << stats.phase1_best_cover_hits
            << " misses=" << stats.phase1_cover_misses
            << " scan_queries=" << stats.phase1_scan_queries
            << " metric_queries=" << stats.phase1_metric_index_queries
            << " qgram_queries=" << stats.phase1_qgram_index_queries
            << " fallback_scans=" << stats.phase1_fallback_scan_queries
            << " metric_dist_calls=" << stats.phase1_metric_distance_calls
            << " metric_build_dist_calls="
            << stats.phase1_metric_build_distance_calls
            << " pigeonhole_queries=" << stats.phase1_pigeonhole_queries
            << " seed_postings="
            << stats.phase1_seed_posting_entries_visited
            << " pigeonhole_candidates="
            << stats.phase1_pigeonhole_candidates
            << " pigeonhole_fallbacks="
            << stats.phase1_pigeonhole_fallbacks
            << " hint_checks=" << stats.phase1_hint_checks
            << " hint_hits=" << stats.phase1_hint_hits
            << " qgram_touched=" << stats.phase1_qgram_touched_candidates
            << " qgram_pruned=" << stats.phase1_qgram_pruned_candidates
            << "\n"
            << "    phase2_rebinding=" << format_ms(stats.phase2_rebinding_ms) << " ms\n"
            << "      index_build=" << format_ms(stats.phase2_index_build_ms) << " ms\n"
            << "      candidate_query=" << format_ms(stats.phase2_candidate_query_ms) << " ms\n"
            << "      exact_verify=" << format_ms(stats.phase2_exact_verify_ms) << " ms\n"
            << "      candidate_query_worker="
            << format_ms(stats.phase2_candidate_query_worker_ms) << " ms\n"
            << "      exact_verify_worker="
            << format_ms(stats.phase2_exact_verify_worker_ms) << " ms\n"
            << "      edge_insert=" << format_ms(stats.phase2_edge_insert_ms) << " ms\n"
            << "    phase3_mbb=" << format_ms(stats.phase3_mbb_ms) << " ms\n"
            << "      parallel_threads=" << stats.phase3_parallel_threads << "\n"
            << "      collect_beacons=" << format_ms(stats.phase3_collect_beacons_ms) << " ms\n"
            << "      collapse_children=" << format_ms(stats.phase3_collapse_children_ms) << " ms\n"
            << "      child_mbb_distance=" << format_ms(stats.phase3_child_mbb_distance_ms) << " ms\n"
            << "      rect_index_build=" << format_ms(stats.phase3_rect_index_build_ms) << " ms\n"
            << "    phase4_attach=" << format_ms(stats.phase4_attach_ms) << " ms\n"
            << "      index_build=" << format_ms(stats.leaf_index_build_ms) << " ms\n"
            << "      candidate_query=" << format_ms(stats.leaf_candidate_query_ms) << " ms\n"
            << "      exact_verify=" << format_ms(stats.leaf_exact_verify_ms) << " ms\n"
            << "      tuple_emit=" << format_ms(stats.leaf_tuple_emit_ms) << " ms\n"
            << "      tuple_merge_sort=" << format_ms(stats.leaf_tuple_merge_sort_ms) << " ms\n"
            << "      populate=" << format_ms(stats.leaf_populate_ms) << " ms\n"
            << "      leaf_beacon_distance=" << format_ms(stats.leaf_beacon_distance_ms) << " ms\n"
            << "    assign_ids=" << format_ms(stats.assign_ids_ms) << " ms\n"
            << "    graph_view=" << format_ms(stats.graph_view_ms) << " ms\n"
            << "    range_join:\n"
            << "      posting_lookup=" << format_ms(stats.range_posting_lookup_ms) << " ms\n"
            << "      seed_union=" << format_ms(stats.range_seed_union_ms) << " ms\n"
            << "      length_filter=" << format_ms(stats.range_length_filter_ms) << " ms\n"
            << "      qgram_query=" << format_ms(stats.range_qgram_query_ms) << " ms\n"
            << "      hybrid_intersection="
            << format_ms(stats.range_hybrid_intersection_ms) << " ms\n"
            << "      full_scan=" << format_ms(stats.range_full_scan_ms) << " ms\n";
  std::cerr << "  Construction range modes: links="
            << build_range_mode_name(range_config_.link_mode)
            << " leaves=" << build_range_mode_name(range_config_.leaf_attach_mode)
            << " phase1=" << phase1_candidate_mode_name(
                   range_config_.phase1_candidate_mode)
            << " phase2_qgram_postfilter="
            << (range_config_.phase2_qgram_postfilter ? "on" : "off")
            << " leaf_qgram_postfilter="
            << (range_config_.leaf_qgram_postfilter ? "on" : "off")
            << " seeds=" << range_config_.range_join.min_seed_len
            << ".." << range_config_.range_join.max_seed_len
            << " candidates="
            << range_candidate_mode_name(range_config_.range_join.candidate_mode)
            << " leaf_direction="
            << leaf_attach_direction_name(stats.leaf_attach_direction_used)
            << " qgram_q=" << range_config_.range_join.qgram_q
            << " auto_max_candidates="
            << range_config_.range_join.auto_pigeonhole_max_candidates
            << " auto_max_ratio_ignored="
            << range_config_.range_join.auto_pigeonhole_max_ratio
            << " auto_hybrid="
            << (range_config_.range_join.auto_hybrid_on_large_candidates
                    ? "true"
                    : "false")
            << "\n";
  std::cerr << "  Phase2 range join: possible=" << stats.phase2_total_possible_pairs
            << " candidates=" << stats.phase2_candidate_pairs
            << " exact_calls=" << stats.phase2_exact_distance_calls
            << " edges=" << stats.phase2_edges_added
            << " distance_batches=" << stats.phase2_distance_batches
            << " fallbacks=" << stats.phase2_full_scan_fallback_count
            << " pigeonhole_queries=" << stats.phase2_pigeonhole_queries
            << " qgram_queries=" << stats.phase2_qgram_queries
            << " hybrid_queries=" << stats.phase2_hybrid_queries
            << " qgram_candidates=" << stats.phase2_qgram_candidate_pairs
            << " qgram_l1_pruned=" << stats.phase2_qgram_pruned_by_l1
            << " base_count_pruned="
            << stats.phase2_base_count_pruned_pairs
            << " length_pruned=" << stats.phase2_length_pruned_pairs
            << " seed_candidates_before_length_filter="
            << stats.phase2_seed_candidate_pairs_before_length_filter
            << " seed_length_pruned="
            << stats.phase2_seed_length_pruned_candidates
            << " pigeonhole_early_abort="
            << stats.phase2_pigeonhole_early_abort_count
            << " range_final_candidates="
            << stats.phase2_range_final_candidate_pairs
            << " required_shared_nonpositive="
            << stats.phase2_required_shared_nonpositive_count
            << " auto_pigeonhole_accepted="
            << stats.phase2_auto_pigeonhole_accepted
            << " auto_pigeonhole_rejected_large="
            << stats.phase2_auto_pigeonhole_rejected_large_candidates
            << " auto_qgram_invoked=" << stats.phase2_auto_qgram_invoked
            << " auto_hybrid_invoked=" << stats.phase2_auto_hybrid_invoked
            << " auto_final_candidates="
            << stats.phase2_auto_final_candidate_pairs
            << " auto_avg_candidate_ratio_ignored="
            << stats.phase2_auto_candidate_ratio_avg
            << " candidate_reduction=" << (stats.phase2_candidate_reduction_ratio * 100.0)
            << "% exact_reduction=" << (stats.phase2_exact_distance_reduction_ratio * 100.0)
            << "%\n";
  std::cerr << "  Leaf range join: possible=" << stats.total_possible_leaf_pairs
            << " candidates=" << stats.leaf_candidate_pairs
            << " exact_calls=" << stats.leaf_exact_distance_calls
            << " attachments=" << stats.leaf_attachments_added
            << " fallbacks=" << stats.leaf_full_scan_fallback_count
            << " pigeonhole_queries=" << stats.leaf_pigeonhole_queries
            << " qgram_queries=" << stats.leaf_qgram_queries
            << " hybrid_queries=" << stats.leaf_hybrid_queries
            << " qgram_candidates=" << stats.leaf_qgram_candidate_pairs
            << " qgram_l1_pruned=" << stats.leaf_qgram_pruned_by_l1
            << " base_count_pruned="
            << stats.leaf_base_count_pruned_pairs
            << " length_pruned=" << stats.leaf_length_pruned_pairs
            << " seed_candidates_before_length_filter="
            << stats.leaf_seed_candidate_pairs_before_length_filter
            << " seed_length_pruned="
            << stats.leaf_seed_length_pruned_candidates
            << " pigeonhole_early_abort="
            << stats.leaf_pigeonhole_early_abort_count
            << " range_final_candidates="
            << stats.leaf_range_final_candidate_pairs
            << " required_shared_nonpositive="
            << stats.leaf_required_shared_nonpositive_count
            << " auto_pigeonhole_accepted="
            << stats.leaf_auto_pigeonhole_accepted
            << " auto_pigeonhole_rejected_large="
            << stats.leaf_auto_pigeonhole_rejected_large_candidates
            << " auto_qgram_invoked=" << stats.leaf_auto_qgram_invoked
            << " auto_hybrid_invoked=" << stats.leaf_auto_hybrid_invoked
            << " auto_final_candidates="
            << stats.leaf_auto_final_candidate_pairs
            << " auto_avg_candidate_ratio_ignored="
            << stats.leaf_auto_candidate_ratio_avg
            << " candidate_reduction=" << (stats.leaf_candidate_reduction_ratio * 100.0)
            << "% exact_reduction=" << (stats.leaf_exact_distance_reduction_ratio * 100.0)
            << "%\n";
  std::cerr << "  Primary layers: " << num_primary_layers() << "\n";
  for (int layer_idx = 0; layer_idx < num_primary_layers(); ++layer_idx) {
    std::cerr << "    W" << layer_idx
              << " radius=" << hierarchy_.primary_radii[static_cast<size_t>(layer_idx)]
              << " nodes=" << primary_layer_size(layer_idx) << "\n";
  }

  const size_t finest_count =
      primary_layer_size(finest_primary_layer_index());
  if (stats_.unique_sequences > 0 && finest_count > 0) {
    double compression =
        1.0 - static_cast<double>(finest_count) / stats_.unique_sequences;
    std::cerr << "  Finest-layer compression: " << (compression * 100.0) << "% ("
              << stats_.unique_sequences << " unique -> " << finest_count
              << " nodes)\n";
  }

  for (int layer_idx = 0; layer_idx + 1 < num_primary_layers(); ++layer_idx) {
    const auto& view = search_graph_view_;
    const size_t layer = static_cast<size_t>(layer_idx);
    size_t total_edges = 0;
    for (uint32_t node_id = view.layer_begin[layer];
         node_id < view.layer_end[layer]; ++node_id) {
      total_edges += view.child_count(node_id);
    }
    const size_t layer_size = primary_layer_size(layer_idx);
    double avg_edges =
        layer_size == 0
            ? 0.0
            : static_cast<double>(total_edges) / layer_size;
    std::cerr << "  Avg W" << layer_idx << " -> W" << (layer_idx + 1)
              << " edges: " << avg_edges << "\n";
  }
}

void BioGeometryIndexBuilder::build(
    const std::vector<std::shared_ptr<BioSequence>>& raw_sequences) {
  build_impl(raw_sequences, false, {}, {}, 0, 0, {});
}

void BioGeometryIndexBuilder::build(
    std::vector<std::shared_ptr<BioSequence>>&& raw_sequences) {
  build_impl(std::move(raw_sequences), true, {}, {}, 0, 0, {});
}

void BioGeometryIndexBuilder::build_reference_windows(
    std::string reference_id,
    std::string reference_sequence,
    size_t window_length,
    size_t stride,
    std::vector<ReferenceContig> reference_contigs) {
  if (window_length == 0) {
    throw std::invalid_argument("reference window length must be positive");
  }
  if (stride == 0) {
    throw std::invalid_argument("reference window stride must be positive");
  }
  build_impl(
      {}, false, std::move(reference_id), std::move(reference_sequence),
      window_length, stride, std::move(reference_contigs));
}

void BioGeometryIndexBuilder::build_impl(
    std::vector<std::shared_ptr<BioSequence>> raw_sequences,
    bool consume_records,
    std::string reference_id,
    std::string reference_sequence,
    size_t reference_window_length,
    size_t reference_stride,
    std::vector<ReferenceContig> reference_contigs) {
  const auto build_start = Clock::now();
  const bool build_from_reference = reference_window_length != 0;
  size_t raw_sequence_count = raw_sequences.size();
  if (build_from_reference) {
    raw_sequence_count = 0;
    if (reference_contigs.empty()) {
      if (reference_sequence.size() >= reference_window_length) {
        raw_sequence_count =
            1 + (reference_sequence.size() - reference_window_length) /
                    reference_stride;
      }
    } else {
      for (const auto& contig : reference_contigs) {
        const size_t contig_length =
            static_cast<size_t>(contig.end - contig.begin);
        if (contig_length >= reference_window_length) {
          raw_sequence_count +=
              1 + (contig_length - reference_window_length) /
                      reference_stride;
        }
      }
    }
  }
  BuildProgressReporter progress(range_config_.progress_interval_seconds);
  stats_ = Statistics{};
  stats_.created_primary_nodes.assign(static_cast<size_t>(num_primary_layers()), 0);
  world_node_count_ = 0;
  sequence_count_ = 0;
  search_graph_view_ = SearchGraphView{};
  build_nodes_.clear();
  build_node_geometry_.clear();
  final_node_ids_.clear();
  primary_layers_.assign(static_cast<size_t>(num_primary_layers()),
                         std::vector<NodeId>());
  extended_layers_.clear();

  std::cerr << "[Build generalized hierarchy] Starting for " << raw_sequence_count
            << " sequences...\n";
  std::cerr << "  Phase 0: Deduplicating sequences...\n";
  progress.begin_phase("phase0_dedup", raw_sequence_count);
  std::vector<std::shared_ptr<BioSequence>> unique_seqs;
  size_t unique_sequence_count = 0;
  {
    ScopedTimer timer(&stats_.phase0_dedup_ms);
    if (build_from_reference) {
      initialize_reference_sequence_store(
          std::move(reference_id), std::move(reference_sequence),
          reference_window_length, reference_stride,
          std::move(reference_contigs), &progress);
      unique_sequence_count = search_graph_view_.sequences.size();
    } else {
      unique_seqs = deduplicate(std::move(raw_sequences));
      unique_sequence_count = unique_seqs.size();
      initialize_sequence_store(unique_seqs, consume_records);
      std::vector<std::shared_ptr<BioSequence>>().swap(unique_seqs);
    }
  }
  progress.finish_phase();
  std::cerr << "    " << raw_sequence_count << " -> "
            << unique_sequence_count << " unique ("
            << stats_.deduplicated << " merged";
  if (build_from_reference) {
    std::cerr << ", " << stats_.invalid_reference_windows
              << " invalid windows skipped";
  }
  std::cerr << ")\n";

  std::cerr << "  Phase 1: Extended hierarchy sketch (top-down)...\n";
  std::cerr << "    Primary radii: ";
  for (size_t i = 0; i < hierarchy_.primary_radii.size(); ++i) {
    if (i) std::cerr << ",";
    std::cerr << hierarchy_.primary_radii[i];
  }
  std::cerr << "\n";
  std::cerr << "    Auxiliary radii: ";
  for (size_t i = 0; i < hierarchy_.auxiliary_radii.size(); ++i) {
    if (i) std::cerr << ",";
    std::cerr << hierarchy_.auxiliary_radii[i];
  }
  std::cerr << "\n";
  {
    ScopedTimer timer(&stats_.phase1_sketch_ms);
    phase1_build_extended_sketch(&progress);
  }

  std::cerr << "    Expanded layers:";
  for (size_t i = 0; i < extended_layers_.size(); ++i) {
    std::cerr << " L" << i << "=" << extended_layers_[i].size();
  }
  std::cerr << "\n";

  std::cerr << "  Phase 2: Inter-tier rebinding (DAG)...\n";
  {
    ScopedTimer timer(&stats_.phase2_rebinding_ms);
    phase2_inter_tier_rebinding(&progress);
  }

  std::cerr << "  Phase 3: Collapse auxiliary tiers + MBB...\n";
  {
    ScopedTimer timer(&stats_.phase3_mbb_ms);
    phase3_collapse_and_compute_mbb(&progress);
  }

  std::cerr << "  Phase 4: Leaf attachment...\n";
  {
    ScopedTimer timer(&stats_.phase4_attach_ms);
    attach_leaves(&progress);
  }
  {
    ScopedTimer timer(&stats_.assign_ids_ms);
    assign_integer_ids();
  }
  {
    ScopedTimer timer(&stats_.graph_view_ms);
    build_search_graph_view();
  }

  std::cerr << "[Build generalized hierarchy] Completed.\n";
  stats_.total_build_ms = elapsed_ms_since(build_start);
  {
    ScopedTimer timer(&stats_.print_summary_ms);
    print_summary();
  }
  stats_.total_build_ms = elapsed_ms_since(build_start);
  release_build_arrays();
}

BioGeometryIndexBuilder::Statistics BioGeometryIndexBuilder::get_statistics() const {
  Statistics stats = stats_;
  stats.phase2_candidate_reduction_ratio =
      reduction_ratio(stats.phase2_total_possible_pairs, stats.phase2_candidate_pairs);
  stats.phase2_exact_distance_reduction_ratio =
      reduction_ratio(stats.phase2_total_possible_pairs, stats.phase2_exact_distance_calls);
  stats.leaf_candidate_reduction_ratio =
      reduction_ratio(stats.total_possible_leaf_pairs, stats.leaf_candidate_pairs);
  stats.leaf_exact_distance_reduction_ratio =
      reduction_ratio(stats.total_possible_leaf_pairs, stats.leaf_exact_distance_calls);
  const size_t phase2_auto_ratio_count =
      stats.phase2_auto_pigeonhole_accepted +
      stats.phase2_auto_pigeonhole_rejected_large_candidates;
  if (phase2_auto_ratio_count > 0) {
    stats.phase2_auto_candidate_ratio_avg =
        stats.phase2_auto_candidate_ratio_sum /
        static_cast<double>(phase2_auto_ratio_count);
  }
  const size_t leaf_auto_ratio_count =
      stats.leaf_auto_pigeonhole_accepted +
      stats.leaf_auto_pigeonhole_rejected_large_candidates;
  if (leaf_auto_ratio_count > 0) {
    stats.leaf_auto_candidate_ratio_avg =
        stats.leaf_auto_candidate_ratio_sum /
        static_cast<double>(leaf_auto_ratio_count);
  }
  const size_t finest_count =
      primary_layer_size(finest_primary_layer_index());
  if (stats.unique_sequences > 0 && finest_count > 0) {
    stats.compression_ratio =
        1.0 - static_cast<double>(finest_count) / stats.unique_sequences;
  }

  if (num_primary_layers() >= 2) {
    size_t total_edges = 0;
    size_t total_nodes = 0;
    for (int layer_idx = 0; layer_idx + 1 < num_primary_layers(); ++layer_idx) {
      const auto& view = search_graph_view_;
      const size_t layer = static_cast<size_t>(layer_idx);
      total_nodes += primary_layer_size(layer_idx);
      for (uint32_t node_id = view.layer_begin[layer];
           node_id < view.layer_end[layer]; ++node_id) {
        total_edges += view.child_count(node_id);
      }
    }
    if (total_nodes > 0) {
      stats.dag_redundancy =
          (static_cast<double>(total_edges) / static_cast<double>(total_nodes) - 1.0) * 100.0;
    }
  }
  return stats;
}

}  // namespace navigamer
