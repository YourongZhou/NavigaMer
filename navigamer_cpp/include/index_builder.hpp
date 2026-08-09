#ifndef NAVIGAMER_INDEX_BUILDER_HPP
#define NAVIGAMER_INDEX_BUILDER_HPP

#include "structure.hpp"
#include "tools.hpp"
#include "range_join.hpp"
#include "phase2_distance_verifier.hpp"
#include "simd_mbb_filter.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace navigamer {

class BuildProgressReporter;
class IndexPersistenceAccess;

struct HierarchyConfig {
  std::vector<int> primary_radii;
  std::vector<int> auxiliary_radii;

  HierarchyConfig() = default;
  explicit HierarchyConfig(std::vector<int> primary_radii_in);
  HierarchyConfig(std::vector<int> primary_radii_in, std::vector<int> auxiliary_radii_in);

  int num_primary_layers() const;
  int num_auxiliary_layers() const;
  int num_expanded_layers() const;
  void validate() const;
};

enum class BuildRangeMode {
  Full,
  Indexed,
};

const char* build_range_mode_name(BuildRangeMode mode);
BuildRangeMode parse_build_range_mode(const std::string& value);

enum class LeafAttachDirection {
  SeqToWorld,
  WorldToSeq,
  Auto,
};

const char* leaf_attach_direction_name(LeafAttachDirection direction);
LeafAttachDirection parse_leaf_attach_direction(const std::string& value);

enum class BuildDistanceMode {
  DP,
  Edlib,
  Auto,
};

const char* build_distance_mode_name(BuildDistanceMode mode);
BuildDistanceMode parse_build_distance_mode(const std::string& value);

enum class Phase1CandidateMode {
  Scan,
  Hybrid,
};

const char* phase1_candidate_mode_name(Phase1CandidateMode mode);
Phase1CandidateMode parse_phase1_candidate_mode(const std::string& value);

struct BuildRangeConfig {
  BuildRangeMode link_mode = BuildRangeMode::Indexed;
  BuildRangeMode leaf_attach_mode = BuildRangeMode::Indexed;
  LeafAttachDirection leaf_attach_direction = LeafAttachDirection::Auto;
  BuildDistanceMode distance_mode = BuildDistanceMode::Edlib;
  Phase1CandidateMode phase1_candidate_mode = Phase1CandidateMode::Hybrid;
  RangeJoinConfig range_join;
  size_t min_rect_index_fanout = 64;
  size_t phase1_metric_min_fanout = 12;
  size_t phase1_qgram_min_fanout = 12;
  size_t phase1_qgram_max_touched = 250000;
  bool phase1_preserve_input_order = true;
  bool phase2_qgram_postfilter = false;
  bool leaf_qgram_postfilter = false;
  int progress_interval_seconds = 600;
  // Sharded construction disables diagnostics for its internal part builders;
  // the outer command still reports the completed shard index.
  bool emit_build_output = true;
};

// Build-time owned vector that can become a read-only view into a persisted
// mmap after loading. Query access remains a plain pointer plus size.
template <typename T>
class FinalArray {
 public:
  using const_iterator = const T*;

  FinalArray() = default;
  FinalArray(std::initializer_list<T> values) : owned_(values) {
    sync_owned_view();
  }
  FinalArray(const FinalArray& other) { copy_from(other); }
  FinalArray& operator=(const FinalArray& other) {
    if (this != &other) copy_from(other);
    return *this;
  }
  FinalArray(FinalArray&& other) noexcept { move_from(std::move(other)); }
  FinalArray& operator=(FinalArray&& other) noexcept {
    if (this != &other) move_from(std::move(other));
    return *this;
  }
  FinalArray& operator=(std::initializer_list<T> values) {
    ensure_owned();
    owned_ = values;
    sync_owned_view();
    return *this;
  }

  size_t size() const { return size_; }
  bool empty() const { return size() == 0; }
  const T* data() const { return data_; }
  const T& operator[](size_t index) const { return data()[index]; }
  T& operator[](size_t index) {
    ensure_owned();
    return owned_[index];
  }
  const T& at(size_t index) const {
    if (index >= size()) throw std::out_of_range("final array index");
    return data()[index];
  }
  T& at(size_t index) {
    ensure_owned();
    return owned_.at(index);
  }
  const_iterator begin() const { return data(); }
  const_iterator end() const {
    const T* first = data();
    return empty() ? first : first + size();
  }
  bool is_mapped() const { return static_cast<bool>(mapped_owner_); }

  bool operator==(const FinalArray& other) const {
    return size() == other.size() &&
           std::equal(begin(), end(), other.begin());
  }
  bool operator!=(const FinalArray& other) const {
    return !(*this == other);
  }

  void reserve(size_t capacity) {
    ensure_owned();
    owned_.reserve(capacity);
    sync_owned_view();
  }
  void resize(size_t size) {
    ensure_owned();
    owned_.resize(size);
    sync_owned_view();
  }
  void assign(size_t count, const T& value) {
    ensure_owned();
    owned_.assign(count, value);
    sync_owned_view();
  }
  void push_back(const T& value) {
    ensure_owned();
    owned_.push_back(value);
    sync_owned_view();
  }
  void clear() {
    ensure_owned();
    owned_.clear();
    sync_owned_view();
  }
  template <typename Iterator>
  void append(Iterator first, Iterator last) {
    ensure_owned();
    owned_.insert(owned_.end(), first, last);
    sync_owned_view();
  }
  void set_owned(std::vector<T> values) {
    mapped_owner_.reset();
    owned_ = std::move(values);
    sync_owned_view();
  }
  void set_mapped(std::shared_ptr<const void> owner,
                  const T* data, size_t size) {
    if (!owner || (!data && size != 0)) {
      throw std::invalid_argument("invalid mapped final array");
    }
    owned_.clear();
    owned_.shrink_to_fit();
    mapped_owner_ = std::move(owner);
    data_ = data;
    size_ = size;
  }

 private:
  void ensure_owned() const {
    if (mapped_owner_) {
      throw std::logic_error("cannot mutate a mapped final array");
    }
  }
  void sync_owned_view() {
    data_ = owned_.data();
    size_ = owned_.size();
  }
  void copy_from(const FinalArray& other) {
    mapped_owner_ = other.mapped_owner_;
    if (mapped_owner_) {
      owned_.clear();
      owned_.shrink_to_fit();
      data_ = other.data_;
      size_ = other.size_;
    } else {
      owned_ = other.owned_;
      sync_owned_view();
    }
  }
  void move_from(FinalArray&& other) {
    const bool mapped = static_cast<bool>(other.mapped_owner_);
    owned_ = std::move(other.owned_);
    mapped_owner_ = std::move(other.mapped_owner_);
    if (mapped) {
      data_ = other.data_;
      size_ = other.size_;
    } else {
      sync_owned_view();
    }
    other.data_ = nullptr;
    other.size_ = 0;
  }

  const T* data_ = nullptr;
  size_t size_ = 0;
  std::vector<T> owned_;
  std::shared_ptr<const void> mapped_owner_;
};

constexpr size_t kReferencePositionBlockSize = 256;

enum class ReferencePositionEncoding : uint8_t {
  Linear = 0,
  Bitset = 1,
  Delta8 = 2,
  Delta16 = 3,
  Absolute32 = 4,
};

// One independently decodable block of monotonically increasing
// representative positions. Dense blocks use a local bitset; sparse blocks
// use the narrowest exact offset representation. LeafId implies both the
// block and the position within it.
struct ReferencePositionBlock {
  uint64_t payload_begin = 0;
  uint32_t base = 0;
  // Payload byte count, or arithmetic step for Linear blocks.
  uint16_t payload_size = 0;
  ReferencePositionEncoding encoding = ReferencePositionEncoding::Bitset;
  uint8_t reserved = 0;

  bool operator==(const ReferencePositionBlock& other) const {
    return payload_begin == other.payload_begin &&
           base == other.base &&
           payload_size == other.payload_size &&
           encoding == other.encoding && reserved == other.reserved;
  }
};
static_assert(sizeof(ReferencePositionBlock) == 16,
              "reference position block must remain 16 bytes");

struct ReferenceOccurrence {
  LeafId sequence_id = INVALID_LEAF_ID;
  uint32_t source_pos = 0;

  bool operator==(const ReferenceOccurrence& other) const {
    return sequence_id == other.sequence_id &&
           source_pos == other.source_pos;
  }
};
static_assert(sizeof(ReferenceOccurrence) == 8,
              "reference occurrence must remain an 8-byte value");

struct ReferenceOccurrenceGroup {
  LeafId sequence_id = INVALID_LEAF_ID;
  uint32_t position_begin = 0;

  bool operator==(const ReferenceOccurrenceGroup& other) const {
    return sequence_id == other.sequence_id &&
           position_begin == other.position_begin;
  }
};
static_assert(sizeof(ReferenceOccurrenceGroup) == 8,
              "reference occurrence group must remain an 8-byte value");

// Canonical, pointer-free sequence storage for a finalized index. SequenceId is
// implicit from the position in records/reference positions, so nodes, beacons,
// leaf links, and search results only need a 32-bit integer reference.
struct SequenceStore {
  std::vector<BioSequence> records;
  FinalArray<ReferencePositionBlock> reference_position_blocks;
  FinalArray<uint8_t> reference_position_payload;
  uint32_t reference_sequence_count = 0;
  // When every representative position is one arithmetic progression, avoid
  // materializing the otherwise per-256-ID position block table.
  bool reference_positions_global_linear = false;
  uint32_t reference_position_begin = 0;
  uint16_t reference_position_step = 0;
  // A sequence with one extra position uses one compact pair. Sequences with
  // two or more extras share one group record and a flat position array.
  FinalArray<ReferenceOccurrence> singleton_occurrences;
  FinalArray<ReferenceOccurrenceGroup> occurrence_groups;
  FinalArray<uint32_t> grouped_occurrence_positions;
  std::vector<ReferenceContig> reference_contigs;
  std::string reference_id;
  // Builds own the reference string. Persisted indexes map the same raw bases
  // directly from the index file so loading does not allocate or eagerly
  // fault the entire reference into RAM.
  std::string reference_sequence;
  FinalArray<char> mapped_reference_sequence;
  size_t fixed_sequence_length = 0;
  bool reference_backed = false;

  size_t size() const {
    return reference_backed ? reference_sequence_count : records.size();
  }
  bool empty() const { return size() == 0; }
  std::string_view reference_view() const {
    if (!mapped_reference_sequence.empty()) {
      return std::string_view(
          mapped_reference_sequence.data(),
          mapped_reference_sequence.size());
    }
    return reference_sequence;
  }
  const BioSequence& at(LeafId id) const {
    return records.at(static_cast<size_t>(id));
  }
  const BioSequence& operator[](LeafId id) const {
    return records[static_cast<size_t>(id)];
  }
  std::string_view sequence(LeafId id) const {
    if (reference_backed) {
      const std::string_view reference = reference_view();
      return std::string_view(
          reference.data() + source_position(id),
          fixed_sequence_length);
    }
    return records[static_cast<size_t>(id)].seq;
  }
  size_t source_position(LeafId id) const {
    if (reference_backed) {
      if (id >= reference_sequence_count) {
        throw std::out_of_range("reference sequence id");
      }
      if (reference_positions_global_linear) {
        return static_cast<size_t>(reference_position_begin) +
               static_cast<size_t>(id) * reference_position_step;
      }
      const size_t block_idx =
          static_cast<size_t>(id) / kReferencePositionBlockSize;
      const size_t in_block =
          static_cast<size_t>(id) % kReferencePositionBlockSize;
      const auto& block = reference_position_blocks.at(block_idx);
      if (in_block == 0) return block.base;
      if (block.encoding == ReferencePositionEncoding::Linear) {
        return static_cast<size_t>(block.base) +
               in_block * block.payload_size;
      }
      const uint8_t* payload =
          reference_position_payload.data() + block.payload_begin;
      const size_t encoded_idx = in_block - 1;
      switch (block.encoding) {
        case ReferencePositionEncoding::Linear:
          break;
        case ReferencePositionEncoding::Bitset: {
          size_t remaining = encoded_idx;
          for (size_t byte_idx = 0;
               byte_idx < block.payload_size; ++byte_idx) {
            uint8_t bits = payload[byte_idx];
            const size_t count = static_cast<size_t>(
                __builtin_popcount(static_cast<unsigned int>(bits)));
            if (remaining >= count) {
              remaining -= count;
              continue;
            }
            while (remaining != 0) {
              bits = static_cast<uint8_t>(bits & (bits - 1));
              --remaining;
            }
            const unsigned int bit = static_cast<unsigned int>(
                __builtin_ctz(static_cast<unsigned int>(bits)));
            return static_cast<size_t>(block.base) +
                   byte_idx * 8 + bit + 1;
          }
          throw std::runtime_error(
              "reference position bitset has too few entries");
        }
        case ReferencePositionEncoding::Delta8:
          return static_cast<size_t>(block.base) +
                 payload[encoded_idx];
        case ReferencePositionEncoding::Delta16: {
          uint16_t delta = 0;
          std::memcpy(&delta, payload + encoded_idx * sizeof(delta),
                      sizeof(delta));
          return static_cast<size_t>(block.base) + delta;
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
    return records.at(static_cast<size_t>(id)).source_pos;
  }
  size_t contig_source_position(size_t source_pos) const {
    const auto& contig = contig_for_position(source_pos);
    return static_cast<size_t>(contig.source_begin) +
           source_pos - contig.begin;
  }
  BwtInterval sa_interval(LeafId id) const {
    if (!reference_backed) {
      return records.at(static_cast<size_t>(id)).bwt_interval;
    }
    (void)id;
    return {};
  }
  const ReferenceContig& contig_for_position(size_t source_pos) const {
    const auto it = std::upper_bound(
        reference_contigs.begin(), reference_contigs.end(), source_pos,
        [](size_t position, const ReferenceContig& contig) {
          return position < contig.begin;
        });
    if (it == reference_contigs.begin()) {
      throw std::out_of_range("reference position has no contig");
    }
    const auto& contig = *std::prev(it);
    if (source_pos < contig.begin || source_pos >= contig.end) {
      throw std::out_of_range("reference position lies outside contigs");
    }
    return contig;
  }
  std::vector<uint32_t> occurrence_positions(LeafId id) const {
    std::vector<uint32_t> positions;
    positions.reserve(additional_occurrence_count(id) + 1);
    for_each_occurrence(
        id, [&](uint32_t source_pos) {
          positions.push_back(source_pos);
        });
    return positions;
  }
  template <typename Visitor>
  void for_each_occurrence(LeafId id, Visitor&& visit) const {
    const uint32_t representative =
        static_cast<uint32_t>(source_position(id));
    bool emitted_representative = false;
    const auto emit_position = [&](uint32_t source_pos) {
      if (!emitted_representative &&
          representative < source_pos) {
        visit(representative);
        emitted_representative = true;
      }
      visit(source_pos);
    };
    const auto singleton = std::lower_bound(
        singleton_occurrences.begin(), singleton_occurrences.end(), id,
        [](const ReferenceOccurrence& occurrence, LeafId sequence_id) {
          return occurrence.sequence_id < sequence_id;
        });
    if (singleton != singleton_occurrences.end() &&
        singleton->sequence_id == id) {
      emit_position(singleton->source_pos);
    } else {
      const auto group = std::lower_bound(
          occurrence_groups.begin(), occurrence_groups.end(), id,
          [](const ReferenceOccurrenceGroup& occurrence_group,
             LeafId sequence_id) {
            return occurrence_group.sequence_id < sequence_id;
          });
      if (group != occurrence_groups.end() && group->sequence_id == id) {
        const size_t group_idx =
            static_cast<size_t>(group - occurrence_groups.begin());
        const size_t position_end =
            group_idx + 1 < occurrence_groups.size()
                ? occurrence_groups[group_idx + 1].position_begin
                : grouped_occurrence_positions.size();
        for (size_t position_idx = group->position_begin;
             position_idx < position_end; ++position_idx) {
          emit_position(grouped_occurrence_positions[position_idx]);
        }
      }
    }
    if (!emitted_representative) visit(representative);
  }
  size_t additional_occurrence_count(LeafId id) const {
    const auto singleton = std::lower_bound(
        singleton_occurrences.begin(), singleton_occurrences.end(), id,
        [](const ReferenceOccurrence& occurrence, LeafId sequence_id) {
          return occurrence.sequence_id < sequence_id;
        });
    if (singleton != singleton_occurrences.end() &&
        singleton->sequence_id == id) {
      return 1;
    }
    const auto group = std::lower_bound(
        occurrence_groups.begin(), occurrence_groups.end(), id,
        [](const ReferenceOccurrenceGroup& occurrence_group,
           LeafId sequence_id) {
          return occurrence_group.sequence_id < sequence_id;
        });
    if (group == occurrence_groups.end() || group->sequence_id != id) {
      return 0;
    }
    const size_t group_idx =
        static_cast<size_t>(group - occurrence_groups.begin());
    const size_t position_end =
        group_idx + 1 < occurrence_groups.size()
            ? occurrence_groups[group_idx + 1].position_begin
            : grouped_occurrence_positions.size();
    return position_end - group->position_begin;
  }
  std::string identifier(LeafId id) const {
    if (!reference_backed) return records.at(static_cast<size_t>(id)).id;
    const size_t global_pos = source_position(id);
    const auto& contig = contig_for_position(global_pos);
    return contig.id + "_" +
           std::to_string(contig_source_position(global_pos));
  }
  BioSequence materialize(LeafId id) const {
    if (!reference_backed) return records.at(static_cast<size_t>(id));
    const auto sequence_view = sequence(id);
    BioSequence sequence_record(
        identifier(id),
        std::string(sequence_view));
    sequence_record.sequence_id = id;
    sequence_record.has_source_pos = true;
    sequence_record.source_pos = source_position(id);
    sequence_record.bwt_interval = sa_interval(id);
    return sequence_record;
  }
};

struct NodeCountOverflowRecord {
  uint32_t link_count = 0;
  uint32_t beacon_count = 0;

  bool operator==(const NodeCountOverflowRecord& other) const {
    return link_count == other.link_count &&
           beacon_count == other.beacon_count;
  }
};
static_assert(sizeof(NodeCountOverflowRecord) == 8,
              "node-count overflow must remain 8 bytes");

// One fixed-size world record. Variable-length relationships live in the
// global arrays in SearchGraphView and are addressed by offset + count.
struct WorldNodeRecord {
  enum class BeaconStorage : uint32_t {
    Delta8 = 0,
    PackedDelta = 1,
    Absolute32 = 2,
    ImplicitCenter = 3,
  };
  enum class LinkStorage : uint32_t {
    Absolute32 = 0,
    Delta16 = 1,
    Delta8 = 2,
    PackedDelta = 3,
  };
  static constexpr uint32_t LINK_COUNT_BITS = 24;
  static constexpr uint32_t BEACON_COUNT_BITS = 4;
  static constexpr uint32_t LINK_COUNT_MASK =
      (uint32_t{1} << LINK_COUNT_BITS) - 1;
  static constexpr uint32_t BEACON_COUNT_MASK =
      (uint32_t{1} << BEACON_COUNT_BITS) - 1;
  static constexpr uint32_t BEACON_COUNT_SHIFT = LINK_COUNT_BITS;
  static constexpr uint32_t STORAGE_SHIFT = 28;
  static constexpr uint32_t STORAGE_MASK = 3;
  static constexpr uint32_t LINK_STORAGE_SHIFT = 30;
  static constexpr uint32_t LINK_STORAGE_MASK =
      uint32_t{3} << LINK_STORAGE_SHIFT;
  static constexpr uint32_t COUNT_OVERFLOW_CODE = BEACON_COUNT_MASK;
  static constexpr uint32_t CHILD_MBB_BEGIN_BITS = 29;
  static constexpr uint32_t CHILD_MBB_BEGIN_MASK =
      (uint32_t{1} << CHILD_MBB_BEGIN_BITS) - 1;
  static constexpr uint32_t PACKED_CHILD_BEGIN_BITS = 28;
  static constexpr uint32_t PACKED_CHILD_BEGIN_MASK =
      (uint32_t{1} << PACKED_CHILD_BEGIN_BITS) - 1;

  // Non-finest nodes address byte base-deltas, per-parent bit-packed deltas,
  // 16-bit forward deltas, or absolute child IDs; finest nodes address
  // 8/16-bit leaf deltas or leaf IDs.
  // Primary layers are explicit, so child and leaf ranges share one
  // offset/count pair; the packed link encoding selects the exact array.
  uint32_t link_begin = 0;
  uint32_t packed_counts = 0;
  // Finest nodes store a 29-bit leaf-distance byte offset plus the exact
  // 1..8-bit cell width. Non-finest nodes use the same layout for child MBBs.
  uint32_t mbb_begin = 0;

  uint32_t child_begin() const {
    return link_storage() == LinkStorage::PackedDelta
               ? link_begin & PACKED_CHILD_BEGIN_MASK
               : link_begin;
  }
  uint32_t leaf_begin() const {
    return link_storage() == LinkStorage::PackedDelta
               ? link_begin & PACKED_CHILD_BEGIN_MASK
               : link_begin;
  }
  uint32_t packed_child_bits() const {
    return (link_begin >> PACKED_CHILD_BEGIN_BITS) + 1;
  }
  void set_packed_child_layout(uint32_t begin, uint32_t bits) {
    if (begin > PACKED_CHILD_BEGIN_MASK) {
      throw std::length_error(
          "packed child-ID storage exceeds 28-bit offset range");
    }
    if (bits == 0 || bits > 16) {
      throw std::invalid_argument(
          "packed child-ID bit width must be 1..16");
    }
    link_begin =
        begin | ((bits - 1) << PACKED_CHILD_BEGIN_BITS);
  }
  uint32_t packed_leaf_bits() const {
    return (link_begin >> PACKED_CHILD_BEGIN_BITS) + 1;
  }
  void set_packed_leaf_layout(uint32_t begin, uint32_t bits) {
    if (begin > PACKED_CHILD_BEGIN_MASK) {
      throw std::length_error(
          "packed leaf-ID storage exceeds 28-bit offset range");
    }
    if (bits == 0 || bits > 16) {
      throw std::invalid_argument(
          "packed leaf-ID bit width must be 1..16");
    }
    link_begin =
        begin | ((bits - 1) << PACKED_CHILD_BEGIN_BITS);
  }
  uint32_t child_mbb_begin() const {
    return mbb_begin & CHILD_MBB_BEGIN_MASK;
  }
  uint32_t child_mbb_bits() const {
    return (mbb_begin >> CHILD_MBB_BEGIN_BITS) + 1;
  }
  void set_child_mbb_layout(uint32_t begin, uint32_t bits) {
    if (begin > CHILD_MBB_BEGIN_MASK) {
      throw std::length_error(
          "packed child MBB storage exceeds 29-bit offset range");
    }
    if (bits == 0 || bits > 8) {
      throw std::invalid_argument("child MBB bit width must be 1..8");
    }
    mbb_begin = begin | ((bits - 1) << CHILD_MBB_BEGIN_BITS);
  }
  uint32_t leaf_mbb_begin() const {
    return mbb_begin & CHILD_MBB_BEGIN_MASK;
  }
  uint32_t leaf_mbb_bits() const {
    return (mbb_begin >> CHILD_MBB_BEGIN_BITS) + 1;
  }
  void set_leaf_mbb_layout(uint32_t begin, uint32_t bits) {
    if (begin > CHILD_MBB_BEGIN_MASK) {
      throw std::length_error(
          "packed leaf MBB storage exceeds 29-bit offset range");
    }
    if (bits == 0 || bits > 8) {
      throw std::invalid_argument("leaf MBB bit width must be 1..8");
    }
    mbb_begin = begin | ((bits - 1) << CHILD_MBB_BEGIN_BITS);
  }
  bool counts_overflow() const {
    return ((packed_counts >> BEACON_COUNT_SHIFT) &
            BEACON_COUNT_MASK) == COUNT_OVERFLOW_CODE;
  }
  uint32_t inline_link_count_or_overflow_index() const {
    return packed_counts & LINK_COUNT_MASK;
  }
  uint32_t inline_beacon_count() const {
    return (packed_counts >> BEACON_COUNT_SHIFT) &
           BEACON_COUNT_MASK;
  }
  BeaconStorage beacon_storage() const {
    return static_cast<BeaconStorage>(
        (packed_counts >> STORAGE_SHIFT) & STORAGE_MASK);
  }
  LinkStorage link_storage() const {
    return static_cast<LinkStorage>(
        packed_counts >> LINK_STORAGE_SHIFT);
  }
  void set_link_storage(LinkStorage storage) {
    packed_counts =
        (packed_counts & ~LINK_STORAGE_MASK) |
        (static_cast<uint32_t>(storage) << LINK_STORAGE_SHIFT);
  }
  void set_inline_counts(uint32_t link_count, uint32_t beacon_count,
                         BeaconStorage storage) {
    if (link_count > LINK_COUNT_MASK ||
        beacon_count >= COUNT_OVERFLOW_CODE) {
      throw std::length_error("node counts exceed inline packed range");
    }
    packed_counts = (packed_counts & LINK_STORAGE_MASK) |
                    link_count |
                    (beacon_count << BEACON_COUNT_SHIFT) |
                    (static_cast<uint32_t>(storage) << STORAGE_SHIFT);
  }
  void set_count_overflow(uint32_t overflow_index,
                          BeaconStorage storage) {
    if (overflow_index > LINK_COUNT_MASK) {
      throw std::length_error("too many node-count overflow records");
    }
    packed_counts = (packed_counts & LINK_STORAGE_MASK) |
                    overflow_index |
                    (COUNT_OVERFLOW_CODE << BEACON_COUNT_SHIFT) |
                    (static_cast<uint32_t>(storage) << STORAGE_SHIFT);
  }
};
static_assert(sizeof(WorldNodeRecord) == 12,
              "scratch world node must remain 12 bytes");

struct PackedWorldNodeLayout {
  static constexpr uint8_t IMPLICIT_PACKED_LINK_FIELDS = 1;
  static constexpr uint8_t IMPLICIT_CENTER_BEACON_STORAGE = 2;
  // Finest-layer packed leaf payloads may be addressed from a small block
  // table instead of carrying an absolute byte offset in every node record.
  // This is only valid together with IMPLICIT_PACKED_LINK_FIELDS.
  static constexpr uint8_t IMPLICIT_LINK_BEGIN = 4;
  // Center-only leaf beacons always have cardinality one, so this four-bit
  // count need not be repeated in each record.
  static constexpr uint8_t IMPLICIT_ONE_BEACON_COUNT = 8;
  // Dense child layouts have packed consecutive children, exactly three
  // packed beacons, and one shared child-MBB width.  The width may be
  // overridden by SearchGraphView's exact one-bit exception map.
  static constexpr uint8_t IMPLICIT_DENSE_CHILD_FIELDS = 16;
  // A consecutive finest layer with dense ternary leaf MBBs derives every
  // leaf count from the center/radius interval. Link storage, link begin,
  // beacon metadata, MBB begin, and MBB width are then all shard-wide facts,
  // so finest nodes require no record bytes at all.
  static constexpr uint8_t IMPLICIT_DENSE_LEAF_FIELDS = 32;

  uint8_t link_begin_bits = WorldNodeRecord::PACKED_CHILD_BEGIN_BITS;
  uint8_t mbb_begin_bits = WorldNodeRecord::CHILD_MBB_BEGIN_BITS;
  uint8_t link_count_bits = WorldNodeRecord::LINK_COUNT_BITS;
  uint8_t record_bytes = sizeof(WorldNodeRecord);
  uint8_t flags = 0;
  uint8_t implicit_packed_link_bits = 0;
  uint8_t implicit_child_mbb_bits = 0;

  static uint8_t bits_for_value(uint64_t maximum) {
    uint8_t bits = 1;
    while ((maximum >>= 1) != 0) ++bits;
    return bits;
  }

  static PackedWorldNodeLayout compact(uint64_t maximum_link_begin,
                                       uint64_t maximum_mbb_begin,
                                       uint64_t maximum_link_count,
                                       bool omit_mbb_begin = false,
                                       bool omit_packed_link_fields = false,
                                       uint8_t implicit_packed_link_bits = 0,
                                       bool omit_center_beacon_storage = false,
                                       bool omit_link_begin = false,
                                       bool omit_beacon_count = false,
                                       bool omit_dense_child_fields = false,
                                       uint8_t implicit_child_mbb_bits = 0,
                                       bool omit_dense_leaf_fields = false) {
    if (omit_mbb_begin && maximum_mbb_begin != 0) {
      throw std::invalid_argument(
          "cannot omit a nonzero world node MBB offset");
    }
    if (omit_packed_link_fields &&
        (implicit_packed_link_bits == 0 ||
         implicit_packed_link_bits > 16)) {
      throw std::invalid_argument(
          "implicit packed link width must be 1..16");
    }
    if (omit_link_begin && !omit_packed_link_fields) {
      throw std::invalid_argument(
          "implicit link begin requires implicit packed link fields");
    }
    if (omit_beacon_count && !omit_center_beacon_storage) {
      throw std::invalid_argument(
          "implicit beacon count requires center-only beacon storage");
    }
    if (omit_dense_child_fields &&
        (!omit_packed_link_fields || !omit_link_begin ||
         omit_center_beacon_storage || omit_beacon_count ||
         implicit_child_mbb_bits == 0 || implicit_child_mbb_bits > 8)) {
      throw std::invalid_argument(
          "dense child fields require implicit packed child links and MBB width");
    }
    if (omit_dense_leaf_fields) {
      if (!omit_mbb_begin || !omit_packed_link_fields ||
          !omit_center_beacon_storage || !omit_beacon_count ||
          omit_dense_child_fields || maximum_link_begin != 0 ||
          maximum_mbb_begin != 0) {
        throw std::invalid_argument(
            "dense leaf fields require implicit leaf links, beacons, and MBBs");
      }
      PackedWorldNodeLayout layout;
      layout.link_begin_bits = 0;
      layout.mbb_begin_bits = 0;
      layout.link_count_bits = 0;
      layout.record_bytes = 0;
      layout.flags = IMPLICIT_DENSE_LEAF_FIELDS;
      layout.implicit_packed_link_bits = implicit_packed_link_bits;
      layout.implicit_child_mbb_bits = 0;
      if (!layout.valid()) {
        throw std::invalid_argument("invalid dense leaf node layout");
      }
      return layout;
    }
    PackedWorldNodeLayout layout;
    layout.link_begin_bits =
        omit_link_begin ? 0 : bits_for_value(maximum_link_begin);
    layout.mbb_begin_bits =
        omit_mbb_begin ? 0 : bits_for_value(maximum_mbb_begin);
    layout.link_count_bits = bits_for_value(maximum_link_count);
    layout.flags =
        (omit_packed_link_fields ? IMPLICIT_PACKED_LINK_FIELDS : 0) |
        (omit_center_beacon_storage ? IMPLICIT_CENTER_BEACON_STORAGE : 0) |
        (omit_link_begin ? IMPLICIT_LINK_BEGIN : 0) |
        (omit_beacon_count ? IMPLICIT_ONE_BEACON_COUNT : 0) |
        (omit_dense_child_fields ? IMPLICIT_DENSE_CHILD_FIELDS : 0);
    layout.implicit_packed_link_bits =
        omit_packed_link_fields ? implicit_packed_link_bits : 0;
    layout.implicit_child_mbb_bits =
        omit_dense_child_fields ? implicit_child_mbb_bits : 0;
    if (layout.link_begin_bits >
            WorldNodeRecord::PACKED_CHILD_BEGIN_BITS ||
        layout.mbb_begin_bits >
            WorldNodeRecord::CHILD_MBB_BEGIN_BITS ||
        layout.link_count_bits > WorldNodeRecord::LINK_COUNT_BITS) {
      throw std::length_error("world node layout exceeds supported range");
    }
    const uint32_t total_bits =
        layout.link_begin_bits +
        (omit_packed_link_fields ? 0 : 4) +
        (omit_packed_link_fields ? 0 : 2) + layout.link_count_bits +
        ((omit_beacon_count || omit_dense_child_fields)
             ? 0
             : WorldNodeRecord::BEACON_COUNT_BITS) +
        ((omit_center_beacon_storage || omit_dense_child_fields) ? 0 : 2) +
        layout.mbb_begin_bits + (omit_dense_child_fields ? 0 : 3);
    layout.record_bytes = static_cast<uint8_t>((total_bits + 7) / 8);
    return layout;
  }

  bool operator==(const PackedWorldNodeLayout& other) const {
    return link_begin_bits == other.link_begin_bits &&
           mbb_begin_bits == other.mbb_begin_bits &&
           link_count_bits == other.link_count_bits &&
           record_bytes == other.record_bytes && flags == other.flags &&
           implicit_packed_link_bits == other.implicit_packed_link_bits &&
           implicit_child_mbb_bits == other.implicit_child_mbb_bits;
  }
  bool operator!=(const PackedWorldNodeLayout& other) const {
    return !(*this == other);
  }
  bool valid() const {
    if (has_implicit_dense_leaf_fields()) {
      return link_begin_bits == 0 && mbb_begin_bits == 0 &&
             link_count_bits == 0 && record_bytes == 0 &&
             flags == IMPLICIT_DENSE_LEAF_FIELDS &&
             implicit_packed_link_bits != 0 &&
             implicit_packed_link_bits <= 16 &&
             implicit_child_mbb_bits == 0;
    }
    if (link_begin_bits > WorldNodeRecord::PACKED_CHILD_BEGIN_BITS ||
        (link_begin_bits == 0 && !has_implicit_link_begin()) ||
        mbb_begin_bits > WorldNodeRecord::CHILD_MBB_BEGIN_BITS ||
        link_count_bits == 0 ||
        link_count_bits > WorldNodeRecord::LINK_COUNT_BITS ||
        (flags & ~(IMPLICIT_PACKED_LINK_FIELDS |
                   IMPLICIT_CENTER_BEACON_STORAGE |
                   IMPLICIT_LINK_BEGIN |
                   IMPLICIT_ONE_BEACON_COUNT |
                   IMPLICIT_DENSE_CHILD_FIELDS |
                   IMPLICIT_DENSE_LEAF_FIELDS)) != 0 ||
        ((flags & IMPLICIT_PACKED_LINK_FIELDS) != 0 &&
         (implicit_packed_link_bits == 0 ||
          implicit_packed_link_bits > 16)) ||
        ((flags & IMPLICIT_PACKED_LINK_FIELDS) == 0 &&
         implicit_packed_link_bits != 0) ||
        (has_implicit_link_begin() &&
         !has_implicit_packed_link_fields()) ||
        (has_implicit_one_beacon_count() &&
         !has_implicit_center_beacon_storage()) ||
        (has_implicit_dense_child_fields() &&
         (!has_implicit_packed_link_fields() ||
          !has_implicit_link_begin() ||
          has_implicit_center_beacon_storage() ||
          has_implicit_one_beacon_count() ||
          implicit_child_mbb_bits == 0 ||
          implicit_child_mbb_bits > 8)) ||
        (!has_implicit_dense_child_fields() &&
         implicit_child_mbb_bits != 0)) {
      return false;
    }
    const uint32_t total_bits =
        link_begin_bits +
        (has_implicit_packed_link_fields() ? 0 : 4) +
        (has_implicit_packed_link_fields() ? 0 : 2) + link_count_bits +
        ((has_implicit_one_beacon_count() ||
          has_implicit_dense_child_fields())
             ? 0
             : WorldNodeRecord::BEACON_COUNT_BITS) +
        ((has_implicit_center_beacon_storage() ||
          has_implicit_dense_child_fields())
             ? 0
             : 2) +
        mbb_begin_bits + (has_implicit_dense_child_fields() ? 0 : 3);
    return record_bytes == (total_bits + 7) / 8;
  }
  bool has_implicit_packed_link_fields() const {
    return (flags & IMPLICIT_PACKED_LINK_FIELDS) != 0 ||
           has_implicit_dense_leaf_fields();
  }
  bool has_implicit_center_beacon_storage() const {
    return (flags & IMPLICIT_CENTER_BEACON_STORAGE) != 0 ||
           has_implicit_dense_leaf_fields();
  }
  bool has_implicit_link_begin() const {
    return (flags & IMPLICIT_LINK_BEGIN) != 0;
  }
  bool has_implicit_one_beacon_count() const {
    return (flags & IMPLICIT_ONE_BEACON_COUNT) != 0 ||
           has_implicit_dense_leaf_fields();
  }
  bool has_implicit_dense_child_fields() const {
    return (flags & IMPLICIT_DENSE_CHILD_FIELDS) != 0;
  }
  bool has_implicit_dense_leaf_fields() const {
    return (flags & IMPLICIT_DENSE_LEAF_FIELDS) != 0;
  }
};
static_assert(sizeof(PackedWorldNodeLayout) == 7,
              "packed node layout header must remain compact");

class PackedWorldNodeRecordRef {
 public:
  using BeaconStorage = WorldNodeRecord::BeaconStorage;
  using LinkStorage = WorldNodeRecord::LinkStorage;

  PackedWorldNodeRecordRef(uint8_t* data,
                           const PackedWorldNodeLayout* layout)
      : data_(data), layout_(layout) {
    reload();
  }

 private:
  void reload() {
    low_ = 0;
    high_ = 0;
    if (layout_->record_bytes == 0) return;
    if (layout_->record_bytes == 9) {
      std::memcpy(&low_, data_, sizeof(low_));
      high_ = data_[sizeof(low_)];
      return;
    }
    const size_t low_bytes =
        std::min<size_t>(layout_->record_bytes, sizeof(low_));
    std::memcpy(&low_, data_, low_bytes);
    if (layout_->record_bytes > sizeof(low_)) {
      std::memcpy(&high_, data_ + sizeof(low_),
                  layout_->record_bytes - sizeof(low_));
    }
  }

 public:

  uint32_t link_begin_value() const {
    if (layout_->has_implicit_link_begin()) return 0;
    return read(link_begin_shift(), layout_->link_begin_bits);
  }
  void set_link_begin_value(uint32_t begin) {
    if (layout_->has_implicit_link_begin()) {
      if (begin != 0) {
        throw std::length_error(
            "packed world node link offset exceeds implicit layout");
      }
      return;
    }
    write(link_begin_shift(), layout_->link_begin_bits, begin,
          "link begin");
  }
  uint32_t child_begin() const { return link_begin_value(); }
  uint32_t leaf_begin() const { return link_begin_value(); }
  uint32_t packed_child_bits() const {
    if (layout_->has_implicit_packed_link_fields()) {
      return layout_->implicit_packed_link_bits;
    }
    return read(link_width_shift(), 4) + 1;
  }
  uint32_t packed_leaf_bits() const { return packed_child_bits(); }
  void set_packed_child_layout(uint32_t begin, uint32_t bits) {
    set_packed_link_layout(begin, bits, "child");
  }
  void set_packed_leaf_layout(uint32_t begin, uint32_t bits) {
    set_packed_link_layout(begin, bits, "leaf");
  }
  uint32_t mbb_begin_value() const {
    if (layout_->has_implicit_dense_leaf_fields()) return 0;
    if (layout_->mbb_begin_bits == 0) return 0;
    return read(mbb_begin_shift(), layout_->mbb_begin_bits);
  }
  void set_mbb_begin_value(uint32_t begin) {
    if (layout_->mbb_begin_bits == 0) {
      if (begin != 0) {
        throw std::length_error(
            "packed world node MBB offset exceeds omitted layout");
      }
      return;
    }
    write(mbb_begin_shift(), layout_->mbb_begin_bits, begin,
          "MBB begin");
  }
  uint32_t child_mbb_begin() const { return mbb_begin_value(); }
  uint32_t leaf_mbb_begin() const { return mbb_begin_value(); }
  uint32_t child_mbb_bits() const {
    if (layout_->has_implicit_dense_leaf_fields()) return 1;
    if (layout_->has_implicit_dense_child_fields()) {
      return layout_->implicit_child_mbb_bits;
    }
    return read(mbb_width_shift(), 3) + 1;
  }
  uint32_t leaf_mbb_bits() const { return child_mbb_bits(); }
  void set_child_mbb_layout(uint32_t begin, uint32_t bits) {
    set_mbb_layout(begin, bits, "child");
  }
  void set_leaf_mbb_layout(uint32_t begin, uint32_t bits) {
    set_mbb_layout(begin, bits, "leaf");
  }
  bool counts_overflow() const {
    if (layout_->has_implicit_dense_leaf_fields()) return false;
    if (layout_->has_implicit_one_beacon_count() ||
        layout_->has_implicit_dense_child_fields()) {
      return false;
    }
    return inline_beacon_count() ==
           WorldNodeRecord::COUNT_OVERFLOW_CODE;
  }
  uint32_t inline_link_count_or_overflow_index() const {
    if (layout_->has_implicit_dense_leaf_fields()) return 0;
    return read(link_count_shift(), layout_->link_count_bits);
  }
  uint32_t inline_link_count_mask() const {
    if (layout_->has_implicit_dense_leaf_fields()) {
      return WorldNodeRecord::LINK_COUNT_MASK;
    }
    return mask(layout_->link_count_bits);
  }
  uint32_t inline_beacon_count() const {
    if (layout_->has_implicit_one_beacon_count()) return 1;
    if (layout_->has_implicit_dense_child_fields()) return 3;
    return read(beacon_count_shift(),
                WorldNodeRecord::BEACON_COUNT_BITS);
  }
  BeaconStorage beacon_storage() const {
    if (layout_->has_implicit_center_beacon_storage()) {
      return BeaconStorage::ImplicitCenter;
    }
    if (layout_->has_implicit_dense_child_fields()) {
      return BeaconStorage::PackedDelta;
    }
    return static_cast<BeaconStorage>(read(beacon_storage_shift(), 2));
  }
  LinkStorage link_storage() const {
    if (layout_->has_implicit_packed_link_fields()) {
      return LinkStorage::PackedDelta;
    }
    return static_cast<LinkStorage>(read(link_storage_shift(), 2));
  }
  void set_link_storage(LinkStorage storage) {
    if (layout_->has_implicit_packed_link_fields()) {
      if (storage != LinkStorage::PackedDelta) {
        throw std::invalid_argument(
            "implicit packed link layout requires packed deltas");
      }
      return;
    }
    write(link_storage_shift(), 2, static_cast<uint32_t>(storage),
          "link storage");
  }
  void set_inline_counts(uint32_t link_count, uint32_t beacon_count,
                         BeaconStorage storage) {
    if (layout_->has_implicit_dense_leaf_fields()) {
      if (link_count > WorldNodeRecord::LINK_COUNT_MASK ||
          beacon_count != 1 || storage != BeaconStorage::ImplicitCenter) {
        throw std::invalid_argument(
            "implicit dense leaf metadata does not match node");
      }
      return;
    }
    if (link_count > inline_link_count_mask() ||
        beacon_count >= WorldNodeRecord::COUNT_OVERFLOW_CODE) {
      throw std::length_error("node counts exceed inline packed range");
    }
    write(link_count_shift(), layout_->link_count_bits, link_count,
          "link count");
    if (layout_->has_implicit_one_beacon_count() ||
        layout_->has_implicit_dense_child_fields()) {
      const uint32_t expected_beacon_count =
          layout_->has_implicit_one_beacon_count() ? 1 : 3;
      if (beacon_count != expected_beacon_count) {
        throw std::invalid_argument(
            "implicit beacon count does not match node");
      }
    } else {
      write(beacon_count_shift(), WorldNodeRecord::BEACON_COUNT_BITS,
            beacon_count, "beacon count");
    }
    if (layout_->has_implicit_center_beacon_storage()) {
      if (storage != BeaconStorage::ImplicitCenter) {
        throw std::invalid_argument(
            "implicit beacon layout requires center beacon");
      }
      return;
    }
    if (layout_->has_implicit_dense_child_fields()) {
      if (storage != BeaconStorage::PackedDelta) {
        throw std::invalid_argument(
            "dense child layout requires packed beacon storage");
      }
      return;
    }
    write(beacon_storage_shift(), 2, static_cast<uint32_t>(storage),
          "beacon storage");
  }
  void set_count_overflow(uint32_t overflow_index,
                          BeaconStorage storage) {
    if (layout_->has_implicit_one_beacon_count() ||
        layout_->has_implicit_dense_child_fields()) {
      throw std::length_error(
          "implicit beacon count cannot represent count overflow");
    }
    if (overflow_index > inline_link_count_mask()) {
      throw std::length_error("too many node-count overflow records");
    }
    write(link_count_shift(), layout_->link_count_bits, overflow_index,
          "node-count overflow index");
    write(beacon_count_shift(), WorldNodeRecord::BEACON_COUNT_BITS,
          WorldNodeRecord::COUNT_OVERFLOW_CODE, "beacon count");
    if (layout_->has_implicit_center_beacon_storage()) {
      if (storage != BeaconStorage::ImplicitCenter) {
        throw std::invalid_argument(
            "implicit beacon layout requires center beacon");
      }
      return;
    }
    write(beacon_storage_shift(), 2, static_cast<uint32_t>(storage),
          "beacon storage");
  }

 private:
  static uint32_t mask(uint32_t bits) {
    return bits == 32 ? std::numeric_limits<uint32_t>::max()
                      : (uint32_t{1} << bits) - 1;
  }
  uint32_t link_begin_shift() const { return 0; }
  uint32_t link_width_shift() const {
    return layout_->link_begin_bits;
  }
  uint32_t link_storage_shift() const {
    return link_width_shift() +
           (layout_->has_implicit_packed_link_fields() ? 0 : 4);
  }
  uint32_t link_count_shift() const {
    return link_storage_shift() +
           (layout_->has_implicit_packed_link_fields() ? 0 : 2);
  }
  uint32_t beacon_count_shift() const {
    return link_count_shift() + layout_->link_count_bits;
  }
  uint32_t beacon_storage_shift() const {
    return beacon_count_shift() +
           ((layout_->has_implicit_one_beacon_count() ||
             layout_->has_implicit_dense_child_fields())
                ? 0
                : WorldNodeRecord::BEACON_COUNT_BITS);
  }
  uint32_t mbb_begin_shift() const {
    return beacon_storage_shift() +
           ((layout_->has_implicit_center_beacon_storage() ||
             layout_->has_implicit_dense_child_fields())
                ? 0
                : 2);
  }
  uint32_t mbb_width_shift() const {
    return mbb_begin_shift() + layout_->mbb_begin_bits;
  }
  uint32_t read(uint32_t bit_offset, uint32_t bits) const {
    if (bits == 0) return 0;
    uint64_t value = 0;
    if (bit_offset >= 64) {
      value = high_ >> (bit_offset - 64);
    } else if (bit_offset + bits <= 64) {
      value = low_ >> bit_offset;
    } else {
      value = (low_ >> bit_offset) |
              (high_ << (64 - bit_offset));
    }
    return static_cast<uint32_t>(value & mask(bits));
  }
  void write(uint32_t bit_offset, uint32_t bits, uint32_t value,
             const char* field) {
    if (bits == 0) {
      if (value != 0) {
        throw std::length_error(std::string("packed world node ") + field +
                                " exceeds omitted layout range");
      }
      return;
    }
    const uint64_t field_mask = mask(bits);
    if (static_cast<uint64_t>(value) > field_mask) {
      throw std::length_error(std::string("packed world node ") + field +
                              " exceeds layout range");
    }
    // Builder finalization can hold more than one proxy to the same record.
    // Refresh before a mutation so one proxy never overwrites another field
    // with an older cached word. Query proxies remain a single-load snapshot.
    reload();
    if (bit_offset >= 64) {
      const uint32_t shift = bit_offset - 64;
      const uint64_t shifted_mask = field_mask << shift;
      high_ = (high_ & ~shifted_mask) |
              (static_cast<uint64_t>(value) << shift);
    } else if (bit_offset + bits <= 64) {
      const uint64_t shifted_mask = field_mask << bit_offset;
      low_ = (low_ & ~shifted_mask) |
             (static_cast<uint64_t>(value) << bit_offset);
    } else {
      const uint32_t low_bits = 64 - bit_offset;
      const uint64_t low_mask = (uint64_t{1} << low_bits) - 1;
      low_ = (low_ & ~(low_mask << bit_offset)) |
             ((static_cast<uint64_t>(value) & low_mask) << bit_offset);
      const uint32_t high_bits = bits - low_bits;
      const uint64_t high_mask = (uint64_t{1} << high_bits) - 1;
      high_ = (high_ & ~high_mask) |
              (static_cast<uint64_t>(value) >> low_bits);
    }
    const size_t low_bytes =
        std::min<size_t>(layout_->record_bytes, sizeof(low_));
    std::memcpy(data_, &low_, low_bytes);
    if (layout_->record_bytes > sizeof(low_)) {
      std::memcpy(data_ + sizeof(low_), &high_,
                  layout_->record_bytes - sizeof(low_));
    }
  }
  void set_packed_link_layout(uint32_t begin, uint32_t bits,
                              const char* kind) {
    if (bits == 0 || bits > 16) {
      throw std::invalid_argument(
          std::string("packed ") + kind + "-ID bit width must be 1..16");
    }
    set_link_begin_value(begin);
    if (layout_->has_implicit_packed_link_fields()) {
      if (bits != layout_->implicit_packed_link_bits) {
        throw std::invalid_argument(
            "packed link width differs from implicit layout");
      }
      return;
    }
    write(link_width_shift(), 4, bits - 1, "link width");
  }
  void set_mbb_layout(uint32_t begin, uint32_t bits,
                      const char* kind) {
    if (bits == 0 || bits > 8) {
      throw std::invalid_argument(
          std::string(kind) + " MBB bit width must be 1..8");
    }
    if (layout_->has_implicit_dense_leaf_fields()) {
      if (begin != 0 || bits != 1) {
        throw std::invalid_argument(
            "implicit dense leaf MBB layout must be one bit at offset zero");
      }
      return;
    }
    set_mbb_begin_value(begin);
    if (layout_->has_implicit_dense_child_fields()) return;
    write(mbb_width_shift(), 3, bits - 1, "MBB width");
  }

  uint8_t* data_ = nullptr;
  const PackedWorldNodeLayout* layout_ = nullptr;
  uint64_t low_ = 0;
  uint64_t high_ = 0;
};
static_assert(sizeof(PackedWorldNodeRecordRef) <= 32,
              "packed node proxy must remain at most 32 bytes");

class PackedWorldNodeArray {
 public:
  size_t size() const { return count_; }
  bool empty() const { return count_ == 0; }
  const uint8_t* data() const { return bytes_.data(); }
  bool is_mapped() const { return bytes_.is_mapped(); }
  const PackedWorldNodeLayout& child_layout() const {
    return child_layout_;
  }
  const PackedWorldNodeLayout& leaf_layout() const {
    return leaf_layout_;
  }
  const FinalArray<uint8_t>& bytes() const { return bytes_; }
  FinalArray<uint8_t>& bytes() { return bytes_; }
  uint32_t finest_node_begin() const { return finest_node_begin_; }

  PackedWorldNodeRecordRef operator[](size_t index) const {
    if (index < finest_node_begin_) {
      if (child_layout_.record_bytes == 0) {
        return PackedWorldNodeRecordRef(nullptr, &child_layout_);
      }
      return PackedWorldNodeRecordRef(
          const_cast<uint8_t*>(bytes_.data()) +
              index * child_layout_.record_bytes,
          &child_layout_);
    }
    if (leaf_layout_.record_bytes == 0) {
      return PackedWorldNodeRecordRef(nullptr, &leaf_layout_);
    }
    return PackedWorldNodeRecordRef(
        const_cast<uint8_t*>(bytes_.data()) + child_record_bytes() +
            (index - finest_node_begin_) * leaf_layout_.record_bytes,
        &leaf_layout_);
  }
  PackedWorldNodeRecordRef at(size_t index) const {
    if (index >= count_) throw std::out_of_range("packed world node index");
    return (*this)[index];
  }
  const uint8_t* record_data(size_t index) const {
    if (index < finest_node_begin_) {
      if (child_layout_.record_bytes == 0) return nullptr;
      return bytes_.data() + index * child_layout_.record_bytes;
    }
    if (leaf_layout_.record_bytes == 0) return nullptr;
    return bytes_.data() + child_record_bytes() +
           (index - finest_node_begin_) * leaf_layout_.record_bytes;
  }

  void assign(size_t count, const WorldNodeRecord&) {
    initialize(count, PackedWorldNodeLayout{});
  }
  void resize(size_t count) {
    initialize(count, PackedWorldNodeLayout{});
  }
  void initialize(size_t count, PackedWorldNodeLayout layout) {
    initialize(count, layout, layout, count);
  }
  void initialize(size_t count, PackedWorldNodeLayout child_layout,
                  PackedWorldNodeLayout leaf_layout,
                  size_t finest_node_begin) {
    if (!child_layout.valid() || !leaf_layout.valid() ||
        finest_node_begin > count ||
        count > std::numeric_limits<NodeId>::max()) {
      throw std::length_error("packed world node array is too large");
    }
    count_ = count;
    child_layout_ = child_layout;
    leaf_layout_ = leaf_layout;
    finest_node_begin_ = static_cast<uint32_t>(finest_node_begin);
    const size_t byte_count = checked_byte_count();
    bytes_.assign(byte_count, uint8_t{0});
  }
  void set_loaded(size_t count, PackedWorldNodeLayout child_layout,
                  PackedWorldNodeLayout leaf_layout,
                  uint32_t finest_node_begin,
                  FinalArray<uint8_t> bytes) {
    if (!child_layout.valid() || !leaf_layout.valid() ||
        finest_node_begin > count ||
        count > std::numeric_limits<NodeId>::max()) {
      throw std::runtime_error("packed world node byte count is invalid");
    }
    count_ = count;
    child_layout_ = child_layout;
    leaf_layout_ = leaf_layout;
    finest_node_begin_ = finest_node_begin;
    if (bytes.size() != checked_byte_count()) {
      throw std::runtime_error("packed world node byte count is invalid");
    }
    bytes_ = std::move(bytes);
  }

 private:
  size_t child_record_bytes() const {
    return static_cast<size_t>(finest_node_begin_) *
           child_layout_.record_bytes;
  }
  size_t checked_byte_count() const {
    const size_t child_bytes = child_record_bytes();
    const size_t leaf_count = count_ - finest_node_begin_;
    if (finest_node_begin_ != 0 &&
        child_bytes / finest_node_begin_ !=
            child_layout_.record_bytes) {
      throw std::length_error("packed child node array is too large");
    }
    if (leaf_layout_.record_bytes != 0 &&
        leaf_count >
            (std::numeric_limits<size_t>::max() - child_bytes) /
                leaf_layout_.record_bytes) {
      throw std::length_error("packed leaf node array is too large");
    }
    return child_bytes + leaf_count * leaf_layout_.record_bytes;
  }

  size_t count_ = 0;
  PackedWorldNodeLayout child_layout_;
  PackedWorldNodeLayout leaf_layout_;
  uint32_t finest_node_begin_ = 0;
  FinalArray<uint8_t> bytes_;
};

template <typename T>
class BuildArrayView {
 public:
  BuildArrayView() = default;
  BuildArrayView(const T* data, size_t size) : data_(data), size_(size) {}
  BuildArrayView(const std::vector<T>& values)
      : data_(values.data()), size_(values.size()) {}

  const T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const T& operator[](size_t index) const { return data_[index]; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ ? data_ + size_ : data_; }

 private:
  const T* data_ = nullptr;
  size_t size_ = 0;
};

// Build-only vectors contain trivial integer values. A 32-bit size/capacity
// pair keeps each mutable handle at 16 bytes, and realloc can grow it in place
// without std::vector's allocate-copy-free peak.
template <typename T>
class CompactBuildVector {
 public:
  static_assert(std::is_trivially_copyable_v<T> &&
                    std::is_trivially_destructible_v<T>,
                "compact build vectors require trivial values");

  CompactBuildVector() = default;
  CompactBuildVector(const CompactBuildVector& other) { copy_from(other); }
  CompactBuildVector& operator=(const CompactBuildVector& other) {
    if (this != &other) copy_from(other);
    return *this;
  }
  CompactBuildVector(CompactBuildVector&& other) noexcept {
    move_from(std::move(other));
  }
  CompactBuildVector& operator=(CompactBuildVector&& other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }
  ~CompactBuildVector() { std::free(data_); }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }
  T* begin() { return data_; }
  const T* begin() const { return data_; }
  T* end() { return data_ ? data_ + size_ : data_; }
  const T* end() const { return data_ ? data_ + size_ : data_; }
  operator BuildArrayView<T>() const {
    return BuildArrayView<T>(data_, size_);
  }

  void clear() { size_ = 0; }
  void truncate(size_t requested_size) {
    if (requested_size > size_) {
      throw std::length_error("compact build-vector truncation overflow");
    }
    size_ = static_cast<uint32_t>(requested_size);
  }
  void release() {
    std::free(data_);
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }
  void reserve(size_t requested_capacity) {
    if (requested_capacity <= capacity_) return;
    if (requested_capacity > std::numeric_limits<uint32_t>::max() ||
        requested_capacity >
            std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::length_error("compact build-vector capacity overflow");
    }
    void* allocation =
        std::realloc(data_, requested_capacity * sizeof(T));
    if (!allocation) throw std::bad_alloc();
    data_ = static_cast<T*>(allocation);
    capacity_ = static_cast<uint32_t>(requested_capacity);
  }
  void push_back(T value) {
    if (size_ == capacity_) {
      size_t grown = capacity_ == 0
                         ? 4
                         : static_cast<size_t>(capacity_) +
                               capacity_ / 2 + 1;
      if (grown > std::numeric_limits<uint32_t>::max()) {
        grown = static_cast<size_t>(size_) + 1;
      }
      if (grown <= size_) grown = static_cast<size_t>(size_) + 1;
      reserve(grown);
    }
    data_[size_++] = value;
  }
  void assign(size_t count, T value) {
    reserve(count);
    if (count != 0) std::fill_n(data_, count, value);
    size_ = static_cast<uint32_t>(count);
  }

 private:
  void copy_from(const CompactBuildVector& other) {
    reserve(other.size_);
    if (other.size_ != 0) {
      std::memcpy(data_, other.data_, other.size_ * sizeof(T));
    }
    size_ = other.size_;
  }
  void move_from(CompactBuildVector&& other) {
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  T* data_ = nullptr;
  uint32_t size_ = 0;
  uint32_t capacity_ = 0;
};
static_assert(sizeof(CompactBuildVector<uint32_t>) == 16,
              "compact build-vector handle must remain 16 bytes");

// Mutable construction record. It intentionally contains only integer
// references: no WorldNode objects are allocated while building the hierarchy.
// The per-node vectors are flattened into SearchGraphView after all build
// phases have produced the same node/edge sets as the original algorithm.
struct BuildWorldNodeRecord {
  LeafId center_sequence_id = INVALID_LEAF_ID;
  uint32_t geometry_index = INVALID_NODE_ID;

  // Phase 1-3 child NodeIds. Finest-layer nodes reuse the same uint32_t
  // storage for LeafIds after their child list is cleared.
  CompactBuildVector<NodeId> child_or_leaf_ids;
};
static_assert(sizeof(BuildWorldNodeRecord) == 24,
              "mutable build node must remain 24 bytes");

// Geometry exists only for primary nodes. Keeping these vectors out of every
// expanded-layer node avoids two empty vector headers on auxiliary nodes.
struct BuildNodeGeometry {
  CompactBuildVector<LeafId> beacon_ids;
  // Bit-packed, dimension-major child MBB distances. Finest nodes use the
  // same storage for one packed distance per leaf.
  CompactBuildVector<uint8_t> link_beacon_dists;
};
static_assert(sizeof(BuildNodeGeometry) == 32,
              "primary-node build geometry must remain 32 bytes");

// One exact periodic center-ID layer. For local node i, the center is
// first + (i / period) * cycle_span + offsets[offset_begin + i % period].
// A zero period selects the ordinary block-base/delta representation.
struct PeriodicCenterLayer {
  enum Pattern : uint8_t {
    Generic = 0,
    Linear16 = 1,
    DefaultPeriod3 = 2,
    DefaultPeriod7 = 3,
  };

  LeafId first = 0;
  LeafId cycle_span = 0;
  uint16_t offset_begin = 0;
  uint8_t period = 0;
  uint8_t pattern = Generic;

  bool operator==(const PeriodicCenterLayer& other) const {
    return first == other.first && cycle_span == other.cycle_span &&
           offset_begin == other.offset_begin &&
           period == other.period && pattern == other.pattern;
  }
};
static_assert(sizeof(PeriodicCenterLayer) == 12,
              "periodic center metadata must remain 12 bytes");

struct SearchGraphView {
  static constexpr uint32_t COARSE_CHILD_MBB_BIN_WIDTH = 12;
  static constexpr uint32_t FINE_CHILD_MBB_BIN_WIDTH = 6;
  // Packed child widths use 1..15 bits for ordinary bit-packed deltas.  The
  // otherwise non-beneficial 16-bit width is a format tag for an exact dense
  // child NodeId range: the base prefix is followed by no payload and child i
  // is base + i.  A generic 16-bit packed payload can never beat Delta16, so
  // reserving this value loses no chosen representation.
  static constexpr uint32_t CONTIGUOUS_CHILD_RANGE_BITS = 16;
  static constexpr size_t CONTIGUOUS_CHILD_OFFSET_TABLE_SIZE = 256;
  inline static const std::array<
      uint8_t, CONTIGUOUS_CHILD_OFFSET_TABLE_SIZE>
      contiguous_child_offset_table = [] {
        std::array<uint8_t, CONTIGUOUS_CHILD_OFFSET_TABLE_SIZE> table{};
        for (size_t offset = 0; offset < table.size(); ++offset) {
          table[offset] = static_cast<uint8_t>(offset);
        }
        return table;
      }();
  // Child MBB values in [0, 10] use the 7-bit width as a format tag. Each
  // beacon-pair distance is stored once, then every child pair is ranked only
  // among base-11 states permitted by the triangle inequality. An odd final
  // dimension remains one exact 4-bit digit.
  static constexpr uint32_t PAIRED_BASE11_CHILD_MBB_BITS = 7;

  static constexpr uint32_t child_mbb_quantization_error(
      uint32_t bin_width) {
    return bin_width / 2;
  }

  // Canonical array representation. NodeId and LeafId are positions in
  // node_records and sequences respectively.
  PackedWorldNodeArray node_records;
  // Center IDs increase within each layer. Every aligned 16-node block stores
  // one absolute base plus fixed-width exact deltas; layer-local widths keep
  // random access O(1) without paying the shard-wide LeafId width per node.
  static constexpr size_t CENTER_ID_BLOCK_SIZE = 8;
  bool center_id_block_bases_16bit = false;
  FinalArray<uint16_t> center_id_block_bases16;
  FinalArray<LeafId> center_id_block_bases;
  FinalArray<uint8_t> center_id_block_deltas;
  std::vector<uint32_t> center_id_block_begins;
  std::vector<uint32_t> center_id_delta_begins;
  std::vector<uint8_t> center_id_delta_bits;
  // Layers whose exact adjacent center-ID deltas repeat store one cycle of
  // byte offsets instead of block bases and bit-packed deltas. The builder
  // verifies every reconstructed center before enabling a layer.
  FinalArray<PeriodicCenterLayer> periodic_center_layers;
  FinalArray<uint8_t> periodic_center_offsets;
  FinalArray<NodeCountOverflowRecord> node_count_overflows;
  SequenceStore sequences;
  std::vector<uint32_t> layer_begin;
  std::vector<uint32_t> layer_end;

  // Base-relative child payloads start with one minimum whole-byte forward
  // base delta from node_id + 1, followed by fixed-width 1..16-bit offsets.
  // Keeping the base adjacent to its payload preserves query locality while
  // avoiding the former 32-bit base in almost every parent segment.
  uint8_t child_base_forward_delta_bytes = 0;
  // When every contiguous child range's base lies within one byte of its
  // 32-node block minimum, keep that minimum once and retain one exact byte
  // per node. This is only a representation change: child IDs remain exact.
  static constexpr uint8_t CHILD_BASE_BLOCK_SIZE = 32;
  bool compact_child_base_blocks = false;
  FinalArray<NodeId> child_base_block_bases;
  // When every non-finest node in a shard has one exact consecutive child
  // range, their base prefixes are a fixed-width dense array.  The prefix for
  // node i is then at i * child_base_byte_count(), so node records need not
  // retain a child-payload offset.  Other shards retain the per-node offset.
  bool implicit_contiguous_child_ranges = false;
  FinalArray<uint8_t> child_id_base_deltas8;
  FinalArray<uint16_t> child_id_deltas16;
  FinalArray<NodeId> child_ids;
  // Signed byte deltas and exact per-node ZigZag-packed deltas share this
  // byte array; LinkStorage distinguishes their interpretation.
  FinalArray<int8_t> leaf_id_deltas8;
  FinalArray<int16_t> leaf_id_deltas16;
  FinalArray<LeafId> leaf_ids;
  // When the finest layout elides its per-node link begin, one exact byte
  // offset per small leaf-node block recovers it. The short in-block scan only
  // reads packed node metadata and keeps the leaf payload contiguous.
  static constexpr uint8_t LEAF_LINK_BEGIN_BLOCK_SIZE = 8;
  FinalArray<uint32_t> leaf_link_begin_blocks;

  // If every finest world's leaf IDs are one exact center-relative consecutive
  // interval, the interval start is reconstructed from this shard-wide radius
  // and no per-leaf ID payload is stored.  The builder enables this only after
  // checking every leaf list, including clipped intervals at both endpoints.
  bool implicit_consecutive_leaf_ids = false;
  LeafId implicit_consecutive_leaf_radius = 0;

  // Each child-center-to-beacon distance d is stored as floor(d / 12) above
  // the last non-finest layer and floor(d / 6) in that final child-world
  // transition. Search decodes the bin midpoint and widens every pruning
  // threshold by the matching error bound, so the lossy representation can
  // only retain extra children, never remove one.
  FinalArray<uint8_t> child_beacon_dists;
  // Explicit beacons only exist above the finest layer. Those NodeIds form a
  // dense prefix, so this side array needs no per-entry node identifier.
  // One shard-local fixed width keeps exact O(1) random access while avoiding
  // a 32-bit offset for every non-finest node.
  static constexpr uint8_t DEFAULT_BEACON_BEGIN_BLOCK_SIZE = 4;
  static constexpr size_t BEACON_BEGIN_BLOCK_SIZE =
      DEFAULT_BEACON_BEGIN_BLOCK_SIZE;
  FinalArray<uint8_t> beacon_begin_blocks;
  uint8_t beacon_begin_block_size = DEFAULT_BEACON_BEGIN_BLOCK_SIZE;
  uint8_t beacon_begin_base_bits = 0;
  uint8_t beacon_begin_delta_bits = 0;
  uint8_t beacon_delta_bits = 16;
  FinalArray<uint8_t> beacon_id_bytes;
  // When every non-leaf world has one of at most 16 exact three-beacon
  // center-relative patterns, a dense 4-bit pattern code replaces both the
  // variable-width beacon payload and its offset table.
  bool dense_beacon_patterns = false;
  uint8_t dense_beacon_pattern_count = 0;
  std::array<int8_t, 16 * 3> dense_beacon_pattern_deltas{};
  // Shards that exceed either the 16-pattern or signed-byte limit retain the
  // same O(1) pattern lookup with minimum-width codes and signed-16 deltas.
  // The common dense mode above remains unchanged on its hot path.
  bool wide_beacon_patterns = false;
  uint16_t wide_beacon_pattern_count = 0;
  uint8_t wide_beacon_pattern_bits = 0;
  FinalArray<int16_t> wide_beacon_pattern_deltas;
  // A dense child layout fixes the common MBB width in the shared node layout;
  // this bit map marks exact nodes that use the one alternate width.
  bool implicit_child_mbb_widths = false;
  uint8_t implicit_child_mbb_exception_bits = 0;
  FinalArray<uint8_t> child_mbb_width_exceptions;

  // Exact child MBB payloads with identical dimensions are interned within a
  // shard. Node records continue to point directly at their shared payload,
  // so query decoding and its conservative pruning bound are unchanged.
  bool interned_child_mbb_payloads = false;
  // In shards whose packed leaf-ID byte stream and leaf-MBB byte stream have
  // the same per-node starts, a leaf MBB begin is exactly leaf_begin().  The
  // node record then omits the redundant MBB offset.  The builder enables
  // this only after checking every finest-layer node.
  bool implicit_leaf_mbb_offsets = false;
  // When every leaf world has at most five links and every exact leaf-to-center
  // distance belongs to one shard-wide three-value alphabet, one base-3 byte
  // per leaf world replaces the individually padded bit streams.
  bool dense_leaf_mbb_ternary = false;
  std::array<uint8_t, 3> dense_leaf_mbb_values{};
  // When every consecutive finest-layer leaf has exact distance
  // 2 * abs(leaf_id - center_id), the dense ternary byte is derived from the
  // clipped interval and omitted. The builder verifies every packed row before
  // enabling this representation.
  bool implicit_shift_leaf_mbb = false;
  FinalArray<uint8_t> leaf_beacon_dists;

  bool has_beacon_patterns() const {
    return dense_beacon_patterns || wide_beacon_patterns;
  }

  size_t beacon_pattern_code_byte_count(size_t node_count) const {
    if (dense_beacon_patterns) return (node_count + 1) / 2;
    if (wide_beacon_patterns) {
      return (node_count * wide_beacon_pattern_bits + 7) / 8;
    }
    return 0;
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline uint16_t wide_beacon_pattern_code(NodeId node_id) const {
    const uint32_t bits = wide_beacon_pattern_bits;
    const size_t bit_offset = static_cast<size_t>(node_id) * bits;
    const size_t byte_offset = bit_offset >> 3;
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    uint32_t word = beacon_id_bytes[byte_offset];
    if (shift + bits > 8) {
      word |= static_cast<uint32_t>(beacon_id_bytes[byte_offset + 1]) << 8;
    }
    if (shift + bits > 16) {
      word |= static_cast<uint32_t>(beacon_id_bytes[byte_offset + 2]) << 16;
    }
    const uint32_t mask = (uint32_t{1} << bits) - 1;
    return static_cast<uint16_t>((word >> shift) & mask);
  }

  void initialize_leaf_link_begin_blocks(size_t finest_node_count) {
    if (finest_node_count == 0) {
      leaf_link_begin_blocks.clear();
      return;
    }
    const size_t block_count =
        (finest_node_count + LEAF_LINK_BEGIN_BLOCK_SIZE - 1) /
        LEAF_LINK_BEGIN_BLOCK_SIZE;
    leaf_link_begin_blocks.assign(block_count, 0);
  }
  bool leaf_link_begins_valid() const {
    const auto& layout = node_records.leaf_layout();
    const size_t finest_count =
        node_records.size() - node_records.finest_node_begin();
    if (!layout.has_implicit_link_begin()) {
      return leaf_link_begin_blocks.empty();
    }
    return finest_count != 0 &&
           leaf_link_begin_blocks.size() ==
               (finest_count + LEAF_LINK_BEGIN_BLOCK_SIZE - 1) /
               LEAF_LINK_BEGIN_BLOCK_SIZE;
  }
  void set_leaf_link_begin(NodeId node_id, uint32_t begin) {
    const uint32_t finest_begin = node_records.finest_node_begin();
    if (!node_records.leaf_layout().has_implicit_link_begin() ||
        node_id < finest_begin ||
        (node_id - finest_begin) % LEAF_LINK_BEGIN_BLOCK_SIZE != 0) {
      throw std::invalid_argument("invalid implicit leaf link begin");
    }
    leaf_link_begin_blocks[(node_id - finest_begin) /
                           LEAF_LINK_BEGIN_BLOCK_SIZE] = begin;
  }
  uint32_t leaf_link_begin(
      NodeId node_id, const PackedWorldNodeRecordRef& node) const {
    if (!node_records.leaf_layout().has_implicit_link_begin()) {
      return node.leaf_begin();
    }
    const uint32_t finest_begin = node_records.finest_node_begin();
    if (node_id < finest_begin ||
        leaf_link_begin_blocks.empty()) {
      throw std::runtime_error("implicit leaf link begin is invalid");
    }
    const uint32_t relative = node_id - finest_begin;
    const uint32_t block_begin =
        relative - relative % LEAF_LINK_BEGIN_BLOCK_SIZE;
    uint32_t begin = leaf_link_begin_blocks[
        block_begin / LEAF_LINK_BEGIN_BLOCK_SIZE];
    const uint32_t bits =
        node_records.leaf_layout().implicit_packed_link_bits;
    for (uint32_t prior = block_begin; prior < relative; ++prior) {
      const auto previous = node_records[finest_begin + prior];
      begin += static_cast<uint32_t>(
          (static_cast<uint64_t>(link_count(previous)) * bits + 7) / 8);
    }
    return begin;
  }
  uint32_t leaf_link_begin(NodeId node_id) const {
    const auto node = node_records[node_id];
    return leaf_link_begin(node_id, node);
  }

  void initialize_beacon_begins(
      size_t non_finest_node_count, uint64_t maximum_begin,
      uint64_t maximum_block_delta,
      uint8_t block_size = DEFAULT_BEACON_BEGIN_BLOCK_SIZE) {
    if (block_size < 2 || block_size > 32 ||
        (block_size & (block_size - 1)) != 0) {
      throw std::invalid_argument("invalid beacon begin block size");
    }
    beacon_begin_block_size = block_size;
    beacon_begin_base_bits = 0;
    beacon_begin_delta_bits = 0;
    if (has_beacon_patterns()) {
      beacon_begin_blocks.clear();
      return;
    }
    if (non_finest_node_count == 0) {
      beacon_begin_blocks.clear();
      return;
    }
    beacon_begin_base_bits =
        PackedWorldNodeLayout::bits_for_value(maximum_begin);
    beacon_begin_delta_bits =
        PackedWorldNodeLayout::bits_for_value(maximum_block_delta);
    const size_t block_count =
        (non_finest_node_count + beacon_begin_block_size - 1) /
        beacon_begin_block_size;
    const size_t record_bits =
        beacon_begin_base_bits +
        (beacon_begin_block_size - 1) * beacon_begin_delta_bits;
    const size_t record_bytes = (record_bits + 7) / 8;
    if (record_bytes == 0 ||
        block_count > std::numeric_limits<size_t>::max() / record_bytes) {
      throw std::length_error("beacon begin block array is too large");
    }
    beacon_begin_blocks.assign(block_count * record_bytes, uint8_t{0});
  }

  bool beacon_begins_valid(size_t non_finest_node_count) const {
    if (dense_beacon_patterns) {
      return non_finest_node_count != 0 &&
             !wide_beacon_patterns &&
             dense_beacon_pattern_count != 0 &&
             dense_beacon_pattern_count <= 16 &&
             wide_beacon_pattern_count == 0 &&
             wide_beacon_pattern_bits == 0 &&
             wide_beacon_pattern_deltas.empty() &&
             beacon_begin_base_bits == 0 && beacon_begin_delta_bits == 0 &&
             beacon_begin_blocks.empty() &&
             beacon_id_bytes.size() ==
                 beacon_pattern_code_byte_count(non_finest_node_count);
    }
    if (wide_beacon_patterns) {
      const uint8_t expected_bits =
          wide_beacon_pattern_count == 0
              ? 0
              : PackedWorldNodeLayout::bits_for_value(
                    wide_beacon_pattern_count - 1);
      return non_finest_node_count != 0 &&
             dense_beacon_pattern_count == 0 &&
             wide_beacon_pattern_count != 0 &&
             wide_beacon_pattern_bits == expected_bits &&
             wide_beacon_pattern_bits <= 16 &&
             wide_beacon_pattern_deltas.size() ==
                 static_cast<size_t>(wide_beacon_pattern_count) * 3 &&
             beacon_begin_base_bits == 0 && beacon_begin_delta_bits == 0 &&
             beacon_begin_blocks.empty() &&
             beacon_id_bytes.size() ==
                 beacon_pattern_code_byte_count(non_finest_node_count);
    }
    if (dense_beacon_pattern_count != 0 ||
        wide_beacon_pattern_count != 0 || wide_beacon_pattern_bits != 0 ||
        !wide_beacon_pattern_deltas.empty()) {
      return false;
    }
    if (non_finest_node_count == 0) {
      return beacon_begin_block_size >= 2 &&
             beacon_begin_block_size <= 32 &&
             (beacon_begin_block_size &
              (beacon_begin_block_size - 1)) == 0 &&
             beacon_begin_base_bits == 0 &&
             beacon_begin_delta_bits == 0 &&
             beacon_begin_blocks.empty();
    }
    if (beacon_begin_block_size < 2 || beacon_begin_block_size > 32 ||
        (beacon_begin_block_size &
         (beacon_begin_block_size - 1)) != 0 ||
        beacon_begin_base_bits == 0 || beacon_begin_base_bits > 32 ||
        beacon_begin_delta_bits == 0 || beacon_begin_delta_bits > 32) {
      return false;
    }
    const size_t block_count =
        (non_finest_node_count + beacon_begin_block_size - 1) /
        beacon_begin_block_size;
    const size_t record_bits =
        beacon_begin_base_bits +
        (beacon_begin_block_size - 1) * beacon_begin_delta_bits;
    const size_t record_bytes = (record_bits + 7) / 8;
    return record_bytes != 0 &&
           block_count <= std::numeric_limits<size_t>::max() / record_bytes &&
           beacon_begin_blocks.size() == block_count * record_bytes;
  }

  void set_beacon_begin(
      NodeId node_id, uint32_t begin, uint32_t block_base) {
    if (has_beacon_patterns()) {
      throw std::logic_error("beacon patterns have no begin offsets");
    }
    if (beacon_begin_base_bits == 0 || beacon_begin_base_bits > 32 ||
        beacon_begin_delta_bits == 0 || beacon_begin_delta_bits > 32 ||
        begin < block_base) {
      throw std::logic_error("beacon begin array is not initialized");
    }
    const uint64_t base_mask =
        beacon_begin_base_bits == 32
            ? std::numeric_limits<uint32_t>::max()
            : (uint64_t{1} << beacon_begin_base_bits) - 1;
    const uint64_t delta_mask =
        beacon_begin_delta_bits == 32
            ? std::numeric_limits<uint32_t>::max()
            : (uint64_t{1} << beacon_begin_delta_bits) - 1;
    const uint32_t delta = begin - block_base;
    if (block_base > base_mask || delta > delta_mask) {
      throw std::length_error("beacon begin exceeds packed width");
    }
    const size_t record_bits =
        beacon_begin_base_bits +
        (beacon_begin_block_size - 1) * beacon_begin_delta_bits;
    const size_t record_bytes = (record_bits + 7) / 8;
    const size_t byte_offset =
        (node_id / beacon_begin_block_size) * record_bytes;
    const size_t in_block = node_id % beacon_begin_block_size;
    const uint32_t shift =
        in_block == 0
            ? 0
            : beacon_begin_base_bits +
                  (in_block - 1) * beacon_begin_delta_bits;
    const uint32_t bits =
        in_block == 0 ? beacon_begin_base_bits : beacon_begin_delta_bits;
    const uint32_t value = in_block == 0 ? block_base : delta;
    for (size_t bit = 0; bit < bits;) {
      const size_t absolute_bit = static_cast<size_t>(shift) + bit;
      const size_t byte = absolute_bit >> 3;
      const uint32_t byte_shift =
          static_cast<uint32_t>(absolute_bit & 7);
      const uint32_t take = std::min<uint32_t>(
          8 - byte_shift, bits - static_cast<uint32_t>(bit));
      const uint32_t mask = (uint32_t{1} << take) - 1;
      beacon_begin_blocks[byte_offset + byte] |= static_cast<uint8_t>(
          ((value >> bit) & mask) << byte_shift);
      bit += take;
    }
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline uint32_t beacon_begin(NodeId node_id) const {
    if (has_beacon_patterns()) return 0;
    const size_t record_bits =
        beacon_begin_base_bits +
        (beacon_begin_block_size - 1) * beacon_begin_delta_bits;
    const size_t record_bytes = (record_bits + 7) / 8;
    const size_t byte_offset =
        (node_id / beacon_begin_block_size) * record_bytes;
    const size_t in_block = node_id % beacon_begin_block_size;
    const uint32_t shift = in_block == 0 ? 0 :
        beacon_begin_base_bits +
        (in_block - 1) * beacon_begin_delta_bits;
    const uint32_t bits =
        in_block == 0 ? beacon_begin_base_bits : beacon_begin_delta_bits;

    // Candidate block sizes are selected so common layouts remain at most one
    // machine-word load; wider records retain the exact generic reader below.
    if (record_bytes <= sizeof(uint64_t)) {
      uint64_t word = 0;
      std::memcpy(
          &word, beacon_begin_blocks.data() + byte_offset, record_bytes);
      const uint64_t mask = bits == 32
          ? std::numeric_limits<uint32_t>::max()
          : (uint64_t{1} << bits) - 1;
      const uint32_t value = static_cast<uint32_t>((word >> shift) & mask);
      if (in_block == 0) return value;
      const uint64_t base_mask = beacon_begin_base_bits == 32
          ? std::numeric_limits<uint32_t>::max()
          : (uint64_t{1} << beacon_begin_base_bits) - 1;
      return static_cast<uint32_t>(word & base_mask) + value;
    }

    const auto read_field = [&](uint32_t field_shift, uint32_t field_bits) {
      uint32_t value = 0;
      for (uint32_t bit = 0; bit < field_bits;) {
        const size_t absolute_bit = static_cast<size_t>(field_shift) + bit;
        const size_t byte = absolute_bit >> 3;
        const uint32_t byte_shift =
            static_cast<uint32_t>(absolute_bit & 7);
        const uint32_t take = std::min<uint32_t>(
            8 - byte_shift, field_bits - bit);
        const uint32_t mask = (uint32_t{1} << take) - 1;
        value |= static_cast<uint32_t>(
            (beacon_begin_blocks[byte_offset + byte] >> byte_shift) & mask)
            << bit;
        bit += take;
      }
      return value;
    };
    const uint32_t base = read_field(0, beacon_begin_base_bits);
    if (in_block == 0) return base;
    return base + read_field(shift, bits);
  }

  void initialize_center_sequence_ids(
      const std::vector<uint8_t>& bits_by_layer,
      LeafId maximum_center_id = std::numeric_limits<LeafId>::max(),
      std::vector<PeriodicCenterLayer> periodic_layers = {},
      std::vector<uint8_t> periodic_offsets = {}) {
    const size_t layer_count = layer_begin.size();
    if (layer_end.size() != layer_count ||
        bits_by_layer.size() != layer_count) {
      throw std::invalid_argument(
          "center ID layer metadata is inconsistent");
    }
    if ((!periodic_layers.empty() &&
         periodic_layers.size() != layer_count) ||
        (periodic_layers.empty() && !periodic_offsets.empty())) {
      throw std::invalid_argument(
          "periodic center ID metadata is inconsistent");
    }
    if (periodic_layers.empty()) {
      periodic_center_layers.clear();
      periodic_center_offsets.clear();
    } else {
      periodic_center_layers.set_owned(std::move(periodic_layers));
      periodic_center_offsets.set_owned(std::move(periodic_offsets));
    }
    center_id_block_begins.assign(layer_count, 0);
    center_id_delta_begins.assign(layer_count, 0);
    center_id_delta_bits = bits_by_layer;
    size_t total_blocks = 0;
    size_t total_delta_bytes = 0;
    for (size_t layer = 0; layer < layer_count; ++layer) {
      if (layer_end[layer] < layer_begin[layer] ||
          bits_by_layer[layer] > 32) {
        throw std::invalid_argument("invalid center ID layer layout");
      }
      center_id_block_begins[layer] = static_cast<uint32_t>(total_blocks);
      center_id_delta_begins[layer] =
          static_cast<uint32_t>(total_delta_bytes);
      if (!periodic_center_layers.empty() &&
          periodic_center_layers[layer].period != 0) {
        if (bits_by_layer[layer] != 0) {
          throw std::invalid_argument(
              "periodic center layer retains an ordinary delta width");
        }
        continue;
      }
      const size_t count = layer_end[layer] - layer_begin[layer];
      const size_t blocks =
          (count + CENTER_ID_BLOCK_SIZE - 1) / CENTER_ID_BLOCK_SIZE;
      total_blocks += blocks;
      const uint64_t slots =
          static_cast<uint64_t>(blocks) * (CENTER_ID_BLOCK_SIZE - 1);
      total_delta_bytes += static_cast<size_t>(
          (slots * bits_by_layer[layer] + 7) / 8);
      if (total_blocks > std::numeric_limits<uint32_t>::max() ||
          total_delta_bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("center ID block storage exceeds 32 bits");
      }
    }
    center_id_block_bases_16bit =
        maximum_center_id <= std::numeric_limits<uint16_t>::max();
    if (center_id_block_bases_16bit) {
      center_id_block_bases16.assign(total_blocks, uint16_t{0});
      center_id_block_bases.clear();
    } else {
      center_id_block_bases.assign(total_blocks, LeafId{0});
      center_id_block_bases16.clear();
    }
    center_id_block_deltas.assign(total_delta_bytes, uint8_t{0});
  }

  LeafId center_id_block_base(size_t base_idx) const {
    return center_id_block_bases_16bit
               ? static_cast<LeafId>(center_id_block_bases16[base_idx])
               : center_id_block_bases[base_idx];
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline bool has_periodic_center_layer(size_t layer) const {
    return !periodic_center_layers.empty() &&
           periodic_center_layers[layer].period != 0;
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline LeafId periodic_center_sequence_id(
      size_t local, size_t layer) const {
    const auto& descriptor = periodic_center_layers[layer];
    switch (descriptor.pattern) {
      case PeriodicCenterLayer::Linear16:
        return static_cast<LeafId>(local << 4);
      case PeriodicCenterLayer::DefaultPeriod3: {
        const uint32_t quotient = static_cast<uint32_t>(local / 3);
        const uint32_t remainder =
            static_cast<uint32_t>(local - quotient * 3);
        static constexpr std::array<uint8_t, 3> offsets = {0, 8, 12};
        return static_cast<LeafId>(quotient * 16 + offsets[remainder]);
      }
      case PeriodicCenterLayer::DefaultPeriod7: {
        const uint32_t quotient = static_cast<uint32_t>(local / 7);
        const uint32_t remainder =
            static_cast<uint32_t>(local - quotient * 7);
        static constexpr std::array<uint8_t, 7> offsets = {
            0, 3, 6, 8, 11, 12, 15};
        return static_cast<LeafId>(quotient * 16 + offsets[remainder]);
      }
      case PeriodicCenterLayer::Generic:
        break;
      default:
        throw std::runtime_error("periodic center pattern is invalid");
    }
    const uint32_t period = descriptor.period;
    uint32_t quotient = 0;
    uint32_t remainder = 0;
    switch (period) {
      case 1:
        quotient = static_cast<uint32_t>(local);
        break;
      case 3:
        quotient = static_cast<uint32_t>(local / 3);
        remainder = static_cast<uint32_t>(local - quotient * 3);
        break;
      case 7:
        quotient = static_cast<uint32_t>(local / 7);
        remainder = static_cast<uint32_t>(local - quotient * 7);
        break;
      default:
        if (period == 0) {
          throw std::runtime_error("periodic center layer is disabled");
        }
        quotient = static_cast<uint32_t>(local / period);
        remainder = static_cast<uint32_t>(local - quotient * period);
        break;
    }
    const uint64_t center =
        static_cast<uint64_t>(descriptor.first) +
        static_cast<uint64_t>(quotient) * descriptor.cycle_span +
        periodic_center_offsets[descriptor.offset_begin + remainder];
    if (center > std::numeric_limits<LeafId>::max()) {
      throw std::runtime_error("periodic center ID exceeds 32 bits");
    }
    return static_cast<LeafId>(center);
  }

  bool center_sequence_ids_valid() const {
    const size_t layer_count = layer_begin.size();
    if (layer_end.size() != layer_count ||
        center_id_block_begins.size() != layer_count ||
        center_id_delta_begins.size() != layer_count ||
        center_id_delta_bits.size() != layer_count ||
        (!periodic_center_layers.empty() &&
         periodic_center_layers.size() != layer_count) ||
        (periodic_center_layers.empty() &&
         !periodic_center_offsets.empty())) {
      return false;
    }
    size_t expected_blocks = 0;
    size_t expected_delta_bytes = 0;
    size_t expected_period_offsets = 0;
    for (size_t layer = 0; layer < layer_count; ++layer) {
      if (layer_end[layer] < layer_begin[layer] ||
          center_id_delta_bits[layer] > 32 ||
          center_id_block_begins[layer] != expected_blocks ||
          center_id_delta_begins[layer] != expected_delta_bytes) {
        return false;
      }
      if (!periodic_center_layers.empty()) {
        const auto& descriptor = periodic_center_layers[layer];
        if (descriptor.period != 0) {
          const size_t count = layer_end[layer] - layer_begin[layer];
          if (count == 0 || center_id_delta_bits[layer] != 0 ||
              descriptor.offset_begin != expected_period_offsets ||
              expected_period_offsets + descriptor.period >
                  periodic_center_offsets.size() ||
              periodic_center_offsets[descriptor.offset_begin] != 0) {
            return false;
          }
          uint8_t previous_offset = 0;
          for (uint32_t offset = 1; offset < descriptor.period; ++offset) {
            const uint8_t current = periodic_center_offsets[
                descriptor.offset_begin + offset];
            if (current <= previous_offset) return false;
            previous_offset = current;
          }
          if ((count > 1 &&
               descriptor.cycle_span <= previous_offset) ||
              (count == 1 && descriptor.cycle_span != 0)) {
            return false;
          }
          const auto offset_equals = [&](size_t offset, uint8_t value) {
            return descriptor.period > offset &&
                   periodic_center_offsets[
                       descriptor.offset_begin + offset] == value;
          };
          if (descriptor.pattern == PeriodicCenterLayer::Linear16) {
            if (descriptor.first != 0 || descriptor.cycle_span != 16 ||
                descriptor.period != 1) return false;
          } else if (descriptor.pattern ==
                     PeriodicCenterLayer::DefaultPeriod3) {
            if (descriptor.first != 0 || descriptor.cycle_span != 16 ||
                descriptor.period != 3 || !offset_equals(1, 8) ||
                !offset_equals(2, 12)) return false;
          } else if (descriptor.pattern ==
                     PeriodicCenterLayer::DefaultPeriod7) {
            static constexpr std::array<uint8_t, 7> expected = {
                0, 3, 6, 8, 11, 12, 15};
            if (descriptor.first != 0 || descriptor.cycle_span != 16 ||
                descriptor.period != expected.size()) return false;
            for (size_t offset = 0; offset < expected.size(); ++offset) {
              if (!offset_equals(offset, expected[offset])) return false;
            }
          } else if (descriptor.pattern != PeriodicCenterLayer::Generic) {
            return false;
          }
          expected_period_offsets += descriptor.period;
          const size_t last_local = count - 1;
          const size_t quotient = last_local / descriptor.period;
          const size_t remainder =
              last_local - quotient * descriptor.period;
          const uint64_t last =
              static_cast<uint64_t>(descriptor.first) +
              static_cast<uint64_t>(quotient) * descriptor.cycle_span +
              periodic_center_offsets[
                  descriptor.offset_begin + remainder];
          if (last > std::numeric_limits<LeafId>::max() ||
              last >= sequences.size()) return false;
          continue;
        }
        if (descriptor.first != 0 || descriptor.cycle_span != 0 ||
            descriptor.offset_begin != 0 ||
            descriptor.pattern != PeriodicCenterLayer::Generic) {
          return false;
        }
      }
      const size_t count = layer_end[layer] - layer_begin[layer];
      const size_t blocks =
          (count + CENTER_ID_BLOCK_SIZE - 1) / CENTER_ID_BLOCK_SIZE;
      expected_blocks += blocks;
      const uint64_t slots =
          static_cast<uint64_t>(blocks) * (CENTER_ID_BLOCK_SIZE - 1);
      expected_delta_bytes += static_cast<size_t>(
          (slots * center_id_delta_bits[layer] + 7) / 8);
    }
    return expected_period_offsets == periodic_center_offsets.size() &&
           (center_id_block_bases_16bit
                ? center_id_block_bases16.size() == expected_blocks &&
                      center_id_block_bases.empty()
                : center_id_block_bases.size() == expected_blocks &&
                      center_id_block_bases16.empty()) &&
           center_id_block_deltas.size() == expected_delta_bytes;
  }

  void set_center_sequence_id(
      NodeId node_id, size_t layer, LeafId center_id) {
    if (layer >= layer_begin.size() || node_id < layer_begin[layer] ||
        node_id >= layer_end[layer]) {
      throw std::out_of_range("center ID node is outside its layer");
    }
    const size_t local = node_id - layer_begin[layer];
    if (has_periodic_center_layer(layer)) {
      if (periodic_center_sequence_id(local, layer) != center_id) {
        throw std::runtime_error(
            "periodic center ID differs from exact build center");
      }
      return;
    }
    const size_t block = local / CENTER_ID_BLOCK_SIZE;
    const size_t in_block = local % CENTER_ID_BLOCK_SIZE;
    const size_t base_idx = center_id_block_begins[layer] + block;
    if (in_block == 0) {
      if (center_id_block_bases_16bit) {
        if (center_id > std::numeric_limits<uint16_t>::max()) {
          throw std::length_error("center ID exceeds 16-bit base layout");
        }
        center_id_block_bases16[base_idx] =
            static_cast<uint16_t>(center_id);
      } else {
        center_id_block_bases[base_idx] = center_id;
      }
      return;
    }
    const LeafId base = center_id_block_base(base_idx);
    if (center_id < base) {
      throw std::runtime_error(
          "center IDs are not monotone within a layer block");
    }
    const uint32_t delta = center_id - base;
    const uint32_t bits = center_id_delta_bits[layer];
    const uint64_t mask =
        bits == 32 ? std::numeric_limits<uint32_t>::max()
                   : bits == 0 ? 0 : (uint64_t{1} << bits) - 1;
    if (delta > mask) {
      throw std::length_error("center ID delta exceeds layer width");
    }
    const size_t slot = block * (CENTER_ID_BLOCK_SIZE - 1) + in_block - 1;
    const size_t bit_offset = slot * bits;
    const size_t byte_offset =
        center_id_delta_begins[layer] + (bit_offset >> 3);
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    const size_t byte_count = (shift + bits + 7) / 8;
    const uint64_t word = static_cast<uint64_t>(delta) << shift;
    for (size_t byte = 0; byte < byte_count; ++byte) {
      center_id_block_deltas[byte_offset + byte] |=
          static_cast<uint8_t>(word >> (byte * 8));
    }
  }

  uint32_t link_count(const PackedWorldNodeRecordRef& node) const {
    if (!node.counts_overflow()) {
      return node.inline_link_count_or_overflow_index();
    }
    return node_count_overflows[
        node.inline_link_count_or_overflow_index()].link_count;
  }
  uint32_t implicit_consecutive_leaf_count(LeafId center) const {
    if (center >= sequences.size()) {
      throw std::out_of_range("implicit leaf center is outside sequences");
    }
    const LeafId begin =
        center > implicit_consecutive_leaf_radius
            ? center - implicit_consecutive_leaf_radius
            : 0;
    const uint64_t end = std::min<uint64_t>(
        sequences.size(), static_cast<uint64_t>(center) +
                              implicit_consecutive_leaf_radius + 1);
    return static_cast<uint32_t>(end - begin);
  }
  static uint8_t consecutive_shift_leaf_mbb_code(
      uint32_t leaf_count_value, uint32_t center_offset) {
    const uint32_t key = (leaf_count_value << 3) | center_offset;
    switch (key) {
      case (1U << 3) | 0U: return 0;
      case (2U << 3) | 0U: return 3;
      case (2U << 3) | 1U: return 1;
      case (3U << 3) | 0U: return 21;
      case (3U << 3) | 1U: return 10;
      case (3U << 3) | 2U: return 5;
      case (4U << 3) | 1U: return 64;
      case (4U << 3) | 2U: return 32;
      case (5U << 3) | 2U: return 194;
      default:
        throw std::runtime_error(
            "implicit shift leaf MBB shape is invalid");
    }
  }
  uint8_t implicit_shift_leaf_mbb_code(
      LeafId center, uint32_t leaf_count_value) const {
    if (!implicit_shift_leaf_mbb || !implicit_consecutive_leaf_ids ||
        dense_leaf_mbb_values != std::array<uint8_t, 3>{0, 2, 4}) {
      throw std::runtime_error("implicit shift leaf MBB is invalid");
    }
    const LeafId begin =
        center > implicit_consecutive_leaf_radius
            ? center - implicit_consecutive_leaf_radius
            : 0;
    return consecutive_shift_leaf_mbb_code(
        leaf_count_value, center - begin);
  }
  uint32_t link_count(NodeId node_id,
                      const PackedWorldNodeRecordRef& node,
                      LeafId center) const {
    if (node_records.leaf_layout().has_implicit_dense_leaf_fields() &&
        node_id >= node_records.finest_node_begin()) {
      if (!implicit_consecutive_leaf_ids || !dense_leaf_mbb_ternary) {
        throw std::runtime_error(
            "implicit dense leaf counts require consecutive ternary leaves");
      }
      return implicit_consecutive_leaf_count(center);
    }
    return link_count(node);
  }
  uint32_t link_count(NodeId node_id,
                      const PackedWorldNodeRecordRef& node) const {
    if (node_records.leaf_layout().has_implicit_dense_leaf_fields() &&
        node_id >= node_records.finest_node_begin()) {
      return link_count(node_id, node, center_sequence_id(node_id));
    }
    return link_count(node);
  }
  uint32_t link_count(NodeId node_id) const {
    const auto node = node_records[node_id];
    return link_count(node_id, node);
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline LeafId center_sequence_id(NodeId node_id, size_t layer) const {
    const size_t local = node_id - layer_begin[layer];
    if (has_periodic_center_layer(layer)) {
      return periodic_center_sequence_id(local, layer);
    }
    const size_t block = local / CENTER_ID_BLOCK_SIZE;
    const size_t in_block = local % CENTER_ID_BLOCK_SIZE;
    const LeafId base = center_id_block_base(
        center_id_block_begins[layer] + block);
    if (in_block == 0) return base;
    const uint32_t bits = center_id_delta_bits[layer];
    if (bits == 0) return base;
    const size_t slot = block * (CENTER_ID_BLOCK_SIZE - 1) + in_block - 1;
    const size_t bit_offset = slot * bits;
    const size_t byte_offset =
        center_id_delta_begins[layer] + (bit_offset >> 3);
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    uint64_t word = 0;
    if (bits <= 25 && byte_offset + sizeof(uint32_t) <=
                          center_id_block_deltas.size()) {
      uint32_t packed = 0;
      std::memcpy(
          &packed, center_id_block_deltas.data() + byte_offset,
          sizeof(packed));
      word = packed;
    } else {
      const size_t byte_count = (shift + bits + 7) / 8;
      for (size_t byte = 0; byte < byte_count; ++byte) {
        word |= static_cast<uint64_t>(
                    center_id_block_deltas[byte_offset + byte])
                << (byte * 8);
      }
    }
    const uint64_t mask =
        bits == 32 ? std::numeric_limits<uint32_t>::max()
                   : (uint64_t{1} << bits) - 1;
    return static_cast<LeafId>(base + ((word >> shift) & mask));
  }

  LeafId center_sequence_id(NodeId node_id) const {
    const auto it = std::upper_bound(layer_end.begin(), layer_end.end(), node_id);
    if (it == layer_end.end()) {
      throw std::out_of_range("center ID node is outside graph layers");
    }
    return center_sequence_id(
        node_id, static_cast<size_t>(it - layer_end.begin()));
  }
  uint32_t child_count(NodeId node_id) const {
    return link_count(node_id);
  }
  uint32_t leaf_count(NodeId node_id) const {
    return link_count(node_id);
  }
  uint32_t beacon_count(const PackedWorldNodeRecordRef& node) const {
    if (!node.counts_overflow()) {
      return node.inline_beacon_count();
    }
    return node_count_overflows[
        node.inline_link_count_or_overflow_index()].beacon_count;
  }
  uint32_t beacon_count(NodeId node_id) const {
    const auto node = node_records[node_id];
    return beacon_count(node);
  }
  void set_node_counts(NodeId node_id, uint32_t link_count_value,
                       uint32_t beacon_count_value,
                       WorldNodeRecord::BeaconStorage storage) {
    auto node = node_records[node_id];
    if (node_records.leaf_layout().has_implicit_dense_leaf_fields() &&
        node_id >= node_records.finest_node_begin()) {
      const uint32_t expected = implicit_consecutive_leaf_count(
          center_sequence_id(node_id));
      if (link_count_value != expected) {
        throw std::invalid_argument(
            "implicit dense leaf count does not match center interval");
      }
      node.set_inline_counts(
          link_count_value, beacon_count_value, storage);
      return;
    }
    if (link_count_value <= node.inline_link_count_mask() &&
        beacon_count_value < WorldNodeRecord::COUNT_OVERFLOW_CODE) {
      node.set_inline_counts(
          link_count_value, beacon_count_value, storage);
      return;
    }
    if (node_count_overflows.size() >
        node.inline_link_count_mask()) {
      throw std::length_error("too many node-count overflow records");
    }
    const uint32_t overflow_index =
        static_cast<uint32_t>(node_count_overflows.size());
    node_count_overflows.push_back(
        {link_count_value, beacon_count_value});
    node.set_count_overflow(overflow_index, storage);
  }

  void initialize_child_base_ids(size_t non_finest_node_count,
                                 uint64_t maximum_forward_delta) {
    if (non_finest_node_count == 0) {
      child_base_forward_delta_bytes = 0;
      return;
    }
    const uint8_t bits =
        PackedWorldNodeLayout::bits_for_value(maximum_forward_delta);
    if (bits > 32) {
      throw std::length_error("child base delta exceeds 32 bits");
    }
    child_base_forward_delta_bytes = (bits + 7) / 8;
  }

  bool child_base_ids_valid(size_t non_finest_node_count) const {
    if (non_finest_node_count == 0) {
      return child_base_forward_delta_bytes == 0 &&
             !compact_child_base_blocks && child_base_block_bases.empty();
    }
    if (child_base_forward_delta_bytes == 0 ||
        child_base_forward_delta_bytes > sizeof(NodeId)) {
      return false;
    }
    return !compact_child_base_blocks ||
           (child_base_forward_delta_bytes > 1 &&
            child_base_block_bases.size() ==
                (non_finest_node_count + CHILD_BASE_BLOCK_SIZE - 1) /
                    CHILD_BASE_BLOCK_SIZE);
  }

  size_t child_base_byte_count() const {
    return compact_child_base_blocks ? 1 : child_base_forward_delta_bytes;
  }

  uint32_t child_begin(
      NodeId node_id, const PackedWorldNodeRecordRef& node) const {
    if (!implicit_contiguous_child_ranges) return node.child_begin();
    const uint64_t begin =
        static_cast<uint64_t>(node_id) * child_base_byte_count();
    if (begin > std::numeric_limits<uint32_t>::max()) {
      throw std::out_of_range("implicit child payload offset exceeds uint32");
    }
    return static_cast<uint32_t>(begin);
  }
  uint32_t child_begin(NodeId node_id) const {
    return child_begin(node_id, node_records[node_id]);
  }

  void append_child_base_id(NodeId node_id, NodeId base) {
    const uint32_t bits = child_base_forward_delta_bytes * 8;
    if (bits == 0 || base <= node_id) {
      throw std::out_of_range("child base does not follow its parent");
    }
    if (compact_child_base_blocks) {
      const size_t block_idx = node_id / CHILD_BASE_BLOCK_SIZE;
      if (block_idx >= child_base_block_bases.size() ||
          base < child_base_block_bases[block_idx] ||
          base - child_base_block_bases[block_idx] >
              std::numeric_limits<uint8_t>::max()) {
        throw std::length_error("child base exceeds compact block range");
      }
      child_id_base_deltas8.push_back(static_cast<uint8_t>(
          base - child_base_block_bases[block_idx]));
      return;
    }
    const uint32_t delta = base - node_id - 1;
    const uint64_t mask =
        bits == 32 ? std::numeric_limits<uint32_t>::max()
                   : (uint64_t{1} << bits) - 1;
    if (delta > mask) {
      throw std::length_error("child base exceeds packed forward range");
    }
    const size_t byte_count = child_base_byte_count();
    for (size_t byte = 0; byte < byte_count; ++byte) {
      child_id_base_deltas8.push_back(
          static_cast<uint8_t>(delta >> (byte * 8)));
    }
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline NodeId child_base_id(NodeId node_id, const uint8_t* segment,
                              size_t byte_count) const {
    if (compact_child_base_blocks) {
      const size_t block_idx = node_id / CHILD_BASE_BLOCK_SIZE;
      if (byte_count != 1 || block_idx >= child_base_block_bases.size()) {
        throw std::out_of_range("compact child base is outside block storage");
      }
      return static_cast<NodeId>(
          child_base_block_bases[block_idx] + segment[0]);
    }
    if (byte_count == 2) {
      uint16_t delta = 0;
      std::memcpy(&delta, segment, sizeof(delta));
      return static_cast<NodeId>(node_id + 1 + delta);
    }
    uint32_t delta = 0;
    for (size_t byte = 0; byte < byte_count; ++byte) {
      delta |= static_cast<uint32_t>(segment[byte]) << (byte * 8);
    }
    return static_cast<NodeId>(node_id + 1 + delta);
  }

  struct ChildIdAccessor {
    NodeId base = 0;
    const uint16_t* deltas16 = nullptr;
    const uint8_t* base_deltas8 = nullptr;
    const uint8_t* packed_deltas = nullptr;
    const NodeId* ids32 = nullptr;
    uint32_t packed_bits = 0;

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline NodeId at(uint32_t offset) const {
      if (base_deltas8) return base + base_deltas8[offset];
      if (deltas16) return base + deltas16[offset];
      if (packed_deltas) {
        if (packed_bits == CONTIGUOUS_CHILD_RANGE_BITS) {
          return base + offset;
        }
        const size_t bit_offset =
            static_cast<size_t>(offset) * packed_bits;
        const size_t byte_offset = bit_offset >> 3;
        const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
        uint32_t word = packed_deltas[byte_offset];
        if (shift + packed_bits > 8) {
          word |= static_cast<uint32_t>(
                      packed_deltas[byte_offset + 1])
                  << 8;
        }
        if (shift + packed_bits > 16) {
          word |= static_cast<uint32_t>(
                      packed_deltas[byte_offset + 2])
                  << 16;
        }
        return base +
               ((word >> shift) &
                ((uint32_t{1} << packed_bits) - 1));
      }
      return ids32[offset];
    }
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline const void* address(uint32_t offset) const {
      if (base_deltas8) {
        return static_cast<const void*>(base_deltas8 + offset);
      }
      if (deltas16) {
        return static_cast<const void*>(deltas16 + offset);
      }
      if (packed_deltas) {
        if (packed_bits == CONTIGUOUS_CHILD_RANGE_BITS) {
          return static_cast<const void*>(packed_deltas);
        }
        return static_cast<const void*>(
            packed_deltas +
            (static_cast<size_t>(offset) * packed_bits >> 3));
      }
      return static_cast<const void*>(ids32 + offset);
    }
  };

  bool child_ids_are_base_delta8(NodeId node_id) const {
    return node_records[node_id].link_storage() ==
           WorldNodeRecord::LinkStorage::Delta8;
  }
  bool child_ids_are_delta16(NodeId node_id) const {
    return node_records[node_id].link_storage() ==
           WorldNodeRecord::LinkStorage::Delta16;
  }
  bool child_ids_are_packed_delta(NodeId node_id) const {
    return node_records[node_id].link_storage() ==
           WorldNodeRecord::LinkStorage::PackedDelta;
  }
  size_t packed_child_byte_count(
      const PackedWorldNodeRecordRef& node,
      uint32_t child_count_value) const {
    if (node.packed_child_bits() == CONTIGUOUS_CHILD_RANGE_BITS) {
      return child_base_byte_count();
    }
    const uint64_t bit_count =
        static_cast<uint64_t>(child_count_value) *
        node.packed_child_bits();
    return child_base_byte_count() +
           static_cast<size_t>((bit_count + 7) / 8);
  }
  size_t packed_child_byte_count(NodeId node_id) const {
    const auto node = node_records[node_id];
    return packed_child_byte_count(node, link_count(node));
  }
  void set_child_ids_base_delta8(NodeId node_id) {
    node_records[node_id].set_link_storage(
        WorldNodeRecord::LinkStorage::Delta8);
  }
  void set_child_ids_delta16(NodeId node_id) {
    node_records[node_id].set_link_storage(
        WorldNodeRecord::LinkStorage::Delta16);
  }
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline ChildIdAccessor child_ids_for(
      NodeId node_id, const PackedWorldNodeRecordRef& node) const {
    switch (node.link_storage()) {
      case WorldNodeRecord::LinkStorage::Delta8: {
        const uint8_t* segment =
            child_id_base_deltas8.data() + child_begin(node_id, node);
        const size_t base_bytes = child_base_byte_count();
        return {child_base_id(node_id, segment, base_bytes), nullptr,
                segment + base_bytes, nullptr,
                nullptr, 0};
      }
      case WorldNodeRecord::LinkStorage::Delta16:
        return {static_cast<NodeId>(node_id + 1),
                child_id_deltas16.data() + child_begin(node_id, node),
                nullptr, nullptr, nullptr, 0};
      case WorldNodeRecord::LinkStorage::PackedDelta: {
        const uint8_t* segment =
            child_id_base_deltas8.data() + child_begin(node_id, node);
        const size_t base_bytes = child_base_byte_count();
        const NodeId base = child_base_id(node_id, segment, base_bytes);
        if (node.packed_child_bits() == CONTIGUOUS_CHILD_RANGE_BITS &&
            link_count(node) <= CONTIGUOUS_CHILD_OFFSET_TABLE_SIZE) {
          return {base, nullptr, contiguous_child_offset_table.data(),
                  nullptr, nullptr, 0};
        }
        return {base, nullptr, nullptr, segment + base_bytes,
                nullptr, node.packed_child_bits()};
      }
      case WorldNodeRecord::LinkStorage::Absolute32:
        return {0, nullptr, nullptr, nullptr,
                child_ids.data() + child_begin(node_id, node), 0};
    }
    throw std::runtime_error("invalid child ID storage");
  }
  inline ChildIdAccessor child_ids_for(NodeId node_id) const {
    const auto node = node_records[node_id];
    return child_ids_for(node_id, node);
  }
  NodeId child_id(NodeId node_id, uint32_t child_offset) const {
    return child_ids_for(node_id).at(child_offset);
  }
  size_t base_delta8_child_edge_count() const {
    size_t count = 0;
    const NodeId non_finest_node_count =
        layer_begin.empty() ? 0 : layer_begin.back();
    for (NodeId node_id = 0;
         node_id < non_finest_node_count; ++node_id) {
      if (child_ids_are_base_delta8(node_id)) {
        count += child_count(node_id);
      }
    }
    return count;
  }
  size_t packed_delta_child_edge_count() const {
    size_t count = 0;
    const NodeId non_finest_node_count =
        layer_begin.empty() ? 0 : layer_begin.back();
    for (NodeId node_id = 0;
         node_id < non_finest_node_count; ++node_id) {
      if (child_ids_are_packed_delta(node_id)) {
        count += child_count(node_id);
      }
    }
    return count;
  }
  size_t edge_count() const {
    return base_delta8_child_edge_count() +
           packed_delta_child_edge_count() +
           child_id_deltas16.size() + child_ids.size();
  }
  size_t leaf_link_count() const {
    size_t count = 0;
    const NodeId finest_begin =
        layer_begin.empty() ? 0 : layer_begin.back();
    for (NodeId node_id = finest_begin;
         node_id < node_records.size(); ++node_id) {
      count += leaf_count(node_id);
    }
    return count;
  }

  uint32_t child_mbb_bits(
      NodeId node_id, const PackedWorldNodeRecordRef& node) const {
    if (layer_begin.empty() || node_id >= layer_begin.back()) {
      throw std::runtime_error("child MBB node width is missing");
    }
    uint32_t bits = node.child_mbb_bits();
    if (implicit_child_mbb_widths) {
      if (implicit_child_mbb_exception_bits > 8) {
        throw std::runtime_error("implicit child MBB width is invalid");
      }
      if (implicit_child_mbb_exception_bits != 0) {
        const size_t byte_index = node_id >> 3;
        if (byte_index >= child_mbb_width_exceptions.size()) {
          throw std::runtime_error("implicit child MBB width is invalid");
        }
        if ((child_mbb_width_exceptions[byte_index] >> (node_id & 7)) &
            1U) {
          bits = implicit_child_mbb_exception_bits;
        }
      }
    }
    if (bits == 0 || bits > 8) {
      throw std::runtime_error("child MBB node width is invalid");
    }
    return bits;
  }
  uint32_t child_mbb_bits(NodeId node_id) const {
    const auto node = node_records[node_id];
    return child_mbb_bits(node_id, node);
  }
  uint32_t child_mbb_bin_width(NodeId node_id) const {
    if (layer_begin.size() < 2 || node_id >= layer_begin.back()) {
      throw std::runtime_error("child MBB node layer is missing");
    }
    const NodeId last_non_finest_begin =
        layer_begin[layer_begin.size() - 2];
    return node_id >= last_non_finest_begin
               ? FINE_CHILD_MBB_BIN_WIDTH
               : COARSE_CHILD_MBB_BIN_WIDTH;
  }
  size_t child_mbb_byte_count(
      NodeId node_id, const PackedWorldNodeRecordRef& node,
      uint32_t child_count_value, uint32_t beacon_count_value) const {
    const uint32_t bits = child_mbb_bits(node_id, node);
    if (bits == PAIRED_BASE11_CHILD_MBB_BITS) {
      const size_t full_pairs = beacon_count_value / 2;
      const uint32_t bin_width = child_mbb_bin_width(node_id);
      uint32_t bits_per_child = (beacon_count_value & 1) ? 4 : 0;
      for (size_t pair = 0; pair < full_pairs; ++pair) {
        bits_per_child += metric_pair_rank_bits(
            child_beacon_dists[node.child_mbb_begin() + pair],
            bin_width);
      }
      return full_pairs + static_cast<size_t>(
          (static_cast<uint64_t>(child_count_value) *
               bits_per_child +
           7) /
          8);
    }
    const uint64_t cells =
        static_cast<uint64_t>(child_count_value) * beacon_count_value;
    return static_cast<size_t>(
        (cells * bits + 7) / 8);
  }
  size_t child_mbb_byte_count(NodeId node_id) const {
    const auto node = node_records[node_id];
    return child_mbb_byte_count(
        node_id, node, link_count(node), beacon_count(node));
  }
  bool child_mbb_range_valid(
      NodeId node_id, const PackedWorldNodeRecordRef& node,
      uint32_t child_count_value, uint32_t beacon_count_value) const {
    const uint32_t begin = node.child_mbb_begin();
    if (begin > child_beacon_dists.size()) return false;
    if (child_mbb_bits(node_id, node) ==
            PAIRED_BASE11_CHILD_MBB_BITS &&
        beacon_count_value / 2 >
            child_beacon_dists.size() - begin) {
      return false;
    }
    return child_mbb_byte_count(
               node_id, node, child_count_value, beacon_count_value) <=
           child_beacon_dists.size() - begin;
  }
  bool child_mbb_range_valid(NodeId node_id) const {
    const auto node = node_records[node_id];
    return child_mbb_range_valid(
        node_id, node, link_count(node), beacon_count(node));
  }
  uint8_t child_beacon_distance(
      NodeId node_id, size_t cell_offset) const {
    const uint64_t cells =
        static_cast<uint64_t>(child_count(node_id)) *
        beacon_count(node_id);
    if (cell_offset >= cells || !child_mbb_range_valid(node_id)) {
      throw std::out_of_range("child MBB cell is outside packed storage");
    }
    const uint32_t bits = child_mbb_bits(node_id);
    return child_beacon_distance_unchecked(
        node_id, cell_offset, bits,
        child_mbb_bin_width(node_id));
  }
  uint8_t child_beacon_distance_unchecked(
      NodeId node_id, size_t cell_offset, uint32_t bits,
      uint32_t bin_width) const {
    const auto node = node_records[node_id];
    return child_beacon_distance_unchecked(
        node.child_mbb_begin(), link_count(node), beacon_count(node),
        cell_offset, bits, bin_width);
  }
  uint8_t child_beacon_distance_unchecked(
      uint32_t begin, size_t children, size_t beacons,
      size_t cell_offset, uint32_t bits, uint32_t bin_width) const {
    uint8_t encoded = 0;
    if (bits == PAIRED_BASE11_CHILD_MBB_BITS) {
      const size_t dimension = cell_offset / children;
      const size_t child = cell_offset % children;
      const size_t pair = dimension / 2;
      const size_t full_pairs = beacons / 2;
      uint32_t bits_per_child = (beacons & 1) ? 4 : 0;
      size_t pair_bit_offset = 0;
      for (size_t prior = 0; prior < full_pairs; ++prior) {
        const uint32_t pair_bits = metric_pair_rank_bits(
            child_beacon_dists[begin + prior], bin_width);
        bits_per_child += pair_bits;
        if (prior < pair) pair_bit_offset += pair_bits;
      }
      const bool full_pair = pair < full_pairs;
      const uint32_t encoded_bits =
          full_pair
              ? metric_pair_rank_bits(
                    child_beacon_dists[begin + pair], bin_width)
              : 4;
      const size_t bit_offset =
          child * bits_per_child + pair_bit_offset;
      const size_t byte_offset = begin + full_pairs + (bit_offset >> 3);
      const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
      uint16_t word = child_beacon_dists[byte_offset];
      if (shift + encoded_bits > 8) {
        word |= static_cast<uint16_t>(
                    child_beacon_dists[byte_offset + 1])
                << 8;
      }
      const uint8_t rank = static_cast<uint8_t>(
          (word >> shift) & ((uint32_t{1} << encoded_bits) - 1));
      const uint8_t code =
          full_pair
              ? metric_pair_code(
                    rank, child_beacon_dists[begin + pair], bin_width)
              : rank;
      encoded = dimension & 1 ? code / 11 : code % 11;
    } else if (bits == 8) {
      encoded = child_beacon_dists[begin + cell_offset];
    } else {
      const size_t bit_offset = cell_offset * bits;
      const size_t byte_offset = bit_offset >> 3;
      const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
      uint16_t word = child_beacon_dists[begin + byte_offset];
      if (shift + bits > 8) {
        word |= static_cast<uint16_t>(
                    child_beacon_dists[
                        begin + byte_offset + 1])
                << 8;
      }
      encoded = static_cast<uint8_t>(
          (word >> shift) & ((uint32_t{1} << bits) - 1));
    }
    const uint32_t midpoint =
        static_cast<uint32_t>(encoded) * bin_width +
        child_mbb_quantization_error(bin_width);
    return static_cast<uint8_t>(std::min<uint32_t>(midpoint, 255));
  }

  uint32_t leaf_mbb_bits(
      NodeId node_id, const PackedWorldNodeRecordRef& node) const {
    if (layer_begin.empty() || node_id < layer_begin.back() ||
        node_id >= node_records.size()) {
      throw std::runtime_error("leaf MBB node width is missing");
    }
    const uint32_t bits = node.leaf_mbb_bits();
    if (bits == 0 || bits > 8) {
      throw std::runtime_error("leaf MBB node width is invalid");
    }
    return bits;
  }
  uint32_t leaf_mbb_bits(NodeId node_id) const {
    const auto node = node_records[node_id];
    return leaf_mbb_bits(node_id, node);
  }
  size_t leaf_mbb_byte_count(
      NodeId node_id, const PackedWorldNodeRecordRef& node,
      uint32_t leaf_count_value, uint32_t beacon_count_value) const {
    if (implicit_shift_leaf_mbb) {
      if (!dense_leaf_mbb_ternary || beacon_count_value != 1 ||
          leaf_count_value > 5) {
        throw std::runtime_error("implicit shift leaf MBB shape is invalid");
      }
      return 0;
    }
    if (dense_leaf_mbb_ternary) {
      if (beacon_count_value != 1 || leaf_count_value > 5) {
        throw std::runtime_error("dense ternary leaf MBB shape is invalid");
      }
      return 1;
    }
    const uint64_t cells =
        static_cast<uint64_t>(leaf_count_value) * beacon_count_value;
    return static_cast<size_t>(
        (cells * leaf_mbb_bits(node_id, node) + 7) / 8);
  }
  size_t leaf_mbb_byte_count(NodeId node_id) const {
    const auto node = node_records[node_id];
    return leaf_mbb_byte_count(
        node_id, node, link_count(node_id, node), beacon_count(node));
  }
  uint32_t leaf_mbb_begin(
      NodeId node_id, const PackedWorldNodeRecordRef& node) const {
    if (implicit_shift_leaf_mbb) {
      if (layer_begin.empty() || node_id < layer_begin.back() ||
          node_id >= node_records.size()) {
        throw std::runtime_error("implicit shift leaf MBB node is missing");
      }
      return 0;
    }
    if (dense_leaf_mbb_ternary) {
      if (layer_begin.empty() || node_id < layer_begin.back() ||
          node_id >= node_records.size()) {
        throw std::runtime_error("dense ternary leaf MBB node is missing");
      }
      return node_id - layer_begin.back();
    }
    if (implicit_leaf_mbb_offsets) {
      if (layer_begin.empty() || node_id < layer_begin.back() ||
          node_id >= node_records.size()) {
        throw std::runtime_error("leaf MBB node offset is missing");
      }
      return leaf_link_begin(node_id, node);
    }
    return node.leaf_mbb_begin();
  }
  uint32_t leaf_mbb_begin(NodeId node_id) const {
    const auto node = node_records[node_id];
    return leaf_mbb_begin(node_id, node);
  }
  bool leaf_mbb_range_valid(
      NodeId node_id, const PackedWorldNodeRecordRef& node,
      uint32_t leaf_count_value, uint32_t beacon_count_value) const {
    const uint32_t begin = leaf_mbb_begin(node_id, node);
    return begin <= leaf_beacon_dists.size() &&
           leaf_mbb_byte_count(
               node_id, node, leaf_count_value, beacon_count_value) <=
               leaf_beacon_dists.size() - begin;
  }
  bool leaf_mbb_range_valid(NodeId node_id) const {
    const auto node = node_records[node_id];
    return leaf_mbb_range_valid(
        node_id, node, link_count(node_id, node), beacon_count(node));
  }
  uint8_t leaf_beacon_distance(
      NodeId node_id, size_t cell_offset) const {
    const uint64_t cells =
        static_cast<uint64_t>(leaf_count(node_id)) *
        beacon_count(node_id);
    if (cell_offset >= cells || !leaf_mbb_range_valid(node_id)) {
      throw std::out_of_range("leaf MBB cell is outside packed storage");
    }
    return leaf_beacon_distance_unchecked(
        node_id, cell_offset, leaf_mbb_bits(node_id));
  }
  uint8_t leaf_beacon_distance_unchecked(
      NodeId node_id, size_t cell_offset, uint32_t bits) const {
    const uint32_t begin = leaf_mbb_begin(node_id);
    if (implicit_shift_leaf_mbb) {
      const LeafId center = center_sequence_id(node_id);
      const LeafId leaf_begin =
          center > implicit_consecutive_leaf_radius
              ? center - implicit_consecutive_leaf_radius
              : 0;
      const size_t center_offset = center - leaf_begin;
      const size_t delta =
          cell_offset > center_offset
              ? cell_offset - center_offset
              : center_offset - cell_offset;
      if (delta > 2) {
        throw std::runtime_error(
            "implicit shift leaf MBB distance is invalid");
      }
      return static_cast<uint8_t>(delta * 2);
    }
    if (dense_leaf_mbb_ternary) {
      const uint32_t leaf_count_value = leaf_count(node_id);
      if (cell_offset >= leaf_count_value) {
        throw std::out_of_range("dense ternary leaf MBB cell is invalid");
      }
      uint8_t packed = leaf_beacon_dists[begin];
      for (size_t index = 0; index < cell_offset; ++index) {
        packed = static_cast<uint8_t>(packed / 3);
      }
      const uint8_t code = static_cast<uint8_t>(packed % 3);
      return dense_leaf_mbb_values[code];
    }
    if (bits == 8) {
      return leaf_beacon_dists[begin + cell_offset];
    }
    const size_t bit_offset = cell_offset * bits;
    const size_t byte_offset = bit_offset >> 3;
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    uint16_t word = leaf_beacon_dists[begin + byte_offset];
    if (shift + bits > 8) {
      word |= static_cast<uint16_t>(
                  leaf_beacon_dists[begin + byte_offset + 1])
              << 8;
    }
    return static_cast<uint8_t>(
        (word >> shift) & ((uint32_t{1} << bits) - 1));
  }

  size_t packed_leaf_byte_count(
      const PackedWorldNodeRecordRef& node,
      uint32_t leaf_count_value) const {
    if (implicit_consecutive_leaf_ids) return 0;
    const uint64_t bit_count =
        static_cast<uint64_t>(leaf_count_value) *
        node.packed_leaf_bits();
    return static_cast<size_t>((bit_count + 7) / 8);
  }
  size_t packed_leaf_byte_count(NodeId node_id) const {
    const auto node = node_records[node_id];
    return packed_leaf_byte_count(node, link_count(node_id, node));
  }
#if defined(__GNUC__) || defined(__clang__)
  __attribute__((always_inline))
#endif
  inline LeafId packed_leaf_id(
      NodeId node_id, uint32_t leaf_offset) const {
    const auto node = node_records[node_id];
    return packed_leaf_id(
        node_id, node, center_sequence_id(node_id), leaf_offset);
  }
  inline LeafId packed_leaf_id(
      NodeId node_id, const PackedWorldNodeRecordRef& node, LeafId center,
      uint32_t leaf_offset) const {
    if (implicit_consecutive_leaf_ids) {
      if (leaf_offset >= link_count(node_id, node, center)) {
        throw std::out_of_range("implicit consecutive leaf offset is invalid");
      }
      const LeafId begin =
          center > implicit_consecutive_leaf_radius
              ? center - implicit_consecutive_leaf_radius
              : 0;
      return begin + leaf_offset;
    }
    const uint32_t bits = node.packed_leaf_bits();
    const uint8_t* data =
        reinterpret_cast<const uint8_t*>(leaf_id_deltas8.data()) +
        leaf_link_begin(node_id, node);
    const size_t bit_offset =
        static_cast<size_t>(leaf_offset) * bits;
    const size_t byte_offset = bit_offset >> 3;
    const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
    uint32_t word = data[byte_offset];
    if (shift + bits > 8) {
      word |= static_cast<uint32_t>(data[byte_offset + 1]) << 8;
    }
    if (shift + bits > 16) {
      word |= static_cast<uint32_t>(data[byte_offset + 2]) << 16;
    }
    const uint32_t zigzag =
        (word >> shift) & ((uint32_t{1} << bits) - 1);
    const int64_t delta =
        (zigzag & 1) != 0
            ? -static_cast<int64_t>((zigzag >> 1) + 1)
            : static_cast<int64_t>(zigzag >> 1);
    return static_cast<LeafId>(
        static_cast<int64_t>(center) + delta);
  }

  LeafId leaf_id(NodeId node_id, uint32_t leaf_offset) const {
    const auto node = node_records[node_id];
    const LeafId center = center_sequence_id(node_id);
    return leaf_id(node_id, node, center, leaf_offset);
  }
  LeafId leaf_id(NodeId node_id, const PackedWorldNodeRecordRef& node,
                 LeafId center,
                 uint32_t leaf_offset) const {
    switch (node.link_storage()) {
      case WorldNodeRecord::LinkStorage::Delta8:
        return center +
               leaf_id_deltas8[leaf_link_begin(node_id, node) + leaf_offset];
      case WorldNodeRecord::LinkStorage::Delta16:
        return center +
               leaf_id_deltas16[leaf_link_begin(node_id, node) + leaf_offset];
      case WorldNodeRecord::LinkStorage::Absolute32:
        return leaf_ids[leaf_link_begin(node_id, node) + leaf_offset];
      case WorldNodeRecord::LinkStorage::PackedDelta:
        return packed_leaf_id(node_id, node, center, leaf_offset);
    }
    return INVALID_LEAF_ID;
  }

  LeafId beacon_sequence_id(NodeId node_id,
                            uint32_t beacon_offset) const {
    const auto node = node_records[node_id];
    const LeafId center = center_sequence_id(node_id);
    return beacon_sequence_id(node_id, node, center, beacon_offset);
  }
  LeafId beacon_sequence_id(
      NodeId node_id, const PackedWorldNodeRecordRef& node,
      LeafId center, uint32_t beacon_offset) const {
    if (has_beacon_patterns() &&
        node_id < node_records.finest_node_begin()) {
      if (beacon_offset >= 3) {
        throw std::out_of_range("beacon pattern offset is invalid");
      }
      if (dense_beacon_patterns) {
        const uint8_t packed = beacon_id_bytes[node_id >> 1];
        const uint8_t code = static_cast<uint8_t>(
            (node_id & 1) == 0 ? packed & 0x0FU : packed >> 4);
        if (code >= dense_beacon_pattern_count) {
          throw std::runtime_error("dense beacon pattern code is invalid");
        }
        return static_cast<LeafId>(
            static_cast<int64_t>(center) +
            dense_beacon_pattern_deltas[code * 3 + beacon_offset]);
      }
      const uint16_t code = wide_beacon_pattern_code(node_id);
      if (code >= wide_beacon_pattern_count) {
        throw std::runtime_error("wide beacon pattern code is invalid");
      }
      return static_cast<LeafId>(
          static_cast<int64_t>(center) +
          wide_beacon_pattern_deltas[code * 3 + beacon_offset]);
    }
    const uint32_t beacon_begin =
        node.beacon_storage() ==
                WorldNodeRecord::BeaconStorage::ImplicitCenter
            ? 0
            : this->beacon_begin(node_id);
    switch (node.beacon_storage()) {
      case WorldNodeRecord::BeaconStorage::Delta8:
        return static_cast<LeafId>(
            static_cast<int64_t>(center) +
            static_cast<int8_t>(beacon_id_bytes[
                beacon_begin + beacon_offset]));
      case WorldNodeRecord::BeaconStorage::PackedDelta: {
        const uint32_t bits = beacon_delta_bits;
        const size_t bit_offset =
            static_cast<size_t>(beacon_offset) * bits;
        const size_t byte_offset = beacon_begin + (bit_offset >> 3);
        const uint32_t shift = static_cast<uint32_t>(bit_offset & 7);
        const size_t byte_count = (shift + bits + 7) / 8;
        uint64_t word = 0;
        for (size_t byte = 0; byte < byte_count; ++byte) {
          word |= static_cast<uint64_t>(static_cast<uint8_t>(
                      beacon_id_bytes[byte_offset + byte]))
                  << (byte * 8);
        }
        const uint64_t mask =
            bits == 32 ? std::numeric_limits<uint32_t>::max()
                       : (uint64_t{1} << bits) - 1;
        const uint32_t zigzag =
            static_cast<uint32_t>((word >> shift) & mask);
        const int64_t delta =
            (zigzag & 1) != 0
                ? -static_cast<int64_t>((zigzag >> 1) + 1)
                : static_cast<int64_t>(zigzag >> 1);
        return static_cast<LeafId>(static_cast<int64_t>(center) + delta);
      }
      case WorldNodeRecord::BeaconStorage::Absolute32: {
        LeafId absolute = 0;
        std::memcpy(
            &absolute,
            beacon_id_bytes.data() + beacon_begin +
                static_cast<size_t>(beacon_offset) * sizeof(LeafId),
            sizeof(absolute));
        return absolute;
      }
      case WorldNodeRecord::BeaconStorage::ImplicitCenter:
        return center;
    }
    return INVALID_LEAF_ID;
  }
};

class BioGeometryIndexBuilder {
  friend class IndexPersistenceAccess;

 public:
  BioGeometryIndexBuilder();
  BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw);
  explicit BioGeometryIndexBuilder(const HierarchyConfig& config);
  BioGeometryIndexBuilder(const HierarchyConfig& config,
                          const BuildRangeConfig& range_config);

  void build(const std::vector<std::shared_ptr<BioSequence>>& raw_sequences);
  void build(std::vector<std::shared_ptr<BioSequence>>&& raw_sequences);
  void build_reference_windows(
      std::string reference_id,
      std::string reference_sequence,
      size_t window_length,
      size_t stride,
      std::vector<ReferenceContig> reference_contigs = {});

  struct Statistics {
    size_t added_sequences = 0;
    size_t unique_sequences = 0;
    size_t deduplicated = 0;
    size_t invalid_reference_windows = 0;
    size_t created_auxiliary_nodes = 0;
    std::vector<size_t> created_primary_nodes;
    double compression_ratio = 0.0;
    double dag_redundancy = 0.0;
    size_t phase2_total_possible_pairs = 0;
    size_t phase2_candidate_pairs = 0;
    size_t phase2_exact_distance_calls = 0;
    size_t phase2_edges_added = 0;
    size_t phase2_full_scan_fallback_count = 0;
    size_t phase2_pigeonhole_queries = 0;
    size_t phase2_qgram_queries = 0;
    size_t phase2_hybrid_queries = 0;
    size_t phase2_qgram_candidate_pairs = 0;
    size_t phase2_qgram_pruned_by_l1 = 0;
    size_t phase2_base_count_pruned_pairs = 0;
    size_t phase2_length_pruned_pairs = 0;
    size_t phase2_seed_candidate_pairs_before_length_filter = 0;
    size_t phase2_seed_length_pruned_candidates = 0;
    size_t phase2_pigeonhole_early_abort_count = 0;
    size_t phase2_range_final_candidate_pairs = 0;
    size_t phase2_required_shared_nonpositive_count = 0;
    size_t phase2_auto_pigeonhole_accepted = 0;
    size_t phase2_auto_pigeonhole_rejected_large_candidates = 0;
    size_t phase2_auto_qgram_invoked = 0;
    size_t phase2_auto_hybrid_invoked = 0;
    size_t phase2_auto_final_candidate_pairs = 0;
    double phase2_auto_candidate_ratio_sum = 0.0;
    double phase2_auto_candidate_ratio_avg = 0.0;
    double phase2_candidate_reduction_ratio = 0.0;
    double phase2_exact_distance_reduction_ratio = 0.0;
    size_t total_possible_leaf_pairs = 0;
    size_t leaf_candidate_pairs = 0;
    size_t leaf_exact_distance_calls = 0;
    size_t leaf_attachments_added = 0;
    size_t leaf_full_scan_fallback_count = 0;
    size_t leaf_pigeonhole_queries = 0;
    size_t leaf_qgram_queries = 0;
    size_t leaf_hybrid_queries = 0;
    size_t leaf_qgram_candidate_pairs = 0;
    size_t leaf_qgram_pruned_by_l1 = 0;
    size_t leaf_base_count_pruned_pairs = 0;
    size_t leaf_length_pruned_pairs = 0;
    size_t leaf_seed_candidate_pairs_before_length_filter = 0;
    size_t leaf_seed_length_pruned_candidates = 0;
    size_t leaf_pigeonhole_early_abort_count = 0;
    size_t leaf_range_final_candidate_pairs = 0;
    size_t leaf_required_shared_nonpositive_count = 0;
    size_t leaf_auto_pigeonhole_accepted = 0;
    size_t leaf_auto_pigeonhole_rejected_large_candidates = 0;
    size_t leaf_auto_qgram_invoked = 0;
    size_t leaf_auto_hybrid_invoked = 0;
    size_t leaf_auto_final_candidate_pairs = 0;
    double leaf_auto_candidate_ratio_sum = 0.0;
    double leaf_auto_candidate_ratio_avg = 0.0;
    double leaf_candidate_reduction_ratio = 0.0;
    double leaf_exact_distance_reduction_ratio = 0.0;
    double total_build_ms = 0.0;
    double phase0_dedup_ms = 0.0;
    double phase1_sketch_ms = 0.0;
    size_t phase1_total_possible_pairs = 0;
    size_t phase1_candidate_pairs = 0;
    size_t phase1_cover_candidate_scans = 0;
    size_t phase1_length_pruned_candidates = 0;
    size_t phase1_lower_bound_pruned_candidates = 0;
    size_t phase1_exact_distance_reused = 0;
    size_t phase1_exact_rejection_reused = 0;
    size_t phase1_cross_layer_distance_reused = 0;
    size_t phase1_exact_distance_calls = 0;
    size_t phase1_best_cover_hits = 0;
    size_t phase1_cover_misses = 0;
    size_t phase1_scan_queries = 0;
    size_t phase1_metric_index_queries = 0;
    size_t phase1_qgram_index_queries = 0;
    size_t phase1_fallback_scan_queries = 0;
    size_t phase1_metric_distance_calls = 0;
    size_t phase1_metric_build_distance_calls = 0;
    size_t phase1_pigeonhole_queries = 0;
    size_t phase1_seed_posting_entries_visited = 0;
    size_t phase1_seed_posting_entries_stored = 0;
    size_t phase1_seed_full_posting_entries = 0;
    size_t phase1_seed_posting_bytes = 0;
    size_t phase1_pigeonhole_candidates = 0;
    size_t phase1_pigeonhole_fallbacks = 0;
    size_t phase1_hint_checks = 0;
    size_t phase1_hint_hits = 0;
    size_t phase1_qgram_touched_candidates = 0;
    size_t phase1_qgram_pruned_candidates = 0;
    double phase2_rebinding_ms = 0.0;
    double phase3_mbb_ms = 0.0;
    double phase4_attach_ms = 0.0;
    double assign_ids_ms = 0.0;
    double graph_view_ms = 0.0;
    double print_summary_ms = 0.0;
    double phase2_index_build_ms = 0.0;
    double phase2_candidate_query_ms = 0.0;
    double phase2_exact_verify_ms = 0.0;
    double phase2_candidate_query_worker_ms = 0.0;
    double phase2_exact_verify_worker_ms = 0.0;
    double phase2_edge_insert_ms = 0.0;
    size_t phase2_distance_batches = 0;
    double leaf_index_build_ms = 0.0;
    double leaf_candidate_query_ms = 0.0;
    double leaf_exact_verify_ms = 0.0;
    double leaf_tuple_emit_ms = 0.0;
    double leaf_tuple_merge_sort_ms = 0.0;
    double leaf_populate_ms = 0.0;
    double leaf_beacon_distance_ms = 0.0;
    double phase3_collect_beacons_ms = 0.0;
    double phase3_collapse_children_ms = 0.0;
    double phase3_child_mbb_distance_ms = 0.0;
    double phase3_rect_index_build_ms = 0.0;
    size_t phase3_parallel_threads = 1;
    double range_posting_lookup_ms = 0.0;
    double range_seed_union_ms = 0.0;
    double range_length_filter_ms = 0.0;
    double range_qgram_query_ms = 0.0;
    double range_hybrid_intersection_ms = 0.0;
    double range_full_scan_ms = 0.0;
    LeafAttachDirection leaf_attach_direction_used = LeafAttachDirection::Auto;
  };
  Statistics get_statistics() const;
  const HierarchyConfig& hierarchy_config() const { return hierarchy_; }
  const BuildRangeConfig& build_range_config() const { return range_config_; }
  int num_primary_layers() const { return hierarchy_.num_primary_layers(); }
  int num_expanded_layers() const { return hierarchy_.num_expanded_layers(); }
  int coarsest_primary_layer_index() const { return 0; }
  int finest_primary_layer_index() const { return std::max(0, num_primary_layers() - 1); }
  size_t num_world_nodes() const { return world_node_count_; }
  size_t num_sequences() const { return sequence_count_; }
  size_t primary_layer_size(int idx) const;
  const SequenceStore& sequence_store() const {
    return search_graph_view_.sequences;
  }
  SequenceStore& sequence_store() {
    return search_graph_view_.sequences;
  }
  bool validate_integer_ids() const;
  const SearchGraphView& search_graph_view() const { return search_graph_view_; }
  bool validate_search_graph_view() const;

 private:
  Statistics stats_;
  HierarchyConfig hierarchy_;
  BuildRangeConfig range_config_;
  size_t world_node_count_ = 0;
  size_t sequence_count_ = 0;
  SearchGraphView search_graph_view_;
  std::vector<int> expanded_radii_;
  std::vector<BuildWorldNodeRecord> build_nodes_;
  std::vector<BuildNodeGeometry> build_node_geometry_;
  std::vector<uint8_t> build_geometry_mbb_bits_;
  bool dense_leaf_mbb_ternary_ = false;
  std::array<uint8_t, 3> dense_leaf_mbb_values_{};
  bool implicit_shift_leaf_mbb_ = false;
  bool implicit_consecutive_leaf_ids_ = false;
  LeafId implicit_consecutive_leaf_radius_ = 0;
  std::vector<std::vector<NodeId>> extended_layers_;
  std::vector<std::vector<NodeId>> primary_layers_;

  std::vector<std::shared_ptr<BioSequence>> deduplicate(
      std::vector<std::shared_ptr<BioSequence>> raw);
  void initialize_sequence_store(
      const std::vector<std::shared_ptr<BioSequence>>& unique_seqs,
      bool consume_records);
  void initialize_reference_sequence_store(
      std::string reference_id,
      std::string reference_sequence,
      size_t window_length,
      size_t stride,
      std::vector<ReferenceContig> reference_contigs,
      BuildProgressReporter* progress);
  void build_impl(
      std::vector<std::shared_ptr<BioSequence>> raw_sequences,
      bool consume_records,
      std::string reference_id,
      std::string reference_sequence,
      size_t reference_window_length,
      size_t reference_stride,
      std::vector<ReferenceContig> reference_contigs = {});

  void phase1_build_extended_sketch(BuildProgressReporter* progress);
  void phase2_inter_tier_rebinding(BuildProgressReporter* progress);
  void phase3_collapse_and_compute_mbb(BuildProgressReporter* progress);

  void attach_leaves(BuildProgressReporter* progress);
  void compact_primary_build_nodes();
  void build_search_graph_view();
  void release_build_arrays();

  void print_summary() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_INDEX_BUILDER_HPP
