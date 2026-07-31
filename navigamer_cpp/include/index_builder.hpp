#ifndef NAVIGAMER_INDEX_BUILDER_HPP
#define NAVIGAMER_INDEX_BUILDER_HPP

#include "structure.hpp"
#include "tools.hpp"
#include "range_join.hpp"
#include "phase2_distance_verifier.hpp"
#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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
};

// A reference-backed leaf needs no BioSequence object. The fixed k-mer is a
// view of reference_sequence at source_pos.
struct ReferenceSequenceRecord {
  uint32_t source_pos = 0;
};
static_assert(sizeof(ReferenceSequenceRecord) == 4,
              "reference leaf record must remain a compact 4-byte value");

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
// implicit from the position in records/reference_records, so nodes, beacons,
// leaf links, and search results only need a 32-bit integer reference.
struct SequenceStore {
  std::vector<BioSequence> records;
  std::vector<ReferenceSequenceRecord> reference_records;
  // A sequence with one extra position uses one compact pair. Sequences with
  // two or more extras share one group record and a flat position array.
  std::vector<ReferenceOccurrence> singleton_occurrences;
  std::vector<ReferenceOccurrenceGroup> occurrence_groups;
  std::vector<uint32_t> grouped_occurrence_positions;
  std::vector<ReferenceContig> reference_contigs;
  std::string reference_id;
  std::string reference_sequence;
  size_t fixed_sequence_length = 0;
  bool reference_backed = false;

  size_t size() const {
    return reference_backed ? reference_records.size() : records.size();
  }
  bool empty() const { return size() == 0; }
  const BioSequence& at(LeafId id) const {
    return records.at(static_cast<size_t>(id));
  }
  const BioSequence& operator[](LeafId id) const {
    return records[static_cast<size_t>(id)];
  }
  std::string_view sequence(LeafId id) const {
    if (reference_backed) {
      const auto& record =
          reference_records[static_cast<size_t>(id)];
      return std::string_view(
          reference_sequence.data() + record.source_pos,
          fixed_sequence_length);
    }
    return records[static_cast<size_t>(id)].seq;
  }
  size_t source_position(LeafId id) const {
    if (reference_backed) {
      return reference_records.at(static_cast<size_t>(id)).source_pos;
    }
    return records.at(static_cast<size_t>(id)).source_pos;
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
           std::to_string(global_pos - contig.begin);
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

// One fixed-size world record. Variable-length relationships live in the
// global arrays in SearchGraphView and are addressed by offset + count.
struct WorldNodeRecord {
  enum class BeaconStorage : uint32_t {
    Delta8 = 0,
    Delta16 = 1,
    Absolute32 = 2,
    ImplicitCenter = 3,
  };
  static constexpr uint32_t BEACON_COUNT_MASK = (uint32_t{1} << 30) - 1;

  LeafId center_sequence_id = INVALID_LEAF_ID;

  uint32_t child_begin = 0;
  uint32_t child_count = 0;
  uint32_t leaf_begin = 0;
  uint32_t leaf_count = 0;
  uint32_t beacon_begin = 0;
  uint32_t beacon_count_and_storage = 0;
  uint32_t mbb_begin = 0;

  uint32_t beacon_count() const {
    return beacon_count_and_storage & BEACON_COUNT_MASK;
  }
  BeaconStorage beacon_storage() const {
    return static_cast<BeaconStorage>(
        beacon_count_and_storage >> 30);
  }
  void set_beacon_layout(uint32_t begin, uint32_t count,
                         BeaconStorage storage) {
    if (count > BEACON_COUNT_MASK) {
      throw std::length_error("node beacon count exceeds packed range");
    }
    beacon_begin = begin;
    beacon_count_and_storage =
        count | (static_cast<uint32_t>(storage) << 30);
  }
};
static_assert(sizeof(WorldNodeRecord) == 32,
              "finalized world node must remain a cache-friendly 32 bytes");

// Mutable construction record. It intentionally contains only integer
// references: no WorldNode objects are allocated while building the hierarchy.
// The per-node vectors are flattened into SearchGraphView after all build
// phases have produced the same node/edge sets as the original algorithm.
struct BuildWorldNodeRecord {
  LeafId center_sequence_id = INVALID_LEAF_ID;
  int radius = 0;
  int expanded_layer_index = -1;
  int primary_layer_index = -1;
  bool is_primary = false;

  // Phase 1-3 child NodeIds. Finest-layer nodes reuse the same uint32_t
  // storage for LeafIds after their child list is cleared.
  std::vector<NodeId> child_or_leaf_ids;
  std::vector<LeafId> beacon_ids;
  // Dimension-major, matching SearchGraphView directly:
  // [beacon_index * child_count + child_index].
  // Stores d(child center, beacon); the child radius reconstructs the MBB.
  std::vector<uint8_t> child_beacon_dists;
  // Dimension-major: [beacon_index * leaf_count + leaf_index].
  std::vector<uint8_t> leaf_beacon_dists;
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
  const_iterator begin() const { return data(); }
  const_iterator end() const {
    const T* first = data();
    return empty() ? first : first + size();
  }
  bool is_mapped() const { return static_cast<bool>(mapped_owner_); }

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

  std::vector<T> owned_;
  std::shared_ptr<const void> mapped_owner_;
  const T* data_ = nullptr;
  size_t size_ = 0;
};

struct SearchGraphView {
  // Canonical array representation. NodeId and LeafId are positions in
  // node_records and sequences respectively.
  FinalArray<WorldNodeRecord> node_records;
  SequenceStore sequences;
  std::vector<uint32_t> layer_begin;
  std::vector<uint32_t> layer_end;

  FinalArray<NodeId> child_ids;
  FinalArray<LeafId> leaf_ids;

  FinalArray<uint8_t> child_beacon_dists;
  FinalArray<int8_t> beacon_deltas8;
  FinalArray<int16_t> beacon_deltas16;
  FinalArray<LeafId> beacon_ids32;

  FinalArray<uint8_t> leaf_beacon_dists;

  LeafId beacon_sequence_id(NodeId node_id,
                            uint32_t beacon_offset) const {
    const auto& node = node_records[node_id];
    switch (node.beacon_storage()) {
      case WorldNodeRecord::BeaconStorage::Delta8:
        return static_cast<LeafId>(
            static_cast<int64_t>(node.center_sequence_id) +
            beacon_deltas8[node.beacon_begin + beacon_offset]);
      case WorldNodeRecord::BeaconStorage::Delta16:
        return static_cast<LeafId>(
            static_cast<int64_t>(node.center_sequence_id) +
            beacon_deltas16[node.beacon_begin + beacon_offset]);
      case WorldNodeRecord::BeaconStorage::Absolute32:
        return beacon_ids32[node.beacon_begin + beacon_offset];
      case WorldNodeRecord::BeaconStorage::ImplicitCenter:
        return node.center_sequence_id;
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
  std::vector<NodeId> final_node_ids_;
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
  void assign_integer_ids();
  void build_search_graph_view();
  void release_build_arrays();

  void print_summary() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_INDEX_BUILDER_HPP
