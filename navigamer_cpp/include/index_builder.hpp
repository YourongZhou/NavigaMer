#ifndef NAVIGAMER_INDEX_BUILDER_HPP
#define NAVIGAMER_INDEX_BUILDER_HPP

#include "structure.hpp"
#include "tools.hpp"
#include "range_join.hpp"
#include "phase2_distance_verifier.hpp"
#include <algorithm>
#include <iterator>
#include <memory>
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

// Canonical, pointer-free sequence storage for a finalized index. SequenceId is
// implicit from the position in records/reference_records, so nodes, beacons,
// leaf links, and search results only need a 32-bit integer reference.
struct SequenceStore {
  std::vector<BioSequence> records;
  std::vector<ReferenceSequenceRecord> reference_records;
  // Every non-representative occurrence is stored. Sorted by
  // (sequence_id, source_pos), so unique windows pay no side-array cost.
  std::vector<ReferenceOccurrence> additional_occurrences;
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
    const auto range = additional_occurrence_range(id);
    positions.reserve(
        static_cast<size_t>(std::distance(range.first, range.second)) + 1);
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
    const auto range = additional_occurrence_range(id);
    bool emitted_representative = false;
    for (auto it = range.first; it != range.second; ++it) {
      if (!emitted_representative &&
          representative < it->source_pos) {
        visit(representative);
        emitted_representative = true;
      }
      visit(it->source_pos);
    }
    if (!emitted_representative) visit(representative);
  }
  std::pair<std::vector<ReferenceOccurrence>::const_iterator,
            std::vector<ReferenceOccurrence>::const_iterator>
  additional_occurrence_range(LeafId id) const {
    const auto first = std::lower_bound(
        additional_occurrences.begin(), additional_occurrences.end(), id,
        [](const ReferenceOccurrence& occurrence, LeafId sequence_id) {
          return occurrence.sequence_id < sequence_id;
        });
    const auto last = std::upper_bound(
        first, additional_occurrences.end(), id,
        [](LeafId sequence_id, const ReferenceOccurrence& occurrence) {
          return sequence_id < occurrence.sequence_id;
        });
    return {first, last};
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
  LeafId center_sequence_id = INVALID_LEAF_ID;
  int radius = 0;
  int expanded_layer_index = -1;
  int primary_layer_index = -1;

  uint32_t child_begin = 0;
  uint32_t child_count = 0;
  uint32_t leaf_begin = 0;
  uint32_t leaf_count = 0;
  uint32_t beacon_begin = 0;
  uint32_t beacon_count = 0;
  uint32_t mbb_begin = 0;
  uint32_t leaf_beacon_begin = 0;
};

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
  // Each uint16_t packs lower in bits 0..7 and upper in bits 8..15.
  std::vector<uint16_t> child_mbb_bounds;
  // Dimension-major: [beacon_index * leaf_count + leaf_index].
  std::vector<uint8_t> leaf_beacon_dists;
};

struct SearchGraphView {
  // Canonical array representation. NodeId and LeafId are positions in
  // node_records and sequences respectively.
  std::vector<WorldNodeRecord> node_records;
  SequenceStore sequences;
  std::vector<uint32_t> layer_begin;
  std::vector<uint32_t> layer_end;

  std::vector<NodeId> child_ids;
  std::vector<LeafId> leaf_ids;

  std::vector<uint8_t> mbb_lo;
  std::vector<uint8_t> mbb_hi;
  std::vector<LeafId> beacon_ids;

  std::vector<uint8_t> leaf_beacon_dists;
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
