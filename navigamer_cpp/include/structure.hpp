#ifndef NAVIGAMER_STRUCTURE_HPP
#define NAVIGAMER_STRUCTURE_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include "mbb_rect_index.hpp"

namespace navigamer {

using NodeId = uint32_t;
using LeafId = uint32_t;

constexpr NodeId INVALID_NODE_ID = UINT32_MAX;
constexpr LeafId INVALID_LEAF_ID = UINT32_MAX;

// Default radius and length constants used by the C++ reference implementation.
constexpr int R_SW = 5;
constexpr int R_MW = 15;
constexpr int R_LW = 30;
constexpr int SEQ_LEN = 250;

// One occurrence of a sequence on the reference genome.
struct RefPosition {
  std::string ref_id;
  int start = 0;
  int end = 0;
  std::string strand = "+";
};

// Optional suffix-array interval placeholder: [start, end).
struct BwtInterval {
  int64_t start = -1;
  int64_t end   = -1;
  bool valid() const { return start >= 0 && end >= start; }
};

// One contig embedded in a flattened reference string. Coordinates are
// half-open global offsets into SequenceStore::reference_sequence.
struct ReferenceContig {
  std::string id;
  uint32_t begin = 0;
  uint32_t end = 0;
  // Coordinate of `begin` in the original contig. Zero for an unsharded
  // reference; non-zero for a stored shard slice.
  uint32_t source_begin = 0;
};

// The atomic database object indexed by NavigaMer.
struct BioSequence {
  std::string id;
  std::string seq;
  LeafId sequence_id = INVALID_LEAF_ID;
  bool has_source_pos = false;
  size_t source_pos = 0;
  std::vector<RefPosition> ref_positions;
  BwtInterval bwt_interval;

  BioSequence() = default;
  BioSequence(std::string seq_id, std::string sequence)
      : id(std::move(seq_id)), seq(std::move(sequence)) {}

  void add_occurrence(const std::string& ref_id, int start, int end,
                      const std::string& strand = "+");
  void set_sa_interval(int64_t sa_start, int64_t sa_end);
  void set_bwt_interval(int64_t bwt_start, int64_t bwt_end);
};

// Metric bounding box for a child world relative to one parent-local beacon.
struct MBB {
  int min_dist = 0;
  int max_dist = 0;
};

// Node in the navigable world DAG.
struct WorldNode {
  std::string node_id;
  NodeId integer_id = INVALID_NODE_ID;
  std::shared_ptr<BioSequence> center_ptr;
  int radius = 0;
  int expanded_layer_index = -1;
  int primary_layer_index = -1;
  bool is_primary = false;

  std::vector<std::shared_ptr<WorldNode>> child_nodes;
  std::vector<std::shared_ptr<BioSequence>> child_leaves;

  // Parent-local beacons and per-child MBB rows used by Appendix C pruning.
  // child_beacon_mbbs[j] is aligned with child_nodes[j].
  std::vector<std::shared_ptr<BioSequence>> beacons;
  std::vector<std::vector<MBB>> child_beacon_mbbs;
  std::shared_ptr<MBBRectIndex> mbb_rect_index;

  // Leaf refinement cache for SW worlds. leaf_beacon_dists[j] is aligned
  // with child_leaves[j] and stores d(child_leaf, beacon_i).
  std::vector<std::vector<int>> leaf_beacon_dists;

  int data_count = 0;

  WorldNode(std::shared_ptr<BioSequence> center, int r, int expanded_layer_idx);

  std::string get_center_sequence() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_STRUCTURE_HPP
