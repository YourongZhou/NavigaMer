#ifndef NAVIGAMER_INDEX_BUILDER_HPP
#define NAVIGAMER_INDEX_BUILDER_HPP

#include "structure.hpp"
#include "tools.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace navigamer {

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

class BioGeometryIndexBuilder {
 public:
  BioGeometryIndexBuilder();
  BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw);
  explicit BioGeometryIndexBuilder(const HierarchyConfig& config);

  void build(const std::vector<std::shared_ptr<BioSequence>>& raw_sequences);

  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_sequences;

  struct Statistics {
    size_t added_sequences = 0;
    size_t unique_sequences = 0;
    size_t deduplicated = 0;
    size_t created_auxiliary_nodes = 0;
    std::vector<size_t> created_primary_nodes;
    double compression_ratio = 0.0;
    double dag_redundancy = 0.0;
  };
  Statistics get_statistics() const;
  const HierarchyConfig& hierarchy_config() const { return hierarchy_; }
  int num_primary_layers() const { return hierarchy_.num_primary_layers(); }
  int num_expanded_layers() const { return hierarchy_.num_expanded_layers(); }
  int coarsest_primary_layer_index() const { return 0; }
  int finest_primary_layer_index() const { return std::max(0, num_primary_layers() - 1); }
  const std::vector<std::shared_ptr<WorldNode>>& primary_layer(int idx) const;
  const std::vector<std::vector<std::shared_ptr<WorldNode>>>& primary_layers() const {
    return primary_layers_;
  }

  std::vector<std::shared_ptr<WorldNode>> find_neighbors(
      const BioSequence& query_seq,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      int radius) const;

 private:
  Statistics stats_;
  HierarchyConfig hierarchy_;
  std::vector<int> expanded_radii_;
  std::vector<std::vector<std::shared_ptr<WorldNode>>> extended_layers_;
  std::vector<std::vector<std::shared_ptr<WorldNode>>> primary_layers_;

  std::vector<std::shared_ptr<BioSequence>> deduplicate(
      const std::vector<std::shared_ptr<BioSequence>>& raw);

  void phase1_build_extended_sketch(const std::vector<std::shared_ptr<BioSequence>>& unique_seqs);
  void phase2_inter_tier_rebinding();
  void phase3_collapse_and_compute_mbb();

  void attach_leaves(const std::vector<std::shared_ptr<BioSequence>>& unique_seqs);

  void print_summary() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_INDEX_BUILDER_HPP
