#ifndef NAVIGAMER_INDEX_BUILDER_HPP
#define NAVIGAMER_INDEX_BUILDER_HPP

#include "structure.hpp"
#include "tools.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace navigamer {

class BioGeometryIndexBuilder {
 public:
  BioGeometryIndexBuilder();
  BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw);

  void build(const std::vector<std::shared_ptr<BioSequence>>& raw_sequences);

  // 各层节点 (1=SW, 2=MW, 3=LW)
  std::vector<std::shared_ptr<WorldNode>> layers[4];  // 0 未用
  std::unordered_map<std::string, std::shared_ptr<BioSequence>> unique_sequences;

  struct Statistics {
    size_t added_sequences = 0;
    size_t unique_sequences = 0;
    size_t deduplicated = 0;
    size_t created_nodes[4] = {0, 0, 0, 0};
    double compression_ratio = 0.0;
    double dag_redundancy = 0.0;
  };
  Statistics get_statistics() const;

  std::vector<std::shared_ptr<WorldNode>> find_neighbors(
      const BioSequence& query_seq,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      int radius) const;

 private:
  Statistics stats_;
  int radius_config[4] = {0, R_SW, R_MW, R_LW};

  // 5 拓展层半径: LW -> INT1 -> MW -> INT2 -> SW
  std::vector<int> extended_radii_;
  std::vector<std::vector<std::shared_ptr<WorldNode>>> extended_layers_;

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
