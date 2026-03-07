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
  static constexpr int BEACON_COUNT = 3;

  // 默认构造：使用默认半径 R_SW=5, R_MW=15, R_LW=30
  BioGeometryIndexBuilder();

  // 指定各层半径构造：r_sw=小世界半径, r_mw=中世界半径, r_lw=大世界半径
  BioGeometryIndexBuilder(int r_sw, int r_mw, int r_lw);

  // 主构建入口：raw_sequences 为输入 reads
  void build(const std::vector<std::shared_ptr<BioSequence>>& raw_sequences);

  // 各层节点 (1=SW, 2=MW, 3=LW)
  std::vector<std::shared_ptr<WorldNode>> layers[4];  // 0 未用
  std::vector<std::shared_ptr<WorldNode>> layer_beacons[4];  // 每层 FPS 选出的 Beacons
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

  // 可扩展：查找 query 在 candidates 中距离 <= radius 的邻居（当前线性扫描）
  std::vector<std::shared_ptr<WorldNode>> find_neighbors(
      const BioSequence& query_seq,
      const std::vector<std::shared_ptr<WorldNode>>& candidates,
      int radius) const;

 private:
  Statistics stats_;
  int radius_config[4] = {0, R_SW, R_MW, R_LW};

  std::vector<std::shared_ptr<BioSequence>> deduplicate(
      const std::vector<std::shared_ptr<BioSequence>>& raw);

  void build_skeleton(const std::vector<std::shared_ptr<BioSequence>>& unique_seqs);
  std::vector<std::shared_ptr<WorldNode>> build_layer_sparse(
      const std::vector<std::shared_ptr<BioSequence>>& items,
      int radius, int layer_level, const std::string& label);
  std::vector<std::shared_ptr<WorldNode>> build_layer_sparse_from_nodes(
      const std::vector<std::shared_ptr<WorldNode>>& items,
      int radius, int layer_level, const std::string& label);

  void dense_wiring();
  void wire_overlap(std::vector<std::shared_ptr<WorldNode>>& parents,
                    const std::vector<std::shared_ptr<WorldNode>>& children);

  void inject_beacons();
  void attach_leaves(const std::vector<std::shared_ptr<BioSequence>>& unique_seqs);

  void print_summary() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_INDEX_BUILDER_HPP
