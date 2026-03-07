#ifndef NAVIGAMER_STRUCTURE_HPP
#define NAVIGAMER_STRUCTURE_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace navigamer {

// 半径与长度常量 (与 Python 一致)
constexpr int R_SW = 5;
constexpr int R_MW = 15;
constexpr int R_LW = 30;
constexpr int SEQ_LEN = 250;

// 参考基因组上的一个出现位置: (ref_id, start, end, strand)
struct RefPosition {
  std::string ref_id;
  int start = 0;
  int end = 0;
  std::string strand = "+";
};

// BWT/SA 区间：叶子节点在 FM-Index 中的 [start, end) 范围
struct BwtInterval {
  int64_t start = -1;
  int64_t end   = -1;
  bool valid() const { return start >= 0 && end >= start; }
};

// 基础数据单元：DNA 序列 + 参考基因组位置列表 + BWT 区间
struct BioSequence {
  std::string id;
  std::string seq;
  std::vector<RefPosition> ref_positions;
  BwtInterval bwt_interval;

  BioSequence() = default;
  BioSequence(std::string seq_id, std::string sequence)
      : id(std::move(seq_id)), seq(std::move(sequence)) {}

  void add_occurrence(const std::string& ref_id, int start, int end,
                      const std::string& strand = "+");
  void set_bwt_interval(int64_t bwt_start, int64_t bwt_end);
};

// DAG 索引节点 (NavigaMer v7 - Multilateration-Enhanced)
// 子节点可能是 WorldNode* 或 BioSequence*（叶子）
struct WorldNode {
  std::string node_id;
  std::shared_ptr<BioSequence> center_ptr;  // 中心序列（唯一 BioSequence）
  int radius = 0;
  int layer = 0;  // 1=SW, 2=MW, 3=LW

  std::vector<std::shared_ptr<WorldNode>> child_nodes;
  std::vector<std::shared_ptr<BioSequence>> child_leaves;

  std::vector<int> beacon_dists;  // 到父层 K 个 Beacons 的距离
  int data_count = 0;

  WorldNode(std::shared_ptr<BioSequence> center, int r, int layer_level);

  std::string get_center_sequence() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_STRUCTURE_HPP
