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

// 度量边界盒：子节点球到某信标 pivot 的距离可能范围（用于剪枝）
struct MBB {
  int min_dist = 0;
  int max_dist = 0;
};

// DAG 索引节点（Top-down 拓展层 + 中间层坍缩后的主层）
// 子节点为 WorldNode*；SW 层带 BioSequence 叶子
struct WorldNode {
  std::string node_id;
  std::shared_ptr<BioSequence> center_ptr;  // 中心序列（唯一 BioSequence）
  int radius = 0;
  // 构建拓展层时 0..4；坍缩完成后 1=SW, 2=MW, 3=LW
  int layer = 0;

  std::vector<std::shared_ptr<WorldNode>> child_nodes;
  std::vector<std::shared_ptr<BioSequence>> child_leaves;

  // 中间层坍缩后：父主层节点 w 的信标（中间层中心序列）及对每个子节点的预计算 MBB
  // child_beacon_mbbs[j] 与 child_nodes[j] 对齐，长度 = beacons.size()
  std::vector<std::shared_ptr<BioSequence>> beacons;
  std::vector<std::vector<MBB>> child_beacon_mbbs;

  int data_count = 0;

  // extended_tier: 0=LW,1=INT1,2=MW,3=INT2,4=SW（仅构建拓展骨架时使用）
  WorldNode(std::shared_ptr<BioSequence> center, int r, int extended_tier);

  std::string get_center_sequence() const;
};

}  // namespace navigamer

#endif  // NAVIGAMER_STRUCTURE_HPP
