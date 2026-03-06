#ifndef NAVIGAMER_TOOLS_HPP
#define NAVIGAMER_TOOLS_HPP

#include "structure.hpp"
#include <vector>
#include <cstddef>

namespace navigamer {

// Levenshtein 编辑距离
int compute_distance(const std::string& a, const std::string& b);
int compute_distance(const BioSequence& a, const BioSequence& b);

// Farthest Point Sampling: 从 candidates 中选 k 个最分散的点
// candidates 为 WorldNode* 或 BioSequence*，通过 get_center_sequence 取序列
// 返回选中的下标
std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<WorldNode>>& nodes, size_t k);
std::vector<size_t> farthest_point_sampling(
    const std::vector<std::shared_ptr<BioSequence>>& sequences, size_t k);

// 随机打乱 [0, n) 的排列（用于骨架构建时的 shuffle）
void shuffle_indices(std::vector<size_t>& indices, unsigned seed);

}  // namespace navigamer

#endif  // NAVIGAMER_TOOLS_HPP
