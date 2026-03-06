"""
工具函数：Edit Distance 计算、2-bit 模拟器、数据生成
"""

import random
from typing import List, Dict, Tuple
from .structure import BioSequence, GenomePointer


def compute_distance(seq_a: BioSequence, seq_b: BioSequence) -> int:
    """
    计算 Levenshtein Distance (编辑距离)
    使用标准的动态规划算法
    
    Args:
        seq_a: 序列 A
        seq_b: 序列 B
        
    Returns:
        编辑距离
    """
    a = seq_a.seq
    b = seq_b.seq
    m, n = len(a), len(b)
    
    # 创建 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 删除
                    dp[i][j-1] + 1,      # 插入
                    dp[i-1][j-1] + 1     # 替换
                )
    
    return dp[m][n]


def _center_for_distance(obj):
    """Extract a sequence object for distance calculation. BioSequence -> self; WorldNode -> center as BioSequence."""
    if isinstance(obj, BioSequence):
        return obj
    from .structure import WorldNode
    if isinstance(obj, WorldNode):
        seq = obj.get_center_sequence()
        return BioSequence("_center", seq)
    if hasattr(obj, 'seq'):
        return obj
    return None


def farthest_point_sampling(candidates: List, k: int, distance_func=None):
    """
    Farthest Point Sampling (FPS): select k anchors that are maximally spread.
    Used for post-build refinement of routing anchors.
    
    Args:
        candidates: List of BioSequence or WorldNode (must have .seq or get_center_sequence).
        k: Number of anchors to select.
        distance_func: (a, b) -> int; defaults to compute_distance.
    
    Returns:
        List of k candidates (subset of candidates).
    """
    if distance_func is None:
        distance_func = compute_distance
    if not candidates or k <= 0:
        return []
    k = min(k, len(candidates))
    centers = [_center_for_distance(c) for c in candidates]
    if any(c is None for c in centers):
        return list(candidates[:k])
    chosen_idx = [random.randint(0, len(candidates) - 1)]
    while len(chosen_idx) < k:
        best_idx = -1
        best_min_dist = -1
        for i in range(len(candidates)):
            if i in chosen_idx:
                continue
            min_d = min(distance_func(centers[i], centers[j]) for j in chosen_idx)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = i
        if best_idx < 0:
            break
        chosen_idx.append(best_idx)
    return [candidates[i] for i in chosen_idx]


class TwoBitSimulator:
    """2-bit 格式模拟器（内存版本）
    模拟从参考基因组提取序列的功能
    """
    
    def __init__(self):
        """初始化模拟器"""
        # 使用字典存储序列数据（模拟磁盘存储）
        # key: (chrom, pos), value: sequence
        self._sequences: Dict[Tuple[str, int], str] = {}
        self._chromosomes: Dict[str, str] = {}  # 存储完整染色体序列
    
    def add_chromosome(self, chrom: str, sequence: str):
        """
        添加染色体序列
        
        Args:
            chrom: 染色体标识
            sequence: 完整序列
        """
        self._chromosomes[chrom] = sequence
    
    def add_sequence(self, chrom: str, pos: int, sequence: str):
        """
        添加序列片段（用于直接存储）
        
        Args:
            chrom: 染色体标识
            pos: 起始位置
            sequence: 序列内容
        """
        self._sequences[(chrom, pos)] = sequence
    
    def extract_sequence(self, chrom: str, pos: int, length: int) -> str:
        """
        从参考基因组提取序列（模拟 O(1) 提取）
        
        Args:
            chrom: 染色体标识
            pos: 起始位置
            length: 提取长度
            
        Returns:
            提取的序列字符串
        """
        # 优先从完整染色体提取
        if chrom in self._chromosomes:
            full_seq = self._chromosomes[chrom]
            if pos + length <= len(full_seq):
                return full_seq[pos:pos + length]
            else:
                # 超出范围，返回可用部分
                return full_seq[pos:]
        
        # 从片段字典提取
        if (chrom, pos) in self._sequences:
            seq = self._sequences[(chrom, pos)]
            if len(seq) >= length:
                return seq[:length]
            else:
                return seq
        
        # 如果都不存在，返回空字符串
        return ""
    
    def get_pointer(self, chrom: str, pos: int, length: int) -> GenomePointer:
        """
        获取 GenomePointer 对象
        
        Args:
            chrom: 染色体标识
            pos: 起始位置
            length: 序列长度
            
        Returns:
            GenomePointer 对象
        """
        seq = self.extract_sequence(chrom, pos, length)
        return GenomePointer(chrom, pos, seq)


def generate_reference_sequence(length: int = 100000, seed: int = None) -> str:
    """
    生成长 DNA 序列
    
    Args:
        length: 序列长度（默认 100,000 bp）
        seed: 随机种子
        
    Returns:
        DNA 序列字符串
    """
    if seed is not None:
        random.seed(seed)
    
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choices(bases, k=length))


def mutate_sequence(sequence: str, mutation_rate: float = 0.03) -> str:
    """
    对序列引入随机突变
    
    Args:
        sequence: 原始序列
        mutation_rate: 突变率（0.01-0.05，即 1%-5%）
        
    Returns:
        突变后的序列
    """
    bases = ['A', 'T', 'C', 'G']
    seq_list = list(sequence)
    length = len(seq_list)
    
    # 计算突变次数
    num_mutations = max(1, int(length * mutation_rate))
    
    for _ in range(num_mutations):
        idx = random.randint(0, len(seq_list) - 1)
        op = random.choice(['sub', 'del', 'ins'])
        
        if op == 'sub':  # 替换
            # 确保替换为不同的碱基
            original = seq_list[idx]
            choices = [b for b in bases if b != original]
            if choices:
                seq_list[idx] = random.choice(choices)
        elif op == 'del':  # 删除
            if len(seq_list) > length // 2:  # 防止删除过多
                del seq_list[idx]
        elif op == 'ins':  # 插入
            seq_list.insert(idx, random.choice(bases))
    
    return ''.join(seq_list)


def generate_reads_with_mutations(
    reference_seq: str,
    num_reads: int,
    read_length: int = 250,
    mutation_rate: float = 0.03,
    seed: int = None
) -> List[BioSequence]:
    """
    从参考序列生成带突变的 Reads
    
    Args:
        reference_seq: 参考序列
        num_reads: 生成的 Reads 数量
        read_length: Read 长度（默认 250）
        mutation_rate: 突变率（默认 0.03，即 3%）
        seed: 随机种子
        
    Returns:
        BioSequence 对象列表
    """
    if seed is not None:
        random.seed(seed)
    
    reads = []
    ref_len = len(reference_seq)
    
    for i in range(num_reads):
        # 随机选择起始位置
        if ref_len > read_length:
            start_pos = random.randint(0, ref_len - read_length)
            # 截取片段
            fragment = reference_seq[start_pos:start_pos + read_length]
        else:
            # 如果参考序列太短，直接使用
            fragment = reference_seq
        
        # 引入突变
        mutated = mutate_sequence(fragment, mutation_rate)
        
        # 创建 BioSequence 对象
        read = BioSequence(f"read_{i:05d}", mutated)
        reads.append(read)
    
    return reads


def generate_random_dna_sequence(seq_id: str, min_len: int = 50, max_len: int = 100) -> BioSequence:
    """
    生成随机 DNA 序列
    
    Args:
        seq_id: 序列 ID
        min_len: 最小长度
        max_len: 最大长度
        
    Returns:
        BioSequence 对象
    """
    length = random.randint(min_len, max_len)
    bases = ['A', 'T', 'C', 'G']
    seq = ''.join(random.choices(bases, k=length))
    return BioSequence(seq_id=seq_id, seq=seq)
