#!/usr/bin/env python3
"""
BioGeometry DAG Index (Prototype)
基于 DAG 和几何度量空间的生物序列索引原型系统
"""

import random
from typing import List, Set, Dict, Any
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import List



class BioSequence:
    """基础数据单元，封装原始字符串"""
    
    def __init__(self, seq_id: str, seq: str):
        """
        Args:
            seq_id: 唯一标识符
            seq: DNA 序列字符串 (A, T, C, G)
        """
        self.id = seq_id
        self.seq = seq
    
    def __repr__(self):
        """返回序列摘要"""
        seq_preview = self.seq[:20] + "..." if len(self.seq) > 20 else self.seq
        return f"BioSequence(id={self.id}, seq={seq_preview}, len={len(self.seq)})"


class GeoNode:
    """通用的图节点类，适用于 SW, MW, LW"""
    
    def __init__(self, node_id: str, center: BioSequence, radius: int, layer_name: str):
        """
        Args:
            node_id: 唯一节点 ID (例如 "SW_1", "MW_5")
            center: 该节点的几何中心序列
            radius: 该节点的覆盖半径 (固定值，由层级配置决定)
            layer_name: "SW", "MW", "LW"
        """
        self.node_id = node_id
        self.center = center
        self.radius = radius
        self.layer_name = layer_name
        self.children = []  # 存储子元素的列表
    
    def is_overlapping(self, query_seq: BioSequence, query_radius: int, distance_func) -> bool:
        """
        判断查询球体与节点球体是否相交
        
        Args:
            query_seq: Query 序列对象
            query_radius: Query 容错半径
            distance_func: 距离计算函数
            
        Returns:
            如果 dist(query, self.center) <= self.radius + query_radius，返回 True
        """
        dist = distance_func(query_seq, self.center)
        return dist <= self.radius + query_radius
    
    def add_child(self, child_obj: Any):
        """将子对象加入 children 列表"""
        self.children.append(child_obj)
    
    def __repr__(self):
        return f"GeoNode(id={self.node_id}, layer={self.layer_name}, radius={self.radius}, children_count={len(self.children)})"

# 必须定义在顶层，因为多进程需要序列化 (Pickle) 函数
def _worker_scan(chunk_data):
    """
    子进程的工作函数：在一个小块数据中搜索
    """
    database_chunk, query, tolerance = chunk_data
    results = []
    # 这里使用的是简单的循环，但因为是在独立进程跑，所以是并行的
    for seq in database_chunk:
        # 注意：这里我们直接调用静态方法，避免传递整个Index对象
        dist = BioGeometryIndex.compute_distance(seq, query)
        if dist <= tolerance:
            results.append(seq)
    return results

def parallel_linear_scan(database: List[BioSequence], query: BioSequence, tolerance: int) -> List[BioSequence]:
    """
    多进程并行版本的暴力搜索
    """
    num_cores = multiprocessing.cpu_count()
    # 简单的任务分片策略
    chunk_size = len(database) // num_cores + 1
    chunks = []
    
    for i in range(0, len(database), chunk_size):
        # 每个 chunk 是一个 tuple: (数据片段, 查询, 容错)
        chunk = database[i:i + chunk_size]
        chunks.append((chunk, query, tolerance))
        
    print(f"  [Parallel] 利用 {num_cores} 个 CPU 核，将数据分为 {len(chunks)} 个块并行计算...")

    final_results = []
    # 启动进程池
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # map 会保持顺序返回结果
        results_list = executor.map(_worker_scan, chunks)
        
        for res in results_list:
            final_results.extend(res)
            
    return final_results

# 定义在顶层
def _worker_connect(args):
    """
    子进程：计算一批 item 应该连到哪些 nodes 上
    """
    items_chunk, nodes_centers, radius = args
    connections = [] # 存储 (item_index_in_chunk, node_index)
    
    for i, item in enumerate(items_chunk):
        item_center = item if isinstance(item, BioSequence) else item.center
        
        for node_idx, center_seq in enumerate(nodes_centers):
            # 这里我们只传 center 序列，不传整个 Node 对象以减少序列化开销
            dist = BioGeometryIndex.compute_distance(item_center, center_seq)
            if dist <= radius:
                connections.append((i, node_idx))
    return connections

class BioGeometryIndex:
    """索引的主控类"""
    
    def __init__(self):
        """初始化索引"""
        self.layers: Dict[int, List] = {
            0: [],  # Layer 0: BioSequence
            1: [],  # Layer 1: SW
            2: [],  # Layer 2: MW
            3: []   # Layer 3: LW
        }
        # 半径配置: {Layer ID: radius}
        self.config = {
            1: 5,   # SW radius
            2: 15,  # MW radius
            3: 30   # LW radius
        }
    
    @staticmethod
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
    
    def build(self, raw_sequences: List[BioSequence]):
        """
        构建索引的主入口
        采用 Bottom-Up + Multi-Assignment 策略
        
        Args:
            raw_sequences: 原始序列列表
        """
        # Layer 0: 存储原始序列
        self.layers[0] = raw_sequences
        
        # Bottom-Up 构建：从 Layer 1 (SW) 开始，逐层向上
        # Layer 1 (SW): 基于 Layer 0 (BioSequence) 构建
        self.layers[1] = self._build_layer(
            items=self.layers[0],
            layer_id=1,
            radius=self.config[1],
            layer_name="SW"
        )
        
        # Layer 2 (MW): 基于 Layer 1 (SW) 构建
        self.layers[2] = self._build_layer(
            items=self.layers[1],
            layer_id=2,
            radius=self.config[2],
            layer_name="MW"
        )
        
        # Layer 3 (LW): 基于 Layer 2 (MW) 构建
        self.layers[3] = self._build_layer(
            items=self.layers[2],
            layer_id=3,
            radius=self.config[3],
            layer_name="LW"
        )
    
    def _build_layer(self, items: List, layer_id: int, radius: int, layer_name: str) -> List[GeoNode]:
        """
        构建单层的内部方法
        包含 Pivot Selection 和 DAG Connection
        
        Args:
            items: 当前层的对象列表（Layer 0 是 BioSequence，其他层是 GeoNode）
            layer_id: 目标层 ID
            radius: 该层的半径
            layer_name: 层名称 ("SW", "MW", "LW")
            
        Returns:
            构建好的 GeoNode 列表
        """
        if not items:
            return []
        
        # ===== Pivot Selection (选中心) =====
        # 使用贪婪策略 (Greedy/Leader Algorithm)
        centers = []
        
        for item in items:
            # 获取当前 item 的中心序列
            if isinstance(item, BioSequence):
                item_center = item
            else:  # GeoNode
                item_center = item.center
            
            # 检查是否距离所有已有 centers 都大于 radius
            is_far_enough = True
            for center_item in centers:
                if isinstance(center_item, BioSequence):
                    center_seq = center_item
                else:  # GeoNode
                    center_seq = center_item.center
                
                dist = self.compute_distance(item_center, center_seq)
                if dist <= radius:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                centers.append(item)
        
        # ===== DAG Connection (建立连接 - 多重归属) =====
        nodes = []
        
        # 根据选出的 centers 创建空的 GeoNode 对象
        for idx, center_item in enumerate(centers):
            if isinstance(center_item, BioSequence):
                center_seq = center_item
            else:  # GeoNode
                center_seq = center_item.center
            
            node_id = f"{layer_name}_{idx}"
            node = GeoNode(
                node_id=node_id,
                center=center_seq,
                radius=radius,
                layer_name=layer_name
            )
            nodes.append(node)
        
        # 准备数据给子进程
        # 我们只传递 center 的序列字符串，因为传递整个对象太慢
        nodes_centers = [n.center for n in nodes]
        
        num_cores = multiprocessing.cpu_count()
        chunk_size = len(items) // num_cores + 1
        chunks_args = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            # 把数据打包：(这一块items, 所有node的中心, 半径)
            chunks_args.append((chunk, nodes_centers, radius))
            
        print(f"  [Build Layer {layer_id}] 启动 {num_cores} 核并行构建连接...")
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = executor.map(_worker_connect, chunks_args)
            
            # 汇总结果：主进程负责把 item 真的塞进 node
            for chunk_idx, chunk_connections in enumerate(results):
                base_idx = chunk_idx * chunk_size
                for local_item_idx, node_idx in chunk_connections:
                    # 还原全局 item
                    real_item = items[base_idx + local_item_idx]
                    # 添加连接
                    nodes[node_idx].add_child(real_item)
                    
        return nodes
    
    def search(self, query_seq: BioSequence, tolerance: int) -> tuple:
        """
        搜索入口
        采用 Top-Down Multilateration Pruning
        
        Args:
            query_seq: 查询序列
            tolerance: 容错半径
            
        Returns:
            (results, stats) 元组
            results: 匹配的序列列表
            stats: 统计信息字典，包含节点访问量
        """
        stats = {
            'lw_visited': 0,
            'mw_visited': 0,
            'sw_visited': 0,
            'mw_candidates': 0,
            'sw_candidates': 0,
            'raw_candidates': 0
        }
        
        # ===== Layer 3 (LW) Search =====
        candidate_mws: Set[GeoNode] = set()
        
        for lw_node in self.layers[3]:
            stats['lw_visited'] += 1
            if lw_node.is_overlapping(query_seq, tolerance, self.compute_distance):
                # 添加所有子节点（MW nodes）
                for mw_node in lw_node.children:
                    candidate_mws.add(mw_node)
        
        stats['mw_candidates'] = len(candidate_mws)
        
        # ===== Layer 2 (MW) Search =====
        candidate_sws: Set[GeoNode] = set()
        
        for mw_node in candidate_mws:
            stats['mw_visited'] += 1
            if mw_node.is_overlapping(query_seq, tolerance, self.compute_distance):
                # 添加所有子节点（SW nodes）
                for sw_node in mw_node.children:
                    candidate_sws.add(sw_node)
        
        stats['sw_candidates'] = len(candidate_sws)
        
        # ===== Layer 1 (SW) Search (Block Retrieval) =====
        final_candidate_sequences: List[BioSequence] = []
        
        for sw_node in candidate_sws:
            stats['sw_visited'] += 1
            if sw_node.is_overlapping(query_seq, tolerance, self.compute_distance):
                # 关键步骤：将该 SW 内的所有 children (Raw Sequences) 全部加入
                # 不再做几何判断，直接批量提取
                for seq in sw_node.children:
                    final_candidate_sequences.append(seq)
        
        stats['raw_candidates'] = len(final_candidate_sequences)
        
        # ===== Verification (Layer 0) =====
        # 去重：使用 set 基于序列 ID 去重
        seen_ids: Set[str] = set()
        unique_candidates: List[BioSequence] = []
        for seq in final_candidate_sequences:
            if seq.id not in seen_ids:
                seen_ids.add(seq.id)
                unique_candidates.append(seq)
        
        # 精确距离验证
        results = []
        for seq in unique_candidates:
            dist = self.compute_distance(seq, query_seq)
            if dist <= tolerance:
                results.append(seq)
        
        return results, stats


def generate_random_dna_sequence(seq_id: str, min_len: int = 50, max_len: int = 100) -> BioSequence:
    """生成随机 DNA 序列"""
    length = random.randint(min_len, max_len)
    bases = ['A', 'T', 'C', 'G']
    seq = ''.join(random.choices(bases, k=length))
    return BioSequence(seq_id=seq_id, seq=seq)


def linear_scan_search(database: List[BioSequence], query: BioSequence, tolerance: int) -> List[BioSequence]:
    """
    暴力搜索：线性扫描所有序列
    
    Args:
        database: 数据库序列列表
        query: 查询序列
        tolerance: 容错半径
        
    Returns:
        匹配的序列列表
    """
    results = []
    for seq in database:
        dist = BioGeometryIndex.compute_distance(seq, query)
        if dist <= tolerance:
            results.append(seq)
    return results

def generate_clustered_data(num_clusters=10, seqs_per_cluster=200, mutation_rate=0.05):
    """
    生成聚类数据（模拟生物家族）：
    先生成 num_clusters 个种子序列，
    然后基于种子进行突变，生成 seqs_per_cluster 个子序列。
    """
    database = []
    cluster_centers = []
    
    # 1. 生成种子 (Ancestors)
    for i in range(num_clusters):
        seed_seq = generate_random_dna_sequence(f"seed_{i}", 100, 100)
        cluster_centers.append(seed_seq)
        # 种子也加入数据库
        # database.append(seed_seq) 
        
    # 2. 生成变异后代 (Descendants)
    bases = ['A', 'T', 'C', 'G']
    global_cnt = 0
    
    for center_seq in cluster_centers:
        original_str = center_seq.seq
        length = len(original_str)
        
        for _ in range(seqs_per_cluster):
            # 复制一份
            mutated_list = list(original_str)
            # 决定突变次数 (根据 mutation_rate)
            # 0.05 * 100 = 5 个突变，正好落在 SW 半径(5) 或 MW 半径(15) 范围内
            num_mutations = max(1, int(length * mutation_rate))
            
            for _ in range(num_mutations):
                idx = random.randint(0, len(mutated_list) - 1)
                op = random.choice(['sub', 'del', 'ins'])
                
                if op == 'sub': # 替换
                    mutated_list[idx] = random.choice(bases)
                elif op == 'del': # 删除
                    if len(mutated_list) > 50:
                        del mutated_list[idx]
                elif op == 'ins': # 插入
                    mutated_list.insert(idx, random.choice(bases))
            
            final_seq_str = "".join(mutated_list)
            seq_obj = BioSequence(f"seq_{global_cnt:05d}", final_seq_str)
            database.append(seq_obj)
            global_cnt += 1
            
    return database, cluster_centers


# def main():
#     """主函数：测试索引构建和搜索"""
#     print("=" * 60)
#     print("BioGeometry DAG Index (Prototype) - Demo")
#     print("=" * 60)
    
#     # ===== 1. 数据生成 =====
#     print("\n[1] 生成测试数据...")
#     random.seed(42)  # 固定随机种子以便复现
#     num_sequences = 1000
#     database = []
    
#     for i in range(num_sequences):
#         seq = generate_random_dna_sequence(f"seq_{i:04d}", min_len=50, max_len=100)
#         database.append(seq)
    
#     print(f"生成了 {len(database)} 条随机 DNA 序列")
#     print(f"示例序列: {database[0]}")
    
#     # ===== 2. 索引构建 =====
#     print("\n[2] 构建索引...")
#     index = BioGeometryIndex()
#     index.build(database)
    
#     # ===== 3. 统计信息 =====
#     print("\n[3] 索引统计信息:")
#     print(f"  Layer 0 (Raw Data): {len(index.layers[0])} 条序列")
#     print(f"  Layer 1 (SW): {len(index.layers[1])} 个节点 (radius={index.config[1]})")
#     print(f"  Layer 2 (MW): {len(index.layers[2])} 个节点 (radius={index.config[2]})")
#     print(f"  Layer 3 (LW): {len(index.layers[3])} 个节点 (radius={index.config[3]})")
    
#     # 计算压缩率
#     compression_sw = len(index.layers[0]) / len(index.layers[1]) if index.layers[1] else 0
#     compression_mw = len(index.layers[1]) / len(index.layers[2]) if index.layers[2] else 0
#     compression_lw = len(index.layers[2]) / len(index.layers[3]) if index.layers[3] else 0
#     print(f"\n压缩率:")
#     print(f"  Layer 0 -> Layer 1: {compression_sw:.2f}x")
#     print(f"  Layer 1 -> Layer 2: {compression_mw:.2f}x")
#     print(f"  Layer 2 -> Layer 3: {compression_lw:.2f}x")
    
#     # ===== 4. 查询测试 =====
#     print("\n[4] 查询测试...")
    
#     # 生成随机查询
#     query = generate_random_dna_sequence("query_0000", min_len=50, max_len=100)
#     tolerance = 5
#     print(f"查询序列: {query}")
#     print(f"容错半径: {tolerance}")
    
#     # 暴力搜索
#     print("\n执行暴力搜索 (Linear Scan)...")
#     linear_results = linear_scan_search(database, query, tolerance)
#     print(f"暴力搜索结果: {len(linear_results)} 条匹配序列")
    
#     # 索引搜索
#     print("\n执行索引搜索 (Index Search)...")
#     index_results, stats = index.search(query, tolerance)
#     print(f"索引搜索结果: {len(index_results)} 条匹配序列")
    
#     # ===== 5. 结果验证 =====
#     print("\n[5] 结果验证...")
    
#     # 提取序列 ID 进行比较
#     linear_ids = {seq.id for seq in linear_results}
#     index_ids = {seq.id for seq in index_results}
    
#     # 检查是否完全一致
#     missing_in_index = linear_ids - index_ids
#     extra_in_index = index_ids - linear_ids
    
#     if not missing_in_index and not extra_in_index:
#         print("✓ 结果完全一致！Recall = 100%")
#     else:
#         print(f"✗ 结果不一致！")
#         if missing_in_index:
#             print(f"  索引搜索遗漏: {len(missing_in_index)} 条序列")
#             print(f"  示例遗漏: {list(missing_in_index)[:5]}")
#         if extra_in_index:
#             print(f"  索引搜索多余: {len(extra_in_index)} 条序列")
#             print(f"  示例多余: {list(extra_in_index)[:5]}")
    
#     # ===== 6. 性能统计 =====
#     print("\n[6] 搜索性能统计:")
#     print(f"  访问的 LW 节点数: {stats['lw_visited']} / {len(index.layers[3])}")
#     print(f"  访问的 MW 节点数: {stats['mw_visited']} / {stats['mw_candidates']}")
#     print(f"  访问的 SW 节点数: {stats['sw_visited']} / {stats['sw_candidates']}")
#     print(f"  候选原始序列数: {stats['raw_candidates']}")
#     print(f"  最终验证序列数: {len(index_results)}")
    
#     # 计算剪枝效率
#     total_nodes = len(index.layers[1]) + len(index.layers[2]) + len(index.layers[3])
#     visited_nodes = stats['lw_visited'] + stats['mw_visited'] + stats['sw_visited']
#     pruning_ratio = (1 - visited_nodes / total_nodes) * 100 if total_nodes > 0 else 0
#     print(f"\n剪枝效率: {pruning_ratio:.2f}% 的节点被剪枝")
#     print(f"  (访问了 {visited_nodes} / {total_nodes} 个节点)")
    
#     print("\n" + "=" * 60)
#     print("Demo 完成！")
#     print("=" * 60)

def main():
    """主函数：测试索引构建、算法性能基准测试"""
    print("=" * 70)
    print(f"BioGeometry DAG Index (Prototype) - Biological Benchmark")
    print("=" * 70)
    
    # ===== 1. 数据生成 (聚类模式) =====
    print("\n[1] 生成模拟生物数据 (Clustered Data)...")
    random.seed(42)
    
    # 配置：20个家族，每家族100条 = 2000条数据
    # 突变率 0.03 意味着平均距离约 3-4，完美契合 SW Radius=5
    t_start = time.perf_counter()
    database, ancestors = generate_clustered_data(
        num_clusters=20, 
        seqs_per_cluster=10, 
        mutation_rate=0.03
    )
    t_gen = time.perf_counter() - t_start
    
    print(f"  完成！生成 {len(database)} 条聚类序列，耗时 {t_gen:.4f}s")
    
    # ===== 2. 索引构建 =====
    print("\n[2] 构建 BioGeometry 索引 (Bottom-Up DAG)...")
    index = BioGeometryIndex()
    
    t_start = time.perf_counter()
    index.build(database)
    t_build = time.perf_counter() - t_start
    
    print(f"  构建完成！耗时: {t_build:.4f}s")
    
    # ===== 3. 结构统计 (预期看到金字塔形状) =====
    print("\n[3] 索引拓扑结构统计:")
    l1_count = len(index.layers[1])
    l2_count = len(index.layers[2])
    l3_count = len(index.layers[3])
    
    print(f"  Layer 3 (LW): {l3_count:>4} nodes (Radius={index.config[3]}) - 顶层入口")
    print(f"  Layer 2 (MW): {l2_count:>4} nodes (Radius={index.config[2]})")
    print(f"  Layer 1 (SW): {l1_count:>4} nodes (Radius={index.config[1]}) - 底层聚类")
    print(f"  Layer 0 (Raw):{len(database):>4} items - 原始数据")
    
    # 验证聚类效果：如果 L1 节点数远小于 L0，说明聚类成功
    compression = len(database) / l1_count if l1_count else 0
    print(f"  [聚类效果] 平均每个 SW 包含 {compression:.1f} 条序列 (理想值应 > 1)")

    # DAG 多重归属检测
    total_sw_refs = sum(len(n.children) for n in index.layers[2])
    dag_redundancy = (total_sw_refs / l1_count - 1) * 100 if l1_count else 0
    print(f"  [DAG特性] 边界冗余度: {dag_redundancy:.1f}% (子节点被多个父节点包含的比例)")

    # ===== 4. 查询性能竞技场 (Query Benchmark) - 修复版 =====
    print("\n[4] 查询性能竞技场 (Query Benchmark)...")
    
    # 修复：能够同时兼容 BioSequence 对象和字符串
    target_ancestor = ancestors[5]
    
    # 如果是 BioSequence 对象，取 .seq；如果是字符串，直接用
    if hasattr(target_ancestor, 'seq'):
        ancestor_str = target_ancestor.seq
    else:
        ancestor_str = target_ancestor
        
    q_list = list(ancestor_str)
    
    # 人为引入 2 个突变 (距离=2)
    if len(q_list) > 5:
        # 确保突变后的碱基和原来不一样
        q_list[0] = 'C' if q_list[0] != 'C' else 'A'
        q_list[1] = 'G' if q_list[1] != 'G' else 'T'
    
    query = BioSequence("query_family_5", "".join(q_list))
    tolerance = 5  # 容错半径
    
    print(f"  Query: 基于家族 F05 的变异序列 (Length={len(query.seq)})")
    print(f"  Tolerance: {tolerance}")
    
    print("-" * 65)
    print(f"{'Method':<25} | {'Time (s)':<12} | {'Matches':<8} | {'Speedup':<10}")
    print("-" * 65)
    
    # --- A. 暴力搜索 (Baseline) ---
    t_start = time.perf_counter()
    res_linear = linear_scan_search(database, query, tolerance)
    t_linear = time.perf_counter() - t_start
    print(f"{'1. Linear Scan (O(N))':<25} | {t_linear:.6f}      | {len(res_linear):<8} | {'1.0x':<10}")
    
    # --- B. BioGeometry 索引搜索 (Algorithmic Opt) ---
    t_start = time.perf_counter()
    res_index, stats = index.search(query, tolerance)
    t_index = time.perf_counter() - t_start
    
    speedup_algo = t_linear / t_index if t_index > 0 else 0
    print(f"{'2. Geometry Index':<25} | {t_index:.6f}      | {len(res_index):<8} | {speedup_algo:.1f}x")
    print("-" * 65)

    # ===== 5. 正确性验证 =====
    print("\n[5] 正确性验证 (Recall Check)...")
    ids_linear = {s.id for s in res_linear}
    ids_index = {s.id for s in res_index}
    
    if ids_linear == ids_index:
        print("  ✓ SUCCESS: 索引搜索结果与暴力搜索完全一致 (Recall = 100%)")
    else:
        print("  ✗ FAILURE: 结果不一致！")
        print(f"    Linear Found: {len(ids_linear)}")
        print(f"    Index Found:  {len(ids_index)}")
        if len(ids_linear) > 0 and len(ids_index) == 0:
            print("    [诊断] 索引搜索未能召回。可能原因：半径设置过小或 DAG 连接未覆盖边界。")

    # ===== 6. 剪枝效率分析 =====
    print("\n[6] 几何剪枝透视 (Pruning Analysis):")
    visited = stats['lw_visited'] + stats['mw_visited'] + stats['sw_visited']
    total_index_nodes = l1_count + l2_count + l3_count
    
    # 计算实际做了多少次 Edit Distance (核心指标)
    # 1. 路由阶段：和 visited 个节点中心做了比对
    # 2. 叶子阶段：和 raw_candidates 个原始序列做了比对
    actual_comparisons = visited + stats['raw_candidates']
    total_sequences = len(database)
    
    comp_reduction = (1 - actual_comparisons / total_sequences) * 100
    
    print(f"  [核心指标] 实际距离计算次数: {actual_comparisons} (暴力搜索需 {total_sequences} 次)")
    print(f"  [核心指标] 计算量节省 (Reduction): {comp_reduction:.2f}%")
    print(f"  --------------------------------------------------")
    print(f"  LW 访问: {stats['lw_visited']} / {l3_count}")
    print(f"  MW 访问: {stats['mw_visited']} / {stats['mw_candidates']} (Candidates)")
    print(f"  SW 访问: {stats['sw_visited']} / {stats['sw_candidates']} (Candidates)")
    print(f"  最终验证: {stats['raw_candidates']} 条原始序列")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
