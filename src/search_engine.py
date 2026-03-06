"""
搜索引擎 (NavigaMer v7 - Multilateration-Enhanced)

核心卖点：0 False Negative + Hierarchical Multilateration (Beacon Pruning)

策略：
  - Adaptive: 0 FN + Sibling Pruning + Beacon Pruning (三角不等式 LB)
  - Greedy: 单路径，可能有 FN
  - Exhaustive: 全路径 0 FN
  - Brute Force: 真值基准

进入条件：dist(Q, N.center) <= R_N + tolerance（宽进）
Beacon Pruning：LB = max_i |V_Q[i] - C.beacon_dists[i]|，若 LB > R_C + tolerance 则剪枝
"""

from typing import List, Set, Dict, Tuple, Union, Optional
from .structure import WorldNode, BioSequence, GenomePointer
from .tools import compute_distance


class SearchStats:
    """搜索性能统计计数器（含 Multilateration 剪枝率）"""

    __slots__ = (
        'node_access_count', 'dist_calc_count', 'leaf_verify_count',
        'layer_breakdown', 'beacon_prune_count', 'candidate_count_for_prune',
    )

    def __init__(self):
        self.node_access_count = 0
        self.dist_calc_count = 0
        self.leaf_verify_count = 0
        self.layer_breakdown: Dict[str, int] = {'LW': 0, 'MW': 0, 'SW': 0}
        self.beacon_prune_count = 0   # 被 Beacon 剪枝掉的子节点数
        self.candidate_count_for_prune = 0  # 参与剪枝判定的候选子节点总数

    @property
    def pruning_rate(self) -> float:
        """被 Beacon 剪枝的节点比例（证明 Multilateration 有效性）"""
        if self.candidate_count_for_prune == 0:
            return 0.0
        return self.beacon_prune_count / self.candidate_count_for_prune

    def to_dict(self) -> Dict:
        return {
            'node_access_count': self.node_access_count,
            'dist_calc_count': self.dist_calc_count,
            'leaf_verify_count': self.leaf_verify_count,
            'layer_breakdown': self.layer_breakdown.copy(),
            'beacon_prune_count': self.beacon_prune_count,
            'candidate_count_for_prune': self.candidate_count_for_prune,
            'pruning_rate': self.pruning_rate,
        }


class BioGeometrySearchEngine:
    """搜索引擎主类 (NavigaMer v7 - Multilateration-Enhanced)"""

    def __init__(self, index_builder):
        self.index = index_builder
        self.layers = index_builder.layers
        self.layer_beacons: Dict[int, List[WorldNode]] = getattr(
            index_builder, 'layer_beacons', {}
        )

    def _center_seq(self, node: WorldNode) -> BioSequence:
        return BioSequence("_center", node.get_center_sequence())

    def _compute_query_beacon_dists(self, query_seq: BioSequence,
                                    layer_id: int,
                                    stats: SearchStats) -> Optional[List[int]]:
        """计算 Query 到指定层 Beacons 的距离向量 V_Q（用于 Beacon Pruning）。"""
        beacons = self.layer_beacons.get(layer_id)
        if not beacons:
            return None
        V_Q = []
        for b in beacons:
            d = compute_distance(query_seq, self._center_seq(b))
            stats.dist_calc_count += 1
            V_Q.append(d)
        return V_Q

    def _beacon_prunable(self, child: WorldNode, V_Q: List[int],
                         tolerance: int, stats: SearchStats) -> bool:
        """Beacon 三角不等式剪枝：LB = max_i |V_Q[i] - child.beacon_dists[i]|，若 LB > R_C + tolerance 则安全剪枝。"""
        if not getattr(child, 'beacon_dists', None) or len(child.beacon_dists) != len(V_Q):
            return False
        lb = max(abs(V_Q[i] - child.beacon_dists[i]) for i in range(len(V_Q)))
        return lb > child.radius + tolerance

    def _compute_anchor_dists(self, query_seq: BioSequence,
                              node: WorldNode,
                              stats: SearchStats) -> List[int]:
        """兼容旧版 per-node routing_anchors（当无 layer_beacons 时使用）。"""
        dists = []
        for anchor in node.routing_anchors:
            if isinstance(anchor, BioSequence):
                ac = anchor
            else:
                ac = BioSequence("_center", anchor.get_center_sequence())
            d = compute_distance(query_seq, ac)
            stats.dist_calc_count += 1
            dists.append(d)
        return dists

    def _anchor_prunable(self, node: WorldNode, child: WorldNode,
                         tolerance: int, q_anchor_dists: List[int]) -> bool:
        """三角不等式安全剪枝（per-node anchors 兼容）。"""
        child_id = node._get_child_id(child)
        fingerprint = node.routing_fingerprints.get(child_id)
        if not fingerprint or len(fingerprint) != len(q_anchor_dists):
            return False
        child_r = child.radius
        for i in range(len(q_anchor_dists)):
            if abs(q_anchor_dists[i] - fingerprint[i]) > child_r + tolerance:
                return True
        return False

    # =====================================================================
    # Greedy Search
    # =====================================================================

    def search_greedy(self, query_seq: BioSequence,
                      tolerance: int) -> Tuple[List[BioSequence], SearchStats]:
        """贪婪单路径搜索：每层选距离最近的一个节点"""
        stats = SearchStats()
        current_nodes = list(self.layers[3])

        for layer_id in [3, 2, 1]:
            layer_name = {3: 'LW', 2: 'MW', 1: 'SW'}[layer_id]
            best_node = None
            min_dist = float('inf')

            for node in current_nodes:
                d = compute_distance(query_seq, self._center_seq(node))
                stats.dist_calc_count += 1
                stats.node_access_count += 1
                stats.layer_breakdown[layer_name] += 1

                if d <= node.radius + tolerance and d < min_dist:
                    min_dist = d
                    best_node = node

            if not best_node:
                return [], stats

            if layer_id == 1:
                results = [c for c in best_node.children
                           if isinstance(c, BioSequence)]
                return results, stats

            # 下一层候选：优先 Beacon Pruning，否则 per-node anchor
            V_Q = self._compute_query_beacon_dists(
                query_seq, layer_id, stats) if self.layer_beacons.get(layer_id) else None
            q_anchor_dists = self._compute_anchor_dists(
                query_seq, best_node, stats) if not V_Q and best_node.routing_anchors else None
            next_candidates = []
            for child in best_node.children:
                if not isinstance(child, WorldNode):
                    continue
                if V_Q is not None:
                    stats.candidate_count_for_prune += 1
                    if self._beacon_prunable(child, V_Q, tolerance, stats):
                        stats.beacon_prune_count += 1
                        continue
                elif q_anchor_dists is not None and self._anchor_prunable(
                        best_node, child, tolerance, q_anchor_dists):
                    continue
                next_candidates.append(child)
            current_nodes = next_candidates

        return [], stats

    # =====================================================================
    # Exhaustive Search
    # =====================================================================

    def search_exhaustive(self, query_seq: BioSequence,
                          tolerance: int) -> Tuple[List[BioSequence], SearchStats]:
        """穷举搜索：遍历所有满足几何约束的路径，保证 0 FN"""
        stats = SearchStats()
        unique_results: Dict[str, BioSequence] = {}
        visited_nodes: Set[str] = set()

        def traverse(node: WorldNode, current_layer: int):
            if node.node_id in visited_nodes:
                return
            visited_nodes.add(node.node_id)

            dist = compute_distance(query_seq, self._center_seq(node))
            stats.dist_calc_count += 1
            stats.node_access_count += 1
            layer_name = {3: 'LW', 2: 'MW', 1: 'SW'}.get(current_layer, 'UNK')
            stats.layer_breakdown[layer_name] += 1

            if dist > node.radius + tolerance:
                return

            for child in node.children:
                if isinstance(child, WorldNode):
                    traverse(child, current_layer - 1)
                elif current_layer == 1:
                    leaf_dist = compute_distance(query_seq, child)
                    stats.dist_calc_count += 1
                    stats.leaf_verify_count += 1
                    if leaf_dist <= tolerance:
                        child_id = getattr(child, 'id', str(id(child)))
                        unique_results[child_id] = child

        for lw_node in self.layers[3]:
            traverse(lw_node, 3)

        return list(unique_results.values()), stats

    # =====================================================================
    # Adaptive Search (v7)
    # =====================================================================

    def search_adaptive(self, query_seq: BioSequence,
                        tolerance: int) -> Tuple[List[BioSequence], SearchStats]:
        """自适应搜索 (v7 Multilateration)：0 FN + Sibling Pruning + Beacon Pruning

        进入条件：dist(Q, N.center) <= R_N + tolerance（宽进）
        Sibling Pruning：d + tolerance <= R_N 时只搜该节点，剪兄弟。
        Beacon Pruning：LB = max_i |V_Q[i] - C.beacon_dists[i]| > R_C + tolerance 则剪枝。
        """
        stats = SearchStats()
        unique_results: Dict[str, BioSequence] = {}
        visited_nodes: Set[str] = set()

        def process_node(node: WorldNode, current_layer: int):
            if current_layer == 1:
                for child in node.children:
                    if isinstance(child, BioSequence):
                        leaf_dist = compute_distance(query_seq, child)
                        stats.dist_calc_count += 1
                        stats.leaf_verify_count += 1
                        if leaf_dist <= tolerance:
                            unique_results[child.id] = child
                return

            world_children = [c for c in node.children
                              if isinstance(c, WorldNode)]
            if not world_children:
                return

            child_layer = current_layer - 1

            # 优先使用层间 Beacon Pruning (Multilateration)
            beacons = self.layer_beacons.get(current_layer)
            if beacons:
                V_Q = self._compute_query_beacon_dists(
                    query_seq, current_layer, stats)
                if V_Q is not None:
                    surviving = []
                    for c in world_children:
                        stats.candidate_count_for_prune += 1
                        if self._beacon_prunable(c, V_Q, tolerance, stats):
                            stats.beacon_prune_count += 1
                            continue
                        surviving.append(c)
                else:
                    surviving = world_children
            else:
                # 回退：per-node routing_anchors
                num_anchors = len(node.routing_anchors)
                if len(world_children) > num_anchors + 1 and num_anchors > 0:
                    q_anchor_dists = self._compute_anchor_dists(
                        query_seq, node, stats)
                    surviving = [
                        c for c in world_children
                        if not self._anchor_prunable(
                            node, c, tolerance, q_anchor_dists)
                    ]
                else:
                    surviving = world_children

            search_layer(surviving, child_layer)

        def search_layer(candidates: List[WorldNode], layer_id: int):
            layer_name = {3: 'LW', 2: 'MW', 1: 'SW'}.get(layer_id, 'UNK')

            contained_node = None
            overlap_nodes = []

            for node in candidates:
                if node.node_id in visited_nodes:
                    continue

                d = compute_distance(query_seq, self._center_seq(node))
                stats.dist_calc_count += 1
                stats.node_access_count += 1
                stats.layer_breakdown[layer_name] += 1

                if d > node.radius + tolerance:
                    continue

                if d + tolerance <= node.radius:
                    contained_node = node
                    break
                else:
                    overlap_nodes.append(node)

            if contained_node:
                visited_nodes.add(contained_node.node_id)
                process_node(contained_node, layer_id)
            else:
                for node in overlap_nodes:
                    if node.node_id not in visited_nodes:
                        visited_nodes.add(node.node_id)
                        process_node(node, layer_id)

        search_layer(list(self.layers[3]), 3)
        return list(unique_results.values()), stats

    # =====================================================================
    # Brute Force
    # =====================================================================

    def search_brute_force(self, query_seq: BioSequence, tolerance: int,
                           all_sequences: List[BioSequence]) -> Tuple[List[BioSequence], SearchStats]:
        """暴力全量扫描，作为 ground truth"""
        stats = SearchStats()
        results = []
        for seq in all_sequences:
            d = compute_distance(query_seq, seq)
            stats.dist_calc_count += 1
            stats.leaf_verify_count += 1
            if d <= tolerance:
                results.append(seq)
        return results, stats

    # =====================================================================
    # Unified Interface
    # =====================================================================

    def search(self, query_seq: BioSequence, tolerance: int,
               mode: str = 'adaptive') -> Tuple[List[BioSequence], SearchStats]:
        if mode == 'greedy':
            return self.search_greedy(query_seq, tolerance)
        elif mode == 'exhaustive':
            return self.search_exhaustive(query_seq, tolerance)
        elif mode == 'adaptive':
            return self.search_adaptive(query_seq, tolerance)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
