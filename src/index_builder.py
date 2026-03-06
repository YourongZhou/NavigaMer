"""
索引构建器 (NavigaMer v7 - Multilateration-Enhanced Edition)

核心理念：Sparse Selection + Dense Wiring + Hierarchical Multilateration (分层多点定位)

流水线（先建骨架，后插信标）：
  Phase 0 — 数据清洗：相同序列合并，Payload 聚合。
  Phase 1 — 骨架生成 (Skeleton)：自底向上稀疏选点，Early Stop 覆盖判定。
  Phase 2 — 稠密连网 (Dense Wiring)：全量重叠连线，保证 0 FN 物理基础。
  Phase 3 — 信标注入 (Beacon Injection)：每层 FPS 选 K 个 Beacons，预计算 beacon_dists。
  Phase 4 — 叶子挂载 (Leaf Attachment)：倒排挂载到 SW 节点。
  Phase 5 — BWT 定位 (可选)：FM-Index 填充 ref_positions。

扩展性接口：
  find_neighbors(query, candidates) — 当前为线性遍历，
  未来可替换为 Minimizer Hash 查表。
"""

import random
import time
from typing import List, Dict, Tuple, Optional, Callable, Iterable, TYPE_CHECKING
from .structure import WorldNode, BioSequence, GenomePointer, R_SW, R_MW, R_LW
from .tools import compute_distance, farthest_point_sampling

if TYPE_CHECKING:
    from .fm_index import FMIndex


def _to_seq_obj(obj) -> Optional[BioSequence]:
    """统一转换为 BioSequence 以便计算距离"""
    if isinstance(obj, BioSequence):
        return obj
    if isinstance(obj, WorldNode):
        return BioSequence("_center", obj.get_center_sequence())
    if isinstance(obj, GenomePointer):
        return BioSequence("_center", obj.get_sequence())
    return None


class BioGeometryIndexBuilder:
    """NavigaMer v7 索引构建器 (Multilateration-Enhanced)

    构建策略：分层稀疏选点 + 全量重叠连网 + 信标注入 (FPS Beacons) + 叶子挂载
    """

    BEACON_COUNT = 3  # 每层 FPS 信标数 K，用于 Zero-Edit-Distance Routing / Beacon Pruning

    def __init__(self):
        self.layers: Dict[int, List[WorldNode]] = {1: [], 2: [], 3: []}
        self.layer_beacons: Dict[int, List[WorldNode]] = {}  # layer_id -> K 个 Beacons (FPS 选出)
        self.radius_config = {1: R_SW, 2: R_MW, 3: R_LW}
        self.unique_sequences: Dict[str, BioSequence] = {}
        self.stats = {
            'added_sequences': 0,
            'unique_sequences': 0,
            'deduplicated': 0,
            'created_nodes': {1: 0, 2: 0, 3: 0},
        }

    # =====================================================================
    # Public API
    # =====================================================================

    def build(self, raw_sequences: List[BioSequence], fm_index: Optional["FMIndex"] = None):
        """Multilateration-Enhanced 流水线；fm_index 仅作 Payload Provider / Verification，非构建依赖"""
        print(f"[Build v7 Multilateration] Starting for {len(raw_sequences)} sequences...")
        t0 = time.perf_counter()

        # Phase 0: 数据清洗
        print("  Phase 0: Deduplicating sequences...")
        unique_seqs = self._deduplicate(raw_sequences)
        print(f"    {len(raw_sequences)} -> {len(unique_seqs)} unique "
              f"({self.stats['deduplicated']} merged)")

        # Phase 1: 骨架生成 — 自底向上稀疏选点
        print("  Phase 1: Skeleton generation (sparse selection)...")
        self._build_skeleton(unique_seqs)
        print(f"    SW={len(self.layers[1])}, MW={len(self.layers[2])}, "
              f"LW={len(self.layers[3])}")

        # Phase 2: 稠密连网 — 全量重叠连线，保证 0 FN
        print("  Phase 2: Dense wiring (exhaustive overlap)...")
        self._dense_wiring()

        # Phase 3: 信标注入 — FPS 选层 Beacons，预计算 beacon_dists（Multilateration 数学基础）
        print("  Phase 3: Beacon injection (FPS)...")
        self._inject_beacons()

        # Phase 4: 叶子挂载 — 倒排挂载到 SW
        print("  Phase 4: Leaf attachment...")
        self._attach_leaves(unique_seqs)

        # Phase 5: BWT 定位 (可选) — 仅作 Payload Provider
        if fm_index is not None:
            print("  Phase 5: Locate ref positions (FM-Index)...")
            self._locate_ref_positions(unique_seqs, fm_index)

        elapsed = time.perf_counter() - t0
        print(f"[Build v7] Completed in {elapsed:.2f}s.")
        self._print_summary()

    # =====================================================================
    # Phase 0: Deduplication
    # =====================================================================

    def _deduplicate(self, raw_sequences: List[BioSequence]) -> List[BioSequence]:
        """完全相同的序列合并到同一 BioSequence，累积 ref_positions"""
        seq_map: Dict[str, BioSequence] = {}
        for seq in raw_sequences:
            self.stats['added_sequences'] += 1
            if seq.seq in seq_map:
                representative = seq_map[seq.seq]
                for occ in seq.ref_positions:
                    representative.add_occurrence(*occ)
                if not seq.ref_positions and not representative.ref_positions:
                    representative.add_occurrence(seq.id, 0, len(seq.seq), '+')
                self.stats['deduplicated'] += 1
            else:
                seq_map[seq.seq] = seq
        self.unique_sequences = {s.id: s for s in seq_map.values()}
        self.stats['unique_sequences'] = len(seq_map)
        return list(seq_map.values())

    # =====================================================================
    # Phase 1: Skeleton Generation (Sparse Selection)
    # =====================================================================

    def _build_skeleton(self, unique_seqs: List[BioSequence]):
        """自底向上分层稀疏选点

        Layer 1 (SW): 从 unique_seqs 中选代表点
        Layer 2 (MW): 从 SW centers 中选代表点
        Layer 3 (LW): 从 MW centers 中选代表点
        """
        shuffled = list(unique_seqs)
        random.shuffle(shuffled)

        sw_nodes = self._build_layer_sparse(
            items=shuffled,
            radius=R_SW,
            layer_level=1,
            label="SW",
        )
        self.layers[1] = sw_nodes

        mw_nodes = self._build_layer_sparse(
            items=sw_nodes,
            radius=R_MW,
            layer_level=2,
            label="MW",
        )
        self.layers[2] = mw_nodes

        lw_nodes = self._build_layer_sparse(
            items=mw_nodes,
            radius=R_LW,
            layer_level=3,
            label="LW",
        )
        self.layers[3] = lw_nodes

    def _build_layer_sparse(self, items, radius: int,
                            layer_level: int, label: str) -> List[WorldNode]:
        """线性扫描稀疏选点

        遍历 items，如果 item 与任一已有节点距离 <= radius 则跳过，
        否则以 item 为中心创建新节点。
        """
        nodes: List[WorldNode] = []
        for i, item in enumerate(items):
            item_seq = _to_seq_obj(item)
            if item_seq is None:
                continue

            covered = False
            for node in self.find_neighbors(item_seq, nodes, radius):
                covered = True
                break

            if not covered:
                center = item if isinstance(item, BioSequence) else item.center_ptr
                new_node = WorldNode(
                    center_ptr=center,
                    radius=radius,
                    layer_level=layer_level,
                )
                nodes.append(new_node)
                self.stats['created_nodes'][layer_level] += 1

            if (i + 1) % 500 == 0:
                print(f"    {label}: scanned {i+1}/{len(items)}, "
                      f"nodes={len(nodes)}", end='\r')

        if len(items) > 500:
            print()
        return nodes

    # =====================================================================
    # Phase 2: Dense Wiring (Exhaustive Overlap)
    # =====================================================================

    def _dense_wiring(self):
        """全量重叠连边

        对相邻层的所有节点对，dist(P, C) <= R_P + R_C 则建立有向边。
        """
        sw_nodes = self.layers[1]
        mw_nodes = self.layers[2]
        lw_nodes = self.layers[3]

        # SW -> MW
        print(f"    Wiring {len(sw_nodes)} SW -> {len(mw_nodes)} MW...")
        for mw in mw_nodes:
            mw.children = []
        self._wire_overlap(parents=mw_nodes, children=sw_nodes)

        # MW -> LW
        print(f"    Wiring {len(mw_nodes)} MW -> {len(lw_nodes)} LW...")
        for lw in lw_nodes:
            lw.children = []
        self._wire_overlap(parents=lw_nodes, children=mw_nodes)

        # 清理无子节点的空壳
        self.layers[2] = [n for n in mw_nodes if n.children]
        self.layers[3] = [n for n in lw_nodes if n.children]
        print(f"    After cleanup: MW={len(self.layers[2])}, "
              f"LW={len(self.layers[3])}")

    def _wire_overlap(self, parents: List[WorldNode],
                      children: List[WorldNode]):
        """几何重叠连边：dist(P.center, C.center) <= R_P + R_C"""
        for parent in parents:
            p_seq = _to_seq_obj(parent.center_ptr)
            if p_seq is None:
                continue
            for child in children:
                c_seq = _to_seq_obj(child.center_ptr)
                if c_seq is None:
                    continue
                d = compute_distance(p_seq, c_seq)
                if d <= parent.radius + child.radius:
                    parent.children.append(child)

    # =====================================================================
    # Phase 3: Beacon Injection (FPS Beacons + beacon_dists for Multilateration)
    # =====================================================================

    def _inject_beacons(self):
        """层间信标注入：每层 (LW/MW) FPS 选 K 个 Beacons，为子层节点预计算 beacon_dists。

        - LW 层 Beacons：用于从 LW 进入 MW 时对 MW 子节点做三角不等式剪枝；每个 MW 节点存到 LW Beacons 的距离。
        - MW 层 Beacons：用于从 MW 进入 SW 时对 SW 子节点做三角不等式剪枝；每个 SW 节点存到 MW Beacons 的距离。
        """
        K = self.BEACON_COUNT

        # LW 层：选出 K 个 Beacons，为所有 MW 节点计算到这些 Beacons 的距离
        lw_nodes = self.layers[3]
        if lw_nodes:
            self.layer_beacons[3] = farthest_point_sampling(
                lw_nodes, min(K, len(lw_nodes)), compute_distance
            )
            for mw in self.layers[2]:
                mw.beacon_dists = [
                    compute_distance(_to_seq_obj(mw), _to_seq_obj(b))
                    for b in self.layer_beacons[3]
                ]
            print(f"    LW beacons: {len(self.layer_beacons[3])}, "
                  f"MW nodes have beacon_dists (len={len(self.layer_beacons[3])})")

        # MW 层：选出 K 个 Beacons，为所有 SW 节点计算到这些 Beacons 的距离
        mw_nodes = self.layers[2]
        if mw_nodes:
            self.layer_beacons[2] = farthest_point_sampling(
                mw_nodes, min(K, len(mw_nodes)), compute_distance
            )
            for sw in self.layers[1]:
                sw.beacon_dists = [
                    compute_distance(_to_seq_obj(sw), _to_seq_obj(b))
                    for b in self.layer_beacons[2]
                ]
            print(f"    MW beacons: {len(self.layer_beacons[2])}, "
                  f"SW nodes have beacon_dists (len={len(self.layer_beacons[2])})")

    # =====================================================================
    # Phase 4: Leaf Attachment
    # =====================================================================

    def _attach_leaves(self, unique_seqs: List[BioSequence]):
        """将叶子序列挂载到所有 dist <= R_SW 的 SW 节点

        同时记录位置信息到 BioSequence.ref_positions。
        """
        sw_nodes = self.layers[1]
        total_links = 0

        for sw in sw_nodes:
            sw_center = _to_seq_obj(sw.center_ptr)
            if sw_center is None:
                continue
            for seq in unique_seqs:
                d = compute_distance(sw_center, seq)
                if d <= sw.radius:
                    sw.children.append(seq)
                    total_links += 1
            sw.data_count = len([c for c in sw.children
                                 if isinstance(c, BioSequence)])

        print(f"    Attached {total_links} leaf-SW links "
              f"(avg {total_links/len(sw_nodes):.1f} per SW)")

    # =====================================================================
    # Phase 5: BWT Locate (optional, Payload Provider only)
    # =====================================================================

    def _locate_ref_positions(self, unique_seqs: List[BioSequence], fm_index: "FMIndex"):
        """用 FM-Index 为每条唯一序列查在 reference 上的所有出现位置，写入 ref_positions"""
        total_pos = 0
        for seq in unique_seqs:
            seq.ref_positions = fm_index.locate(seq.seq)
            total_pos += len(seq.ref_positions)
        print(f"    Located {total_pos} ref positions for {len(unique_seqs)} sequences")

    # =====================================================================
    # Scalability Interface
    # =====================================================================

    def find_neighbors(self, query_seq: BioSequence,
                       candidates: List[WorldNode],
                       radius: int) -> List[WorldNode]:
        """查找 query 在 candidates 中距离 <= radius 的邻居

        当前实现：线性遍历。
        未来实现：Minimizer Hash 查表。
        """
        result = []
        for node in candidates:
            node_seq = _to_seq_obj(node.center_ptr)
            if node_seq is None:
                continue
            d = compute_distance(query_seq, node_seq)
            if d <= radius:
                result.append(node)
        return result

    # =====================================================================
    # Statistics & Summary
    # =====================================================================

    def _print_summary(self):
        sw_count = len(self.layers[1])
        mw_count = len(self.layers[2])
        lw_count = len(self.layers[3])
        print(f"  Layer 1 (SW): {sw_count} nodes")
        print(f"  Layer 2 (MW): {mw_count} nodes")
        print(f"  Layer 3 (LW): {lw_count} nodes")

        if sw_count > 0 and mw_count > 0:
            total_refs = sum(
                sum(1 for c in mw.children if isinstance(c, WorldNode))
                for mw in self.layers[2]
            )
            avg_parents = total_refs / sw_count if sw_count else 0
            print(f"  Avg parents per SW: {avg_parents:.2f}")

        raw = self.stats['added_sequences']
        unique = self.stats['unique_sequences']
        if unique > 0:
            compression = 1 - sw_count / unique
            print(f"  Compression: {compression:.1%} "
                  f"({unique} unique -> {sw_count} SW)")

    def get_statistics(self) -> Dict:
        raw_count = self.stats['added_sequences']
        unique_count = self.stats['unique_sequences']
        sw_count = len(self.layers[1])
        mw_count = len(self.layers[2])
        lw_count = len(self.layers[3])

        comp_sw = unique_count / sw_count if sw_count > 0 else 0
        comp_mw = sw_count / mw_count if mw_count > 0 else 0
        comp_lw = mw_count / lw_count if lw_count > 0 else 0

        compression_ratio = 1 - sw_count / unique_count if unique_count > 0 else 0

        dag_redundancy = 0
        avg_parents_per_sw = 0
        if sw_count > 0 and mw_count > 0:
            total_sw_refs = sum(
                sum(1 for c in mw.children if isinstance(c, WorldNode))
                for mw in self.layers[2]
            )
            avg_parents_per_sw = total_sw_refs / sw_count
            dag_redundancy = (avg_parents_per_sw - 1) * 100

        return {
            'raw_count': raw_count,
            'unique_count': unique_count,
            'sw_count': sw_count,
            'mw_count': mw_count,
            'lw_count': lw_count,
            'compression_sw': comp_sw,
            'compression_mw': comp_mw,
            'compression_lw': comp_lw,
            'compression_ratio': compression_ratio,
            'dag_redundancy': dag_redundancy,
            'avg_parents_per_sw': avg_parents_per_sw,
        }
