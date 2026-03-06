"""
核心数据结构定义 (NavigaMer v7 - Multilateration-Enhanced Edition)

设计原则：
  - Sparse Selection + Dense Wiring
  - Hierarchical Multilateration (分层多点定位) + Beacon Pruning
  - 轻量化节点，支持 Payload（倒排位置列表）
  - DAG 结构支持 Multi-Parent，0 False Negative
"""

import uuid
from typing import List, Any, Optional, Union, Dict, Tuple


R_SW = 5
R_MW = 15
R_LW = 30
SEQ_LEN = 250
SKELETON_SAMPLE_SIZE = 1000000


class BioSequence:
    """基础数据单元，封装原始 DNA 序列 + 参考基因组位置列表 (BWT/FM-Index 定位结果)"""

    __slots__ = ('id', 'seq', 'ref_positions')

    def __init__(self, seq_id: str, seq: str):
        self.id = seq_id
        self.seq = seq
        self.ref_positions: List[Tuple[str, int, int, str]] = []

    @property
    def occurrences(self) -> List[Tuple[str, int, int, str]]:
        """位置信息列表 (ref_id, start, end, strand)，与 ref_positions 一致，用于论文/API 统一命名"""
        return self.ref_positions

    def add_occurrence(self, ref_id: str, start: int, end: int, strand: str = '+'):
        """记录该序列在参考基因组上的一个出现位置 (ref_id, start, end, strand)"""
        self.ref_positions.append((ref_id, start, end, strand))

    def __repr__(self):
        seq_preview = self.seq[:20] + "..." if len(self.seq) > 20 else self.seq
        occ = f", ref_pos={len(self.ref_positions)}" if self.ref_positions else ""
        return f"BioSequence(id={self.id}, seq={seq_preview}, len={len(self.seq)}{occ})"


class GenomePointer:
    """基因组指针（内存模拟版本）"""

    __slots__ = ('chrom', 'pos', 'seq')

    def __init__(self, chrom: str, pos: int, seq: str):
        self.chrom = chrom
        self.pos = pos
        self.seq = seq

    def get_sequence(self) -> str:
        return self.seq

    def __repr__(self):
        return f"GenomePointer(chrom={self.chrom}, pos={self.pos}, len={len(self.seq)})"


class WorldNode:
    """DAG 索引节点 (NavigaMer v7 - Multilateration-Enhanced)

    轻量化设计：
      - center_ptr 指向唯一 BioSequence
      - children 为子节点列表（WorldNode 或 BioSequence）
      - beacon_dists: 本节点到当前层 K 个 Beacons 的预计算距离（用于三角不等式 LB 剪枝）
      - routing_anchors / routing_fingerprints 保留兼容，与 layer_beacons + beacon_dists 二选一使用
    """

    __slots__ = (
        'node_id', 'center_ptr', 'radius', 'layer',
        'children', 'routing_anchors', 'routing_fingerprints',
        'beacon_dists', 'data_count',
    )

    def __init__(self, center_ptr: Union[BioSequence, GenomePointer],
                 radius: int, layer_level: int):
        layer_name = {1: "SW", 2: "MW", 3: "LW"}.get(layer_level, "UNK")
        self.node_id = f"{layer_name}_{uuid.uuid4().hex[:8]}"

        self.center_ptr = center_ptr
        self.radius = radius
        self.layer = layer_level

        self.children: List[Union['WorldNode', BioSequence]] = []

        self.routing_anchors: List[Union['WorldNode', BioSequence]] = []
        self.routing_fingerprints: Dict[str, List[int]] = {}
        self.beacon_dists: List[int] = []  # 到父层 K 个 Beacons 的距离，用于 Beacon Pruning

        self.data_count = 0

    def get_center_sequence(self) -> str:
        if isinstance(self.center_ptr, BioSequence):
            return self.center_ptr.seq
        elif isinstance(self.center_ptr, GenomePointer):
            return self.center_ptr.get_sequence()
        return str(self.center_ptr)

    def _get_child_id(self, child: Union['WorldNode', BioSequence]) -> str:
        if isinstance(child, WorldNode):
            return child.node_id
        return getattr(child, 'id', str(id(child)))

    def add_child(self, child: 'WorldNode'):
        self.children.append(child)

    def add_child_with_fingerprint(self, child: Union['WorldNode', BioSequence],
                                   fingerprint: List[int]):
        self.children.append(child)
        cid = self._get_child_id(child)
        self.routing_fingerprints[cid] = fingerprint

    def __repr__(self):
        layer_name = {1: "SW", 2: "MW", 3: "LW"}.get(self.layer, "UNK")
        return (f"WorldNode(id={self.node_id}, layer={layer_name}, "
                f"radius={self.radius}, children={len(self.children)}, "
                f"data_count={self.data_count})")
