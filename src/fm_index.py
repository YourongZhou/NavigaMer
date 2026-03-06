"""
FM-Index / Suffix Array 定位层 (BWT-based position locator)

对 reference 建 SA，用精确匹配查询序列在参考基因组上的所有出现位置。
用于 NavigaMer 搜索命中后解析基因组坐标。
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    import pydivsufsort
except ImportError:
    pydivsufsort = None


_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """DNA 反向互补序列"""
    return seq.translate(_COMPLEMENT)[::-1]


class FMIndex:
    """
    基于 Suffix Array 的 FM-Index 定位器。
    对 reference 建 SA，支持 pattern 精确匹配并返回所有起始位置。
    """

    def __init__(self, ref_seq: str, ref_id: str = "ref"):
        """
        Args:
            ref_seq: 参考基因组序列 (单条，如 E. coli 染色体)
            ref_id: 参考 ID，输出位置时使用 (如 "NC_000913.3")
        """
        if pydivsufsort is None:
            raise ImportError("pydivsufsort is required for FMIndex. pip install pydivsufsort")
        self.ref_id = ref_id
        self.ref_seq = ref_seq
        self._bytes = ref_seq.encode("ascii") if isinstance(ref_seq, str) else ref_seq
        self._sa: Optional[np.ndarray] = None
        self._build()

    def _build(self):
        """构建 Suffix Array"""
        self._sa = pydivsufsort.divsufsort(self._bytes)

    def search_all(self, pattern: str) -> List[int]:
        """
        返回 pattern 在 reference 上所有精确匹配的起始位置 (0-based)。

        Args:
            pattern: 查询序列 (DNA 字符串)

        Returns:
            起始位置列表，空则未匹配
        """
        if not pattern:
            return []
        p = pattern.encode("ascii") if isinstance(pattern, str) else pattern
        result = pydivsufsort.sa_search(self._bytes, self._sa, p)
        if result is None or result[0] == 0:
            return []
        count, first_sa = result
        positions = [int(self._sa[first_sa + i]) for i in range(count)]
        return sorted(positions)

    def locate(
        self,
        pattern: str,
        both_strands: bool = True,
    ) -> List[Tuple[str, int, int, str]]:
        """
        返回 pattern 在 reference 上的所有出现位置，格式为 (ref_id, start, end, strand)。

        Args:
            pattern: 查询序列
            both_strands: 是否同时查正向与反向互补链

        Returns:
            [(ref_id, start, end, strand), ...]，start/end 为 0-based 左闭右开
        """
        out: List[Tuple[str, int, int, str]] = []
        n = len(pattern)

        # 正向链
        for start in self.search_all(pattern):
            out.append((self.ref_id, start, start + n, "+"))

        # 反向互补链
        if both_strands and pattern:
            rc = reverse_complement(pattern)
            if rc != pattern:
                for start in self.search_all(rc):
                    out.append((self.ref_id, start, start + len(rc), "-"))

        return out
