"""
I/O 工具：Reference/Reads 加载与 TSV 输出 (NavigaMer v7 Multilateration)

支持两种输入模式：
  - FASTA + FASTQ 文件路径
  - 纯字符串 (reference string, query string)

输出规范：query_id, hit_id, distance, ref_positions (json_string), ...
"""

import json
import os
from typing import Tuple, List, Iterable, Union
from .structure import BioSequence


def load_reference(path_or_string: str) -> Tuple[str, str]:
    """
    加载参考序列。

    Args:
        path_or_string: FASTA 文件路径，或直接为序列字符串

    Returns:
        (ref_id, sequence)。若为字符串输入则 ref_id="ref"。
    """
    if os.path.isfile(path_or_string):
        return _load_fasta(path_or_string)
    return ("ref", path_or_string.strip())


def _load_fasta(path: str) -> Tuple[str, str]:
    """读取单条序列的 FASTA 文件，返回 (id, seq)"""
    ref_id = "ref"
    seq_parts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                ref_id = line[1:].split()[0]
            else:
                seq_parts.append(line)
    return (ref_id, "".join(seq_parts))


def load_reads(
    path_or_string: str,
    ref_id: str = "ref",
) -> Union[List[BioSequence], Iterable[BioSequence]]:
    """
    加载 reads。

    Args:
        path_or_string: FASTQ 文件路径，或直接为一条 query 序列字符串
        ref_id: 用于 occurrence 的 ref 标识（文件模式时可选）

    Returns:
        BioSequence 列表，或流式迭代器（大文件时 yield）
    """
    if os.path.isfile(path_or_string):
        return _load_fastq(path_or_string)
    # 单条字符串 -> 单条 BioSequence
    s = path_or_string.strip()
    return [BioSequence("query_0", s)]


def _load_fastq(path: str) -> List[BioSequence]:
    """读取 FASTQ，返回 BioSequence 列表（id 取第一行 @ 后内容）"""
    reads = []
    with open(path, "r") as f:
        while True:
            id_line = f.readline()
            if not id_line:
                break
            if not id_line.startswith("@"):
                continue
            seq_id = id_line[1:].strip().split()[0]
            seq = f.readline().strip()
            f.readline()  # +
            f.readline()  # qual
            if seq:
                reads.append(BioSequence(seq_id, seq))
    return reads


def write_tsv(
    rows: List[dict],
    output_path: str,
    columns: List[str] = None,
) -> None:
    """
    写入 TSV，格式对齐 spaced_seed 风格。

    每行通常包含：read_id, read_len, ref_id, strand, query_start,
    reference_start, aligned_length, score, edit_distance, [query_fragment, reference_fragment]
    """
    if not rows:
        return
    if columns is None:
        columns = list(rows[0].keys())
    with open(output_path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in columns) + "\n")


def search_results_to_tsv_rows(
    query_id: str,
    query_seq: str,
    query_start: int,
    hit: BioSequence,
    edit_distance: int,
) -> List[dict]:
    """
    将单次搜索命中的 BioSequence 展开为 TSV 行（每条 ref_position 一行）。

    列包含：query_id, hit_id, distance, ref_positions (json_string)，及与 spaced_seed 对齐的字段。
    """
    aligned_len = len(hit.seq)
    score = aligned_len - edit_distance
    ref_positions_json = json.dumps(
        [(ref_id, start, end, strand) for ref_id, start, end, strand in hit.ref_positions]
    )
    rows = []
    if hit.ref_positions:
        for ref_id, start, end, strand in hit.ref_positions:
            rows.append(_one_tsv_row(
                query_id, query_seq, query_start, hit, edit_distance,
                score, ref_positions_json, ref_id, start, end, strand,
            ))
    else:
        rows.append(_one_tsv_row(
            query_id, query_seq, query_start, hit, edit_distance,
            score, ref_positions_json, "", 0, 0, "+",
        ))
    return rows


def _one_tsv_row(query_id, query_seq, query_start, hit, edit_distance,
                 score, ref_positions_json, ref_id, start, end, strand):
    return {
        "query_id": query_id,
        "hit_id": hit.id,
        "distance": edit_distance,
        "ref_positions": ref_positions_json,
        "read_id": query_id,
        "read_len": len(query_seq),
        "ref_id": ref_id,
        "strand": strand,
        "query_start": query_start,
        "reference_start": start,
        "aligned_length": end - start,
        "score": score,
        "edit_distance": edit_distance,
        "query_fragment": query_seq,
        "reference_fragment": hit.seq,
    }
