#!/usr/bin/env python3
"""
统一 indexer 对比工具 —— 支持 strobemer / spaced_seed / tensor_sketch 三种方法。

输入模式：
  1. FASTA + FASTQ 文件
  2. 两个字符串 (--ref_seq / --query_seq)

输出格式 (TSV)：
  read_id  read_len  ref_id  strand  query_start  reference_start
  aligned_length  score  query_fragment  reference_fragment

容错参数：
  strobemer  — order(2/3), k, strobe_w_min_offset, strobe_w_max_offset, w, mode
  spaced_seed — weight, homology_length, similarity, seeds(直接指定 mask)
  tensor_sketch — sketch_dim, tuple_size, window_size, hnsw_ef_search, top_k
"""

import argparse
import os
import sys
import subprocess
import tempfile
import csv
from pathlib import Path

METHODS_DIR = Path(__file__).resolve().parent

# ─────────────────────── 通用 I/O ───────────────────────

def write_temp_fasta(seq, name="manual_ref"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False)
    f.write(f">{name}\n{seq}\n")
    f.close()
    return f.name


def write_temp_fastq(seq, name="manual_query"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fq", delete=False)
    f.write(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n")
    f.close()
    return f.name


UNIFIED_HEADER = [
    "read_id", "read_len", "ref_id", "strand",
    "query_start", "reference_start", "aligned_length",
    "score", "query_fragment", "reference_fragment",
]


# ─────────────────────── Strobemer ───────────────────────

def run_strobemer(args, ref_path, query_path, out_tsv):
    """调用 strobemers/StrobeMap 并将 MUMmer-like 输出转为统一 TSV。"""
    strobe_dir = METHODS_DIR / "ksahlin" / "strobemers"
    strobemap = strobe_dir / "StrobeMap"

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, str(strobemap),
            "--queries", query_path,
            "--references", ref_path,
            "--outfolder", tmpdir,
            "--prefix", "matches",
            "--k", str(args.strobe_k),
            "--strobe_w_min_offset", str(args.strobe_w_min_offset),
            "--strobe_w_max_offset", str(args.strobe_w_max_offset),
            "--w", str(args.strobe_w),
            "--n", str(args.strobe_order),
        ]
        if args.strobe_mode == "kmer":
            cmd.append("--kmer_index")
        elif args.strobe_mode == "randstrobe":
            cmd.append("--randstrobe_index")
        elif args.strobe_mode == "minstrobe":
            cmd.append("--minstrobe_index")
        # hybridstrobe is default

        cmd.append("--rev_comp")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(strobe_dir) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=str(strobe_dir))
        if proc.returncode != 0:
            print(f"[strobemer] stderr:\n{proc.stderr}", file=sys.stderr)
            raise RuntimeError(f"StrobeMap failed (rc={proc.returncode})")

        raw_tsv = os.path.join(tmpdir, "matches.tsv")
        _convert_strobemer_output(raw_tsv, out_tsv)
        print(f"[strobemer] stdout:\n{proc.stdout}", file=sys.stderr)


def _convert_strobemer_output(raw_path, out_tsv):
    """将 MUMmer-like 格式转为统一 TSV。"""
    with open(raw_path) as fin, open(out_tsv, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(UNIFIED_HEADER)

        current_query = None
        is_reverse = False
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                parts = line[2:].strip().split()
                current_query = parts[0] if parts else "unknown"
                is_reverse = "Reverse" in line
            else:
                fields = line.strip().split()
                if len(fields) < 4:
                    continue
                ref_id, ref_pos, query_pos, match_len = (
                    fields[0], fields[1], fields[2], fields[3],
                )
                strand = "-" if is_reverse else "+"
                writer.writerow([
                    current_query, "0", ref_id, strand,
                    query_pos, ref_pos, match_len,
                    match_len,  # score = match_length
                    "", "",
                ])


# ─────────────────────── Spaced Seed ───────────────────────

def run_spaced_seed(args, ref_path, query_path, out_tsv):
    """调用 ales_fasta_fastq_align_parallel.py 或直接用 API。"""
    ss_dir = METHODS_DIR / "spaced_seed"

    is_query_fastq = any(query_path.lower().endswith(ext)
                         for ext in (".fq", ".fastq", ".fq.gz", ".fastq.gz"))

    if is_query_fastq:
        _run_spaced_seed_file(args, ref_path, query_path, out_tsv, ss_dir)
    else:
        _run_spaced_seed_fasta_as_fastq(args, ref_path, query_path, out_tsv, ss_dir)


def _run_spaced_seed_file(args, ref_path, query_path, out_tsv, ss_dir):
    cmd = [
        sys.executable, str(ss_dir / "ales_fasta_fastq_align_parallel.py"),
        "--fasta", ref_path,
        "--fastq", query_path,
        "--output", out_tsv,
        "--weight", str(args.ss_weight),
        "--homology-length", str(args.ss_homology_length),
        "--similarity", str(args.ss_similarity),
        "--k-seeds", str(args.ss_k_seeds),
    ]
    if args.ss_seeds:
        cmd.extend(["--seeds"] + args.ss_seeds)

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(ss_dir))
    if proc.returncode != 0:
        print(f"[spaced_seed] stderr:\n{proc.stderr}", file=sys.stderr)
        raise RuntimeError(f"spaced_seed failed (rc={proc.returncode})")
    print(f"[spaced_seed] stderr:\n{proc.stderr}", file=sys.stderr)


def _run_spaced_seed_fasta_as_fastq(args, ref_path, query_path, out_tsv, ss_dir):
    """query 是 FASTA 时，先转为 FASTQ 再调用。"""
    sys.path.insert(0, str(ss_dir))
    from ales_spaced_seed import ALeSSpacedSeed

    def read_fasta_simple(path):
        records = []
        name, chunks = None, []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        records.append((name, "".join(chunks).upper()))
                    name = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
        if name is not None:
            records.append((name, "".join(chunks).upper()))
        return records

    tmp_fq = write_temp_fastq_from_fasta(query_path)
    try:
        _run_spaced_seed_file(args, ref_path, tmp_fq, out_tsv, ss_dir)
    finally:
        os.unlink(tmp_fq)


def write_temp_fastq_from_fasta(fasta_path):
    """将 FASTA 文件转为临时 FASTQ 文件。"""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fq", delete=False)
    name, chunks = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(chunks).upper()
                    f.write(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n")
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seq = "".join(chunks).upper()
        f.write(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n")
    f.close()
    return f.name


# ─────────────────────── Tensor Sketch ───────────────────────

def run_tensor_sketch(args, ref_path, query_path, out_tsv):
    """调用编译好的 mgsketch_index 二进制。"""
    ts_dir = METHODS_DIR / "tensor-sketch-alignment" / "metagraph"
    binary = ts_dir / "mgsketch_index"

    if not binary.exists():
        raise FileNotFoundError(f"mgsketch_index 未编译: {binary}")

    cmd = [
        str(binary),
        "--output_tsv", out_tsv,
        "--window_size", str(args.ts_window_size),
        "--stride", str(args.ts_stride),
        "--sketch_dim", str(args.ts_sketch_dim),
        "--tuple_size", str(args.ts_tuple_size),
        "--seed", str(args.ts_seed),
        "--hnsw_M", str(args.ts_hnsw_M),
        "--hnsw_ef_construction", str(args.ts_hnsw_ef_construction),
        "--hnsw_ef_search", str(args.ts_hnsw_ef_search),
        "--top_k", str(args.ts_top_k),
    ]

    if args.ref_seq:
        cmd.extend(["--ref_seq", args.ref_seq])
    else:
        cmd.extend(["--ref_fasta", ref_path])

    if args.query_seq:
        cmd.extend(["--query_seq", args.query_seq])
    else:
        cmd.extend(["--query_fastq", query_path])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[tensor_sketch] stderr:\n{proc.stderr}", file=sys.stderr)
        raise RuntimeError(f"mgsketch_index failed (rc={proc.returncode})")
    print(f"[tensor_sketch] stdout:\n{proc.stdout}", file=sys.stderr)


# ─────────────────────── CLI ───────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="统一 indexer 对比工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入
    inp = p.add_argument_group("输入")
    inp.add_argument("--ref_fasta", type=str, default=None, help="Reference FASTA 文件")
    inp.add_argument("--query_fastq", type=str, default=None, help="Query FASTQ/FASTA 文件")
    inp.add_argument("--ref_seq", type=str, default=None, help="直接输入 reference 序列字符串")
    inp.add_argument("--query_seq", type=str, default=None, help="直接输入 query 序列字符串")

    # 方法选择
    p.add_argument("--method", type=str, required=True,
                   choices=["strobemer", "spaced_seed", "tensor_sketch", "all"],
                   help="要运行的方法")
    p.add_argument("--output_dir", type=str, default="./comparison_output",
                   help="输出目录")

    # Strobemer 参数
    st = p.add_argument_group("Strobemer 参数")
    st.add_argument("--strobe_k", type=int, default=15, help="Strobe 长度 (k-mer size)")
    st.add_argument("--strobe_order", type=int, default=2, choices=[2, 3],
                    help="Strobe 分段数 (order)，只能是 2 或 3")
    st.add_argument("--strobe_w_min_offset", type=int, default=20,
                    help="第二个匹配位置的最小偏移")
    st.add_argument("--strobe_w_max_offset", type=int, default=70,
                    help="第二个匹配位置的最大偏移")
    st.add_argument("--strobe_w", type=int, default=1,
                    help="降采样窗口 (1=不降采样)")
    st.add_argument("--strobe_mode", type=str, default="hybridstrobe",
                    choices=["kmer", "randstrobe", "minstrobe", "hybridstrobe"],
                    help="Strobemer 模式")

    # Spaced Seed 参数
    ss = p.add_argument_group("Spaced Seed 参数")
    ss.add_argument("--ss_weight", type=int, default=11,
                    help="Seed 中 care 位数量 (weight)")
    ss.add_argument("--ss_homology_length", type=int, default=64,
                    help="同源区域长度 (N)")
    ss.add_argument("--ss_similarity", type=float, default=0.8,
                    help="碱基相似度 (p)，0~1")
    ss.add_argument("--ss_k_seeds", type=int, default=1,
                    help="设计的 seed 数量")
    ss.add_argument("--ss_seeds", nargs="+", default=None,
                    help="直接指定 spaced-seed masks (如 111011 1110011)")

    # Tensor Sketch 参数
    ts = p.add_argument_group("Tensor Sketch 参数")
    ts.add_argument("--ts_window_size", type=int, default=150,
                    help="切片窗口大小")
    ts.add_argument("--ts_stride", type=int, default=1,
                    help="滑动步长")
    ts.add_argument("--ts_sketch_dim", type=int, default=128,
                    help="Sketch 向量维度 (D)")
    ts.add_argument("--ts_tuple_size", type=int, default=3,
                    help="元组大小 (t)")
    ts.add_argument("--ts_seed", type=int, default=42,
                    help="随机种子")
    ts.add_argument("--ts_hnsw_M", type=int, default=16, help="HNSW M")
    ts.add_argument("--ts_hnsw_ef_construction", type=int, default=200,
                    help="HNSW ef_construction")
    ts.add_argument("--ts_hnsw_ef_search", type=int, default=50,
                    help="HNSW ef_search (增大可提高召回率/容错)")
    ts.add_argument("--ts_top_k", type=int, default=5, help="返回 Top K 结果")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 确定输入文件
    ref_path = args.ref_fasta
    query_path = args.query_fastq
    tmp_files = []

    if args.ref_seq and not ref_path:
        ref_path = write_temp_fasta(args.ref_seq)
        tmp_files.append(ref_path)
    if args.query_seq and not query_path:
        query_path = write_temp_fastq(args.query_seq)
        tmp_files.append(query_path)

    if not ref_path and not args.ref_seq:
        parser.error("必须提供 --ref_fasta 或 --ref_seq")
    if not query_path and not args.query_seq:
        parser.error("必须提供 --query_fastq 或 --query_seq")

    os.makedirs(args.output_dir, exist_ok=True)

    methods = (
        ["strobemer", "spaced_seed", "tensor_sketch"]
        if args.method == "all"
        else [args.method]
    )

    try:
        for method in methods:
            out_tsv = os.path.join(args.output_dir, f"{method}_output.tsv")
            print(f"\n{'='*60}", file=sys.stderr)
            print(f">>> 运行 {method}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            if method == "strobemer":
                run_strobemer(args, ref_path, query_path, out_tsv)
            elif method == "spaced_seed":
                run_spaced_seed(args, ref_path, query_path, out_tsv)
            elif method == "tensor_sketch":
                run_tensor_sketch(args, ref_path, query_path, out_tsv)

            print(f"[{method}] 输出: {out_tsv}", file=sys.stderr)
    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.unlink(f)


if __name__ == "__main__":
    main()
