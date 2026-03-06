#!/usr/bin/env python3
"""
容错参数对比测试 —— 自动生成带突变的测试数据，扫描不同参数组合，画图对比三个 indexer 的召回率。

测试流程：
  1. 从 E. coli 参考基因组截取一段 reference
  2. 对每个错误率 (1%, 3%, 5%, 10%, 15%, 20%) 生成若干 query reads
  3. 对每个算法 × 每组参数 × 每个错误率 运行 indexer
  4. 统计召回率 (有匹配的 query 比例) 和匹配质量
  5. 画图对比
"""

import os
import sys
import random
import tempfile
import subprocess
import csv
import json
import time
from pathlib import Path
from collections import defaultdict

METHODS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = METHODS_DIR.parent

# ─────────────────────── 数据生成 ───────────────────────

def load_reference(fasta_path, max_len=50000):
    """从 FASTA 加载 reference 序列（截取前 max_len bp）"""
    seq_parts = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip().upper())
    full_seq = "".join(seq_parts)
    return full_seq[:max_len]


def mutate_sequence(sequence, mutation_rate, rng):
    bases = ['A', 'T', 'C', 'G']
    seq_list = list(sequence)
    length = len(seq_list)
    num_mutations = max(1, int(length * mutation_rate))

    for _ in range(num_mutations):
        idx = rng.randint(0, len(seq_list) - 1)
        op = rng.choice(['sub', 'del', 'ins'])
        if op == 'sub':
            original = seq_list[idx]
            choices = [b for b in bases if b != original]
            if choices:
                seq_list[idx] = rng.choice(choices)
        elif op == 'del':
            if len(seq_list) > length // 2:
                del seq_list[idx]
        elif op == 'ins':
            seq_list.insert(idx, rng.choice(bases))
    return ''.join(seq_list)


def generate_test_data(reference, mutation_rate, num_reads, read_length, seed=42):
    """生成带突变的 reads，返回 [(read_id, seq, true_start), ...]"""
    rng = random.Random(seed)
    reads = []
    ref_len = len(reference)
    for i in range(num_reads):
        start = rng.randint(0, ref_len - read_length)
        fragment = reference[start:start + read_length]
        mutated = mutate_sequence(fragment, mutation_rate, rng)
        reads.append((f"read_{i:04d}", mutated, start))
    return reads


def write_fasta(path, name, seq):
    with open(path, 'w') as f:
        f.write(f">{name}\n{seq}\n")


def write_fastq(path, reads):
    with open(path, 'w') as f:
        for rid, seq, _ in reads:
            f.write(f"@{rid}\n{seq}\n+\n{'I' * len(seq)}\n")


# ─────────────────────── 算法运行 ───────────────────────

def run_strobemer(ref_fa, query_fq, out_tsv, params):
    strobe_dir = METHODS_DIR / "ksahlin" / "strobemers"
    cmd = [
        sys.executable, str(strobe_dir / "StrobeMap"),
        "--queries", query_fq,
        "--references", ref_fa,
        "--outfolder", str(Path(out_tsv).parent),
        "--prefix", Path(out_tsv).stem,
        "--k", str(params.get("k", 15)),
        "--strobe_w_min_offset", str(params.get("w_min", 20)),
        "--strobe_w_max_offset", str(params.get("w_max", 70)),
        "--w", str(params.get("w", 1)),
        "--n", str(params.get("order", 2)),
        "--rev_comp",
    ]
    mode = params.get("mode", "hybridstrobe")
    if mode == "kmer":
        cmd.append("--kmer_index")
    elif mode == "randstrobe":
        cmd.append("--randstrobe_index")
    elif mode == "minstrobe":
        cmd.append("--minstrobe_index")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(strobe_dir) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(strobe_dir))
    if proc.returncode != 0:
        return {"error": proc.stderr[:500]}

    raw_tsv = str(Path(out_tsv).parent / (Path(out_tsv).stem + ".tsv"))
    return parse_strobemer_output(raw_tsv)


def parse_strobemer_output(raw_path):
    """解析 StrobeMap 输出，返回 {read_id: [matches]}"""
    results = defaultdict(list)
    if not os.path.exists(raw_path):
        return results
    current_query = None
    with open(raw_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                parts = line[2:].strip().split()
                current_query = parts[0] if parts else None
            else:
                fields = line.strip().split()
                if len(fields) >= 4 and current_query:
                    results[current_query].append({
                        "ref_pos": int(fields[1]),
                        "query_pos": int(fields[2]),
                        "match_len": int(fields[3]),
                    })
    return results


def run_spaced_seed(ref_fa, query_fq, out_tsv, params):
    ss_dir = METHODS_DIR / "spaced_seed"
    cmd = [
        sys.executable, str(ss_dir / "ales_fasta_fastq_align_parallel.py"),
        "--fasta", ref_fa,
        "--fastq", query_fq,
        "--output", out_tsv,
        "--weight", str(params.get("weight", 11)),
        "--homology-length", str(params.get("homology_length", 64)),
        "--similarity", str(params.get("similarity", 0.8)),
        "--k-seeds", str(params.get("k_seeds", 1)),
        "--tries", str(params.get("tries", 40)),
        "--indel-trials", str(params.get("indel_trials", 40)),
        "--no-revcomp",
    ]
    if params.get("seeds"):
        cmd.extend(["--seeds"] + params["seeds"])

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ss_dir))
    if proc.returncode != 0:
        return {"error": proc.stderr[:500]}

    return parse_spaced_seed_output(out_tsv)


def parse_spaced_seed_output(tsv_path):
    results = defaultdict(list)
    if not os.path.exists(tsv_path):
        return results
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rid = row.get("read_id", "")
            if rid and rid != "*":
                results[rid].append({
                    "ref_start": int(row.get("reference_start", 0)),
                    "score": float(row.get("score", 0)),
                    "aligned_length": int(row.get("aligned_length", 0)),
                })
    return results


def run_tensor_sketch(ref_fa, query_fq, out_tsv, params):
    binary = METHODS_DIR / "tensor-sketch-alignment" / "metagraph" / "mgsketch_index"
    if not binary.exists():
        return {"error": f"mgsketch_index not found at {binary}"}

    cmd = [
        str(binary),
        "--ref_fasta", ref_fa,
        "--query_fastq", query_fq,
        "--output_tsv", out_tsv,
        "--window_size", str(params.get("window_size", 150)),
        "--stride", str(params.get("stride", 1)),
        "--sketch_dim", str(params.get("sketch_dim", 128)),
        "--tuple_size", str(params.get("tuple_size", 3)),
        "--seed", str(params.get("seed", 42)),
        "--hnsw_M", str(params.get("hnsw_M", 16)),
        "--hnsw_ef_construction", str(params.get("hnsw_ef_construction", 200)),
        "--hnsw_ef_search", str(params.get("hnsw_ef_search", 50)),
        "--top_k", str(params.get("top_k", 5)),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr[:500]}

    return parse_tensor_sketch_output(out_tsv)


def parse_tensor_sketch_output(tsv_path):
    results = defaultdict(list)
    if not os.path.exists(tsv_path):
        return results
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rid = row.get("read_id", "")
            if rid:
                results[rid].append({
                    "ref_start": int(row.get("reference_start", 0)),
                    "sketch_l2_dist": float(row.get("sketch_l2_dist", 0)),
                    "edit_dist": int(row.get("edit_dist", 0)),
                })
    return results


# ─────────────────────── 参数配置 ───────────────────────

STROBEMER_CONFIGS = {
    "k=20": {"k": 20, "order": 2, "w_min": 20, "w_max": 70},
    "k=15 (default)": {"k": 15, "order": 2, "w_min": 20, "w_max": 70},
    "k=10": {"k": 10, "order": 2, "w_min": 20, "w_max": 70},
}

SPACED_SEED_CONFIGS = {
    "w=14": {"weight": 14, "similarity": 0.8, "k_seeds": 1, "homology_length": 64},
    "w=11 (default)": {"weight": 11, "similarity": 0.8, "k_seeds": 1, "homology_length": 64},
    "w=8": {"weight": 8, "similarity": 0.8, "k_seeds": 1, "homology_length": 64},
}

TENSOR_SKETCH_CONFIGS = {
    "ef=20": {"sketch_dim": 128, "tuple_size": 3, "hnsw_ef_search": 20, "top_k": 5, "window_size": 150},
    "ef=50 (default)": {"sketch_dim": 128, "tuple_size": 3, "hnsw_ef_search": 50, "top_k": 5, "window_size": 150},
    "ef=200": {"sketch_dim": 128, "tuple_size": 3, "hnsw_ef_search": 200, "top_k": 5, "window_size": 150},
}

MUTATION_RATES = [0.01, 0.05, 0.10, 0.15]
NUM_READS = 10
READ_LENGTH = 200
REF_LENGTH = 5000


# ─────────────────────── 主测试逻辑 ───────────────────────

def compute_recall(match_results, reads, ref_seq, tolerance=50):
    """
    计算召回率：如果某个 read 的匹配位置与真实起始位置差距 <= tolerance，则视为正确。
    对于没有位置信息的算法（如只返回分数），只要有匹配就算命中。
    """
    if isinstance(match_results, dict) and "error" in match_results:
        return 0.0, 0

    total = len(reads)
    if total == 0:
        return 0.0, 0

    hit_count = 0
    for rid, seq, true_start in reads:
        matches = match_results.get(rid, [])
        if not matches:
            continue

        found = False
        for m in matches:
            ref_pos = m.get("ref_pos", m.get("ref_start", -1))
            if ref_pos >= 0 and abs(ref_pos - true_start) <= tolerance:
                found = True
                break
            elif ref_pos < 0:
                found = True
                break
        if found:
            hit_count += 1

    return hit_count / total, hit_count


def run_benchmark():
    ecoli_fa = PROJECT_DIR / "data" / "ecoli" / "fasta" / "ecoli.fa"
    if not ecoli_fa.exists():
        print(f"[错误] 找不到 E. coli 参考基因组: {ecoli_fa}")
        sys.exit(1)

    print("=" * 70)
    print("容错参数对比测试")
    print("=" * 70)

    print(f"\n[1/5] 加载参考序列 (前 {REF_LENGTH} bp)...")
    reference = load_reference(str(ecoli_fa), max_len=REF_LENGTH)
    print(f"  参考序列长度: {len(reference)} bp")

    output_dir = METHODS_DIR / "benchmark_output"
    output_dir.mkdir(exist_ok=True)

    all_results = {
        "strobemer": {},
        "spaced_seed": {},
        "tensor_sketch": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_fa = os.path.join(tmpdir, "ref.fa")
        write_fasta(ref_fa, "ecoli_segment", reference)

        for mr_idx, mutation_rate in enumerate(MUTATION_RATES):
            pct = int(mutation_rate * 100)
            print(f"\n[2/5] 错误率 {pct}% ({mr_idx+1}/{len(MUTATION_RATES)})")

            reads = generate_test_data(reference, mutation_rate, NUM_READS, READ_LENGTH, seed=42 + mr_idx)
            query_fq = os.path.join(tmpdir, f"query_{pct}pct.fq")
            write_fastq(query_fq, reads)

            # --- Strobemer ---
            for cfg_name, params in STROBEMER_CONFIGS.items():
                out_tsv = os.path.join(tmpdir, f"strobe_{pct}_{cfg_name}.tsv")
                t0 = time.time()
                results = run_strobemer(ref_fa, query_fq, out_tsv, params)
                elapsed = time.time() - t0

                if isinstance(results, dict) and "error" in results:
                    recall, hits = 0.0, 0
                    print(f"    [strobemer] {cfg_name}: 错误 - {results['error'][:100]}")
                else:
                    recall, hits = compute_recall(results, reads, reference)
                    print(f"    [strobemer] {cfg_name}: 召回={recall:.1%} ({hits}/{NUM_READS}) 耗时={elapsed:.1f}s")

                key = (cfg_name, mutation_rate)
                all_results["strobemer"][key] = {
                    "recall": recall, "hits": hits, "time": elapsed,
                    "params": params, "mutation_rate": mutation_rate,
                }

            # --- Spaced Seed ---
            for cfg_name, params in SPACED_SEED_CONFIGS.items():
                out_tsv = os.path.join(tmpdir, f"ss_{pct}_{cfg_name}.tsv")
                t0 = time.time()
                results = run_spaced_seed(ref_fa, query_fq, out_tsv, params)
                elapsed = time.time() - t0

                if isinstance(results, dict) and "error" in results:
                    recall, hits = 0.0, 0
                    print(f"    [spaced_seed] {cfg_name}: 错误 - {results['error'][:100]}")
                else:
                    recall, hits = compute_recall(results, reads, reference)
                    print(f"    [spaced_seed] {cfg_name}: 召回={recall:.1%} ({hits}/{NUM_READS}) 耗时={elapsed:.1f}s")

                key = (cfg_name, mutation_rate)
                all_results["spaced_seed"][key] = {
                    "recall": recall, "hits": hits, "time": elapsed,
                    "params": params, "mutation_rate": mutation_rate,
                }

            # --- Tensor Sketch ---
            for cfg_name, params in TENSOR_SKETCH_CONFIGS.items():
                out_tsv = os.path.join(tmpdir, f"ts_{pct}_{cfg_name}.tsv")
                t0 = time.time()
                results = run_tensor_sketch(ref_fa, query_fq, out_tsv, params)
                elapsed = time.time() - t0

                if isinstance(results, dict) and "error" in results:
                    recall, hits = 0.0, 0
                    print(f"    [tensor_sketch] {cfg_name}: 错误 - {results['error'][:100]}")
                else:
                    recall, hits = compute_recall(results, reads, reference)
                    print(f"    [tensor_sketch] {cfg_name}: 召回={recall:.1%} ({hits}/{NUM_READS}) 耗时={elapsed:.1f}s")

                key = (cfg_name, mutation_rate)
                all_results["tensor_sketch"][key] = {
                    "recall": recall, "hits": hits, "time": elapsed,
                    "params": params, "mutation_rate": mutation_rate,
                }

    # 保存原始数据
    results_json = output_dir / "benchmark_results.json"
    serializable = {}
    for method, data in all_results.items():
        serializable[method] = {}
        for (cfg, mr), val in data.items():
            serializable[method][f"{cfg}|{mr}"] = val
    with open(results_json, 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[3/5] 原始数据已保存: {results_json}")

    # 画图
    print("\n[4/5] 生成对比图表...")
    plot_results(all_results, output_dir)

    print(f"\n[5/5] 完成！所有结果保存在: {output_dir}")


# ─────────────────────── 画图 ───────────────────────

def plot_results(all_results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 8,
        'figure.dpi': 150,
    })

    mutation_pcts = [int(mr * 100) for mr in MUTATION_RATES]

    # --- 图 1: 三个子图，每个算法一个 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    method_configs = {
        "strobemer": (STROBEMER_CONFIGS, "Strobemer"),
        "spaced_seed": (SPACED_SEED_CONFIGS, "Spaced Seed"),
        "tensor_sketch": (TENSOR_SKETCH_CONFIGS, "Tensor Sketch"),
    }

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    for ax_idx, (method, (configs, title)) in enumerate(method_configs.items()):
        ax = axes[ax_idx]
        for cfg_idx, cfg_name in enumerate(configs.keys()):
            recalls = []
            for mr in MUTATION_RATES:
                key = (cfg_name, mr)
                val = all_results[method].get(key, {})
                recalls.append(val.get("recall", 0.0))

            ax.plot(mutation_pcts, recalls,
                    marker=markers[cfg_idx % len(markers)],
                    color=colors[cfg_idx],
                    linewidth=2, markersize=6,
                    label=cfg_name)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Mutation Rate (%)")
        if ax_idx == 0:
            ax.set_ylabel("Recall")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(mutation_pcts)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', framealpha=0.9)

    fig.suptitle("Error Tolerance Benchmark: Recall vs Mutation Rate", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path1 = output_dir / "recall_by_method.png"
    fig.savefig(path1, bbox_inches='tight')
    plt.close(fig)
    print(f"  图 1 已保存: {path1}")

    # --- 图 2: 三个算法的默认配置对比 ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    default_cfgs = {
        "Strobemer": ("strobemer", "k=15 (default)"),
        "Spaced Seed": ("spaced_seed", "w=11 (default)"),
        "Tensor Sketch": ("tensor_sketch", "ef=50 (default)"),
    }

    for idx, (label, (method, cfg_name)) in enumerate(default_cfgs.items()):
        recalls = []
        for mr in MUTATION_RATES:
            key = (cfg_name, mr)
            val = all_results[method].get(key, {})
            recalls.append(val.get("recall", 0.0))
        ax2.plot(mutation_pcts, recalls,
                 marker=markers[idx], color=colors[idx],
                 linewidth=2.5, markersize=8,
                 label=label)

    ax2.set_title("Default Config Comparison: Recall vs Mutation Rate", fontweight='bold')
    ax2.set_xlabel("Mutation Rate (%)")
    ax2.set_ylabel("Recall")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(mutation_pcts)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    plt.tight_layout()
    path2 = output_dir / "recall_default_comparison.png"
    fig2.savefig(path2, bbox_inches='tight')
    plt.close(fig2)
    print(f"  图 2 已保存: {path2}")

    # --- 图 3: 每个算法的最佳容错配置对比 ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    best_cfgs = {
        "Strobemer (best)": ("strobemer", "k=10"),
        "Spaced Seed (best)": ("spaced_seed", "w=8"),
        "Tensor Sketch (best)": ("tensor_sketch", "ef=200"),
    }

    for idx, (label, (method, cfg_name)) in enumerate(best_cfgs.items()):
        recalls = []
        for mr in MUTATION_RATES:
            key = (cfg_name, mr)
            val = all_results[method].get(key, {})
            recalls.append(val.get("recall", 0.0))
        ax3.plot(mutation_pcts, recalls,
                 marker=markers[idx], color=colors[idx + 3],
                 linewidth=2.5, markersize=8,
                 label=label)

    ax3.set_title("Best Tolerance Config Comparison: Recall vs Mutation Rate", fontweight='bold')
    ax3.set_xlabel("Mutation Rate (%)")
    ax3.set_ylabel("Recall")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xticks(mutation_pcts)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    plt.tight_layout()
    path3 = output_dir / "recall_best_comparison.png"
    fig3.savefig(path3, bbox_inches='tight')
    plt.close(fig3)
    print(f"  图 3 已保存: {path3}")

    # --- 图 4: 运行时间对比 ---
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, (method, (configs, title)) in enumerate(method_configs.items()):
        ax = axes4[ax_idx]
        for cfg_idx, cfg_name in enumerate(configs.keys()):
            times = []
            for mr in MUTATION_RATES:
                key = (cfg_name, mr)
                val = all_results[method].get(key, {})
                times.append(val.get("time", 0.0))

            ax.plot(mutation_pcts, times,
                    marker=markers[cfg_idx % len(markers)],
                    color=colors[cfg_idx],
                    linewidth=2, markersize=6,
                    label=cfg_name)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Mutation Rate (%)")
        if ax_idx == 0:
            ax.set_ylabel("Time (seconds)")
        ax.set_xticks(mutation_pcts)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9)

    fig4.suptitle("Runtime vs Mutation Rate", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path4 = output_dir / "runtime_by_method.png"
    fig4.savefig(path4, bbox_inches='tight')
    plt.close(fig4)
    print(f"  图 4 已保存: {path4}")


if __name__ == "__main__":
    run_benchmark()
