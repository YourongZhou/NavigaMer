#!/usr/bin/env python3
"""
NavigaMer V3.0 仿真基准测试 —— 四个实验对比 NavigaMer C++ / Strobemer / Spaced Seed / Tensor Sketch.

实验一: 鲁棒性边界 (Recall vs Error Rate)
实验二: 搜索空间缩减 (ROC-like: Candidate Fraction vs Recall)
实验三: 全面性 (Hit Recall 多拷贝)
实验四: 候选集质量 (Edit Distance 分布密度图)
"""

import os
import sys
import random
import tempfile
import subprocess
import csv
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

METHODS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = METHODS_DIR.parent
NAVIGAMER_CPP = PROJECT_DIR / "navigamer_cpp"
OUTPUT_DIR = METHODS_DIR / "v3_benchmark_output"

# ─────────────────────── Data generation ───────────────────────

def load_reference(fasta_path, max_len=10000):
    seq_parts = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip().upper())
    return "".join(seq_parts)[:max_len]


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
        elif op == 'del' and len(seq_list) > length // 2:
            del seq_list[idx]
        elif op == 'ins':
            seq_list.insert(idx, rng.choice(bases))
    return ''.join(seq_list)


def generate_test_data(reference, mutation_rate, num_reads, read_length, seed=42, return_original=False):
    """Returns [(read_id, seq, true_start), ...] or with original_fragment if return_original."""
    rng = random.Random(seed)
    reads = []
    ref_len = len(reference)
    for i in range(num_reads):
        start = rng.randint(0, max(0, ref_len - read_length))
        fragment = reference[start:start + read_length]
        if len(fragment) < read_length:
            continue
        mutated = mutate_sequence(fragment, mutation_rate, rng)
        if return_original:
            reads.append((f"read_{i:04d}", mutated, start, fragment))
        else:
            reads.append((f"read_{i:04d}", mutated, start))
    return reads


def write_fasta(path, name, seq):
    with open(path, 'w') as f:
        f.write(f">{name}\n{seq}\n")


def write_fastq(path, reads):
    with open(path, 'w') as f:
        for r in reads:
            rid, seq = r[0], r[1]
            f.write(f"@{rid}\n{seq}\n+\n{'I' * len(seq)}\n")


# ─────────────────────── Method runners ───────────────────────

def run_navigamer(ref_fa, query_fq, out_tsv, params):
    binary = NAVIGAMER_CPP / "navigamer"
    if not binary.exists():
        return {"error": f"navigamer not found at {binary}"}
    window = params.get("window_size", 200)
    stride = params.get("stride", 1)
    tol = params.get("tolerance", 5)
    cmd = [str(binary), "benchmark", "--ref", ref_fa, "--reads", query_fq,
           "--tolerance", str(tol), "--window", str(window), "--stride", str(stride), "--out", out_tsv]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(NAVIGAMER_CPP))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[:500], "runtime": elapsed}
    return parse_navigamer_output(out_tsv, elapsed)


def parse_navigamer_output(tsv_path, runtime=0.0):
    hits = defaultdict(list)
    total_leaf_verify = 0
    total_dist_calcs = 0
    if not os.path.exists(tsv_path):
        return {"hits": hits, "candidates_count": 0, "runtime": runtime}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rid = row.get("query_id") or row.get("read_id", "")
            if not rid:
                continue
            try:
                total_leaf_verify += int(row.get("leaf_verify_count", 0))
                total_dist_calcs += int(row.get("dist_calcs", 0))
            except (ValueError, TypeError):
                pass
            ref_start = row.get("reference_start", "")
            if ref_start.isdigit() and row.get("hit_id"):
                hits[rid].append({
                    "ref_start": int(ref_start),
                    "ref_pos": int(ref_start),
                    "edit_dist": int(row.get("edit_distance") or 0),
                    "score": int(row.get("score") or 0),
                })
    return {"hits": hits, "candidates_count": total_leaf_verify, "runtime": runtime}


def run_strobemer(ref_fa, query_fq, out_tsv, params):
    strobe_dir = METHODS_DIR / "ksahlin" / "strobemers"
    cmd = [sys.executable, str(strobe_dir / "StrobeMap"), "--queries", query_fq,
           "--references", ref_fa, "--outfolder", str(Path(out_tsv).parent),
           "--prefix", Path(out_tsv).stem, "--k", str(params.get("k", 15)),
           "--strobe_w_min_offset", str(params.get("w_min", 20)),
           "--strobe_w_max_offset", str(params.get("w_max", 70)),
           "--w", str(params.get("w", 1)), "--n", str(params.get("order", 2)), "--rev_comp"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(strobe_dir) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(strobe_dir))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[:500], "hits": defaultdict(list), "candidates_count": 0, "runtime": elapsed}
    raw = Path(out_tsv).parent / (Path(out_tsv).stem + ".tsv")
    hits = parse_strobemer_output(str(raw))
    total_matches = sum(len(v) for v in hits.values())
    return {"hits": hits, "candidates_count": total_matches, "runtime": elapsed}


def parse_strobemer_output(raw_path):
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
                        "ref_pos": int(fields[1]), "query_pos": int(fields[2]),
                        "match_len": int(fields[3]), "ref_start": int(fields[1]),
                    })
    return results


def run_spaced_seed(ref_fa, query_fq, out_tsv, params):
    ss_dir = METHODS_DIR / "spaced_seed"
    cmd = [sys.executable, str(ss_dir / "ales_fasta_fastq_align_parallel.py"),
           "--fasta", ref_fa, "--fastq", query_fq, "--output", out_tsv,
           "--weight", str(params.get("weight", 11)),
           "--homology-length", str(params.get("homology_length", 64)),
           "--similarity", str(params.get("similarity", 0.8)),
           "--k-seeds", str(params.get("k_seeds", 1)), "--no-revcomp"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ss_dir))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[:500], "hits": defaultdict(list), "candidates_count": 0, "runtime": elapsed}
    hits = parse_spaced_seed_output(out_tsv)
    total = sum(len(v) for v in hits.values())
    return {"hits": hits, "candidates_count": total, "runtime": elapsed}


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
        return {"error": f"mgsketch_index not found", "hits": defaultdict(list), "candidates_count": 0, "runtime": 0}
    cmd = [str(binary), "--ref_fasta", ref_fa, "--query_fastq", query_fq, "--output_tsv", out_tsv,
           "--window_size", str(params.get("window_size", 150)), "--stride", str(params.get("stride", 1)),
           "--sketch_dim", str(params.get("sketch_dim", 128)), "--tuple_size", str(params.get("tuple_size", 3)),
           "--hnsw_ef_search", str(params.get("hnsw_ef_search", 50)), "--top_k", str(params.get("top_k", 5))]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[:500], "hits": defaultdict(list), "candidates_count": 0, "runtime": elapsed}
    hits = parse_tensor_sketch_output(out_tsv)
    top_k = params.get("top_k", 5)
    num_queries = len(hits)
    total_candidates = num_queries * top_k
    return {"hits": hits, "candidates_count": total_candidates, "runtime": elapsed}


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
                    "edit_dist": int(row.get("edit_dist", 0)),
                    "sketch_l2_dist": float(row.get("sketch_l2_dist", 0)),
                })
    return results


# ─────────────────────── Metrics ───────────────────────

POS_TOLERANCE = 50


def sequence_recall(match_results, reads, tolerance=POS_TOLERANCE):
    """Recall: fraction of reads with at least one hit within tolerance of true_start."""
    if isinstance(match_results, dict) and "error" in match_results:
        return 0.0, 0
    hits_dict = match_results.get("hits", match_results) if isinstance(match_results, dict) else match_results
    total = len(reads)
    if total == 0:
        return 0.0, 0
    hit_count = 0
    for r in reads:
        rid, true_start = r[0], r[2]
        matches = hits_dict.get(rid, [])
        for m in matches:
            ref_pos = m.get("ref_pos", m.get("ref_start", -1))
            if ref_pos >= 0 and abs(ref_pos - true_start) <= tolerance:
                hit_count += 1
                break
    return hit_count / total, hit_count


def hit_recall(match_results, ground_truth_positions):
    """ground_truth_positions: dict read_id -> set of (ref_id, start) true positions.
    Returns |Reported ∩ True| / |True|."""
    if isinstance(match_results, dict) and "error" in match_results:
        return 0.0
    hits_dict = match_results.get("hits", match_results)
    all_true = set()
    all_reported = set()
    for rid, true_positions in ground_truth_positions.items():
        for (ref_id, start) in true_positions:
            all_true.add((ref_id, start))
        for m in hits_dict.get(rid, []):
            ref_start = m.get("ref_start", m.get("ref_pos", None))
            if ref_start is not None:
                all_reported.add(("ref", ref_start))
    if not all_true:
        return 1.0
    return len(all_reported & all_true) / len(all_true)


def candidate_fraction(candidates_count, total_db_size):
    if total_db_size <= 0:
        return 0.0
    return candidates_count / total_db_size


# ─────────────────────── Configs ───────────────────────

# 简单 case：短 ref、少 reads，便于快速看效果（不用全 ecoli）
MUTATION_RATES = [0.0, 0.05, 0.15]
REF_LENGTH = 3000
READ_LENGTH = 200
NUM_READS_EXP1 = 5
NUM_READS_EXP2 = 5

DEFAULT_PARAMS = {
    "navigamer": {"tolerance": 5, "window_size": 200, "stride": 1},
    "strobemer": {"k": 15, "order": 2, "w_min": 20, "w_max": 70},
    "spaced_seed": {"weight": 11, "similarity": 0.8, "homology_length": 64, "k_seeds": 1},
    "tensor_sketch": {"hnsw_ef_search": 50, "top_k": 5, "window_size": 150, "sketch_dim": 128, "tuple_size": 3},
}

EXP2_KNOB_SCAN = {
    "navigamer": [{"tolerance": r, "window_size": 200, "stride": 1} for r in [2, 5, 10, 20]],
    "strobemer": [{"k": k, "order": 2, "w_min": 20, "w_max": 70} for k in [20, 15, 10]],
    "spaced_seed": [{"weight": w, "similarity": 0.8, "homology_length": 64, "k_seeds": 1} for w in [14, 11, 8]],
    "tensor_sketch": [{"hnsw_ef_search": ef, "top_k": 5, "window_size": 150, "sketch_dim": 128, "tuple_size": 3} for ef in [20, 50, 200]],
}


# ─────────────────────── Experiment 1: Robustness ───────────────────────

def run_experiment_1(ecoli_fa, output_dir):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = load_reference(str(ecoli_fa), max_len=REF_LENGTH)
    results = {"navigamer": [], "strobemer": [], "spaced_seed": [], "tensor_sketch": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_fa = os.path.join(tmpdir, "ref.fa")
        write_fasta(ref_fa, "ref", reference)

        for mr in MUTATION_RATES:
            reads = generate_test_data(reference, mr, NUM_READS_EXP1, READ_LENGTH, seed=42 + int(mr * 100))
            query_fq = os.path.join(tmpdir, f"q_{int(mr*100)}.fq")
            write_fastq(query_fq, reads)

            for method, runner in [
                ("navigamer", lambda rf, qf, ot, p: run_navigamer(rf, qf, ot, p)),
                ("strobemer", lambda rf, qf, ot, p: run_strobemer(rf, qf, ot, p)),
                ("spaced_seed", lambda rf, qf, ot, p: run_spaced_seed(rf, qf, ot, p)),
                ("tensor_sketch", lambda rf, qf, ot, p: run_tensor_sketch(rf, qf, ot, p)),
            ]:
                out_tsv = os.path.join(tmpdir, f"{method}_{int(mr*100)}.tsv")
                res = runner(ref_fa, query_fq, out_tsv, DEFAULT_PARAMS[method])
                rec, _ = sequence_recall(res, reads)
                results[method].append({"mutation_rate": mr, "recall": rec})

    with open(output_dir / "exp1_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_exp1(results, output_dir)
    return results


def plot_exp1(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    xs = [int(mr * 100) for mr in MUTATION_RATES]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (method, label) in enumerate([("navigamer", "NavigaMer"), ("strobemer", "Strobemer"),
                                         ("spaced_seed", "Spaced Seed"), ("tensor_sketch", "Tensor Sketch")]):
        ys = [r["recall"] for r in results[method]]
        ax.plot(xs, ys, marker='o', label=label, linewidth=2)
    ax.set_xlabel("Mutation Rate (%)")
    ax.set_ylabel("Sequence Recall")
    ax.set_title("Exp1: Robustness (Default Params)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "exp1_robustness.png", bbox_inches='tight')
    plt.close(fig)


# ─────────────────────── Experiment 2: Search space ───────────────────────

def run_experiment_2(ecoli_fa, output_dir):
    reference = load_reference(str(ecoli_fa), max_len=REF_LENGTH)
    fixed_error = 0.10
    reads = generate_test_data(reference, fixed_error, NUM_READS_EXP2, READ_LENGTH, seed=99)
    total_windows_ref = max(0, (len(reference) - READ_LENGTH) // 1 + 1)
    total_windows_nm = max(0, (len(reference) - 200) // 1 + 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_fa = os.path.join(tmpdir, "ref.fa")
        write_fasta(ref_fa, "ref", reference)
        query_fq = os.path.join(tmpdir, "query.fq")
        write_fastq(query_fq, reads)

        series = {"navigamer": [], "strobemer": [], "spaced_seed": [], "tensor_sketch": []}
        for method in ["navigamer", "strobemer", "spaced_seed", "tensor_sketch"]:
            configs = EXP2_KNOB_SCAN[method]
            for idx, params in enumerate(configs):
                out_tsv = os.path.join(tmpdir, f"{method}_{idx}.tsv")
                if method == "navigamer":
                    res = run_navigamer(ref_fa, query_fq, out_tsv, params)
                elif method == "strobemer":
                    res = run_strobemer(ref_fa, query_fq, out_tsv, params)
                elif method == "spaced_seed":
                    res = run_spaced_seed(ref_fa, query_fq, out_tsv, params)
                else:
                    res = run_tensor_sketch(ref_fa, query_fq, out_tsv, params)
                rec, _ = sequence_recall(res, reads)
                cand = res.get("candidates_count", 0)
                if method == "navigamer":
                    frac = candidate_fraction(cand, total_windows_nm)
                elif method == "tensor_sketch":
                    frac = candidate_fraction(cand, total_windows_ref * len(reads))
                else:
                    frac = candidate_fraction(cand, total_windows_ref * len(reads))
                series[method].append({"recall": rec, "candidate_fraction": max(1e-6, frac), "params": params})

    with open(output_dir / "exp2_search_space.json", "w") as f:
        def ser(o):
            if isinstance(o, (int, float, str, bool, type(None))):
                return o
            if isinstance(o, dict):
                return {k: ser(v) for k, v in o.items()}
            if isinstance(o, list):
                return [ser(x) for x in o]
            return str(o)
        json.dump(ser(series), f, indent=2)
    plot_exp2(series, output_dir)
    return series


def plot_exp2(series, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, label in [("navigamer", "NavigaMer"), ("strobemer", "Strobemer"),
                         ("spaced_seed", "Spaced Seed"), ("tensor_sketch", "Tensor Sketch")]:
        pts = series[method]
        xs = [p["candidate_fraction"] for p in pts]
        ys = [p["recall"] for p in pts]
        ax.plot(xs, ys, marker='o', label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Candidate Fraction (log)")
    ax.set_ylabel("Recall")
    ax.set_title("Exp2: Search Space vs Recall (10% Error)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "exp2_search_space_roc.png", bbox_inches='tight')
    plt.close(fig)


# ─────────────────────── Experiment 3: Comprehensiveness ───────────────────────

def run_experiment_3(ecoli_fa, output_dir):
    reference = load_reference(str(ecoli_fa), max_len=REF_LENGTH)
    # Multi-copy: build ref with same 200bp fragment at 3 positions; one query = mutated fragment
    rng = random.Random(42)
    fragment_len = 200
    start0 = rng.randint(0, len(reference) - fragment_len - 1)
    fragment = reference[start0:start0 + fragment_len]
    gap = 100
    ref_multi = fragment + "N" * gap + fragment + "N" * gap + fragment
    true_positions = [0, fragment_len + gap, 2 * (fragment_len + gap)]
    query_seq = mutate_sequence(fragment, 0.05, rng)
    ground_truth = {"read_multi": set(("ref", p) for p in true_positions)}

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_fa = os.path.join(tmpdir, "ref.fa")
        write_fasta(ref_fa, "ref", ref_multi)
        query_fq = os.path.join(tmpdir, "q.fq")
        with open(query_fq, "w") as f:
            f.write(f"@read_multi\n{query_seq}\n+\n{'I'*len(query_seq)}\n")

        hit_recalls = {}
        for method, params in DEFAULT_PARAMS.items():
            out_tsv = os.path.join(tmpdir, f"{method}.tsv")
            if method == "navigamer":
                res = run_navigamer(ref_fa, query_fq, out_tsv, params)
            elif method == "strobemer":
                res = run_strobemer(ref_fa, query_fq, out_tsv, params)
            elif method == "spaced_seed":
                res = run_spaced_seed(ref_fa, query_fq, out_tsv, params)
            else:
                res = run_tensor_sketch(ref_fa, query_fq, out_tsv, params)
            hr = hit_recall(res, ground_truth)
            hit_recalls[method] = hr

    with open(output_dir / "exp3_comprehensiveness.json", "w") as f:
        json.dump(hit_recalls, f, indent=2)
    plot_exp3(hit_recalls, output_dir)
    return hit_recalls


def plot_exp3(hit_recalls, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ["NavigaMer", "Strobemer", "Spaced Seed", "Tensor Sketch"]
    keys = ["navigamer", "strobemer", "spaced_seed", "tensor_sketch"]
    vals = [hit_recalls.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=['#4C72B0', '#55A868', '#DD8452', '#C44E52'])
    ax.set_ylabel("Hit Recall")
    ax.set_title("Exp3: Comprehensiveness (Multi-copy)")
    ax.set_ylim(0, 1.05)
    fig.savefig(output_dir / "exp3_hit_recall.png", bbox_inches='tight')
    plt.close(fig)


# ─────────────────────── Experiment 4: Candidate quality ───────────────────────

def run_experiment_4(ecoli_fa, output_dir):
    reference = load_reference(str(ecoli_fa), max_len=REF_LENGTH)
    fixed_error = 0.10
    reads = generate_test_data(reference, fixed_error, 8, READ_LENGTH, seed=88, return_original=True)
    target_recall = 0.98

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_fa = os.path.join(tmpdir, "ref.fa")
        write_fasta(ref_fa, "ref", reference)
        query_fq = os.path.join(tmpdir, "q.fq")
        write_fastq(query_fq, reads)

        dists_by_method = {}
        for method in ["navigamer", "strobemer", "spaced_seed", "tensor_sketch"]:
            params = DEFAULT_PARAMS[method].copy()
            if method == "navigamer":
                params["tolerance"] = 15
            elif method == "strobemer":
                params["k"] = 10
            elif method == "spaced_seed":
                params["weight"] = 8
            else:
                params["hnsw_ef_search"] = 200
            out_tsv = os.path.join(tmpdir, f"{method}.tsv")
            if method == "navigamer":
                res = run_navigamer(ref_fa, query_fq, out_tsv, params)
            elif method == "strobemer":
                res = run_strobemer(ref_fa, query_fq, out_tsv, params)
            elif method == "spaced_seed":
                res = run_spaced_seed(ref_fa, query_fq, out_tsv, params)
            else:
                res = run_tensor_sketch(ref_fa, query_fq, out_tsv, params)
            hits_dict = res.get("hits", {})
            dists = []
            for r in reads:
                rid, query_seq, true_start, orig = r[0], r[1], r[2], r[3]
                for m in hits_dict.get(rid, []):
                    ref_start = m.get("ref_start", m.get("ref_pos"))
                    if ref_start is None:
                        continue
                    ref_frag = reference[ref_start:ref_start + len(query_seq)] if ref_start + len(query_seq) <= len(reference) else ""
                    if ref_frag:
                        ed = levenshtein(query_seq, ref_frag)
                        dists.append(ed)
            dists_by_method[method] = dists if dists else [0]

    with open(output_dir / "exp4_candidate_quality.json", "w") as f:
        json.dump({k: v for k, v in dists_by_method.items()}, f, indent=2)
    plot_exp4(dists_by_method, output_dir)
    return dists_by_method


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+ (0 if a[i-1]==b[j-1] else 1))
    return dp[m][n]


def plot_exp4(dists_by_method, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, label in [("navigamer", "NavigaMer"), ("strobemer", "Strobemer"),
                         ("spaced_seed", "Spaced Seed"), ("tensor_sketch", "Tensor Sketch")]:
        d = dists_by_method.get(method, [])
        if not d:
            continue
        d = np.array(d)
        d = d[d <= 100]
        if len(d) == 0:
            continue
        ax.hist(d, bins=min(30, max(5, len(d)//5)), alpha=0.5, density=True, label=label)
    ax.set_xlabel("Edit Distance (query vs candidate)")
    ax.set_ylabel("Density")
    ax.set_title("Exp4: Candidate Quality Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "exp4_distance_density.png", bbox_inches='tight')
    plt.close(fig)


# ─────────────────────── Summary plot ───────────────────────

def plot_summary(exp1, exp2, exp3, exp4, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    xs = [int(mr*100) for mr in MUTATION_RATES]
    for i, (method, label) in enumerate([("navigamer", "NavigaMer"), ("strobemer", "Strobemer"),
                                         ("spaced_seed", "Spaced Seed"), ("tensor_sketch", "Tensor Sketch")]):
        ys = [r["recall"] for r in exp1[method]]
        axes[0, 0].plot(xs, ys, marker='o', label=label)
    axes[0, 0].set_xlabel("Mutation Rate (%)")
    axes[0, 0].set_ylabel("Recall")
    axes[0, 0].set_title("Exp1: Robustness")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.05, 1.05)

    for method, label in [("navigamer", "NavigaMer"), ("strobemer", "Strobemer"),
                          ("spaced_seed", "Spaced Seed"), ("tensor_sketch", "Tensor Sketch")]:
        pts = exp2[method]
        axes[0, 1].plot([p["candidate_fraction"] for p in pts], [p["recall"] for p in pts], marker='o', label=label)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Candidate Fraction")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].set_title("Exp2: Search Space")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    labels = ["NavigaMer", "Strobemer", "Spaced Seed", "Tensor Sketch"]
    keys = ["navigamer", "strobemer", "spaced_seed", "tensor_sketch"]
    axes[1, 0].bar(labels, [exp3.get(k, 0) for k in keys], color=['#4C72B0', '#55A868', '#DD8452', '#C44E52'])
    axes[1, 0].set_ylabel("Hit Recall")
    axes[1, 0].set_title("Exp3: Comprehensiveness")
    axes[1, 0].set_ylim(0, 1.05)

    for method, label in [("navigamer", "NavigaMer"), ("strobemer", "Strobemer"),
                          ("spaced_seed", "Spaced Seed"), ("tensor_sketch", "Tensor Sketch")]:
        d = exp4.get(method, [])
        d = [x for x in d if x <= 80]
        if d:
            axes[1, 1].hist(d, bins=20, alpha=0.5, label=label)
    axes[1, 1].set_xlabel("Edit Distance")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Exp4: Candidate Quality")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("NavigaMer V3.0 Benchmark Summary", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "summary.png", bbox_inches='tight')
    plt.close(fig)


# ─────────────────────── Main ───────────────────────

def main():
    parser = argparse.ArgumentParser(description="NavigaMer V3.0 Benchmark")
    parser.add_argument("--exp", type=str, default="all", choices=["1", "2", "3", "4", "all"],
                        help="Run experiment 1, 2, 3, 4, or all")
    parser.add_argument("--ref", type=str, default=None, help="E. coli FASTA (default: data/ecoli/fasta/ecoli.fa)")
    args = parser.parse_args()
    ecoli_fa = args.ref or str(PROJECT_DIR / "data" / "ecoli" / "fasta" / "ecoli.fa")
    if not os.path.exists(ecoli_fa):
        print(f"Reference not found: {ecoli_fa}")
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1_data = exp2_data = exp3_data = exp4_data = None
    if args.exp in ("1", "all"):
        print("Running Experiment 1: Robustness...")
        exp1_data = run_experiment_1(ecoli_fa, OUTPUT_DIR)
    if args.exp in ("2", "all"):
        print("Running Experiment 2: Search Space...")
        exp2_data = run_experiment_2(ecoli_fa, OUTPUT_DIR)
    if args.exp in ("3", "all"):
        print("Running Experiment 3: Comprehensiveness...")
        exp3_data = run_experiment_3(ecoli_fa, OUTPUT_DIR)
    if args.exp in ("4", "all"):
        print("Running Experiment 4: Candidate Quality...")
        exp4_data = run_experiment_4(ecoli_fa, OUTPUT_DIR)

    if args.exp == "all" and (exp1_data is not None or exp2_data is not None or exp3_data is not None or exp4_data is not None):
        if exp1_data is None:
            with open(OUTPUT_DIR / "exp1_robustness.json") as f:
                exp1_data = json.load(f)
        if exp2_data is None:
            with open(OUTPUT_DIR / "exp2_search_space.json") as f:
                exp2_data = json.load(f)
        if exp3_data is None:
            with open(OUTPUT_DIR / "exp3_comprehensiveness.json") as f:
                exp3_data = json.load(f)
        if exp4_data is None:
            with open(OUTPUT_DIR / "exp4_candidate_quality.json") as f:
                exp4_data = json.load(f)
        print("Generating summary plot...")
        plot_summary(exp1_data, exp2_data, exp3_data, exp4_data, OUTPUT_DIR)

    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
