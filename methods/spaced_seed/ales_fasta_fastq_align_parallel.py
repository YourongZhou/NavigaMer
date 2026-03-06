import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import os
import sys
from collections import defaultdict

from ales_spaced_seed import ALeSSpacedSeed


VALID_FASTA_EXT = {".fa", ".fasta", ".fna"}
VALID_FASTQ_EXT = {".fq", ".fastq"}
_PARALLEL_CTX = {}


def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def _is_ext(path, valid_exts):
    name = path.lower()
    for ext in valid_exts:
        if name.endswith(ext) or name.endswith(ext + ".gz"):
            return True
    return False


def resolve_fasta_files(path_or_dir):
    if os.path.isdir(path_or_dir):
        files = []
        for name in sorted(os.listdir(path_or_dir)):
            p = os.path.join(path_or_dir, name)
            if os.path.isfile(p) and _is_ext(p, VALID_FASTA_EXT):
                files.append(p)
        if not files:
            raise FileNotFoundError(f"目录中未找到 FASTA 文件: {path_or_dir}")
        return files
    if not os.path.isfile(path_or_dir):
        raise FileNotFoundError(f"FASTA 路径不存在: {path_or_dir}")
    if not _is_ext(path_or_dir, VALID_FASTA_EXT):
        raise ValueError(f"不是 FASTA 文件: {path_or_dir}")
    return [path_or_dir]


def resolve_fastq_files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                fp = os.path.join(p, name)
                if os.path.isfile(fp) and _is_ext(fp, VALID_FASTQ_EXT):
                    out.append(fp)
        else:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"FASTQ 路径不存在: {p}")
            if not _is_ext(p, VALID_FASTQ_EXT):
                raise ValueError(f"不是 FASTQ 文件: {p}")
            out.append(p)
    if not out:
        raise FileNotFoundError("未找到可用 FASTQ 文件")
    return sorted(out)


def read_fasta(file_path):
    records = []
    name = None
    chunks = []
    with _open_text(file_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks).upper()))
                name = line[1:].split()[0] if line[1:].strip() else f"record_{len(records)+1}"
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    return records


def iter_fastq(file_path, max_reads=None):
    def _detect_pair_suffix(head_line, path):
        tokens = head_line[1:].strip().split()
        for token in tokens:
            if token.endswith("/1") or token.endswith("/2"):
                return token[-2:]
        base = os.path.basename(path).lower()
        if base.endswith("_1.fastq") or base.endswith("_1.fq") or base.endswith("_1.fastq.gz") or base.endswith("_1.fq.gz"):
            return "/1"
        if base.endswith("_2.fastq") or base.endswith("_2.fq") or base.endswith("_2.fastq.gz") or base.endswith("_2.fq.gz"):
            return "/2"
        return ""

    count = 0
    with _open_text(file_path) as fh:
        while True:
            head = fh.readline()
            if not head:
                break
            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()
            if not seq or not plus or not qual:
                raise ValueError(f"FASTQ 格式错误: {file_path}")
            if not head.startswith("@") or not plus.startswith("+"):
                raise ValueError(f"FASTQ 四行记录格式错误: {file_path}")
            rid = head[1:].strip().split()[0]
            suffix = _detect_pair_suffix(head, file_path)
            if suffix and not (rid.endswith("/1") or rid.endswith("/2")):
                rid = rid + suffix
            yield rid, seq.strip().upper(), qual.strip()
            count += 1
            if max_reads is not None and count >= max_reads:
                break


def reverse_complement(seq):
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


def reverse_complement_with_gaps(seq):
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    out = []
    for ch in reversed(seq):
        if ch == "-":
            out.append("-")
        else:
            out.append(ch.translate(comp))
    return "".join(out)


def build_ref_index(ales, reference, masks):
    idx_by_mask = {}
    for mask in masks:
        mlen = len(mask)
        table = defaultdict(list)
        if len(reference) >= mlen:
            for i in range(len(reference) - mlen + 1):
                key = ales.apply_mask(reference[i : i + mlen], mask)
                table[key].append(i)
        idx_by_mask[mask] = table
    return idx_by_mask


def find_hits_with_index(
    ales,
    query,
    masks,
    ref_index,
    max_ref_hits_per_key=None,
    max_hit_candidates=None,
):
    hits = []
    for mask in masks:
        mlen = len(mask)
        if len(query) < mlen:
            continue
        table = ref_index[mask]
        for q in range(len(query) - mlen + 1):
            q_key = ales.apply_mask(query[q : q + mlen], mask)
            refs = table.get(q_key)
            if not refs:
                continue
            if max_ref_hits_per_key is not None and len(refs) > max_ref_hits_per_key:
                continue
            for r in refs:
                hits.append((mask, q, r))
                if max_hit_candidates is not None and len(hits) >= max_hit_candidates:
                    return hits
    return hits
def align_one_strand(ales, query, reference, masks, ref_index, args):
    hits = find_hits_with_index(
        ales,
        query,
        masks,
        ref_index,
        max_ref_hits_per_key=args.max_ref_hits_per_key,
        max_hit_candidates=args.max_hit_candidates,
    )
    if not hits:
        return None

    hsps = []
    for mask, q_pos, r_pos in hits:
        ext = ales._ungapped_xdrop_extend(
            query,
            reference,
            q_pos,
            r_pos,
            len(mask),
            match_score=args.match_score,
            mismatch_score=args.mismatch_score,
            xdrop=args.xdrop,
        )
        hsps.append(
            {
                "mask": mask,
                "query_start": ext["q_start"],
                "query_end": ext["q_end"],
                "reference_start": ext["r_start"],
                "reference_end": ext["r_end"],
                "score": ext["score"],
            }
        )

    uniq = {}
    for h in hsps:
        key = (h["query_start"], h["query_end"], h["reference_start"], h["reference_end"], h["mask"])
        if key not in uniq or h["score"] > uniq[key]["score"]:
            uniq[key] = h
    hsps = list(uniq.values())

    gapped = ales._gapped_extension(
        query,
        reference,
        hsps,
        match_score=args.match_score,
        mismatch_score=args.mismatch_score,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
        local_window=args.local_window,
        retire_distance=args.retire_distance,
        diag_band=args.diag_band,
        min_triplets=args.min_triplets,
    )
    if gapped is None:
        return None

    q_start, q_end = gapped["q_start"], gapped["q_end"]
    r_start, r_end = gapped["r_start"], gapped["r_end"]
    return {
        "mask": gapped.get("mask", "N/A"),
        "query_start": q_start,
        "query_end": q_end,
        "reference_start": r_start,
        "reference_end": r_end,
        "aligned_length": gapped["aligned_length"],
        "score": gapped.get("score", 0),
        "query_fragment": query[q_start : q_end + 1],
        "reference_fragment": reference[r_start : r_end + 1],
        "aligned_query": gapped.get("aligned_query", query[q_start : q_end + 1]),
        "aligned_reference": gapped.get("aligned_reference", reference[r_start : r_end + 1]),
    }


def choose_better(a, b):
    if a is None:
        return b
    if b is None:
        return a
    ka = (a["aligned_length"], a.get("score", 0))
    kb = (b["aligned_length"], b.get("score", 0))
    return b if kb > ka else a


def _process_single_read(rid, read_seq, references, masks, ales, args):
    best = None
    for ref_name, ref_seq, ref_idx in references:
        hit_fwd = align_one_strand(ales, read_seq, ref_seq, masks, ref_idx, args)
        if hit_fwd is not None:
            hit_fwd["ref_id"] = ref_name
            hit_fwd["strand"] = "+"
            best = choose_better(best, hit_fwd)

        if not args.no_revcomp:
            rc = reverse_complement(read_seq)
            hit_rev = align_one_strand(ales, rc, ref_seq, masks, ref_idx, args)
            if hit_rev is not None:
                orig_q_start = len(read_seq) - 1 - hit_rev["query_end"]
                orig_q_end = len(read_seq) - 1 - hit_rev["query_start"]
                hit_rev["query_start"] = orig_q_start
                hit_rev["query_end"] = orig_q_end
                hit_rev["query_fragment"] = reverse_complement(hit_rev["query_fragment"])
                hit_rev["aligned_query"] = reverse_complement_with_gaps(hit_rev.get("aligned_query", hit_rev["query_fragment"]))
                hit_rev["aligned_reference"] = hit_rev.get("aligned_reference", hit_rev["reference_fragment"])[::-1]
                hit_rev["ref_id"] = ref_name
                hit_rev["strand"] = "-"
                best = choose_better(best, hit_rev)

    if best is not None:
        row = (
            f"{rid}\t{len(read_seq)}\t{best['ref_id']}\t{best['strand']}\t{best['mask']}\t"
            f"{best['query_start']}\t{best['reference_start']}\t{best['aligned_length']}\t"
            f"{best.get('score', 0)}\t{best['query_fragment']}\t{best['reference_fragment']}\n"
        )
        return True, row

    return False, f"{rid}\t{len(read_seq)}\t*\t*\t*\t-1\t-1\t0\t0\t\t\n"


def _init_parallel_worker(args, masks, references):
    global _PARALLEL_CTX
    worker_ales = ALeSSpacedSeed(
        weight=args.weight,
        homology_length=args.homology_length,
        similarity=args.similarity,
        k_seeds=args.k_seeds,
        upper_bound=args.upper_bound,
        random_seed=args.random_seed,
        estimate_trials=args.estimate_trials,
    )
    _PARALLEL_CTX = {
        "args": args,
        "masks": masks,
        "references": references,
        "ales": worker_ales,
    }


def _map_read_worker(task):
    rid, read_seq = task
    ctx = _PARALLEL_CTX
    return _process_single_read(
        rid,
        read_seq,
        ctx["references"],
        ctx["masks"],
        ctx["ales"],
        ctx["args"],
    )


def map_reads(args):
    fasta_files = resolve_fasta_files(args.fasta)
    fastq_files = resolve_fastq_files(args.fastq)

    references = []
    for fp in fasta_files:
        recs = read_fasta(fp)
        if not recs:
            raise ValueError(f"FASTA 无有效序列: {fp}")
        references.extend(recs)

    ales = ALeSSpacedSeed(
        weight=args.weight,
        homology_length=args.homology_length,
        similarity=args.similarity,
        k_seeds=args.k_seeds,
        upper_bound=args.upper_bound,
        random_seed=args.random_seed,
        estimate_trials=args.estimate_trials,
    )

    if args.seeds:
        masks = args.seeds
        print(
            f"[ALeS] 使用外部指定 seeds={masks}; 跳过设计阶段",
            file=sys.stderr,
        )
    else:
        design = ales.design_seeds(tries=args.tries, indel_trials=args.indel_trials)
        masks = design["seeds"]
        print(
            f"[ALeS] w={ales.w}, k={ales.k}, p={ales.p}, N={ales.N}, m={design['m']}, M={design['M']}, seeds={masks}, sens={design['sensitivity']:.6f}",
            file=sys.stderr,
        )

    ref_indices = []
    for ref_name, ref_seq in references:
        idx = build_ref_index(ales, ref_seq, masks)
        ref_indices.append((ref_name, ref_seq, idx))

    workers = max(1, int(args.workers))
    chunk_size = max(1, int(args.task_chunk_size))
    use_parallel = workers > 1
    mode = f"parallel(workers={workers}, chunk={chunk_size})" if use_parallel else "sequential"
    print(f"[MODE] {mode}", file=sys.stderr)

    total = 0
    mapped = 0
    executor = None
    if use_parallel:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_parallel_worker,
            initargs=(args, masks, ref_indices),
        )

    with open(args.output, "w", encoding="utf-8", newline="") as out:
        out.write(
            "read_id\tread_len\tref_id\tstrand\tmask\tquery_start\treference_start\taligned_length\tscore\tquery_fragment\treference_fragment\n"
        )


        for fq in fastq_files:
            print(f"[FASTQ] processing {fq}", file=sys.stderr)
            if use_parallel:
                task_iter = ((rid, read_seq) for rid, read_seq, _qual in iter_fastq(fq, max_reads=args.max_reads))
                for is_mapped, row in executor.map(_map_read_worker, task_iter, chunksize=chunk_size):
                    total += 1
                    if is_mapped:
                        mapped += 1
                    out.write(row)
                    if total % args.progress_every == 0:
                        rate = (mapped / total) * 100.0 if total > 0 else 0.0
                        print(f"[PROGRESS] reads={total}, mapped={mapped}, rate={rate:.2f}%", file=sys.stderr)
            else:
                for rid, read_seq, _qual in iter_fastq(fq, max_reads=args.max_reads):
                    is_mapped, row = _process_single_read(rid, read_seq, ref_indices, masks, ales, args)
                    total += 1
                    if is_mapped:
                        mapped += 1
                    out.write(row)
                    if total % args.progress_every == 0:
                        rate = (mapped / total) * 100.0 if total > 0 else 0.0
                        print(f"[PROGRESS] reads={total}, mapped={mapped}, rate={rate:.2f}%", file=sys.stderr)

    if executor is not None:
        executor.shutdown(wait=True)

    rate = (mapped / total) * 100.0 if total > 0 else 0.0
    print(f"[DONE] reads={total}, mapped={mapped}, rate={rate:.2f}%", file=sys.stderr)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="使用 ALeSSpacedSeed 对 FASTQ reads 与 FASTA reference 进行比对")
    parser.add_argument("--fasta", required=True, help="reference FASTA 文件或目录")
    parser.add_argument("--fastq", nargs="+", required=True, help="FASTQ 文件或目录（可多个）")
    parser.add_argument("--output", required=True, help="输出 TSV 文件")

    parser.add_argument("--weight", type=int, default=11)
    parser.add_argument("--homology-length", type=int, default=64)
    parser.add_argument("--similarity", type=float, default=0.8)
    parser.add_argument("--k-seeds", type=int, default=1)
    parser.add_argument("--upper-bound", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--estimate-trials", type=int, default=100_000)
    parser.add_argument("--tries", type=int, default=40)
    parser.add_argument("--indel-trials", type=int, default=40)
    parser.add_argument("--seeds", nargs="+", default=None, help="直接指定 spaced-seed masks（如 111011 1110011），指定后跳过设计")

    parser.add_argument("--match-score", type=int, default=1)
    parser.add_argument("--mismatch-score", type=int, default=-1)
    parser.add_argument("--xdrop", type=int, default=5)
    parser.add_argument("--gap-open", type=int, default=-5)
    parser.add_argument("--gap-extend", type=int, default=-1)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--retire-distance", type=int, default=128)
    parser.add_argument("--diag-band", type=int, default=64)
    parser.add_argument("--min-triplets", type=int, default=3)
    parser.add_argument("--max-ref-hits-per-key", type=int, default=300, help="单个 seed key 在 reference 中允许的最大命中数；超过则视为低复杂度并跳过")
    parser.add_argument("--max-hit-candidates", type=int, default=8000, help="每条 read 保留的最大 seed 命中候选数")

    parser.add_argument("--max-reads", type=int, default=None, help="仅处理前 N 条 reads（调试用）")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-revcomp", action="store_true", help="关闭反向互补链比对")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数，1 为串行")
    parser.add_argument("--task-chunk-size", type=int, default=16, help="并行任务分块大小")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    map_reads(args)


if __name__ == "__main__":
    main()
