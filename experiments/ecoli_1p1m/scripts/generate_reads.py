#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple


DNA_BASES = "ACGT"
MAIN_TAUS = (1, 2, 3, 5)
HARD_TAUS = (2, 5)
ORACLE_PREFIXES = (50_000, 100_000)
READ_LENGTH = 150


def load_single_fasta(path: Path) -> Tuple[str, str]:
    header = None
    parts: List[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is None:
                header = line[1:].split()[0]
            continue
        parts.append(line.upper())
    if header is None:
        raise ValueError(f"missing FASTA header: {path}")
    sequence = "".join(parts)
    if not sequence:
        raise ValueError(f"empty FASTA sequence: {path}")
    invalid = sorted(set(sequence) - set(DNA_BASES))
    if invalid:
        raise ValueError(f"reference contains non-ACGT bases: {''.join(invalid)}")
    return header, sequence


def mutate_substitutions(sequence: str, edit_count: int,
                         rng: random.Random) -> Tuple[str, List[dict]]:
    if edit_count == 0:
        return sequence, []
    positions = rng.sample(range(len(sequence)), edit_count)
    bases = list(sequence)
    script = []
    for position in sorted(positions):
        original = bases[position]
        alternatives = [base for base in DNA_BASES if base != original]
        replacement = rng.choice(alternatives)
        bases[position] = replacement
        script.append({
            "op": "sub",
            "position": position,
            "from": original,
            "to": replacement,
        })
    return "".join(bases), script


def mutate_mixed(sequence: str, edit_count: int,
                 rng: random.Random) -> Tuple[str, List[dict]]:
    if edit_count == 0:
        return sequence, []
    current = list(sequence)
    script = []
    for _ in range(edit_count):
        allowed_ops = ["sub", "ins"]
        if len(current) > 1:
            allowed_ops.append("del")
        op = rng.choice(allowed_ops)
        if op == "sub":
            position = rng.randrange(len(current))
            original = current[position]
            replacement = rng.choice([base for base in DNA_BASES if base != original])
            current[position] = replacement
            script.append({
                "op": "sub",
                "position": position,
                "from": original,
                "to": replacement,
            })
        elif op == "ins":
            position = rng.randrange(len(current) + 1)
            base = rng.choice(DNA_BASES)
            current.insert(position, base)
            script.append({
                "op": "ins",
                "position": position,
                "base": base,
            })
        else:
            position = rng.randrange(len(current))
            original = current.pop(position)
            script.append({
                "op": "del",
                "position": position,
                "base": original,
            })
    return "".join(current), script


def global_kmer_frequencies(sequence: str, k: int) -> List[int]:
    counts = [0] * (4 ** k)
    code = 0
    mask = (1 << (2 * k)) - 1
    if len(sequence) < k:
        return counts
    for index, base in enumerate(sequence):
        code = ((code << 2) | "ACGT".index(base)) & mask
        if index + 1 >= k:
            counts[code] += 1
    return counts


def encoded_kmer_stream(sequence: str, k: int) -> List[int]:
    if len(sequence) < k:
        return []
    codes: List[int] = []
    code = 0
    mask = (1 << (2 * k)) - 1
    for index, base in enumerate(sequence):
        code = ((code << 2) | "ACGT".index(base)) & mask
        if index + 1 >= k:
            codes.append(code)
    return codes


def hard_window_starts(sequence: str, read_length: int, limit: int) -> List[int]:
    k = 5
    frequencies = global_kmer_frequencies(sequence, k)
    kmer_codes = encoded_kmer_stream(sequence, k)
    scored = []
    max_start = len(sequence) - read_length
    if max_start < 0:
        return []
    window_kmer_count = read_length - k + 1
    if window_kmer_count <= 0:
        raise ValueError("read length shorter than k-mer length")

    current_score = sum(frequencies[code] for code in kmer_codes[:window_kmer_count])
    counts = {}
    unique_count = 0
    for code in kmer_codes[:window_kmer_count]:
        count = counts.get(code, 0) + 1
        counts[code] = count
        if count == 1:
            unique_count += 1
    scored.append((-current_score, unique_count, 0))

    for start in range(1, max_start + 1):
        outgoing = kmer_codes[start - 1]
        incoming = kmer_codes[start + window_kmer_count - 1]
        current_score += frequencies[incoming] - frequencies[outgoing]

        remaining = counts[outgoing] - 1
        if remaining == 0:
            del counts[outgoing]
            unique_count -= 1
        else:
            counts[outgoing] = remaining

        count = counts.get(incoming, 0) + 1
        counts[incoming] = count
        if count == 1:
            unique_count += 1
        scored.append((-current_score, unique_count, start))

    scored.sort()
    return [start for _, _, start in scored[:limit]]


def sampled_starts(max_start: int, count: int, seed: int) -> List[int]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    return [rng.randrange(max_start + 1) for _ in range(count)]


def write_fastq(path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for read_id, sequence in rows:
            handle.write(f"@{read_id}\n{sequence}\n+\n{'I' * len(sequence)}\n")


def write_truth(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "read_id\tfamily\ttau\tsource_start\tsource_end\tseed\tsequence_length\tedit_script_json\n"
        )
        for row in rows:
            handle.write(
                f"{row['read_id']}\t{row['family']}\t{row['tau']}\t"
                f"{row['source_start']}\t{row['source_end']}\t{row['seed']}\t"
                f"{row['sequence_length']}\t"
                f"{json.dumps(row['edit_script'], separators=(',', ':'))}\n"
            )


def build_dataset(sequence: str, family: str, tau: int, count: int,
                  seed: int, hard_starts: Sequence[int] = ()) -> Tuple[List[Tuple[str, str]], List[dict]]:
    max_start = len(sequence) - READ_LENGTH
    if max_start < 0:
        raise ValueError("reference shorter than read length")
    starts = list(hard_starts) if hard_starts else sampled_starts(max_start, count, seed)
    if hard_starts and len(starts) < count:
        raise ValueError("not enough hard starts to satisfy requested count")
    starts = starts[:count]
    reads = []
    truth = []
    for index, start in enumerate(starts):
        source = sequence[start:start + READ_LENGTH]
        row_seed = seed * 1_000_003 + index
        rng = random.Random(row_seed)
        if family == "exact":
            mutated, script = source, []
        elif family == "substitution":
            mutated, script = mutate_substitutions(source, tau, rng)
        else:
            mutated, script = mutate_mixed(source, tau, rng)
        read_id = f"{family}_tau{tau}_read{index:05d}"
        reads.append((read_id, mutated))
        truth.append({
            "read_id": read_id,
            "family": family,
            "tau": tau,
            "source_start": start,
            "source_end": start + READ_LENGTH,
            "seed": row_seed,
            "sequence_length": len(mutated),
            "edit_script": script,
        })
    return reads, truth


def mixed_op_coverage(truth_rows: Sequence[dict]) -> set:
    coverage = set()
    for row in truth_rows:
        for op in row["edit_script"]:
            coverage.add(op["op"])
    return coverage


def generate_all(reference_path: Path, output_dir: Path, seed: int) -> None:
    _, reference = load_single_fasta(reference_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    hard_starts = hard_window_starts(reference, READ_LENGTH, 5_000)

    datasets = []
    datasets.append(("main_exact_10000", "exact", 0, 10_000, reference, seed + 11, ()))
    for tau in MAIN_TAUS:
        datasets.append((f"main_substitution_tau{tau}_10000", "substitution", tau, 10_000, reference, seed + 100 + tau, ()))
        datasets.append((f"main_mixed_tau{tau}_10000", "mixed", tau, 10_000, reference, seed + 200 + tau, ()))
    for tau in HARD_TAUS:
        datasets.append((f"main_hard_tau{tau}_5000", "mixed", tau, 5_000, reference, seed + 300 + tau, hard_starts))

    for prefix in ORACLE_PREFIXES:
        prefix_sequence = reference[:prefix]
        for tau in MAIN_TAUS:
            datasets.append((f"oracle_prefix{prefix}_mixed_tau{tau}_1000", "mixed", tau, 1_000, prefix_sequence, seed + prefix + tau, ()))

    manifest_rows = []
    for dataset_name, family, tau, count, sequence, row_seed, starts in datasets:
        reads, truth = build_dataset(sequence, family, tau, count, row_seed, starts)
        if family == "mixed":
            coverage = mixed_op_coverage(truth)
            required = {"sub", "ins", "del"}
            if coverage != required:
                missing = ",".join(sorted(required - coverage))
                raise RuntimeError(
                    f"mixed dataset {dataset_name} missing edit operations: {missing}"
                )
        fastq_path = output_dir / f"{dataset_name}.fq"
        truth_path = output_dir / f"{dataset_name}.truth.tsv"
        write_fastq(fastq_path, reads)
        write_truth(truth_path, truth)
        manifest_rows.append({
            "dataset": dataset_name,
            "family": family,
            "tau": tau,
            "count": count,
            "fastq": fastq_path.name,
            "truth": truth_path.name,
            "seed": row_seed,
        })

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "reference_path": str(reference_path),
            "read_length": READ_LENGTH,
            "seed": seed,
            "datasets": manifest_rows,
        }, handle, indent=2)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args(argv)
    generate_all(args.ref, args.out_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
