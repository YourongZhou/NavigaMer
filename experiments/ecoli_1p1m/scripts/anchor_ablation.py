#!/usr/bin/env python3

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import edlib


@dataclass(frozen=True)
class Window:
    window_id: int
    start: int
    sequence: str


@dataclass(frozen=True)
class EnvelopeMetrics:
    envelope_size: int
    true_neighbor_count: int
    false_negative_count: int
    source_recovered: bool


def edit_distance(lhs: str, rhs: str) -> int:
    return int(edlib.align(lhs, rhs, task="distance")["editDistance"])


def load_fasta_sequence(path: Path) -> str:
    parts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(">"):
            continue
        parts.append(line.strip().upper())
    if not parts:
        raise ValueError(f"empty FASTA: {path}")
    return "".join(parts)


def make_windows(reference: str, window_length: int, stride: int) -> List[Window]:
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    windows = []
    for start in range(0, len(reference) - window_length + 1, stride):
        windows.append(
            Window(len(windows), start, reference[start:start + window_length])
        )
    if not windows:
        raise ValueError("no reference windows generated")
    return windows


def mutate_substitutions(sequence: str, edits: int, rng: random.Random) -> str:
    bases = "ACGT"
    chars = list(sequence)
    positions = list(range(len(chars)))
    rng.shuffle(positions)
    for pos in positions[: min(edits, len(chars))]:
        old = chars[pos]
        choices = [base for base in bases if base != old]
        chars[pos] = rng.choice(choices)
    return "".join(chars)


def precompute_anchor_distances(windows: Sequence[Window]) -> List[List[int]]:
    matrix = [[0] * len(windows) for _ in windows]
    for i, lhs in enumerate(windows):
        for j in range(i + 1, len(windows)):
            dist = edit_distance(lhs.sequence, windows[j].sequence)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return matrix


def select_anchor_ids(
    strategy: str,
    query: str,
    source_idx: int,
    windows: Sequence[Window],
    anchor_distances: Sequence[Sequence[int]],
    anchor_count: int,
    rng: random.Random,
) -> List[int]:
    if anchor_count <= 0:
        return []
    candidates = [idx for idx in range(len(windows)) if idx != source_idx]
    if len(candidates) < anchor_count:
        raise ValueError("not enough non-source windows for anchors")
    if strategy == "random":
        return rng.sample(candidates, anchor_count)

    if strategy == "coordinate_proximal":
        source_start = windows[source_idx].start
        ranked = sorted(
            candidates,
            key=lambda idx: (abs(windows[idx].start - source_start), idx),
        )
        return ranked[:anchor_count]

    query_distances = [(edit_distance(query, windows[idx].sequence), idx)
                       for idx in candidates]
    if strategy == "proximal":
        query_distances.sort(key=lambda item: (item[0], item[1]))
        return [idx for _, idx in query_distances[:anchor_count]]
    if strategy == "far":
        query_distances.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in query_distances[:anchor_count]]
    if strategy == "actual":
        # Minimal proxy for NavigaMer's discovered nearby beacons: use the
        # nearest edit-distance anchors available around the source locus.
        local_radius = max(10, anchor_count * 4)
        local = [
            (edit_distance(query, windows[idx].sequence), idx)
            for idx in candidates
            if abs(idx - source_idx) <= local_radius
        ]
        local.sort(key=lambda item: (item[0], item[1]))
        selected = [idx for _, idx in local[:anchor_count]]
        if len(selected) < anchor_count:
            selected.extend(
                idx for _, idx in sorted(query_distances, key=lambda item: (item[0], item[1]))
                if idx not in selected
            )
        return selected[:anchor_count]
    raise ValueError(f"unknown anchor strategy: {strategy}")


def evaluate_anchor_set(
    query: str,
    source_idx: int,
    windows: Sequence[Window],
    anchor_distances: Sequence[Sequence[int]],
    anchor_ids: Sequence[int],
    tolerance: int,
) -> EnvelopeMetrics:
    query_anchor_distances = {
        anchor_id: edit_distance(query, windows[anchor_id].sequence)
        for anchor_id in anchor_ids
    }
    envelope = []
    true_neighbors = []
    for window in windows:
        passes = True
        for anchor_id in anchor_ids:
            if abs(query_anchor_distances[anchor_id] -
                   anchor_distances[window.window_id][anchor_id]) > tolerance:
                passes = False
                break
        if passes:
            envelope.append(window.window_id)
        if edit_distance(query, window.sequence) <= tolerance:
            true_neighbors.append(window.window_id)
    false_negatives = len(set(true_neighbors) - set(envelope))
    return EnvelopeMetrics(
        envelope_size=len(envelope),
        true_neighbor_count=len(true_neighbors),
        false_negative_count=false_negatives,
        source_recovered=source_idx in envelope,
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def run_experiment(
    reference_path: Path,
    out_path: Path,
    window_length: int,
    stride: int,
    query_count: int,
    query_edits: int,
    tolerance: int,
    anchor_counts: Sequence[int],
    strategies: Sequence[str],
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    reference = load_fasta_sequence(reference_path)
    windows = make_windows(reference, window_length, stride)
    anchor_distances = precompute_anchor_distances(windows)
    if query_count > len(windows):
        raise ValueError("query_count cannot exceed number of windows")
    source_ids = rng.sample(range(len(windows)), query_count)
    queries = [
        mutate_substitutions(windows[source_idx].sequence, query_edits, rng)
        for source_idx in source_ids
    ]

    rows = []
    for strategy in strategies:
        for anchor_count in anchor_counts:
            envelope_sizes = []
            true_counts = []
            false_negatives = 0
            source_recovered = 0
            for ordinal, source_idx in enumerate(source_ids):
                local_rng = random.Random(seed + 1009 * ordinal + 37 * anchor_count)
                anchor_ids = select_anchor_ids(
                    strategy, queries[ordinal], source_idx, windows,
                    anchor_distances, anchor_count, local_rng
                )
                metrics = evaluate_anchor_set(
                    queries[ordinal], source_idx, windows,
                    anchor_distances, anchor_ids, tolerance
                )
                envelope_sizes.append(metrics.envelope_size)
                true_counts.append(metrics.true_neighbor_count)
                false_negatives += metrics.false_negative_count
                source_recovered += 1 if metrics.source_recovered else 0
            window_count = len(windows)
            mean_envelope = mean([float(v) for v in envelope_sizes])
            rows.append({
                "strategy": strategy,
                "anchor_count": str(anchor_count),
                "query_count": str(query_count),
                "window_count": str(window_count),
                "mean_envelope_size": fmt(mean_envelope),
                "mean_exact_calls": fmt(mean_envelope),
                "mean_true_neighbor_count": fmt(mean([float(v) for v in true_counts])),
                "false_negative_count_total": str(false_negatives),
                "source_recovery_rate": fmt(source_recovered / query_count),
                "bound_check_pass_rate": fmt(mean_envelope / window_count),
                "pruning_ratio": fmt(1.0 - mean_envelope / window_count),
                "reference_path": str(reference_path),
                "window_length": str(window_length),
                "stride": str(stride),
                "query_edits": str(query_edits),
                "tolerance": str(tolerance),
                "seed": str(seed),
            })

    fields = [
        "strategy", "anchor_count", "query_count", "window_count",
        "mean_envelope_size", "mean_exact_calls", "mean_true_neighbor_count",
        "false_negative_count_total", "source_recovery_rate",
        "bound_check_pass_rate", "pruning_ratio", "reference_path",
        "window_length", "stride", "query_edits", "tolerance", "seed",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_csv_ints(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part]


def parse_csv_strings(value: str) -> List[str]:
    return [part for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anchor selection ablation using exact edit-distance envelopes."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=150)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--query-edits", type=int, default=3)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--anchor-counts", default="1,2,4,8")
    parser.add_argument(
        "--strategies",
        default="random,far,proximal,coordinate_proximal,actual",
    )
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    run_experiment(
        reference_path=args.reference,
        out_path=args.out,
        window_length=args.window_length,
        stride=args.stride,
        query_count=args.query_count,
        query_edits=args.query_edits,
        tolerance=args.tolerance,
        anchor_counts=parse_csv_ints(args.anchor_counts),
        strategies=parse_csv_strings(args.strategies),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
