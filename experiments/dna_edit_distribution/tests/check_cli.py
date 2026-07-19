#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--plot-script", type=Path, required=True)
    return parser.parse_args()


def run_program(binary, output_dir, threads):
    subprocess.run(
        [
            str(binary),
            "--length",
            "12",
            "--pairs",
            "64",
            "--seed",
            "20260719",
            "--threads",
            str(threads),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )


def read_counts(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert list(rows[0]) == [
        "edit_distance",
        "count",
        "probability",
        "cumulative_probability",
    ]
    distances = [int(row["edit_distance"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    cumulative = [float(row["cumulative_probability"]) for row in rows]
    assert min(distances) == 0
    assert max(distances) == 12
    assert sum(counts) == 64
    assert all(0 <= distance <= 12 for distance in distances)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
    assert math.isclose(cumulative[-1], 1.0, abs_tol=1e-12)
    for count, probability in zip(counts, probabilities):
        assert math.isclose(probability, count / 64, abs_tol=1e-12)
    return counts


def nearest_rank(counts, probability):
    target = math.ceil(probability * sum(counts))
    cumulative = 0
    for distance, count in enumerate(counts):
        cumulative += count
        if cumulative >= target:
            return distance
    raise AssertionError("quantile target exceeds histogram")


def read_summary(path, counts):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["metric", "value"]
    summary = {row["metric"]: row["value"] for row in rows}
    required = {
        "length",
        "num_pairs",
        "seed",
        "mean",
        "standard_deviation",
        "min",
        "median",
        "max",
        "mode",
        "q05",
        "q95",
        "elapsed_seconds",
        "pairs_per_second",
    }
    assert required <= summary.keys()
    assert summary["length"] == "12"
    assert summary["num_pairs"] == "64"
    total = sum(counts)
    mean = sum(distance * count for distance, count in enumerate(counts)) / total
    variance = (
        sum((distance - mean) ** 2 * count for distance, count in enumerate(counts))
        / total
    )
    nonzero = [distance for distance, count in enumerate(counts) if count]
    assert math.isclose(float(summary["mean"]), mean, abs_tol=1e-12)
    assert math.isclose(
        float(summary["standard_deviation"]), math.sqrt(variance), abs_tol=1e-12
    )
    assert int(summary["min"]) == min(nonzero)
    assert int(summary["median"]) == nearest_rank(counts, 0.50)
    assert int(summary["max"]) == max(nonzero)
    assert int(summary["mode"]) == max(range(len(counts)), key=counts.__getitem__)
    assert int(summary["q05"]) == nearest_rank(counts, 0.05)
    assert int(summary["q95"]) == nearest_rank(counts, 0.95)
    assert float(summary["elapsed_seconds"]) > 0
    assert float(summary["pairs_per_second"]) > 0


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="dna-edit-distribution-test-") as tmp:
        root = Path(tmp)
        parallel = root / "parallel"
        serial = root / "serial"
        run_program(args.binary, parallel, 4)
        run_program(args.binary, serial, 1)

        parallel_counts = read_counts(parallel / "histogram.csv")
        serial_counts = read_counts(serial / "histogram.csv")
        assert parallel_counts == serial_counts
        read_summary(parallel / "summary.csv", parallel_counts)

        with (parallel / "run_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["length"] == 12
        assert metadata["num_pairs"] == 64
        assert metadata["seed"] == 20260719
        assert metadata["requested_threads"] == 4
        assert metadata["actual_threads"] >= 1
        assert metadata["wfa2_lib_version"] == "2.3.6"
        assert metadata["wfa2_lib_commit"] == (
            "0db345a8fe862fd7873d3354c499da385583a65a"
        )
        assert metadata["distance_metric"] == "exact global Levenshtein"
        assert metadata["alignment_scope"] == "score_only"
        assert metadata["heuristic"] == "none"
        assert metadata["compiler"]
        assert metadata["started_at_utc"]
        assert metadata["finished_at_utc"]
        assert metadata["elapsed_seconds"] > 0
        assert metadata["pairs_per_second"] > 0

        subprocess.run(
            [
                sys.executable,
                str(args.plot_script),
                "--histogram",
                str(parallel / "histogram.csv"),
                "--output-dir",
                str(parallel),
            ],
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        assert (parallel / "edit_distance_distribution.png").stat().st_size > 0
        assert (parallel / "edit_distance_distribution.pdf").stat().st_size > 0


if __name__ == "__main__":
    main()
