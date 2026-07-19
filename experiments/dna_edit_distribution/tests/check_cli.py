#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
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
    assert min(distances) == 0
    assert max(distances) == 12
    assert sum(counts) == 64
    assert all(0 <= distance <= 12 for distance in distances)
    return counts


def read_summary(path):
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
        read_summary(parallel / "summary.csv")

        with (parallel / "run_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["length"] == 12
        assert metadata["num_pairs"] == 64
        assert metadata["seed"] == 20260719
        assert metadata["requested_threads"] == 4
        assert metadata["actual_threads"] >= 1
        assert metadata["wfa2_lib_version"] == "2.3.6"


if __name__ == "__main__":
    main()
