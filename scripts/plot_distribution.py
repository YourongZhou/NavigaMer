#!/usr/bin/env python3
"""Plot the discrete PMF produced by dna_edit_distribution."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a random-DNA edit-distance probability mass function."
    )
    parser.add_argument(
        "--histogram",
        type=Path,
        required=True,
        help="path to histogram.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="directory for PNG and PDF outputs (default: results)",
    )
    return parser.parse_args()


def read_histogram(path):
    required_columns = {
        "edit_distance",
        "count",
        "probability",
        "cumulative_probability",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != required_columns:
            raise ValueError(
                "histogram must contain edit_distance,count,probability,"
                "cumulative_probability"
            )
        rows = list(reader)

    if not rows:
        raise ValueError("histogram contains no rows")
    distances = [int(row["edit_distance"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    cumulative = [float(row["cumulative_probability"]) for row in rows]

    if distances != list(range(len(distances))):
        raise ValueError("edit-distance bins must be consecutive and start at zero")
    if any(count < 0 for count in counts):
        raise ValueError("histogram counts must be nonnegative")
    num_pairs = sum(counts)
    if num_pairs <= 0:
        raise ValueError("histogram count sum must be positive")
    if any(not math.isfinite(probability) or probability < 0.0
           for probability in probabilities):
        raise ValueError("histogram probabilities must be finite and nonnegative")

    expected = [count / num_pairs for count in counts]
    if any(abs(observed - wanted) > 1e-12
           for observed, wanted in zip(probabilities, expected)):
        raise ValueError("histogram probabilities do not match counts")
    if abs(cumulative[-1] - 1.0) > 1e-12:
        raise ValueError("final cumulative probability is not one")
    return distances, probabilities, num_pairs


def main():
    args = parse_args()
    distances, probabilities, num_pairs = read_histogram(args.histogram)
    length = distances[-1]
    mean = sum(
        distance * probability
        for distance, probability in zip(distances, probabilities)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(
        distances,
        probabilities,
        width=0.85,
        color="#4C78A8",
        edgecolor="#2F4B66",
        linewidth=0.35,
        label="PMF",
    )
    axis.axvline(
        mean,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean:.2f}",
    )
    axis.set_xlabel("Edit distance")
    axis.set_ylabel("Probability")
    axis.set_title(
        f"Random DNA edit-distance distribution (L={length}, pairs={num_pairs:,})"
    )
    axis.set_xlim(-0.75, length + 0.75)
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()

    figure.savefig(
        args.output_dir / "edit_distance_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "edit_distance_distribution.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
