#!/usr/bin/env python3
"""Plot the current geom benchmark evidence against planned Results 1-6."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_RUN_DIR = Path(
    "/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/"
    "geom_final_benchmark_contained_reuse_20260703"
)

PALETTE = {
    "nav": "#4C78A8",
    "pre": "#9E9E9E",
    "rand": "#72B7B2",
    "spaced": "#F58518",
    "load": "#D9D9D9",
    "hot": "#4C78A8",
    "good": "#7CBF8A",
    "partial": "#F2C078",
    "missing": "#D6D6D6",
    "warn": "#D95F5F",
    "text": "#222222",
}


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def first(rows: Sequence[Dict[str, str]], **conds: str) -> Dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in conds.items()):
            return row
    raise KeyError(f"no row matched {conds}")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def annotate_bar(ax, x: float, y: float, text: str, *, dy: float = 0.04, size: int = 6) -> None:
    ax.text(x, y * (1 + dy), text, ha="center", va="bottom", fontsize=size)


def build_mapping_rows() -> List[Dict[str, str]]:
    return [
        {
            "result": "Result 1",
            "claim": "Exact candidate retrieval / no false negatives",
            "figure_panel": "A, B",
            "status": "supported_current",
            "current_evidence": "All current geom and exact-verified baseline rows report fn_count=0; C++ recall and distance-bound guards passed.",
            "gap": "Broader threshold/dataset sweep can strengthen the final paper table.",
        },
        {
            "result": "Result 2",
            "claim": "Proximal anchors tighten multilateration envelopes",
            "figure_panel": "A only",
            "status": "missing",
            "current_evidence": "No anchor strategy ablation TSV was found in the current geom final run.",
            "gap": "Run random/far/proximal/actual anchor sweep with envelope size and exact-call counts.",
        },
        {
            "result": "Result 3",
            "claim": "Hierarchical worlds reduce search cost while preserving exactness",
            "figure_panel": "A, C",
            "status": "partial_current",
            "current_evidence": "One L4 geom configuration shows stable source-sorted q64/q128/q256 hot-query scaling with fn_count=0.",
            "gap": "Needs flat/1/2/3/4-layer ablation or uniform hierarchy sweep for the final hierarchy claim.",
        },
        {
            "result": "Result 4",
            "claim": "Contained/overlap/corner paths remain recall safe",
            "figure_panel": "A, E",
            "status": "partial_current",
            "current_evidence": "Contained reuse and boundary workloads completed with fn_count=0; random_windows is explicitly a negative-control row.",
            "gap": "Needs deterministic contained/overlap/uncovered class counts and fallback overhead.",
        },
        {
            "result": "Result 5",
            "claim": "Compact, high-quality error-tolerant candidates vs seed baselines",
            "figure_panel": "A, D, E",
            "status": "preliminary_supported",
            "current_evidence": "q64 source-sorted comparison includes randstrobe and spaced seed candidate retrieval plus exact verify, both with fn_count=0.",
            "gap": "Needs q-gram/pigeonhole and broader error/repeat sweeps for a final candidate-quality claim.",
        },
        {
            "result": "Result 6",
            "claim": "System locality and throughput benefit under cache/reuse",
            "figure_panel": "A, C, D, F",
            "status": "supported_current_boundary",
            "current_evidence": "Hot-query speedups are separated from cold load; q64/q128/q256 source-sorted rows and peak RSS are recorded.",
            "gap": "Needs perf counters such as LLC misses and bytes/query for the memory-bound mechanism claim.",
        },
    ]


def plot_result_tiles(ax, mapping_rows: Sequence[Dict[str, str]]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    status_color = {
        "supported_current": PALETTE["good"],
        "supported_current_boundary": PALETTE["good"],
        "preliminary_supported": PALETTE["partial"],
        "partial_current": PALETTE["partial"],
        "missing": PALETTE["missing"],
    }
    for idx, row in enumerate(mapping_rows):
        color = status_color[row["status"]]
        ax.add_patch(Rectangle((idx + 0.04, 0.08), 0.92, 0.78, facecolor=color, edgecolor="white", lw=1.5))
        result_no = row["result"].split()[-1]
        status = row["status"].replace("_current", "").replace("_", "\n")
        ax.text(idx + 0.5, 0.62, f"R{result_no}", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(idx + 0.5, 0.34, status, ha="center", va="center", fontsize=6.5, linespacing=1.0)
    ax.text(0, 0.98, "Current evidence mapped to planned Results 1-6", fontsize=8, fontweight="bold", va="top")


def plot_correctness(ax, summary_rows: Sequence[Dict[str, str]]) -> None:
    rows = [row for row in summary_rows if row["result_kind"] in {"navigamer", "baseline"}]
    labels = []
    fn = []
    mismatch = []
    for row in rows:
        method = row["method"]
        if method == "geom_L4_leaf5_a0p5_contained_reuse":
            method = "NavigaMer"
        label = f"{method}\n{row['dataset']} q{row['query_count']}"
        labels.append(label)
        fn.append(f(row, "fn_count"))
        mismatch.append(f(row, "mismatch_count"))
    y = list(range(len(labels)))
    ax.scatter([0] * len(y), y, s=82, color=PALETTE["good"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter([1] * len(y), y, s=82, color=PALETTE["good"], edgecolor="white", linewidth=0.8, zorder=3)
    for idx, (fn_v, mm_v) in enumerate(zip(fn, mismatch)):
        ax.text(0, idx, f"{int(fn_v)}", va="center", ha="center", fontsize=6, color="white", fontweight="bold")
        ax.text(1, idx, f"{int(mm_v)}", va="center", ha="center", fontsize=6, color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FN", "Mismatch"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("Reported count")
    ax.set_title("Result 1: zero FN / mismatch across current rows", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="y", color="#F1F1F1", lw=0.6)


def plot_source_scaling(ax, raw_rows: Sequence[Dict[str, str]]) -> None:
    rows = sorted(raw_rows, key=lambda row: int(row["query_count"]))
    q = [int(row["query_count"]) for row in rows]
    mean = [f(row, "mean_query_ms") for row in rows]
    p95 = [f(row, "p95_query_ms") for row in rows]
    ax.plot(q, mean, marker="o", color=PALETTE["nav"], lw=1.8, label="Mean")
    ax.plot(q, p95, marker="s", color="#555555", lw=1.4, label="P95")
    for x, yv in zip(q, mean):
        ax.text(x, yv + 8, f"{yv:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(q)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylabel("Latency per query (ms)")
    ax.set_xlabel("Number of source-sorted queries")
    ax.set_title("Results 3/6: source-sorted scaling is stable", loc="left", fontsize=8, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)


def plot_baseline_comparison(ax, comparison_rows: Sequence[Dict[str, str]]) -> None:
    order = [
        ("pre_contained_geom_q64_hot", "Pre-contained\ngeom"),
        ("randstrobe_candidate_plus_exact_verify", "Randstrobe\n+ exact verify"),
        ("spaced_seed_candidate_plus_exact_verify", "Spaced seed\n+ exact verify"),
    ]
    current = f(comparison_rows[0], "current_navigamer_hot_ms")
    labels = ["NavigaMer\ncontained reuse"]
    values = [current / 1000.0]
    colors = [PALETTE["nav"]]
    speedups = [""]
    for key, label in order:
        row = first(comparison_rows, comparison=key)
        labels.append(label)
        values.append(f(row, "baseline_ms") / 1000.0)
        colors.append(PALETTE["pre"] if "pre" in key else (PALETTE["rand"] if "rand" in key else PALETTE["spaced"]))
        speedups.append(f"{f(row, 'speedup_x'):.2f}x")
    x = list(range(len(values)))
    ax.bar(x, values, color=colors, width=0.68)
    for idx, value in enumerate(values):
        annotate_bar(ax, idx, value, f"{value:.2f}s\n{speedups[idx]}".strip(), dy=0.08, size=6)
    ax.set_yscale("log")
    ax.set_ylabel("q64 hot / query+verify time (s, log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title("Results 5/6: q64 comparison, cold load excluded", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6, which="both")


def plot_workload_boundary(ax, raw_rows: Sequence[Dict[str, str]]) -> None:
    order = [
        ("source_sorted_stride1", "source-sorted\nstride1"),
        ("source_sorted_mutated_tau5", "mutated\ntau5"),
        ("real_dup_4x", "real dup\n4x"),
        ("random_windows", "random\nwindows"),
    ]
    rows_by_dataset = {row["dataset"]: row for row in raw_rows}
    values = [f(rows_by_dataset[key], "mean_query_ms") for key, _ in order]
    colors = [PALETTE["nav"], "#7BAFDE", "#A0CBE8", PALETTE["warn"]]
    x = list(range(len(order)))
    ax.bar(x, values, color=colors, width=0.68)
    for idx, value in enumerate(values):
        annotate_bar(ax, idx, value, f"{value:.1f} ms", dy=0.1, size=6)
    ax.set_yscale("log")
    ax.set_ylabel("Mean query latency (ms, log)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in order], fontsize=6)
    ax.set_title("Results 4/5: workload boundary, not a random-query claim", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6, which="both")
    dup = rows_by_dataset["real_dup_4x"].get("verified_result_cache_hit_count", "192")
    ax.text(2, values[2] * 0.42, f"cache hits={dup}", ha="center", va="top", fontsize=6, color="#333333")


def plot_cold_hot(ax, nav_rows: Sequence[Dict[str, str]]) -> None:
    selected = [
        ("source_sorted_stride1", "64", "stride1\nq64"),
        ("source_sorted_stride1", "128", "stride1\nq128"),
        ("source_sorted_stride1", "256", "stride1\nq256"),
        ("source_sorted_mutated_tau5", "64", "tau5\nq64"),
        ("random_windows", "64", "random\nq64"),
    ]
    rows = [first(nav_rows, dataset=dataset, query_count=q) for dataset, q, _ in selected]
    labels = [label for _, _, label in selected]
    load_s = [f(row, "cold_load_ms") / 1000.0 for row in rows]
    hot_s = [f(row, "hot_query_ms") / 1000.0 for row in rows]
    x = list(range(len(rows)))
    ax.bar(x, load_s, color=PALETTE["load"], width=0.68, label="Cold index load")
    ax.bar(x, hot_s, bottom=load_s, color=PALETTE["hot"], width=0.68, label="Hot query")
    for idx, row in enumerate(rows):
        rss_gb = f(row, "max_rss_kb") / (1024 * 1024)
        ax.text(idx, load_s[idx] + hot_s[idx] + 10, f"{rss_gb:.1f} GB", ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("One-shot process time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title("Result 6: one-shot runs are load dominated", loc="left", fontsize=8, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6)
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)


def make_figure(run_dir: Path, out_prefix: Path) -> None:
    setup_style()
    summary = read_tsv(run_dir / "geom_final_summary.tsv")
    comparison = read_tsv(run_dir / "geom_final_comparison.tsv")
    q64 = read_tsv(run_dir / "geom_source_sorted_stride1_q64.tsv")[0]
    q128 = read_tsv(run_dir / "geom_source_sorted_stride1_q128.tsv")[0]
    q256 = read_tsv(run_dir / "geom_source_sorted_stride1_q256.tsv")[0]
    optional = read_tsv(run_dir / "geom_q64_optional_tau5_realdup_random.tsv")
    source_rows = [q64, q128, q256]
    workload_rows = [q64] + optional
    nav_summary = [row for row in summary if row["result_kind"] == "navigamer"]
    mapping_rows = build_mapping_rows()

    write_tsv(
        out_prefix.with_name(out_prefix.name + "_result_mapping.tsv"),
        ["result", "claim", "figure_panel", "status", "current_evidence", "gap"],
        mapping_rows,
    )

    fig = plt.figure(figsize=(10.8, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[0.7, 1.15, 1.15, 1.15])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])
    ax_f = fig.add_subplot(grid[3, :])

    plot_result_tiles(ax_a, mapping_rows)
    plot_correctness(ax_b, summary)
    plot_baseline_comparison(ax_c, comparison)
    plot_source_scaling(ax_d, source_rows)
    plot_workload_boundary(ax_e, workload_rows)
    plot_cold_hot(ax_f, nav_summary)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list("ABCDEF")):
        panel_label(ax, label)

    fig.suptitle(
        "Current NavigaMer geom benchmark evidence mapped to Results 1-6",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.02,
        -0.02,
        "Exact edit-distance verification is integrated for NavigaMer rows; randstrobe/spaced-seed bars include exact verifier wall time. "
        "Cold load is shown separately and is not counted in hot-query speedups.",
        fontsize=6.5,
        ha="left",
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=DEFAULT_RUN_DIR / "figures" / "geom_final_result_1to6_map",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(args.run_dir, args.out_prefix)
    print(args.out_prefix.with_suffix(".svg"))
    print(args.out_prefix.with_suffix(".pdf"))
    print(args.out_prefix.with_suffix(".png"))
    print(args.out_prefix.with_suffix(".tiff"))
    print(args.out_prefix.with_name(args.out_prefix.name + "_result_mapping.tsv"))


if __name__ == "__main__":
    main()
