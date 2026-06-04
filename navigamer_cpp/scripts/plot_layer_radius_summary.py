#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / ".tmp_experiments" / "layer_sweep_5k_summary.csv"
FIG_DIR = ROOT / "navigamer_cpp" / "figures"


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "L": int(row["L"]),
                    "r_leaf": int(row["r_leaf"]),
                    "alpha": row["alpha"],
                    "radius_schedule": row["radius_schedule"],
                    "avg_query_time_ms": float(row["avg_query_time_ms"]),
                    "avg_world_access": float(row["avg_world_access"]),
                    "avg_edge_access": float(row["avg_edge_access"]),
                    "avg_anchor_distance": float(row["avg_anchor_distance"]),
                }
            )
    return rows


def grouped(rows, r_leaf):
    subset = [r for r in rows if r["r_leaf"] == r_leaf]
    subset.sort(key=lambda r: (r["alpha"], r["L"]))
    return subset


def plot_time(rows):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    alphas = ["0.500000", "0.700000"]
    labels = {"0.500000": "alpha=0.5", "0.700000": "alpha=0.7"}
    colors = {"0.500000": "#0f766e", "0.700000": "#b45309"}

    for ax, r_leaf in zip(axes, [4, 8, 12]):
        subset = grouped(rows, r_leaf)
        for alpha in alphas:
            alpha_rows = [r for r in subset if r["alpha"] == alpha]
            xs = [r["L"] for r in alpha_rows]
            ys = [r["avg_query_time_ms"] for r in alpha_rows]
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=colors[alpha], label=labels[alpha])
        ax.set_title(f"r_leaf = {r_leaf}")
        ax.set_xlabel("Primary Layers L")
        ax.grid(True, alpha=0.25)
        ax.set_xticks([2, 3, 4, 5])

    axes[0].set_ylabel("Average query time (ms)")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Query Time vs Primary-Layer Count", fontsize=14)
    fig.tight_layout()
    return fig


def plot_access(rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    metrics = [
        ("avg_world_access", "world_access", "#2563eb"),
        ("avg_edge_access", "edge_access", "#dc2626"),
        ("avg_anchor_distance", "anchor_distance", "#7c3aed"),
    ]

    for ax, r_leaf in zip(axes, [4, 8, 12]):
        subset = grouped(rows, r_leaf)
        alpha_rows = [r for r in subset if r["alpha"] == "0.500000"]
        xs = [r["L"] for r in alpha_rows]
        for key, label, color in metrics:
            ys = [r[key] for r in alpha_rows]
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=label)
        ax.set_title(f"r_leaf = {r_leaf} (alpha=0.5)")
        ax.set_xlabel("Primary Layers L")
        ax.grid(True, alpha=0.25)
        ax.set_xticks([2, 3, 4, 5])

    axes[0].set_ylabel("Average count per query")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("Access-Cost Breakdown vs Primary-Layer Count", fontsize=14)
    fig.tight_layout()
    return fig


def main():
    rows = load_rows(SUMMARY_CSV)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_time = plot_time(rows)
    fig_time.savefig(FIG_DIR / "layer_radius_query_time_vs_L.png", dpi=200, bbox_inches="tight")
    plt.close(fig_time)

    fig_access = plot_access(rows)
    fig_access.savefig(FIG_DIR / "layer_radius_access_breakdown_vs_L.png", dpi=200, bbox_inches="tight")
    plt.close(fig_access)

    print(FIG_DIR / "layer_radius_query_time_vs_L.png")
    print(FIG_DIR / "layer_radius_access_breakdown_vs_L.png")


if __name__ == "__main__":
    main()
