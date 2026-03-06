#!/usr/bin/env python3
"""从 benchmark_results.json 重新画图（无需重跑算法）"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_output"
RESULTS_JSON = OUTPUT_DIR / "benchmark_results.json"

MUTATION_RATES = [0.01, 0.05, 0.10, 0.15]

STROBEMER_CFGS = ["k=20", "k=15 (default)", "k=10"]
SPACED_SEED_CFGS = ["w=14", "w=11 (default)", "w=8"]
TENSOR_SKETCH_CFGS = ["ef=20", "ef=50 (default)", "ef=200"]

OLD_TO_NEW = {
    "k=15 (默认)": "k=15 (default)",
    "w=11 (默认)": "w=11 (default)",
    "ef=50 (默认)": "ef=50 (default)",
}


def load_data():
    with open(RESULTS_JSON) as f:
        raw = json.load(f)

    data = {"strobemer": {}, "spaced_seed": {}, "tensor_sketch": {}}
    for method, entries in raw.items():
        for compound_key, val in entries.items():
            cfg_name, mr_str = compound_key.rsplit("|", 1)
            cfg_name = OLD_TO_NEW.get(cfg_name, cfg_name)
            mr = float(mr_str)
            data[method][(cfg_name, mr)] = val
    return data


def plot_all(data):
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'legend.fontsize': 9, 'figure.dpi': 150,
    })

    mutation_pcts = [int(mr * 100) for mr in MUTATION_RATES]
    colors = plt.cm.Set2(np.linspace(0, 0.8, 3))
    markers = ['o', 's', '^']

    method_info = [
        ("strobemer", STROBEMER_CFGS, "Strobemer (vary k)"),
        ("spaced_seed", SPACED_SEED_CFGS, "Spaced Seed (vary weight)"),
        ("tensor_sketch", TENSOR_SKETCH_CFGS, "Tensor Sketch (vary ef_search)"),
    ]

    # ── Figure 1: per-method subplots ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax_idx, (method, cfgs, title) in enumerate(method_info):
        ax = axes[ax_idx]
        for ci, cfg in enumerate(cfgs):
            recalls = [data[method].get((cfg, mr), {}).get("recall", 0.0) for mr in MUTATION_RATES]
            ax.plot(mutation_pcts, recalls, marker=markers[ci], color=colors[ci],
                    linewidth=2.2, markersize=7, label=cfg)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Mutation Rate (%)")
        if ax_idx == 0:
            ax.set_ylabel("Recall")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks(mutation_pcts)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', framealpha=0.9)

    fig.suptitle("Error Tolerance: Recall vs Mutation Rate (per method)", fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = OUTPUT_DIR / "recall_by_method.png"
    fig.savefig(p1, bbox_inches='tight')
    plt.close(fig)
    print(f"  [1] {p1}")

    # ── Figure 2: default configs head-to-head ──
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    defaults = [
        ("Strobemer k=15", "strobemer", "k=15 (default)"),
        ("SpacedSeed w=11", "spaced_seed", "w=11 (default)"),
        ("TensorSketch ef=50", "tensor_sketch", "ef=50 (default)"),
    ]
    c2 = plt.cm.tab10([0, 1, 2])
    for i, (label, method, cfg) in enumerate(defaults):
        recalls = [data[method].get((cfg, mr), {}).get("recall", 0.0) for mr in MUTATION_RATES]
        ax2.plot(mutation_pcts, recalls, marker=markers[i], color=c2[i],
                 linewidth=2.5, markersize=8, label=label)
    ax2.set_title("Default Config Comparison", fontweight='bold', fontsize=14)
    ax2.set_xlabel("Mutation Rate (%)")
    ax2.set_ylabel("Recall")
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xticks(mutation_pcts)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    plt.tight_layout()
    p2 = OUTPUT_DIR / "recall_default_comparison.png"
    fig2.savefig(p2, bbox_inches='tight')
    plt.close(fig2)
    print(f"  [2] {p2}")

    # ── Figure 3: best tolerance configs head-to-head ──
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    bests = [
        ("Strobemer k=10", "strobemer", "k=10"),
        ("SpacedSeed w=8", "spaced_seed", "w=8"),
        ("TensorSketch ef=200", "tensor_sketch", "ef=200"),
    ]
    c3 = plt.cm.Dark2([0, 1, 2])
    for i, (label, method, cfg) in enumerate(bests):
        recalls = [data[method].get((cfg, mr), {}).get("recall", 0.0) for mr in MUTATION_RATES]
        ax3.plot(mutation_pcts, recalls, marker=markers[i], color=c3[i],
                 linewidth=2.5, markersize=8, label=label)
    ax3.set_title("Best Tolerance Config Comparison", fontweight='bold', fontsize=14)
    ax3.set_xlabel("Mutation Rate (%)")
    ax3.set_ylabel("Recall")
    ax3.set_ylim(-0.05, 1.08)
    ax3.set_xticks(mutation_pcts)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    plt.tight_layout()
    p3 = OUTPUT_DIR / "recall_best_comparison.png"
    fig3.savefig(p3, bbox_inches='tight')
    plt.close(fig3)
    print(f"  [3] {p3}")

    # ── Figure 4: runtime bar chart ──
    fig4, ax4 = plt.subplots(figsize=(10, 5.5))
    all_cfgs = [
        ("Strobe k=20", "strobemer", "k=20"),
        ("Strobe k=15", "strobemer", "k=15 (default)"),
        ("Strobe k=10", "strobemer", "k=10"),
        ("SS w=14", "spaced_seed", "w=14"),
        ("SS w=11", "spaced_seed", "w=11 (default)"),
        ("SS w=8", "spaced_seed", "w=8"),
        ("TS ef=20", "tensor_sketch", "ef=20"),
        ("TS ef=50", "tensor_sketch", "ef=50 (default)"),
        ("TS ef=200", "tensor_sketch", "ef=200"),
    ]
    avg_times = []
    labels = []
    bar_colors = []
    method_color = {"strobemer": "#4C72B0", "spaced_seed": "#DD8452", "tensor_sketch": "#55A868"}
    for label, method, cfg in all_cfgs:
        times = [data[method].get((cfg, mr), {}).get("time", 0.0) for mr in MUTATION_RATES]
        avg_times.append(np.mean(times))
        labels.append(label)
        bar_colors.append(method_color[method])

    x = np.arange(len(labels))
    bars = ax4.bar(x, avg_times, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax4.set_ylabel("Avg Time (seconds)")
    ax4.set_title("Average Runtime per Config", fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.2, axis='y')

    for bar, t in zip(bars, avg_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{t:.1f}s", ha='center', va='bottom', fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Strobemer"),
        Patch(facecolor="#DD8452", label="Spaced Seed"),
        Patch(facecolor="#55A868", label="Tensor Sketch"),
    ]
    ax4.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    p4 = OUTPUT_DIR / "runtime_comparison.png"
    fig4.savefig(p4, bbox_inches='tight')
    plt.close(fig4)
    print(f"  [4] {p4}")


if __name__ == "__main__":
    data = load_data()
    plot_all(data)
    print("\nDone!")
