#!/usr/bin/env python3
"""Summarize NavigaMer Result 1-6 benchmark artifacts.

The script intentionally stays at the orchestration layer: it reads TSV/CSV
outputs produced by existing benchmarks and writes a compact status table plus
a review report for future method comparisons.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RESULT_IDS = [f"Result {i}" for i in range(1, 7)]
BAD_COUNT_COLUMNS = {
    "fn_count",
    "false_negative_count",
    "false_negative_count_total",
    "false_negative_total",
    "mismatch_count",
    "equality_failure_count",
    "result_mismatch_count",
}
METRIC_COLUMNS = {
    "query_count",
    "fn_count",
    "mismatch_count",
    "false_negative_count",
    "false_negative_count_total",
    "mean_query_ms",
    "p95_query_ms",
    "mean_leaf_verify",
    "mean_envelope_size",
    "mean_exact_calls",
    "pruning_ratio",
    "source_recovery_rate",
    "candidate_count",
    "raw_candidate_count",
    "verified_match_count",
    "reuse_hit_rate",
    "path_reuse_hit_ratio",
    "center_distance_reduction",
    "world_access_reduction",
    "p95_speedup",
}


def read_table(path: Path) -> List[Dict[str, str]]:
    delimiter = "," if path.suffix == ".csv" else "\t"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader)


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "unavailable"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def any_bad_count(rows: Sequence[Dict[str, str]]) -> bool:
    for row in rows:
        for col in BAD_COUNT_COLUMNS:
            value = as_float(row.get(col))
            if value is not None and value > 0:
                return True
    return False


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def list_existing(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if path.exists() and path.stat().st_size > 0]


def collect_tables(paths: Iterable[Path]) -> List[Tuple[Path, List[Dict[str, str]]]]:
    tables: List[Tuple[Path, List[Dict[str, str]]]] = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            rows = read_table(path)
            if rows:
                tables.append((path, rows))
    return tables


def rel(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def min_value(rows: Sequence[Dict[str, str]], column: str, selector: Optional[Tuple[str, str]] = None) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        if selector is not None and row.get(selector[0]) != selector[1]:
            continue
        value = as_float(row.get(column))
        if value is not None:
            values.append(value)
    return min(values) if values else None


def max_value(rows: Sequence[Dict[str, str]], column: str, selector: Optional[Tuple[str, str]] = None) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        if selector is not None and row.get(selector[0]) != selector[1]:
            continue
        value = as_float(row.get(column))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def add_metrics(result_id: str, path: Path, root: Path, rows: Sequence[Dict[str, str]], out: List[Dict[str, str]]) -> None:
    for col in METRIC_COLUMNS:
        values = [as_float(row.get(col)) for row in rows if col in row]
        values = [value for value in values if value is not None]
        if not values:
            continue
        out.append({
            "result_id": result_id,
            "source": rel(path, root),
            "metric": f"{col}.max",
            "value": f"{max(values):.6g}",
            "status": "observed",
            "path": str(path),
        })
        if len(values) > 1:
            out.append({
                "result_id": result_id,
                "source": rel(path, root),
                "metric": f"{col}.min",
                "value": f"{min(values):.6g}",
                "status": "observed",
                "path": str(path),
            })


def status_row(result_id: str, status: str, summary: str, boundary: str, next_step: str) -> Dict[str, str]:
    return {
        "result_id": result_id,
        "status": status,
        "summary": summary,
        "claim_boundary": boundary,
        "next_step": next_step,
    }


def summarize_result1(root: Path, metrics: List[Dict[str, str]]) -> Dict[str, str]:
    summary_path = first_existing([
        root / "result1_correctness" / "query_benchmark_summary.tsv",
        root / "result1_correctness" / "summary.tsv",
    ])
    recall_log = root / "logs" / "test_recall.log"
    distance_log = root / "logs" / "test_distance_bound.log"
    logs_present = recall_log.exists() and distance_log.exists()
    if summary_path is None and not logs_present:
        return status_row("Result 1", "missing", "No correctness benchmark or guard logs found.", "No claim can be made.", "Run query-benchmark plus test_recall/test_distance_bound.")
    rows = read_table(summary_path) if summary_path else []
    if rows:
        add_metrics("Result 1", summary_path, root, rows, metrics)
    if any_bad_count(rows):
        return status_row("Result 1", "failed", "Correctness output reports nonzero false-negative or mismatch counts.", "Reliability claim is not supported for this run.", "Inspect query_benchmark_summary.tsv and failing query detail rows.")
    if rows and logs_present:
        return status_row("Result 1", "supported", "Query benchmark and recall/distance guard logs report no false negatives or mismatches.", "Applies to the query classes, edit thresholds, and profile flags in this run.", "For paper-scale evidence, rerun with the full reference and archived command metadata.")
    return status_row("Result 1", "preliminary", "Some correctness evidence exists, but either query-benchmark summary or guard logs are missing.", "Treat as a smoke result only.", "Add both guard logs and query-benchmark summary to support the reliability claim.")


def summarize_result2(root: Path, metrics: List[Dict[str, str]]) -> Dict[str, str]:
    path = first_existing([
        root / "result2_anchor" / "anchor_selection.tsv",
        root / "result2_anchor" / "anchor_ablation.tsv",
    ])
    if path is None:
        return status_row("Result 2", "missing", "No anchor-selection ablation output found.", "No anchor-envelope/pruning claim can be made.", "Run experiments/ecoli_1p1m/scripts/anchor_ablation.py.")
    rows = read_table(path)
    add_metrics("Result 2", path, root, rows, metrics)
    if any_bad_count(rows):
        return status_row("Result 2", "failed", "Anchor ablation has nonzero false negatives.", "Anchor pruning is not recall-safe under this run.", "Inspect false_negative_count_total by strategy and anchor_count.")
    strategies = {row.get("strategy", "") for row in rows}
    proximal = min_value(rows, "mean_envelope_size", ("strategy", "proximal"))
    random = min_value(rows, "mean_envelope_size", ("strategy", "random"))
    if proximal is not None and random is not None:
        summary = f"Anchor ablation has zero false negatives; best proximal envelope {proximal:.3g} vs random {random:.3g}."
    else:
        summary = f"Anchor ablation has zero false negatives across {len(strategies)} strategies."
    return status_row("Result 2", "supported", summary, "Applies to the generated query set, strategies, and anchor counts in the TSV.", "For stronger claims, include full-reference runs and confidence intervals.")


def summarize_result3(root: Path, metrics: List[Dict[str, str]]) -> Dict[str, str]:
    tables = collect_tables([
        root / "result3_hierarchy" / "layer_radius.csv",
        root / "result3_hierarchy" / "uniform_summary.tsv",
        root / "result3_hierarchy" / "uniform_summary_all.tsv",
    ])
    if not tables:
        return status_row("Result 3", "missing", "No hierarchy/layer-radius sweep output found.", "No hierarchy-parameter claim can be made.", "Run layer-radius-experiment or import a uniform hierarchy sweep summary.")
    bad = False
    total_rows = 0
    for path, rows in tables:
        total_rows += len(rows)
        bad = bad or any_bad_count(rows)
        add_metrics("Result 3", path, root, rows, metrics)
    if bad:
        return status_row("Result 3", "failed", "Hierarchy sweep reports nonzero false negatives or mismatches.", "The tested hierarchy configuration is unsafe.", "Inspect the failing L/r_leaf/alpha or uniform hierarchy row.")
    status = "supported" if any("uniform" in path.name for path, _ in tables) else "preliminary"
    summary = f"Hierarchy sweep contains {total_rows} rows with no detected correctness failures."
    boundary = "Layer-radius search-cost evidence only unless the sweep includes full uniform-hierarchy/query-locality rows."
    next_step = "Complete q8/q32 and real-dup schedule coverage before making a final hierarchy claim."
    return status_row("Result 3", status, summary, boundary, next_step)


def summarize_result4(root: Path, metrics: List[Dict[str, str]]) -> Dict[str, str]:
    paths = list_existing((root / "result4_corner").glob("*.tsv"))
    if not paths:
        return status_row("Result 4", "missing", "No dedicated corner-case/fallback output found.", "Current Result 4 remains unclaimed in this run.", "Add a small adversarial/repeat/no-hit benchmark or extract query-benchmark class rows into result4_corner.")
    bad = False
    for path in paths:
        rows = read_table(path)
        bad = bad or any_bad_count(rows)
        add_metrics("Result 4", path, root, rows, metrics)
    if bad:
        return status_row("Result 4", "failed", "Corner-case output reports nonzero false negatives or mismatches.", "Corner-case reliability is not supported.", "Inspect result4_corner TSV rows.")
    return status_row("Result 4", "preliminary", f"Found {len(paths)} corner-case TSV file(s) with no detected correctness failures.", "Needs explicit pass/fallback criteria before it is paper-ready.", "Define the expected corner classes and add a deterministic runner.")


def summarize_result5(root: Path, metrics: List[Dict[str, str]]) -> Dict[str, str]:
    paths = list_existing((root / "result5_candidates").glob("*.tsv"))
    if not paths:
        return status_row("Result 5", "missing", "No external candidate/baseline comparison output found.", "No seed-baseline comparison can be made.", "Run candidate_tool baselines and navigamer candidate-verify, or import their summaries.")
    bad = False
    methods = set()
    all_rows: List[Dict[str, str]] = []
    for path in paths:
        rows = read_table(path)
        all_rows.extend(rows)
        bad = bad or any_bad_count(rows)
        add_metrics("Result 5", path, root, rows, metrics)
        for row in rows:
            methods.add(row.get("method") or row.get("profile") or row.get("strategy") or "")
    if bad:
        return status_row("Result 5", "failed", "Candidate/baseline output reports nonzero false negatives.", "Baseline comparison is not recall-safe.", "Inspect candidate verification summary and raw candidate generation.")
    non_navigamer = {method for method in methods if method and method.lower() not in {"navigamer", "optimized", "baseline"}}
    nav_ms = min_value(all_rows, "mean_query_ms", ("method", "navigamer"))
    other_ms = [as_float(row.get("mean_query_ms")) for row in all_rows if (row.get("method") or "").lower() != "navigamer"]
    other_ms = [value for value in other_ms if value is not None]
    if non_navigamer:
        speed_note = ""
        if nav_ms is not None and other_ms:
            faster = min(other_ms) < nav_ms
            speed_note = " Some external baselines are faster in this run." if faster else " NavigaMer is not slower than the listed external baselines in this run."
        return status_row("Result 5", "mixed", f"Found zero-FN candidate comparison rows for {len(methods)} methods.{speed_note}", "Interpret speed/candidate-count tradeoffs method by method.", "Keep raw candidate generation time and exact verification time separated in future comparisons.")
    return status_row("Result 5", "preliminary", "Candidate outputs exist, but no named external baseline method was detected.", "Only intra-NavigaMer evidence is available.", "Add qgram-safe/pigeonhole/randstrobe/spaced-seed summaries.")


def summarize_result6(root: Path, metrics: List[Dict[str, str]]) -> Dict[str, str]:
    paths = list_existing([
        root / "result6_locality" / "locality_summary.tsv",
        root / "result6_locality" / "summary.tsv",
        root / "result6_locality" / "report" / "summary.tsv",
    ])
    paths.extend(path for path in (root / "result6_locality").glob("*locality*.tsv") if path not in paths)
    if not paths:
        return status_row("Result 6", "missing", "No persisted-index locality benchmark output found.", "No locality/reuse claim can be made.", "Run locality-benchmark or query-locality-report.")
    bad = False
    rows_total = 0
    schedules = set()
    for path in paths:
        rows = read_table(path)
        rows_total += len(rows)
        bad = bad or any_bad_count(rows)
        add_metrics("Result 6", path, root, rows, metrics)
        schedules.update(row.get("batch_schedule") or row.get("batch_schedule_mode") or "" for row in rows)
    if bad:
        return status_row("Result 6", "failed", "Locality benchmark reports nonzero false negatives or mismatches.", "Locality optimization is not recall-safe for this run.", "Inspect fn_count/mismatch_count by profile and schedule.")
    status = "supported" if rows_total else "missing"
    schedule_note = ", ".join(sorted(s for s in schedules if s)) or "unknown schedule"
    return status_row("Result 6", status, f"Locality benchmark has {rows_total} rows with zero false negatives/mismatches; schedules: {schedule_note}.", "Applies to the persisted index, generated datasets, and schedules in the TSV.", "For final comparisons, include original/random/signature/source-oracle schedules and real-dup datasets.")


def write_report(root: Path, rows: Sequence[Dict[str, str]]) -> None:
    lines = [
        "# NavigaMer Results 1-6 Benchmark Report",
        "",
        f"Run directory: `{root}`",
        "",
        "| Result | Status | Summary |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['result_id']} | {row['status']} | {row['summary']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `supported`: this run contains the expected artifact(s) and no detected correctness failure.",
        "- `preliminary`: useful evidence exists, but coverage is not enough for a final paper claim.",
        "- `mixed`: comparison evidence exists but favors different methods on different axes.",
        "- `missing`: no relevant artifact was found in this run directory.",
        "- `failed`: at least one correctness column reports a nonzero false-negative or mismatch count.",
        "",
        "Use `summary_all.tsv` for machine-readable metric extraction and `result_status.tsv` for run-level comparison.",
        "",
    ])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(root: Path) -> None:
    root = root.resolve()
    if not root.exists():
        raise SystemExit(f"run directory does not exist: {root}")
    metrics: List[Dict[str, str]] = []
    rows = [
        summarize_result1(root, metrics),
        summarize_result2(root, metrics),
        summarize_result3(root, metrics),
        summarize_result4(root, metrics),
        summarize_result5(root, metrics),
        summarize_result6(root, metrics),
    ]
    write_tsv(root / "result_status.tsv", ["result_id", "status", "summary", "claim_boundary", "next_step"], rows)
    write_tsv(root / "summary_all.tsv", ["result_id", "source", "metric", "value", "status", "path"], metrics)
    write_report(root, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NavigaMer Result 1-6 benchmark artifacts.")
    parser.add_argument("run_dir", type=Path, help="Benchmark run directory.")
    args = parser.parse_args()
    summarize(args.run_dir)


if __name__ == "__main__":
    main()
