#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_table(path: Path, delimiter: str = "\t") -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_table(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: str) -> float:
    if value in ("", "NA", "nan", "NaN"):
        raise ValueError("not numeric")
    return float(value)


def numeric_values(rows: Iterable[dict], field: str) -> List[float]:
    values = []
    for row in rows:
        try:
            values.append(as_float(row.get(field, "")))
        except ValueError:
            continue
    return values


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def fmt(value: float | str | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def summarize_oracle(base: Path, out: Path) -> List[dict]:
    rows = read_table(base / "collected" / "oracle_summary_all.tsv")
    output = []
    for row in rows:
        oracle_reads = row.get("oracle_read_count", "0")
        if oracle_reads in ("", "0", "NA"):
            continue
        false_negatives = int(float(row.get("false_negative_count_total", "0") or 0))
        recall = row.get("mean_recall", "NA")
        status = "PASS" if false_negatives == 0 and recall not in ("", "NA") and float(recall) == 1.0 else "FAIL"
        output.append({
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "variant": row.get("variant", ""),
            "read_count": row.get("read_count", ""),
            "true_neighbor_count_total": row.get("true_neighbor_count_total", ""),
            "false_negative_count_total": str(false_negatives),
            "mean_recall": recall,
            "status": status,
        })
    write_table(out / "result1_no_fn_oracle.tsv", [
        "dataset", "method", "variant", "read_count",
        "true_neighbor_count_total", "false_negative_count_total",
        "mean_recall", "status",
    ], output)
    return output


def summarize_candidates(base: Path, out: Path) -> List[dict]:
    rows = read_table(base / "collected" / "main_summary_all.tsv")
    fields = [
        "dataset", "method", "variant", "read_count",
        "raw_candidate_count_mean", "raw_candidate_count_p95",
        "raw_candidate_count_p99", "accepted_candidate_count_mean",
        "accepted_candidate_count_p95", "accepted_candidate_count_p99",
        "retrieval_milliseconds_mean", "retrieval_milliseconds_p95",
        "retrieval_milliseconds_p99", "verification_milliseconds_mean",
        "verification_milliseconds_p95", "verification_milliseconds_p99",
        "total_milliseconds_mean", "total_milliseconds_p95",
        "total_milliseconds_p99", "oracle_read_count",
        "false_negative_count_total", "mean_recall",
        "mean_raw_candidate_blowup", "mean_accepted_candidate_blowup",
    ]
    output = [{field: row.get(field, "") for field in fields}
              for row in sorted(rows, key=lambda r: (
                  r.get("dataset", ""), r.get("method", ""),
                  r.get("variant", "")))]
    write_table(out / "result5_candidate_retrieval.tsv", fields, output)
    return output


def read_truth_by_id(path: Path) -> dict:
    return {row.get("read_id", ""): row for row in read_table(path)}


def parse_sam_alignments(path: Path) -> dict:
    alignments = {}
    if not path.exists():
        return alignments
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            try:
                flag = int(parts[1])
                pos0 = int(parts[3]) - 1 if parts[3] != "0" else -1
                mapq = int(parts[4]) if parts[4].isdigit() else 0
            except ValueError:
                continue
            alignments.setdefault(parts[0], []).append({
                "flag": flag,
                "rname": parts[2],
                "pos0": pos0,
                "mapq": mapq,
            })
    return alignments


def is_primary(flag: int) -> bool:
    return (flag & 0x100) == 0 and (flag & 0x800) == 0


def is_mapped(flag: int) -> bool:
    return (flag & 0x4) == 0


def source_recovered(alignment: dict, truth: dict, tolerance: int = 5) -> bool:
    if not is_mapped(alignment["flag"]):
        return False
    try:
        source_start = int(truth.get("source_start", ""))
    except ValueError:
        return False
    return abs(int(alignment["pos0"]) - source_start) <= tolerance


def extract_index_path_from_meta(meta_path: Path) -> str:
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    command = meta.get("command", [])
    if not isinstance(command, list):
        return ""
    for idx, token in enumerate(command):
        if token == "--index" and idx + 1 < len(command):
            return str(command[idx + 1])
    return ""


def elapsed_seconds_from_meta(meta_path: Path) -> str:
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    value = meta.get("elapsed_seconds", "")
    try:
        return fmt(float(value))
    except (TypeError, ValueError):
        return str(value)


def navigamer_row_recovers_source(row: dict, truth: dict,
                                  tolerance: int = 5) -> bool:
    try:
        source_start = int(truth.get("source_start", ""))
    except ValueError:
        return False
    starts = []
    try:
        starts.append(int(float(row.get("reference_start", ""))))
    except ValueError:
        pass
    ref_positions = row.get("ref_positions", "")
    if ref_positions:
        try:
            parsed = json.loads(ref_positions)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, list) and len(item) >= 2:
                    try:
                        starts.append(int(float(item[1])))
                    except (TypeError, ValueError):
                        continue
    return any(abs(start - source_start) <= tolerance for start in starts)


def summarize_navigamer_persisted_retrieval(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "dataset", "method", "variant", "read_count",
        "hit_row_count", "result_read_count", "result_rate",
        "source_recovered_count", "source_recovery_rate", "elapsed_seconds",
        "query_time_ms_mean", "query_time_ms_p50", "query_time_ms_p95",
        "dist_calcs_mean", "leaf_verify_count_mean", "candidate_count_mean",
        "result_count_mean", "index_path",
    ]
    output = []
    for root in sorted(base.glob("navigamer_main_rerun_v*")):
        if not root.is_dir():
            continue
        for work_dir in sorted(root.glob("*_work")):
            result_path = work_dir / "navigamer_benchmark.tsv"
            rows = read_table(result_path)
            if not rows:
                continue
            dataset = work_dir.name[:-5]
            truth = read_truth_by_id(base / "reads" / f"{dataset}.truth.tsv")
            if not truth:
                continue
            per_read_rows = {}
            for row in rows:
                read_id = row.get("read_id") or row.get("query_id")
                if read_id:
                    per_read_rows.setdefault(read_id, []).append(row)
            first_rows = [items[0] for _, items in sorted(per_read_rows.items())]
            read_count = len(truth)
            result_read_count = sum(1 for read_id in truth if read_id in per_read_rows)
            recovered_count = 0
            for read_id, truth_row in truth.items():
                if any(navigamer_row_recovers_source(row, truth_row)
                       for row in per_read_rows.get(read_id, [])):
                    recovered_count += 1
            query_times = numeric_values(first_rows, "query_time_ms")
            dist_calcs = numeric_values(first_rows, "dist_calcs")
            leaf_verify = numeric_values(first_rows, "leaf_verify_count")
            candidates = numeric_values(first_rows, "candidate_count_for_prune")
            result_counts = numeric_values(first_rows, "result_count")
            meta_path = work_dir / "run_meta.json"
            output.append({
                "source": root.name,
                "dataset": dataset,
                "method": "NavigaMer",
                "variant": "persisted-query-index",
                "read_count": str(read_count),
                "hit_row_count": str(len(rows)),
                "result_read_count": str(result_read_count),
                "result_rate": fmt(result_read_count / read_count) if read_count else "0",
                "source_recovered_count": str(recovered_count),
                "source_recovery_rate": fmt(recovered_count / read_count) if read_count else "0",
                "elapsed_seconds": elapsed_seconds_from_meta(meta_path),
                "query_time_ms_mean": fmt(sum(query_times) / len(query_times)) if query_times else "0",
                "query_time_ms_p50": fmt(percentile(query_times, 0.50)),
                "query_time_ms_p95": fmt(percentile(query_times, 0.95)),
                "dist_calcs_mean": fmt(sum(dist_calcs) / len(dist_calcs)) if dist_calcs else "0",
                "leaf_verify_count_mean": fmt(sum(leaf_verify) / len(leaf_verify)) if leaf_verify else "0",
                "candidate_count_mean": fmt(sum(candidates) / len(candidates)) if candidates else "0",
                "result_count_mean": fmt(sum(result_counts) / len(result_counts)) if result_counts else "0",
                "index_path": extract_index_path_from_meta(meta_path),
            })
    output.sort(key=lambda row: (row["source"], row["dataset"]))
    write_table(out / "result5_navigamer_persisted_retrieval.tsv", fields, output)
    return output


def summarize_mapper_end_to_end(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "dataset", "method", "read_count", "alignment_count",
        "mapped_read_count", "mapped_rate", "source_recovered_count",
        "source_recovery_rate", "wall_seconds", "max_rss_kb", "cpu_percent",
        "build_wall_seconds", "build_max_rss_kb", "index_bytes",
    ]
    output = []
    for root in sorted(base.glob("mapper_baselines_v*")):
        runs_dir = root / "runs"
        build_rows = {
            row.get("method", ""): row
            for row in read_table(root / "indexes" / "build_summary.tsv")
        }
        for sam_path in sorted(runs_dir.glob("*.sam")):
            parts = sam_path.name.split(".")
            if len(parts) < 3:
                continue
            dataset = ".".join(parts[:-2])
            method = parts[-2]
            truth = read_truth_by_id(runs_dir / f"{dataset}.truth.tsv")
            if not truth:
                continue
            alignments = parse_sam_alignments(sam_path)
            read_count = len(truth)
            alignment_count = sum(len(items) for items in alignments.values())
            mapped_count = 0
            recovered_count = 0
            for read_id, truth_row in truth.items():
                read_alignments = alignments.get(read_id, [])
                primary = [
                    aln for aln in read_alignments
                    if is_primary(aln["flag"])
                ]
                if any(is_mapped(aln["flag"]) for aln in primary):
                    mapped_count += 1
                if any(source_recovered(aln, truth_row) for aln in primary):
                    recovered_count += 1
            time_log = parse_time_log(sam_path.with_suffix(".time.log"))
            build = build_rows.get(method, {})
            output.append({
                "source": root.name,
                "dataset": dataset,
                "method": method,
                "read_count": str(read_count),
                "alignment_count": str(alignment_count),
                "mapped_read_count": str(mapped_count),
                "mapped_rate": fmt(mapped_count / read_count) if read_count else "0",
                "source_recovered_count": str(recovered_count),
                "source_recovery_rate": fmt(recovered_count / read_count) if read_count else "0",
                "wall_seconds": parse_wall_seconds(
                    time_log.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", "")
                ),
                "max_rss_kb": time_log.get("Maximum resident set size (kbytes)", ""),
                "cpu_percent": time_log.get("Percent of CPU this job got", ""),
                "build_wall_seconds": build.get("build_wall_seconds", ""),
                "build_max_rss_kb": build.get("max_rss_kb", ""),
                "index_bytes": build.get("index_bytes", ""),
            })
    write_table(out / "result5_mapper_end_to_end.tsv", fields, output)
    return output


def summarize_anchor_selection(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "strategy", "anchor_count", "query_count", "window_count",
        "mean_envelope_size", "false_negative_count_total",
        "source_recovery_rate", "pruning_ratio", "mean_exact_calls",
        "mean_true_neighbor_count", "bound_check_pass_rate",
        "window_length", "stride", "query_edits", "tolerance", "seed",
    ]
    output = []
    for root in sorted(base.glob("anchor_ablation_v*")):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.tsv")):
            rows = read_table(path)
            if not rows:
                continue
            required = {"strategy", "anchor_count", "mean_envelope_size"}
            if not required.issubset(rows[0].keys()):
                continue
            source = f"{root.name}/{path.stem}"
            for row in rows:
                output.append({
                    "source": source,
                    **{field: row.get(field, "") for field in fields if field != "source"},
                })
    output.sort(key=lambda row: (
        row.get("source", ""),
        row.get("strategy", ""),
        int(row.get("anchor_count", "0") or 0),
    ))
    write_table(out / "result2_anchor_selection.tsv", fields, output)
    return output


def summarize_corner_paths(base: Path, out: Path) -> List[dict]:
    fields = [
        "dataset", "query_count", "contained_count", "overlap_count",
        "uncovered_count", "contained_fraction", "overlap_fraction",
        "uncovered_fraction", "mbb_rect_fallback_count_total",
        "leaf_verify_count_mean", "result_count_mean", "query_time_ms_mean",
        "query_time_ms_p95",
    ]
    by_class_fields = [
        "dataset", "query_path_class", "query_count", "fraction",
        "mbb_rect_fallback_count_total", "leaf_verify_count_mean",
        "result_count_mean", "query_time_ms_mean", "query_time_ms_p95",
    ]
    output = []
    by_class_output = []
    for corner_dir in sorted(base.glob("corner_v*")):
        if not corner_dir.is_dir():
            continue
        for path in sorted(corner_dir.glob("*.tsv")):
            dataset_name = f"{corner_dir.name}/{path.stem}"
            table_rows = read_table(path)
            if not table_rows:
                continue
            if "query_id" not in table_rows[0] or "query_path_class" not in table_rows[0]:
                continue
            rows_by_query = {}
            for row in table_rows:
                query_id = row.get("query_id", "")
                if query_id and query_id not in rows_by_query:
                    rows_by_query[query_id] = row
            rows = list(rows_by_query.values())
            total = len(rows)
            counts = {"contained": 0, "overlap": 0, "uncovered": 0}
            for row in rows:
                klass = row.get("query_path_class", "unclassified")
                if klass in counts:
                    counts[klass] += 1
            fallback_total = sum(
                int(float(row.get("mbb_rect_fallback_count", "0") or 0))
                for row in rows
            )
            leaf_values = numeric_values(rows, "leaf_verify_count")
            result_values = numeric_values(rows, "result_count")
            time_values = numeric_values(rows, "query_time_ms")
            output.append({
                "dataset": dataset_name,
                "query_count": str(total),
                "contained_count": str(counts["contained"]),
                "overlap_count": str(counts["overlap"]),
                "uncovered_count": str(counts["uncovered"]),
                "contained_fraction": fmt(counts["contained"] / total) if total else "0",
                "overlap_fraction": fmt(counts["overlap"] / total) if total else "0",
                "uncovered_fraction": fmt(counts["uncovered"] / total) if total else "0",
                "mbb_rect_fallback_count_total": str(fallback_total),
                "leaf_verify_count_mean": fmt(sum(leaf_values) / len(leaf_values)) if leaf_values else "",
                "result_count_mean": fmt(sum(result_values) / len(result_values)) if result_values else "",
                "query_time_ms_mean": fmt(sum(time_values) / len(time_values)) if time_values else "",
                "query_time_ms_p95": fmt(percentile(time_values, 0.95)) if time_values else "",
            })
            for klass in ("contained", "overlap", "uncovered"):
                class_rows = [
                    row for row in rows
                    if row.get("query_path_class", "unclassified") == klass
                ]
                leaf_class = numeric_values(class_rows, "leaf_verify_count")
                result_class = numeric_values(class_rows, "result_count")
                time_class = numeric_values(class_rows, "query_time_ms")
                fallback_class = sum(
                    int(float(row.get("mbb_rect_fallback_count", "0") or 0))
                    for row in class_rows
                )
                by_class_output.append({
                    "dataset": dataset_name,
                    "query_path_class": klass,
                    "query_count": str(len(class_rows)),
                    "fraction": fmt(len(class_rows) / total) if total else "0",
                    "mbb_rect_fallback_count_total": str(fallback_class),
                    "leaf_verify_count_mean": fmt(sum(leaf_class) / len(leaf_class)) if leaf_class else "",
                    "result_count_mean": fmt(sum(result_class) / len(result_class)) if result_class else "",
                    "query_time_ms_mean": fmt(sum(time_class) / len(time_class)) if time_class else "",
                    "query_time_ms_p95": fmt(percentile(time_class, 0.95)) if time_class else "",
                })
    write_table(out / "result4_corner_paths.tsv", fields, output)
    write_table(out / "result4_corner_paths_by_class.tsv",
                by_class_fields, by_class_output)
    return output


def summarize_builds(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "method", "variant", "build_wall_seconds", "index_bytes",
        "reference_length", "window_length", "stride", "number_of_windows",
        "notes",
    ]
    output = []
    for row in read_table(base / "candidates_main_shared" / "build_summary.tsv"):
        output.append({
            "source": "baseline",
            "method": row.get("method", ""),
            "variant": row.get("variant", ""),
            "build_wall_seconds": row.get("wall_seconds", ""),
            "index_bytes": row.get("index_bytes", ""),
            "reference_length": row.get("reference_length", ""),
            "window_length": row.get("window_length", ""),
            "stride": row.get("stride", ""),
            "number_of_windows": row.get("number_of_windows", ""),
            "notes": row.get("parameters", ""),
        })

    build_rows = read_table(base / "navigamer_main_rebuilt" / "build_scale.csv",
                            delimiter=",")
    if build_rows:
        row = build_rows[-1]
        navidx_files = sorted((base / "navigamer_main_rebuilt").glob("*.navidx"))
        index_bytes = navidx_files[-1].stat().st_size if navidx_files else 0
        total_ms = as_float(row.get("total_build_ms", "0"))
        output.append({
            "source": "NavigaMer",
            "method": "NavigaMer",
            "variant": "adaptive",
            "build_wall_seconds": fmt(total_ms / 1000.0),
            "index_bytes": str(index_bytes),
            "reference_length": row.get("prefix_len", ""),
            "window_length": "150",
            "stride": "1",
            "number_of_windows": row.get("window_count", ""),
            "notes": "world_node_count="
            + row.get("world_node_count", "")
            + ";finest_world_count="
            + row.get("finest_world_count", ""),
        })

    write_table(out / "index_build_summary.tsv", fields, output)
    return output


def summarize_hierarchy_ablation(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "L", "radius_schedule", "query_count", "no_fn_count",
        "no_fn_rate", "query_time_ms_mean", "query_time_ms_p50",
        "query_time_ms_p95", "world_access_mean", "world_access_p95",
        "edge_access_mean", "anchor_distance_mean", "candidate_count_mean",
        "result_count_mean",
    ]
    grouped = {}
    for root in sorted(base.glob("layer_ablation_v*")):
        if not root.is_dir():
            continue
        for path in sorted(list(root.glob("*.csv")) + list(root.glob("*.tsv"))):
            rows = read_table(path, delimiter=",")
            if not rows:
                continue
            if "L" not in rows[0] or "radius_schedule" not in rows[0]:
                continue
            source = f"{root.name}/{path.stem}"
            for row in rows:
                key = (source, row.get("L", ""), row.get("radius_schedule", ""))
                grouped.setdefault(key, []).append(row)

    output = []
    for (source, layer_count, schedule), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), item[0][2])
    ):
        times = numeric_values(rows, "query_time_ms")
        world = numeric_values(rows, "world_access_count")
        edge = numeric_values(rows, "edge_access_count")
        anchor = numeric_values(rows, "anchor_distance_count")
        candidates = numeric_values(rows, "candidate_count")
        results = numeric_values(rows, "result_count")
        no_fn_count = sum(
            1 for row in rows
            if row.get("no_fn", "") in ("1", "true", "True", "PASS")
        )
        total = len(rows)
        output.append({
            "source": source,
            "L": layer_count,
            "radius_schedule": schedule,
            "query_count": str(total),
            "no_fn_count": str(no_fn_count),
            "no_fn_rate": fmt(no_fn_count / total) if total else "0",
            "query_time_ms_mean": fmt(sum(times) / len(times)) if times else "",
            "query_time_ms_p50": fmt(percentile(times, 0.5)) if times else "",
            "query_time_ms_p95": fmt(percentile(times, 0.95)) if times else "",
            "world_access_mean": fmt(sum(world) / len(world)) if world else "",
            "world_access_p95": fmt(percentile(world, 0.95)) if world else "",
            "edge_access_mean": fmt(sum(edge) / len(edge)) if edge else "",
            "anchor_distance_mean": fmt(sum(anchor) / len(anchor)) if anchor else "",
            "candidate_count_mean": fmt(sum(candidates) / len(candidates)) if candidates else "",
            "result_count_mean": fmt(sum(results) / len(results)) if results else "",
        })

    write_table(out / "result3_hierarchy_ablation.tsv", fields, output)
    return output


def summarize_locality(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "condition", "query_count", "mean_query_ms",
        "median_query_ms", "p95_query_ms", "wall_seconds", "max_rss_kb",
        "cpu_percent", "mean_prev_world_jaccard",
        "median_prev_world_jaccard", "mean_world_visit_count",
        "mean_leaf_visit_count",
    ]
    output = []
    trace_roots = [
        ("", base / "path_trace_v1"),
        ("locality_minimal_v1/", base / "locality_minimal_v1" / "runs"),
    ]
    for prefix, trace_dir in trace_roots:
        for path in sorted(trace_dir.glob("*.path_trace.tsv")):
            rows = read_table(path)
            world_visits = numeric_values(rows, "world_visit_count")
            leaf_visits = numeric_values(rows, "leaf_visit_count")
            jaccards = numeric_values(rows, "prev_world_jaccard")
            output.append({
                "source": "path_trace",
                "condition": prefix + path.name.removesuffix(".path_trace.tsv"),
                "query_count": str(len(rows)),
                "mean_query_ms": "",
                "median_query_ms": "",
                "p95_query_ms": "",
                "wall_seconds": "",
                "max_rss_kb": "",
                "cpu_percent": "",
                "mean_prev_world_jaccard": fmt(sum(jaccards) / len(jaccards)) if jaccards else "",
                "median_prev_world_jaccard": fmt(percentile(jaccards, 0.5)) if jaccards else "",
                "mean_world_visit_count": fmt(sum(world_visits) / len(world_visits)) if world_visits else "",
                "mean_leaf_visit_count": fmt(sum(leaf_visits) / len(leaf_visits)) if leaf_visits else "",
            })

    for timing_root in (base / "locality_minimal_v1" / "runs",
                        base / "prefetch_ab_v1"):
        if not timing_root.exists():
            continue
        for path in sorted(timing_root.glob("*.tsv")):
            rows = read_table(path)
            per_query_rows = {}
            for index, row in enumerate(rows):
                query_id = row.get("query_id") or row.get("read_id") or str(index)
                per_query_rows.setdefault(query_id, row)
            unique_rows = list(per_query_rows.values())
            times = numeric_values(unique_rows, "query_time_ms")
            if not times:
                continue
            time_log_path = path.with_suffix(".time.log")
            if not time_log_path.exists():
                time_log_path = path.with_suffix(".stderr_time.log")
            time_log = parse_time_log(time_log_path)
            output.append({
                "source": "timing",
                "condition": path.stem,
                "query_count": str(len(times)),
                "mean_query_ms": fmt(sum(times) / len(times)),
                "median_query_ms": fmt(percentile(times, 0.5)),
                "p95_query_ms": fmt(percentile(times, 0.95)),
                "wall_seconds": parse_wall_seconds(
                    time_log.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", "")
                ),
                "max_rss_kb": time_log.get("Maximum resident set size (kbytes)", ""),
                "cpu_percent": time_log.get("Percent of CPU this job got", ""),
                "mean_prev_world_jaccard": "",
                "median_prev_world_jaccard": "",
                "mean_world_visit_count": "",
                "mean_leaf_visit_count": "",
            })

    write_table(out / "result6_locality_prefetch.tsv", fields, output)
    return output


def parse_time_log(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ": " not in line:
            continue
        key, value = line.strip().split(": ", 1)
        values[key.strip()] = value.strip().rstrip("%")
    return values


def parse_wall_seconds(value: str) -> str:
    if not value:
        return ""
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return fmt(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
        if len(parts) == 2:
            return fmt(int(parts[0]) * 60 + float(parts[1]))
        return fmt(float(value))
    except ValueError:
        return ""


def summarize_system_ab(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "query_count", "query_time_ms_mean", "query_time_ms_p50",
        "query_time_ms_p95", "wall_seconds", "max_rss_kb", "cpu_percent",
        "major_page_faults", "minor_page_faults", "dist_calcs_mean",
        "leaf_verify_count_mean", "mbb_scan_child_checks_mean",
        "mbb_rect_index_queries_mean", "mbb_rect_candidate_children_mean",
        "center_distance_calls_after_qgram_mean", "qgram_prune_ratio_mean",
        "result_count_mean",
    ]
    output = []
    for root in sorted(base.glob("system_ab_v*")):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.tsv")):
            if path.name == "run_status.tsv":
                continue
            rows_by_query = {}
            for row in read_table(path):
                query_id = row.get("query_id", "")
                if query_id and query_id not in rows_by_query:
                    rows_by_query[query_id] = row
            rows = list(rows_by_query.values())
            if not rows:
                continue
            time_log = parse_time_log(path.with_suffix(".time.log"))
            times = numeric_values(rows, "query_time_ms")
            output.append({
                "source": f"{root.name}/{path.stem}",
                "query_count": str(len(rows)),
                "query_time_ms_mean": fmt(sum(times) / len(times)) if times else "",
                "query_time_ms_p50": fmt(percentile(times, 0.5)) if times else "",
                "query_time_ms_p95": fmt(percentile(times, 0.95)) if times else "",
                "wall_seconds": parse_wall_seconds(
                    time_log.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", "")
                ),
                "max_rss_kb": time_log.get("Maximum resident set size (kbytes)", ""),
                "cpu_percent": time_log.get("Percent of CPU this job got", ""),
                "major_page_faults": time_log.get("Major (requiring I/O) page faults", ""),
                "minor_page_faults": time_log.get("Minor (reclaiming a frame) page faults", ""),
                "dist_calcs_mean": fmt(sum(numeric_values(rows, "dist_calcs")) / len(rows)),
                "leaf_verify_count_mean": fmt(sum(numeric_values(rows, "leaf_verify_count")) / len(rows)),
                "mbb_scan_child_checks_mean": fmt(sum(numeric_values(rows, "mbb_scan_child_checks")) / len(rows)),
                "mbb_rect_index_queries_mean": fmt(sum(numeric_values(rows, "mbb_rect_index_queries")) / len(rows)),
                "mbb_rect_candidate_children_mean": fmt(sum(numeric_values(rows, "mbb_rect_candidate_children")) / len(rows)),
                "center_distance_calls_after_qgram_mean": fmt(
                    sum(numeric_values(rows, "center_distance_calls_after_qgram")) / len(rows)
                ),
                "qgram_prune_ratio_mean": fmt(sum(numeric_values(rows, "qgram_prune_ratio")) / len(rows)),
                "result_count_mean": fmt(sum(numeric_values(rows, "result_count")) / len(rows)),
            })
    write_table(out / "result6_system_ab.tsv", fields, output)
    return output


def parse_perf_stat(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        count = parts[0].strip()
        event = parts[2].strip()
        if not event:
            continue
        values[event] = "" if count.startswith("<") else count
    return values


def ratio(numerator: str, denominator: str) -> str:
    try:
        num = float(numerator)
        den = float(denominator)
    except ValueError:
        return ""
    if den == 0:
        return ""
    return fmt(num / den)


def per_query(value: str, query_count: int) -> str:
    if query_count <= 0:
        return ""
    try:
        return fmt(float(value) / query_count)
    except ValueError:
        return ""


def summarize_perf_stat(base: Path, out: Path) -> List[dict]:
    fields = [
        "source", "query_count", "wall_seconds", "max_rss_kb", "cpu_percent",
        "cycles", "instructions", "ipc", "cache_references", "cache_misses",
        "cache_miss_rate", "branches", "branch_misses", "branch_miss_rate",
        "page_faults", "minor_faults", "major_faults", "cycles_per_query",
        "cache_misses_per_query",
    ]
    output = []
    for root in sorted(base.glob("perf_stat_*")):
        if not root.is_dir():
            continue
        for perf_path in sorted(root.glob("*.perf.tsv")):
            stem = perf_path.name.removesuffix(".perf.tsv")
            query_rows = {
                row.get("query_id", "")
                for row in read_table(root / f"{stem}.tsv")
                if row.get("query_id", "")
            }
            query_count = len(query_rows)
            time_log = parse_time_log(root / f"{stem}.time.log")
            perf = parse_perf_stat(perf_path)
            cycles = perf.get("cycles", "")
            instructions = perf.get("instructions", "")
            cache_refs = perf.get("cache-references", "")
            cache_misses = perf.get("cache-misses", "")
            branches = perf.get("branches", "")
            branch_misses = perf.get("branch-misses", "")
            has_instruction_count = instructions not in ("", "0")
            has_cache_count = cache_refs not in ("", "0") or cache_misses not in ("", "0")
            output.append({
                "source": f"{root.name}/{stem}",
                "query_count": str(query_count),
                "wall_seconds": parse_wall_seconds(
                    time_log.get("Elapsed (wall clock) time (h:mm:ss or m:ss)", "")
                ),
                "max_rss_kb": time_log.get("Maximum resident set size (kbytes)", ""),
                "cpu_percent": time_log.get("Percent of CPU this job got", ""),
                "cycles": cycles,
                "instructions": instructions,
                "ipc": ratio(instructions, cycles) if has_instruction_count else "",
                "cache_references": cache_refs,
                "cache_misses": cache_misses,
                "cache_miss_rate": ratio(cache_misses, cache_refs),
                "branches": branches,
                "branch_misses": branch_misses,
                "branch_miss_rate": ratio(branch_misses, branches),
                "page_faults": perf.get("page-faults", ""),
                "minor_faults": perf.get("minor-faults", ""),
                "major_faults": perf.get("major-faults", ""),
                "cycles_per_query": per_query(cycles, query_count),
                "cache_misses_per_query": (
                    per_query(cache_misses, query_count) if has_cache_count else ""
                ),
            })

    write_table(out / "result6_perf_stat.tsv", fields, output)
    return output


def summarize_validation_logs(base: Path, out: Path) -> List[dict]:
    checks = {
        "test_recall": "ALL PASSED",
        "test_distance_bound": "ALL PASSED",
        "test_build_range_equivalence": "build range equivalence tests passed",
        "test_mbb_filter_equivalence": "MBB filter equivalence tests passed",
        "test_query_benchmark_gate": "query benchmark gate tests passed",
    }
    fields = ["check", "status", "evidence", "log_path"]
    log_dir = base / "results_summary_v1" / "validation_logs"
    output = []
    for check, marker in checks.items():
        log_path = log_dir / f"{check}.log"
        if not log_path.exists():
            output.append({
                "check": check,
                "status": "MISSING",
                "evidence": "",
                "log_path": str(log_path),
            })
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        status = "PASS" if marker in text else "FAIL"
        evidence = ""
        for line in reversed(text.splitlines()):
            if marker in line or "Summary:" in line or "passed" in line:
                evidence = line.strip()
                break
        output.append({
            "check": check,
            "status": status,
            "evidence": evidence,
            "log_path": str(log_path),
        })
    write_table(out / "result1_module_equivalence.tsv", fields, output)
    return output


def write_evidence_matrix(out: Path,
                          oracle_rows: Sequence[dict],
                          anchor_rows: Sequence[dict],
                          candidate_rows: Sequence[dict],
                          navigamer_persisted_rows: Sequence[dict],
                          mapper_rows: Sequence[dict],
                          corner_rows: Sequence[dict],
                          build_rows: Sequence[dict],
                          hierarchy_rows: Sequence[dict],
                          locality_rows: Sequence[dict],
                          system_rows: Sequence[dict],
                          perf_rows: Sequence[dict],
                          validation_rows: Sequence[dict]) -> None:
    fields = [
        "result", "claim", "status", "key_evidence", "tables", "caveat",
    ]
    navigamer_oracle = [
        row for row in oracle_rows
        if row.get("method") == "NavigaMer" and row.get("variant") == "adaptive"
    ]
    navigamer_zero_fn = all(
        row.get("false_negative_count_total") == "0"
        for row in navigamer_oracle
    ) if navigamer_oracle else False
    candidate_methods = len({row.get("method", "") for row in candidate_rows})
    rows = [
        {
            "result": "Result 1",
            "claim": "correctness/no false negatives",
            "status": "supported" if navigamer_zero_fn and validation_rows else "incomplete",
            "key_evidence": (
                f"oracle_rows={len(oracle_rows)}; "
                f"navigamer_zero_fn={navigamer_zero_fn}"
            ),
            "tables": "result1_no_fn_oracle.tsv; result1_module_equivalence.tsv",
            "caveat": "oracle evidence is limited to tested query/reference subsets",
        },
        {
            "result": "Result 2",
            "claim": "proximal anchors tighten candidate envelopes",
            "status": "supported" if anchor_rows else "missing",
            "key_evidence": (
                f"anchor_rows={len(anchor_rows)}; "
                f"all_zero_fn={all(row.get('false_negative_count_total') == '0' for row in anchor_rows) if anchor_rows else False}"
            ),
            "tables": "result2_anchor_selection.tsv",
            "caveat": "actual-anchor rows are a proxy unless runtime anchor tracing is added",
        },
        {
            "result": "Result 3",
            "claim": "hierarchy/world ablation preserves no-FN and exposes cost tradeoffs",
            "status": "preliminary" if hierarchy_rows else "missing",
            "key_evidence": (
                f"hierarchy_rows={len(hierarchy_rows)}; "
                f"all_no_fn={all(row.get('no_fn_rate') == '1' for row in hierarchy_rows) if hierarchy_rows else False}"
            ),
            "tables": "result3_hierarchy_ablation.tsv",
            "caveat": "current dense evidence is sparse-window 1.1M, not full stride-1 multi-L",
        },
        {
            "result": "Result 4",
            "claim": "contained/overlap/uncovered path classes are measured",
            "status": "supported" if corner_rows else "missing",
            "key_evidence": f"corner_rows={len(corner_rows)}",
            "tables": "result4_corner_paths.tsv; result4_corner_paths_by_class.tsv",
            "caveat": "fallback behavior is represented by tested synthetic/error query sets",
        },
        {
            "result": "Result 5",
            "claim": "candidate retrieval and mapper baselines",
            "status": "supported" if candidate_rows and mapper_rows else "incomplete",
            "key_evidence": (
                f"candidate_rows={len(candidate_rows)}; "
                f"mapper_rows={len(mapper_rows)}; "
                f"navigamer_persisted_rows={len(navigamer_persisted_rows)}; "
                f"candidate_methods={candidate_methods}"
            ),
            "tables": "result5_candidate_retrieval.tsv; result5_navigamer_persisted_retrieval.tsv; result5_mapper_end_to_end.tsv",
            "caveat": "mapper source recovery is primary-locus recovery, not candidate no-FN",
        },
        {
            "result": "Result 6",
            "claim": "locality/prefetch/system behavior",
            "status": "preliminary" if locality_rows or system_rows or perf_rows else "missing",
            "key_evidence": (
                f"locality_rows={len(locality_rows)}; "
                f"system_rows={len(system_rows)}; perf_rows={len(perf_rows)}"
            ),
            "tables": "result6_locality_prefetch.tsv; result6_system_ab.tsv; result6_perf_stat.tsv",
            "caveat": "local host lacks hardware perf counters; remote VM exposed cycles but not reliable cache counters",
        },
        {
            "result": "Build/Persistence",
            "claim": "index build time and persisted index sizes are recorded",
            "status": "supported" if build_rows else "missing",
            "key_evidence": f"build_rows={len(build_rows)}",
            "tables": "index_build_summary.tsv",
            "caveat": "full E. coli NavigaMer persisted index is still separate from 1.1M evidence",
        },
    ]
    write_table(out / "result0_evidence_matrix.tsv", fields, rows)


def write_status(out: Path, oracle_rows: Sequence[dict],
                 anchor_rows: Sequence[dict],
                 candidate_rows: Sequence[dict],
                 navigamer_persisted_rows: Sequence[dict],
                 mapper_rows: Sequence[dict],
                 corner_rows: Sequence[dict],
                 build_rows: Sequence[dict],
                 hierarchy_rows: Sequence[dict],
                 locality_rows: Sequence[dict],
                 system_rows: Sequence[dict],
                 perf_rows: Sequence[dict],
                 validation_rows: Sequence[dict]) -> None:
    navigamer_oracle = [
        row for row in oracle_rows
        if row["method"] == "NavigaMer" and row["variant"] == "adaptive"
    ]
    navigamer_fn = sum(int(row["false_negative_count_total"])
                      for row in navigamer_oracle)
    min_recall = min((float(row["mean_recall"]) for row in navigamer_oracle),
                     default=0.0)
    lines = [
        "# E. coli 1.1M Results Status",
        "",
        "## Result 1",
        (
            f"NavigaMer oracle rows: {len(navigamer_oracle)}; "
            f"zero false negatives: {navigamer_fn == 0}; "
            f"minimum recall: {fmt(min_recall)}."
        ),
        (
            "Module equivalence checks passing: "
            f"{sum(1 for row in validation_rows if row['status'] == 'PASS')}/"
            f"{len(validation_rows)}."
        ),
        "",
        "## Result 2",
        (
            f"Anchor-selection rows: {len(anchor_rows)}; "
            f"all zero-FN: "
            f"{all(row.get('false_negative_count_total') == '0' for row in anchor_rows) if anchor_rows else False}."
        ),
        "",
        "## Result 5",
        (
            f"Candidate retrieval summary rows: {len(candidate_rows)} across "
            f"{len({row['method'] for row in candidate_rows})} methods."
        ),
        f"NavigaMer persisted-index retrieval rows: {len(navigamer_persisted_rows)}.",
        f"Mapper end-to-end rows: {len(mapper_rows)}.",
        "",
        "## Result 4",
        f"Corner/path classification rows: {len(corner_rows)}.",
        "",
        "## Result 3 Preliminary",
        (
            f"Hierarchy ablation rows: {len(hierarchy_rows)}; "
            f"all no-FN in summarized rows: "
            f"{all(row.get('no_fn_rate') == '1' for row in hierarchy_rows) if hierarchy_rows else False}."
        ),
        "",
        "## Build / Persistence",
        f"Index build rows summarized: {len(build_rows)}.",
        "",
        "## Result 6 Preliminary",
        (
            f"Locality/prefetch diagnostic rows summarized: {len(locality_rows)}. "
            f"System A/B rows summarized: {len(system_rows)}. "
            f"Remote perf-stat rows summarized: {len(perf_rows)}. "
            "Perf hardware counters were unavailable on the local host if absent."
        ),
        "",
    ]
    (out / "results_status.md").write_text("\n".join(lines), encoding="utf-8")


def first_row(rows: Sequence[dict], **matches: str) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in matches.items()):
            return row
    return {}


def safe_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator in (None, 0):
        return "NA"
    return fmt(numerator / denominator)


def row_float(row: dict, field: str) -> float | None:
    try:
        return as_float(row.get(field, ""))
    except ValueError:
        return None


def write_claims(out: Path,
                 oracle_rows: Sequence[dict],
                 anchor_rows: Sequence[dict],
                 candidate_rows: Sequence[dict],
                 navigamer_persisted_rows: Sequence[dict],
                 mapper_rows: Sequence[dict],
                 corner_class_rows: Sequence[dict],
                 build_rows: Sequence[dict],
                 hierarchy_rows: Sequence[dict],
                 locality_rows: Sequence[dict],
                 system_rows: Sequence[dict],
                 perf_rows: Sequence[dict],
                 validation_rows: Sequence[dict]) -> List[dict]:
    fields = [
        "result", "status", "claim", "key_numbers", "evidence_tables",
        "manuscript_safe_wording", "caveat", "next_action",
    ]
    claims = []

    def add(result: str, status: str, claim: str, key_numbers: str,
            evidence: str, wording: str, caveat: str,
            next_action: str = "") -> None:
        claims.append({
            "result": result,
            "status": status,
            "claim": claim,
            "key_numbers": key_numbers,
            "evidence_tables": evidence,
            "manuscript_safe_wording": wording,
            "caveat": caveat,
            "next_action": next_action,
        })

    navigamer_oracle = [
        row for row in oracle_rows
        if row.get("method") == "NavigaMer" and row.get("variant") == "adaptive"
    ]
    total_fn = sum(
        int(float(row.get("false_negative_count_total", "0") or 0))
        for row in navigamer_oracle
    )
    recall_values = numeric_values(navigamer_oracle, "mean_recall")
    add(
        "Result 1", "supported" if navigamer_oracle and total_fn == 0 else "incomplete",
        "NavigaMer returns zero false negatives on tested oracle subsets.",
        (
            f"oracle_rows={len(navigamer_oracle)}; total_FN={total_fn}; "
            f"min_recall={fmt(min(recall_values)) if recall_values else 'NA'}; "
            f"module_checks_PASS="
            f"{sum(row.get('status') == 'PASS' for row in validation_rows)}/"
            f"{len(validation_rows)}"
        ),
        "result1_no_fn_oracle.tsv; result1_module_equivalence.tsv",
        (
            "Across the tested E. coli 1.1M oracle subsets, NavigaMer produced "
            "zero false negatives relative to exhaustive edit-distance range search."
        ),
        "Oracle subsets are sampled prefixes/query sets, not every possible 150-mer in the 1.1M reference.",
        "Scale oracle checks if a broader correctness claim is required.",
    )

    proximal_ratios = []
    far_ratios = []
    for source in sorted({row.get("source", "") for row in anchor_rows}):
        by_key = {
            (row.get("strategy", ""), row.get("anchor_count", "")):
            row_float(row, "mean_envelope_size")
            for row in anchor_rows if row.get("source") == source
        }
        proximal = by_key.get(("proximal", "2"))
        random = by_key.get(("random", "2"))
        far = by_key.get(("far", "2"))
        proximal_ratios.append(safe_ratio(random, proximal))
        far_ratios.append(safe_ratio(far, proximal))
    add(
        "Result 2", "supported" if anchor_rows else "missing",
        "Proximal anchors produce much tighter candidate envelopes than random anchors while preserving no-FN.",
        (
            f"rows={len(anchor_rows)}; "
            f"all_FN0={all(row.get('false_negative_count_total') == '0' for row in anchor_rows) if anchor_rows else False}; "
            f"random/proximal_envelope_ratio_at_2anchors={', '.join(proximal_ratios)}; "
            f"far/proximal_ratio_at_2anchors={', '.join(far_ratios)}"
        ),
        "result2_anchor_selection.tsv",
        (
            "With two anchors, proximal-anchor envelopes were substantially smaller "
            "than random-anchor envelopes in the tested 1.1M ablation settings, "
            "with zero false negatives."
        ),
        "The current actual-anchor trace remains a proxy; far anchors are also competitive at high anchor counts and should not be described as failing.",
        "Add runtime anchor tracing if claiming actual online anchor selection rather than controlled ablation.",
    )

    edge_ratios = []
    for source in sorted({row.get("source", "") for row in hierarchy_rows}):
        l3 = first_row(hierarchy_rows, source=source, L="3")
        l4 = first_row(hierarchy_rows, source=source, L="4")
        edge_ratios.append(safe_ratio(
            row_float(l4, "edge_access_mean"),
            row_float(l3, "edge_access_mean"),
        ))
    add(
        "Result 3", "preliminary" if hierarchy_rows else "missing",
        "Hierarchy ablation preserves no-FN; added layers show limited latency benefit and higher traversal overhead in 1.1M sparse-window tests.",
        (
            f"rows={len(hierarchy_rows)}; "
            f"all_noFN={all(row.get('no_fn_rate') == '1' for row in hierarchy_rows) if hierarchy_rows else False}; "
            f"L4_vs_L3_edge_access_ratio={', '.join(edge_ratios)}"
        ),
        "result3_hierarchy_ablation.tsv",
        (
            "Across stride100/50/25 sparse-window hierarchy ablations, L=2/3/4 "
            "all preserved no-FN; L=4 increased edge-access overhead without a "
            "clear p95 latency win."
        ),
        "This does not yet prove the hierarchy is necessary versus a true flat/all-beacon baseline; it is sparse-window 1.1M evidence.",
        "Run/include flat and single-layer baselines if Result 3 must support hierarchy necessity strongly.",
    )

    path_classes = {}
    for row in corner_class_rows:
        entry = path_classes.setdefault(
            row.get("query_path_class", ""),
            {"queries": 0, "fallback": 0, "p95": []},
        )
        entry["queries"] += int(float(row.get("query_count", "0") or 0))
        entry["fallback"] += int(float(row.get("mbb_rect_fallback_count_total", "0") or 0))
        p95 = row_float(row, "query_time_ms_p95")
        if p95 is not None:
            entry["p95"].append(p95)
    path_summary = "; ".join(
        f"{name}:q={values['queries']},fallback={values['fallback']},"
        f"max_p95={fmt(max(values['p95'])) if values['p95'] else 'NA'}"
        for name, values in sorted(path_classes.items())
    )
    add(
        "Result 4", "supported" if corner_class_rows else "missing",
        "Contained, overlap, and uncovered path classes are observed and summarized separately.",
        f"rows={len(corner_class_rows)}; {path_summary}",
        "result4_corner_paths.tsv; result4_corner_paths_by_class.tsv",
        (
            "The tested query sets include contained, overlap, and uncovered cases; "
            "uncovered queries are present but relatively rare in the summarized "
            "path-class counts."
        ),
        "Fallback count is zero in the current summarized rows, so this supports path classification more than fallback-overhead stress testing.",
        "Add a dedicated fallback-triggering set if the paper needs a strong fallback-overhead claim.",
    )

    add(
        "Result 5a", "supported" if navigamer_persisted_rows else "missing",
        "Persisted-index NavigaMer retrieval recovers the source locus for all tested 1.1M reads.",
        (
            f"persisted_rows={len(navigamer_persisted_rows)}; "
            f"all_source_recovery_1={all(row.get('source_recovery_rate') == '1' for row in navigamer_persisted_rows) if navigamer_persisted_rows else False}; "
            f"datasets={len({row.get('dataset', '') for row in navigamer_persisted_rows})}"
        ),
        "result5_navigamer_persisted_retrieval.tsv",
        (
            "Using the persisted 1.1M NavigaMer index, every tested read set had "
            "source_recovery_rate=1.0."
        ),
        "This is source-locus recovery under the generated truth set, not a complete all-neighbor oracle over the full reference.",
        "Keep this separate from mapper primary-locus recovery and from exhaustive oracle recall.",
    )

    nav_tau5 = first_row(candidate_rows, dataset="main_mixed_tau5_10000",
                         method="NavigaMer", variant="adaptive")
    qgram_tau5 = first_row(candidate_rows, dataset="main_mixed_tau5_10000",
                           method="qgram-safe", variant="q5")
    pigeon_tau5 = first_row(candidate_rows, dataset="main_mixed_tau5_10000",
                            method="pigeonhole", variant="tau5")
    contig_tau5 = first_row(candidate_rows, dataset="main_mixed_tau5_10000",
                            method="contig", variant="k23")
    add(
        "Result 5b", "mixed" if candidate_rows else "missing",
        "Current q-gram/pigeonhole candidate baselines are faster and smaller than NavigaMer under the present 1.1M implementation/parameters.",
        (
            "mixed_tau5 raw_mean: "
            f"NavigaMer={fmt(row_float(nav_tau5, 'raw_candidate_count_mean'))}, "
            f"qgram_q5={fmt(row_float(qgram_tau5, 'raw_candidate_count_mean'))}, "
            f"pigeonhole_tau5={fmt(row_float(pigeon_tau5, 'raw_candidate_count_mean'))}, "
            f"contig_k23={fmt(row_float(contig_tau5, 'raw_candidate_count_mean'))}; "
            "p95_ms: "
            f"NavigaMer={fmt(row_float(nav_tau5, 'total_milliseconds_p95'))}, "
            f"qgram_q5={fmt(row_float(qgram_tau5, 'total_milliseconds_p95'))}, "
            f"pigeonhole_tau5={fmt(row_float(pigeon_tau5, 'total_milliseconds_p95'))}"
        ),
        "result5_candidate_retrieval.tsv",
        (
            "In the current 1.1M candidate-retrieval benchmark, safe q-gram and "
            "pigeonhole filters dominate NavigaMer on candidate count and p95 time; "
            "NavigaMer should not be claimed faster here."
        ),
        "TensorSketch rows report low accepted candidates but this is not equivalent to source/no-FN recovery without a stronger recovery analysis.",
        "Either improve NavigaMer query path or frame NavigaMer around correctness/geometry rather than speed for this dataset.",
    )

    minimap_tau5 = first_row(mapper_rows, dataset="main_mixed_tau5_10000",
                             method="minimap2")
    strobe_tau5 = first_row(mapper_rows, dataset="main_mixed_tau5_10000",
                            method="strobealign")
    add(
        "Result 5c", "supported" if mapper_rows else "missing",
        "Native mappers map all reads but primary source-locus recovery declines under harder errors.",
        (
            f"mapper_rows={len(mapper_rows)}; "
            f"mixed_tau5_minimap2_source={fmt(row_float(minimap_tau5, 'source_recovery_rate'))}; "
            f"mixed_tau5_strobealign_source={fmt(row_float(strobe_tau5, 'source_recovery_rate'))}"
        ),
        "result5_mapper_end_to_end.tsv",
        (
            "BWA-MEM2, minimap2, and strobealign all map the tested reads, but "
            "primary source-locus recovery drops below 1.0 under tau5/hard reads."
        ),
        "Mapper primary-locus recovery is not candidate no-FN; repeat-equivalent mappings may be biologically acceptable.",
        "Use as end-to-end mapper context, not as an index-level candidate-retrieval oracle.",
    )

    random_path = first_row(
        locality_rows, source="path_trace",
        condition="locality_minimal_v1/random128_prefetch_off_singlethread")
    sorted_path = first_row(
        locality_rows, source="path_trace",
        condition="locality_minimal_v1/source_sorted128_prefetch_off_singlethread")
    random_time = first_row(locality_rows, source="timing",
                            condition="random128_prefetch_off_singlethread")
    sorted_time = first_row(locality_rows, source="timing",
                            condition="source_sorted128_prefetch_off_singlethread")
    sorted_on = first_row(locality_rows, source="timing",
                          condition="source_sorted128_prefetch_on_singlethread")
    add(
        "Result 6", "preliminary" if locality_rows or system_rows or perf_rows else "missing",
        "Similar query ordering increases path reuse, but current prefetch/order settings do not measurably improve runtime.",
        (
            "singlethread_128: "
            f"random_jaccard={fmt(row_float(random_path, 'mean_prev_world_jaccard'))}, "
            f"sorted_jaccard={fmt(row_float(sorted_path, 'mean_prev_world_jaccard'))}, "
            f"random_wall={fmt(row_float(random_time, 'wall_seconds'))}s, "
            f"sorted_off_wall={fmt(row_float(sorted_time, 'wall_seconds'))}s, "
            f"sorted_on_wall={fmt(row_float(sorted_on, 'wall_seconds'))}s; "
            f"locality_rows={len(locality_rows)}"
        ),
        "result6_locality_prefetch.tsv; result6_system_ab.tsv; result6_perf_stat.tsv",
        (
            "Source-sorted queries show much higher world-path reuse than random "
            "order, but current prefetch and ordering do not produce a measurable "
            "wall-clock speedup on 1.1M."
        ),
        "Hardware cache counters are unavailable/unreliable on the accessible machines, so memory-bound/cache-miss claims remain unsupported.",
        "Need a physical machine with valid perf counters or a stronger software locality metric before claiming cache-aware throughput gains.",
    )

    navigamer_build_rows = [
        row for row in build_rows if row.get("method") == "NavigaMer"
    ]
    index_sizes = numeric_values(navigamer_build_rows, "index_bytes")
    add(
        "Build/Persistence", "supported" if build_rows else "missing",
        "Persisted index build artifacts and sizes are recorded for 1.1M experiments.",
        (
            f"build_rows={len(build_rows)}; "
            f"navigamer_rows={len(navigamer_build_rows)}; "
            f"max_navidx_bytes={fmt(max(index_sizes)) if index_sizes else 'NA'}"
        ),
        "index_build_summary.tsv; result5_navigamer_persisted_retrieval.tsv",
        (
            "The 1.1M NavigaMer comparison uses a persisted .navidx index and "
            "records build/index-size metadata."
        ),
        "Full E. coli persisted-index evidence remains separate from this 1.1M Results set.",
        "Keep 1.1M and full-reference claims explicitly separated.",
    )

    write_table(out / "results_claims_1p1m.tsv", fields, claims)
    md_lines = [
        "# E. coli 1.1M Claim-Level Results",
        "",
        "This file maps manuscript-level claims to the current 1.1M evidence tables.",
    ]
    for claim in claims:
        md_lines.extend([
            "",
            f"## {claim['result']} - {claim['status']}",
            f"**Claim:** {claim['claim']}",
            "",
            f"**Key numbers:** {claim['key_numbers']}",
            "",
            f"**Safe wording:** {claim['manuscript_safe_wording']}",
            "",
            f"**Evidence:** {claim['evidence_tables']}",
            "",
            f"**Caveat:** {claim['caveat']}",
        ])
        if claim["next_action"]:
            md_lines.extend(["", f"**Next action:** {claim['next_action']}"])
    (out / "results_claims_1p1m.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    return claims


def write_manuscript_draft(out: Path, claims: Sequence[dict]) -> None:
    by_result = {row.get("result", ""): row for row in claims}

    def wording(result: str) -> str:
        return by_result.get(result, {}).get("manuscript_safe_wording", "")

    def key(result: str) -> str:
        return by_result.get(result, {}).get("key_numbers", "")

    lines = [
        "# Draft 1.1M Results Section",
        "",
        "This draft is limited to the E. coli 1.1M evidence currently summarized in `results_summary_v1`. It is written from the claim-level table and keeps unsupported speed or cache claims out of the main text.",
        "",
        "## Result 1. NavigaMer preserved exact candidate retrieval in oracle checks",
        "",
        wording("Result 1"),
        "",
        f"Key evidence: {key('Result 1')}.",
        "",
        "## Result 2. Proximal anchors tightened candidate envelopes",
        "",
        wording("Result 2"),
        "",
        f"Key evidence: {key('Result 2')}.",
        "",
        "## Result 3. Hierarchy ablations preserved recall but did not yet prove a hierarchy advantage",
        "",
        wording("Result 3"),
        "",
        f"Key evidence: {key('Result 3')}. This result should be presented as an ablation of safe layer settings, not as proof that the hierarchy is necessary.",
        "",
        "## Result 4. Boundary path classes were observed under the 1.1M query sets",
        "",
        wording("Result 4"),
        "",
        f"Key evidence: {key('Result 4')}. This result supports path classification; the current data do not yet support a strong fallback-overhead claim.",
        "",
        "## Result 5. Persisted NavigaMer retrieval was correct, but current safe filters were faster",
        "",
        wording("Result 5a"),
        "",
        wording("Result 5b"),
        "",
        wording("Result 5c"),
        "",
        f"Key evidence: {key('Result 5a')}. {key('Result 5b')}. {key('Result 5c')}. This comparison should not be interpreted as a throughput advantage for NavigaMer under the current 1.1M implementation.",
        "",
        "## Result 6. Query ordering increased path reuse without a measured prefetch speedup",
        "",
        wording("Result 6"),
        "",
        f"Key evidence: {key('Result 6')}. The current data support a locality signal in the traversal paths, but not a cache-miss or throughput-improvement claim.",
        "",
        "## Claim-evidence map",
        "",
    ]
    for row in claims:
        lines.extend([
            f"- {row.get('result', '')}: {row.get('claim', '')}",
            f"  Evidence: {row.get('evidence_tables', '')}. Status: {row.get('status', '')}.",
        ])
    lines.extend([
        "",
        "## Limitations to keep out of main claims",
        "",
        "- Do not claim a general full-reference no-FN guarantee from the 1.1M sampled oracle rows alone.",
        "- Do not claim that the current hierarchy ablation proves hierarchy necessity versus a true flat/all-beacon index.",
        "- Do not claim that NavigaMer is faster than safe q-gram or pigeonhole filters on the current 1.1M benchmark.",
        "- Do not claim cache-miss reduction from the current perf-stat rows, because accessible machines did not expose reliable hardware cache counters.",
        "",
    ])
    (out / "results_manuscript_draft_1p1m.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def summarize(base: Path, out: Path) -> None:
    raise_csv_field_limit()
    out.mkdir(parents=True, exist_ok=True)
    oracle_rows = summarize_oracle(base, out)
    anchor_rows = summarize_anchor_selection(base, out)
    candidate_rows = summarize_candidates(base, out)
    navigamer_persisted_rows = summarize_navigamer_persisted_retrieval(base, out)
    mapper_rows = summarize_mapper_end_to_end(base, out)
    corner_rows = summarize_corner_paths(base, out)
    corner_class_rows = read_table(out / "result4_corner_paths_by_class.tsv")
    build_rows = summarize_builds(base, out)
    hierarchy_rows = summarize_hierarchy_ablation(base, out)
    locality_rows = summarize_locality(base, out)
    system_rows = summarize_system_ab(base, out)
    perf_rows = summarize_perf_stat(base, out)
    validation_rows = summarize_validation_logs(base, out)
    write_evidence_matrix(out, oracle_rows, anchor_rows, candidate_rows,
                          navigamer_persisted_rows, mapper_rows, corner_rows, build_rows,
                          hierarchy_rows, locality_rows, system_rows,
                          perf_rows, validation_rows)
    write_status(out, oracle_rows, anchor_rows, candidate_rows,
                 navigamer_persisted_rows, mapper_rows,
                 corner_rows, build_rows, hierarchy_rows, locality_rows,
                 system_rows, perf_rows,
                 validation_rows)
    claim_rows = write_claims(out, oracle_rows, anchor_rows, candidate_rows,
                              navigamer_persisted_rows, mapper_rows,
                              corner_class_rows, build_rows, hierarchy_rows,
                              locality_rows, system_rows, perf_rows,
                              validation_rows)
    write_manuscript_draft(out, claim_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize E. coli 1.1M comparison artifacts into Results tables."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.base_dir, args.out_dir)


if __name__ == "__main__":
    main()
