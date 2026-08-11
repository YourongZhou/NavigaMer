#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi

RUN_DIR="$1"

for required in result_status.tsv summary_all.tsv report.md; do
  if [[ ! -s "$RUN_DIR/$required" ]]; then
    echo "missing required summary artifact: $RUN_DIR/$required" >&2
    exit 1
  fi
done

python3 - "$RUN_DIR" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
bad_columns = {
    "fn_count",
    "false_negative_count",
    "false_negative_count_total",
    "false_negative_total",
    "mismatch_count",
    "equality_failure_count",
    "result_mismatch_count",
}
errors = []


def as_float(value):
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


status_path = root / "result_status.tsv"
with status_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row.get("status") == "failed":
            errors.append(f"{status_path}: {row.get('result_id')} status=failed")

for path in root.rglob("*"):
    if path.suffix not in {".tsv", ".csv"}:
        continue
    if path.name in {"commands.tsv", "summary_all.tsv"}:
        continue
    delimiter = "," if path.suffix == ".csv" else "\t"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row_index, row in enumerate(reader, start=2):
            for column in bad_columns.intersection(row.keys()):
                value = as_float(row.get(column))
                if value is not None and value > 0:
                    errors.append(f"{path}:{row_index}: {column}={row[column]}")

if errors:
    print("correctness check failed:", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    sys.exit(1)

print(f"OK: no nonzero false-negative or mismatch counters found under {root}")
PY
