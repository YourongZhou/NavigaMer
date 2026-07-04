#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

RUN_DIR="$TMP_DIR/run"
mkdir -p \
  "$RUN_DIR/logs" \
  "$RUN_DIR/result1_correctness" \
  "$RUN_DIR/result2_anchor" \
  "$RUN_DIR/result3_hierarchy" \
  "$RUN_DIR/result5_candidates" \
  "$RUN_DIR/result6_locality"

printf 'PASS recall guard\n' > "$RUN_DIR/logs/test_recall.log"
printf 'PASS distance bound guard\n' > "$RUN_DIR/logs/test_distance_bound.log"

cat > "$RUN_DIR/result1_correctness/query_benchmark_summary.tsv" <<'TSV'
profile	class	query_count	fn_count	mismatch_count	mean_query_ms	p95_query_ms
optimized	substitution	4	0	0	1.5	2.0
optimized	insertion	4	0	0	1.7	2.2
TSV

cat > "$RUN_DIR/result2_anchor/anchor_selection.tsv" <<'TSV'
strategy	anchor_count	query_count	window_count	mean_envelope_size	false_negative_count_total	source_recovery_rate	pruning_ratio
proximal	2	8	100	12.5	0	1.0	0.875
random	2	8	100	48.0	0	1.0	0.520
TSV

cat > "$RUN_DIR/result3_hierarchy/layer_radius.csv" <<'CSV'
L,r_leaf,alpha,query_count,false_negative_count,mean_leaf_verify
3,80,0.7,8,0,120
CSV

cat > "$RUN_DIR/result5_candidates/candidate_summary.tsv" <<'TSV'
method	query_count	false_negative_count_total	candidate_count	mean_query_ms
navigamer	8	0	120	1.8
qgram-safe	8	0	400	1.2
TSV

cat > "$RUN_DIR/result6_locality/locality_summary.tsv" <<'TSV'
dataset	scenario	profile	batch_schedule	query_count	fn_count	mismatch_count	mean_query_ms	p95_query_ms	mean_leaf_verify	reuse_hit_rate
source_sorted_mutated_tau5	random_windows	optimized	source-oracle	8	0	0	2.3	3.1	88	0.6
TSV

python3 "$ROOT_DIR/scripts/summarize_results_1to6.py" "$RUN_DIR"

test -s "$RUN_DIR/summary_all.tsv"
test -s "$RUN_DIR/result_status.tsv"
test -s "$RUN_DIR/report.md"

python3 - "$RUN_DIR/result_status.tsv" <<'PY'
import csv
import sys

rows = {
    row["result_id"]: row
    for row in csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t")
}
assert rows["Result 1"]["status"] == "supported", rows["Result 1"]
assert rows["Result 2"]["status"] == "supported", rows["Result 2"]
assert rows["Result 5"]["status"] == "mixed", rows["Result 5"]
assert rows["Result 6"]["status"] == "supported", rows["Result 6"]
PY

bash "$ROOT_DIR/scripts/check_results_1to6.sh" "$RUN_DIR"

BAD_DIR="$TMP_DIR/bad"
cp -R "$RUN_DIR" "$BAD_DIR"
cat > "$BAD_DIR/result6_locality/locality_summary.tsv" <<'TSV'
dataset	scenario	profile	batch_schedule	query_count	fn_count	mismatch_count
source_sorted_mutated_tau5	random_windows	optimized	source-oracle	8	1	0
TSV
if bash "$ROOT_DIR/scripts/check_results_1to6.sh" "$BAD_DIR" >/dev/null 2>&1; then
  echo "checker should fail on nonzero fn_count" >&2
  exit 1
fi

DRY_DIR="$TMP_DIR/dry"
bash "$ROOT_DIR/scripts/run_results_1to6_pipeline.sh" \
  --quick \
  --dry-run \
  --label contract \
  --out-dir "$DRY_DIR" \
  --ref ACGTACGTACGTACGTACGTACGTACGTACGT \
  --index "$TMP_DIR/example.navidx" \
  --threads 2

test -s "$DRY_DIR/commands.tsv"
grep -q 'query-benchmark' "$DRY_DIR/commands.tsv"
grep -q 'locality-benchmark' "$DRY_DIR/commands.tsv"
test -s "$DRY_DIR/result_status.tsv"

DEFAULT_DRY_DIR="$TMP_DIR/default_dry"
bash "$ROOT_DIR/scripts/run_results_1to6_pipeline.sh" \
  --quick \
  --dry-run \
  --skip-build \
  --out-dir "$DEFAULT_DRY_DIR" \
  --threads 2
grep -q 'ACGTGCACTGATTGCATAGCTACGAAAAAAAAAAAACCCCGGGGCCCCCCCCCGGGCCCC' "$DEFAULT_DRY_DIR/commands.tsv"
grep -q -- '--locality-datasets same_template ' "$DEFAULT_DRY_DIR/commands.tsv"
if grep -q 'same_template\\,random_windows' "$DEFAULT_DRY_DIR/commands.tsv"; then
  echo "default no-index quick smoke should not include random_windows" >&2
  exit 1
fi
