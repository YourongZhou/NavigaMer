#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="quick"
LABEL="$(date +%Y%m%d_%H%M%S)"
OUT_DIR=""
REF_INPUT="ACGTGCACTGATTGCATAGCTACGAAAAAAAAAAAACCCCGGGGCCCCCCCCCGGGCCCC"
INDEX_PATH=""
THREADS="$(nproc 2>/dev/null || echo 1)"
DRY_RUN=0
SKIP_BUILD=0
RUN_EXTERNAL_BASELINES=0
NAVIGAMER_BIN="$ROOT_DIR/navigamer_cpp/navigamer"

usage() {
  cat <<'USAGE'
usage: scripts/run_results_1to6_pipeline.sh [options]

Options:
  --quick                         Run a small smoke/contract benchmark matrix (default).
  --full                          Run a paper-scale-oriented matrix; provide --ref and preferably --index.
  --label LABEL                   Label embedded in the default output directory name.
  --out-dir DIR                   Output directory for all artifacts.
  --ref FASTA_OR_LITERAL          Reference FASTA path or literal DNA sequence.
  --index NAVIDX                  Persisted NavigaMer index for Result 6 locality runs.
  --threads N                     Thread count recorded/passed to supported commands.
  --navigamer-bin PATH            NavigaMer binary path.
  --skip-build                    Do not build C++ binaries before running.
  --run-external-baselines        Record/run external baseline comparison hook commands.
  --dry-run                       Record commands and write summaries without executing benchmarks.
  -h, --help                      Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      MODE="quick"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --ref)
      REF_INPUT="$2"
      shift 2
      ;;
    --index)
      INDEX_PATH="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --navigamer-bin)
      NAVIGAMER_BIN="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --run-external-baselines)
      RUN_EXTERNAL_BASELINES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$ROOT_DIR/benchmark_runs/${LABEL}_${MODE}"
fi

mkdir -p \
  "$OUT_DIR/logs" \
  "$OUT_DIR/inputs" \
  "$OUT_DIR/result1_correctness" \
  "$OUT_DIR/result2_anchor" \
  "$OUT_DIR/result3_hierarchy" \
  "$OUT_DIR/result4_corner" \
  "$OUT_DIR/result5_candidates" \
  "$OUT_DIR/result6_locality"

COMMANDS_TSV="$OUT_DIR/commands.tsv"
printf 'timestamp\tstep\tcommand\n' > "$COMMANDS_TSV"

quote_cmd() {
  local quoted=""
  local part
  for part in "$@"; do
    printf -v quoted '%s%q ' "$quoted" "$part"
  done
  printf '%s' "${quoted% }"
}

record_command() {
  local step="$1"
  shift
  printf '%s\t%s\t%s\n' "$(date -Iseconds)" "$step" "$(quote_cmd "$@")" >> "$COMMANDS_TSV"
}

run_cmd() {
  local step="$1"
  shift
  record_command "$step" "$@"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN $step: $(quote_cmd "$@")"
    return 0
  fi
  echo "RUN $step"
  "$@" > "$OUT_DIR/logs/${step}.log" 2>&1
}

write_status_note() {
  local path="$1"
  local result="$2"
  local status="$3"
  local evidence="$4"
  printf 'result_id\tstatus\tevidence\n' > "$path"
  printf '%s\t%s\t%s\n' "$result" "$status" "$evidence" >> "$path"
}

REF_FOR_CPP="$REF_INPUT"
REF_FOR_PY="$REF_INPUT"
if [[ -f "$REF_INPUT" ]]; then
  REF_FOR_PY="$REF_INPUT"
else
  REF_FOR_PY="$OUT_DIR/inputs/reference.fa"
  {
    printf '>pipeline_reference\n'
    printf '%s\n' "$REF_INPUT"
  } > "$REF_FOR_PY"
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_cmd make_navigamer make -C "$ROOT_DIR/navigamer_cpp" "-j$THREADS"
  run_cmd make_test_recall make -C "$ROOT_DIR/navigamer_cpp" test_recall
  run_cmd make_test_distance_bound make -C "$ROOT_DIR/navigamer_cpp" test_distance_bound
  run_cmd make_test_query_benchmark make -C "$ROOT_DIR/navigamer_cpp" test_query_benchmark
fi

run_cmd test_recall "$ROOT_DIR/navigamer_cpp/test_recall"
run_cmd test_distance_bound "$ROOT_DIR/navigamer_cpp/test_distance_bound"
run_cmd test_query_benchmark_gate "$ROOT_DIR/navigamer_cpp/test_query_benchmark_gate"

if [[ "$MODE" == "quick" ]]; then
  QB_WINDOW=12
  QB_STRIDE=4
  QB_LENGTH=12
  QB_TOL=1
  QB_COUNT=1
  QB_WARMUP=1
  QB_MEASURED=2
  ANCHOR_QUERY_COUNT=8
  LAYER_QUERIES=2
  LOCALITY_COUNT=8
else
  QB_WINDOW=150
  QB_STRIDE=1
  QB_LENGTH=150
  QB_TOL=5
  QB_COUNT=8
  QB_WARMUP=2
  QB_MEASURED=10
  ANCHOR_QUERY_COUNT=64
  LAYER_QUERIES=32
  LOCALITY_COUNT=256
fi

run_cmd result1_query_benchmark "$NAVIGAMER_BIN" query-benchmark \
  --ref "$REF_FOR_CPP" \
  --window "$QB_WINDOW" \
  --stride "$QB_STRIDE" \
  --query-length "$QB_LENGTH" \
  --tolerance "$QB_TOL" \
  --threads "$THREADS" \
  --queries-per-class "$QB_COUNT" \
  --warmup-iterations "$QB_WARMUP" \
  --measured-iterations "$QB_MEASURED" \
  --cold-cache-bytes 0 \
  --query-benchmark-ablations 1 \
  --proximal-oracle 1 \
  --proximal-oracle-k 1,2,4 \
  --out "$OUT_DIR/result1_correctness/query_benchmark_detail.tsv" \
  --summary-out "$OUT_DIR/result1_correctness/query_benchmark_summary.tsv" \
  --json-out "$OUT_DIR/result1_correctness/query_benchmark_summary.json"

run_cmd result2_anchor_ablation python3 "$ROOT_DIR/experiments/ecoli_1p1m/scripts/anchor_ablation.py" \
  --reference "$REF_FOR_PY" \
  --out "$OUT_DIR/result2_anchor/anchor_selection.tsv" \
  --window-length "$QB_WINDOW" \
  --stride "$QB_STRIDE" \
  --query-count "$ANCHOR_QUERY_COUNT" \
  --query-edits "$QB_TOL" \
  --tolerance "$QB_TOL" \
  --anchor-counts 1,2,4 \
  --strategies random,proximal,actual \
  --seed 20260703

run_cmd result3_layer_radius "$NAVIGAMER_BIN" layer-radius-experiment \
  --ref "$REF_FOR_CPP" \
  --length "$QB_LENGTH" \
  --tolerance "$QB_TOL" \
  --query-edits "$QB_TOL" \
  --queries-per-cell "$LAYER_QUERIES" \
  --stride "$QB_STRIDE" \
  --seed 20260703 \
  --L-values 2,3 \
  --r-leaf-values 2,4 \
  --alpha-values 0.5,0.7 \
  --out "$OUT_DIR/result3_hierarchy/layer_radius.csv"

write_status_note \
  "$OUT_DIR/result4_corner/corner_status.tsv" \
  "Result 4" \
  "preliminary" \
  "query-benchmark covers deterministic no_hit, low_complexity, single_hit, and multi_hit classes; add a dedicated adversarial corner runner for final claims"

if [[ "$RUN_EXTERNAL_BASELINES" -eq 1 ]]; then
  run_cmd make_candidate_tool make -C "$ROOT_DIR/experiments/ecoli_1p1m" candidate_tool
  write_status_note \
    "$OUT_DIR/result5_candidates/external_baseline_hook.tsv" \
    "Result 5" \
    "preliminary" \
    "candidate_tool built; add dataset-specific candidate generation and candidate-verify commands for each baseline"
else
  write_status_note \
    "$OUT_DIR/result5_candidates/external_baseline_hook.tsv" \
    "Result 5" \
    "skipped" \
    "external baselines were not requested; pass --run-external-baselines to enable the hook"
fi

if [[ -n "$INDEX_PATH" ]]; then
  if [[ "$MODE" == "quick" ]]; then
    LOCALITY_DATASETS="same_template,random_windows"
    LOCALITY_SCHEDULES="original,source-oracle"
  else
    LOCALITY_DATASETS="source_sorted_mutated_tau5,source_sorted_mutated_tau8,real_dup_1x,real_dup_4x,real_dup_16x"
    LOCALITY_SCHEDULES="original,random,minimizer,qgram-signature,router-signature,source-oracle"
  fi
  run_cmd result6_locality "$NAVIGAMER_BIN" locality-benchmark \
    --index "$INDEX_PATH" \
    --ref "$REF_FOR_CPP" \
    --out "$OUT_DIR/result6_locality/locality_summary.tsv" \
    --query-count "$LOCALITY_COUNT" \
    --query-length "$QB_LENGTH" \
    --query-edits "$QB_TOL" \
    --tolerance "$QB_TOL" \
    --seed 20260703 \
    --locality-profiles baseline,path_reuse,optimized \
    --locality-datasets "$LOCALITY_DATASETS" \
    --batch-schedules "$LOCALITY_SCHEDULES"
else
  if [[ "$MODE" == "quick" ]]; then
    REPORT_LOCALITY_DATASETS="same_template"
  else
    REPORT_LOCALITY_DATASETS="same_template,random_windows"
  fi
  run_cmd result6_locality_report "$NAVIGAMER_BIN" query-locality-report \
    --ref "$REF_FOR_CPP" \
    --out-dir "$OUT_DIR/result6_locality/report" \
    --window "$QB_WINDOW" \
    --stride "$QB_STRIDE" \
    --query-count "$LOCALITY_COUNT" \
    --query-length "$QB_LENGTH" \
    --query-edits "$QB_TOL" \
    --tolerance "$QB_TOL" \
    --seed 20260703 \
    --locality-profiles baseline,path_reuse,optimized \
    --locality-datasets "$REPORT_LOCALITY_DATASETS" \
    --batch-schedules original,source-oracle
fi

python3 "$ROOT_DIR/scripts/summarize_results_1to6.py" "$OUT_DIR"
bash "$ROOT_DIR/scripts/check_results_1to6.sh" "$OUT_DIR"

echo "Benchmark pipeline complete: $OUT_DIR"
