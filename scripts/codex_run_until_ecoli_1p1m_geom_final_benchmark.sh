#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/codex_run_until_ecoli_1p1m_geom_final_benchmark.sh [options]

Run the repository Codex automation loop with the E. coli 1.1M geom final
benchmark goal: rerun and summarize geom_L4_leaf5_a0p5 contained-path-reuse
benchmarks, then compare hot query time against exact-verified randstrobe and
spaced-seed baselines.

Options are passed through to scripts/codex_autorun.sh after this wrapper's
own setup. Common examples:

  scripts/codex_run_until_ecoli_1p1m_geom_final_benchmark.sh --sleep 5h10m
  scripts/codex_run_until_ecoli_1p1m_geom_final_benchmark.sh --sleep 0s --max-attempts 1

This wrapper:
  1. copies CODEX_ECOLI_1P1M_GEOM_FINAL_BENCHMARK_GOAL.md into CODEX_GOAL.md;
  2. reopens CODEX_PROGRESS.md by setting State to in_progress;
  3. pins Codex to gpt-5.5 and high reasoning effort unless the caller already
     supplied explicit CODEX_* overrides;
  4. invokes scripts/codex_autorun.sh with CODEX_GOAL.md.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
GOAL_TEMPLATE="$REPO/CODEX_ECOLI_1P1M_GEOM_FINAL_BENCHMARK_GOAL.md"
GOAL_FILE="$REPO/CODEX_GOAL.md"
PROGRESS_FILE="$REPO/CODEX_PROGRESS.md"
AUTORUN="$REPO/scripts/codex_autorun.sh"

if [[ ! -f "$GOAL_TEMPLATE" ]]; then
  printf 'error: missing goal template: %s\n' "$GOAL_TEMPLATE" >&2
  exit 66
fi
if [[ ! -f "$PROGRESS_FILE" ]]; then
  printf 'error: missing progress file: %s\n' "$PROGRESS_FILE" >&2
  exit 66
fi
if [[ ! -x "$AUTORUN" ]]; then
  printf 'error: missing executable autorun script: %s\n' "$AUTORUN" >&2
  exit 66
fi

cp "$GOAL_TEMPLATE" "$GOAL_FILE"

if grep -qE '^[[:space:]]*-[[:space:]]*State:' "$PROGRESS_FILE"; then
  sed -i '0,/^[[:space:]]*-[[:space:]]*State:.*$/s//- State: in_progress/' "$PROGRESS_FILE"
else
  printf '\n## Current Status\n\n- State: in_progress\n' >> "$PROGRESS_FILE"
fi

if grep -qE '^[[:space:]]*-[[:space:]]*Last updated:' "$PROGRESS_FILE"; then
  sed -i "0,/^[[:space:]]*-[[:space:]]*Last updated:.*$/s//- Last updated: $(date +%F)/" "$PROGRESS_FILE"
fi

export CODEX_MODEL_OVERRIDE="${CODEX_MODEL_OVERRIDE:-gpt-5.5}"
export CODEX_MODEL_REASONING_EFFORT="${CODEX_MODEL_REASONING_EFFORT:-high}"

exec "$AUTORUN" \
  --repo "$REPO" \
  --prompt-file "$GOAL_FILE" \
  --progress-file "$PROGRESS_FILE" \
  "$@"
