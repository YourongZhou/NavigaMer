#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/codex_wait_strobe_spaced_then_real_locality_v2.sh [options] [-- v2-options]

Wait for the current E. coli 1.1M strobemer/spaced-seed autorun to finish,
write a handoff file, then launch the E. coli 1.1M real-read locality V2 goal.

Options:
  --pid PID             Process id to wait for. May be repeated.
  --pid-file PATH       File containing one PID per line. Default:
                        .codex_logs/ecoli_1p1m_strobe_spaced_autorun.pid.
  --poll SECONDS        Poll interval while waiting. Default: 30.
  --require-complete    Require CODEX_PROGRESS.md to be complete before
                        launching V2.
  --handoff PATH        Handoff file to write. Default:
                        CODEX_ECOLI_STROBE_TO_REAL_LOCALITY_HANDOFF.md.
  -h, --help            Show this help.

Everything after `--` is passed to
scripts/codex_run_until_ecoli_1p1m_real_locality_v2.sh.

Examples:
  scripts/codex_wait_strobe_spaced_then_real_locality_v2.sh --pid 12345 -- --sleep 5h10m
  scripts/codex_wait_strobe_spaced_then_real_locality_v2.sh --poll 60 -- --sleep 0s --max-attempts 1
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
POLL_SECONDS=30
REQUIRE_COMPLETE=0
DEFAULT_PID_FILE="$REPO/.codex_logs/ecoli_1p1m_strobe_spaced_autorun.pid"
HANDOFF_FILE="$REPO/CODEX_ECOLI_STROBE_TO_REAL_LOCALITY_HANDOFF.md"
PIDS=()
V2_ARGS=()
PID_FILE_SUPPLIED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      V2_ARGS=("$@")
      break
      ;;
    --pid)
      [[ $# -ge 2 ]] || { printf 'error: --pid requires a value\n' >&2; exit 64; }
      PIDS+=("$2")
      shift 2
      ;;
    --pid-file)
      [[ $# -ge 2 ]] || { printf 'error: --pid-file requires a path\n' >&2; exit 64; }
      DEFAULT_PID_FILE="$2"
      PID_FILE_SUPPLIED=1
      shift 2
      ;;
    --poll)
      [[ $# -ge 2 ]] || { printf 'error: --poll requires seconds\n' >&2; exit 64; }
      [[ "$2" =~ ^[0-9]+$ && "$2" -gt 0 ]] || {
        printf 'error: --poll must be a positive integer\n' >&2
        exit 64
      }
      POLL_SECONDS="$2"
      shift 2
      ;;
    --require-complete)
      REQUIRE_COMPLETE=1
      shift
      ;;
    --handoff)
      [[ $# -ge 2 ]] || { printf 'error: --handoff requires a path\n' >&2; exit 64; }
      HANDOFF_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

is_pid() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

process_alive() {
  local pid="$1"
  is_pid "$pid" && kill -0 "$pid" 2>/dev/null
}

read_pid_file() {
  local path="$1"
  [[ -f "$path" ]] || return 0
  while IFS= read -r pid; do
    [[ -z "$pid" || "$pid" =~ ^# ]] && continue
    PIDS+=("$pid")
  done < "$path"
}

dedupe_pids() {
  printf '%s\n' "$@" | awk 'NF && !seen[$0]++'
}

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  read_pid_file "$DEFAULT_PID_FILE"
fi

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  mapfile -t PIDS < <(dedupe_pids "${PIDS[@]}")
fi

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  if [[ "$PID_FILE_SUPPLIED" -eq 1 ]]; then
    printf 'error: no pid found in supplied pid file: %s\n' "$DEFAULT_PID_FILE" >&2
    exit 66
  fi
  printf 'no strobemer/spaced-seed autorun pid found; launching V2 now\n'
else
  printf 'waiting for strobemer/spaced-seed autorun pid(s): %s\n' "${PIDS[*]}"
  while true; do
    alive=()
    for pid in "${PIDS[@]}"; do
      if process_alive "$pid"; then
        alive+=("$pid")
      fi
    done
    if [[ "${#alive[@]}" -eq 0 ]]; then
      break
    fi
    printf '[%s] still running: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${alive[*]}"
    sleep "$POLL_SECONDS"
  done
fi

progress_state="$(
  sed -n 's/^[[:space:]]*-[[:space:]]*State:[[:space:]]*//p' "$REPO/CODEX_PROGRESS.md" |
    head -n 1
)"

if [[ "$REQUIRE_COMPLETE" -eq 1 &&
      ! "$progress_state" =~ ^(complete|completed|done|finished)$ ]]; then
  printf 'error: strobemer/spaced-seed progress state is not complete: %s\n' "${progress_state:-unknown}" >&2
  exit 65
fi

latest_log=""
if [[ -L "$REPO/.codex_logs/latest.log" || -f "$REPO/.codex_logs/latest.log" ]]; then
  latest_log="$REPO/.codex_logs/latest.log"
elif [[ -L "$REPO/.codex/logs/latest.log" || -f "$REPO/.codex/logs/latest.log" ]]; then
  latest_log="$REPO/.codex/logs/latest.log"
fi

{
  printf '# Codex E. coli Strobemer/Spaced-Seed To Real-Locality V2 Handoff\n\n'
  printf '%s\n' "- generated_at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  printf '%s\n' "- waited_pids: ${PIDS[*]:-none}"
  printf '%s\n' "- progress_state_after_strobe_spaced_goal: ${progress_state:-unknown}"
  printf '%s\n' "- next_goal: CODEX_ECOLI_1P1M_REAL_LOCALITY_V2_GOAL.md"
  if [[ -n "$latest_log" ]]; then
    printf '%s\n\n' "- latest_log: $latest_log"
    printf '## Latest Strobemer/Spaced-Seed Log Tail\n\n```text\n'
    tail -n 160 "$latest_log" || true
    printf '\n```\n'
  else
    printf '%s\n' '- latest_log: none found'
  fi
  printf '\n## Git Status Before V2 Goal\n\n```text\n'
  git -C "$REPO" status --short
  printf '```\n'
} > "$HANDOFF_FILE"

printf 'wrote V2 handoff: %s\n' "$HANDOFF_FILE"

exec "$REPO/scripts/codex_run_until_ecoli_1p1m_real_locality_v2.sh" "${V2_ARGS[@]}"

