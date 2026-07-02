#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/codex_wait_diagnostic_then_fast_no_fn.sh [options] [-- fast-options]

Wait for an existing diagnostic Codex autorun to finish, write a short
diagnostic handoff file, then launch the strict fast/no-FN automation goal.

Options:
  --pid PID             Process id to wait for. May be repeated.
  --pid-file PATH       File containing one PID per line.
  --discover            Auto-discover running scripts/codex_autorun.sh jobs in
                        this repository when no --pid is supplied. Default.
  --no-discover         Do not auto-discover; start fast goal immediately if no
                        PID is supplied.
  --poll SECONDS        Poll interval while waiting. Default: 30.
  --require-complete    Require CODEX_PROGRESS.md to be complete after the
                        diagnostic run before launching the fast goal.
  --handoff PATH        Handoff file to write. Default:
                        CODEX_DIAGNOSTIC_HANDOFF.md.
  -h, --help            Show this help.

Everything after `--` is passed to scripts/codex_run_until_fast_no_fn.sh.

Examples:
  scripts/codex_wait_diagnostic_then_fast_no_fn.sh --pid 12345 -- --sleep 5h10m
  scripts/codex_wait_diagnostic_then_fast_no_fn.sh --discover -- --sleep 0s --max-attempts 1
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd -P)"
POLL_SECONDS=30
DISCOVER=1
REQUIRE_COMPLETE=0
HANDOFF_FILE="$REPO/CODEX_DIAGNOSTIC_HANDOFF.md"
PIDS=()
FAST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      FAST_ARGS=("$@")
      break
      ;;
    --pid)
      [[ $# -ge 2 ]] || { printf 'error: --pid requires a value\n' >&2; exit 64; }
      PIDS+=("$2")
      shift 2
      ;;
    --pid-file)
      [[ $# -ge 2 ]] || { printf 'error: --pid-file requires a path\n' >&2; exit 64; }
      while IFS= read -r pid; do
        [[ -z "$pid" || "$pid" =~ ^# ]] && continue
        PIDS+=("$pid")
      done < "$2"
      shift 2
      ;;
    --discover)
      DISCOVER=1
      shift
      ;;
    --no-discover)
      DISCOVER=0
      shift
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

discover_autorun_pids() {
  local self="$$"
  pgrep -af "scripts/codex_autorun.sh --repo $REPO" 2>/dev/null |
    awk -v self="$self" '$1 != self {print $1}' || true
}

dedupe_pids() {
  printf '%s\n' "$@" | awk 'NF && !seen[$0]++'
}

if [[ "${#PIDS[@]}" -eq 0 && "$DISCOVER" -eq 1 ]]; then
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && PIDS+=("$pid")
  done < <(discover_autorun_pids)
fi

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  mapfile -t PIDS < <(dedupe_pids "${PIDS[@]}")
fi

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  printf 'waiting for diagnostic autorun pid(s): %s\n' "${PIDS[*]}"
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
else
  printf 'no diagnostic autorun pid found; preparing handoff and launching fast goal now\n'
fi

progress_state="$(
  sed -n 's/^[[:space:]]*-[[:space:]]*State:[[:space:]]*//p' "$REPO/CODEX_PROGRESS.md" |
    head -n 1
)"

if [[ "$REQUIRE_COMPLETE" -eq 1 &&
      ! "$progress_state" =~ ^(complete|completed|done|finished)$ ]]; then
  printf 'error: diagnostic progress state is not complete: %s\n' "${progress_state:-unknown}" >&2
  exit 65
fi

latest_log=""
if [[ -L "$REPO/.codex_logs/latest.log" || -f "$REPO/.codex_logs/latest.log" ]]; then
  latest_log="$REPO/.codex_logs/latest.log"
elif [[ -L "$REPO/.codex/logs/latest.log" || -f "$REPO/.codex/logs/latest.log" ]]; then
  latest_log="$REPO/.codex/logs/latest.log"
fi

{
  printf '# Codex Diagnostic Handoff\n\n'
  printf '%s\n' "- generated_at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  printf '%s\n' "- waited_pids: ${PIDS[*]:-none}"
  printf '%s\n' "- progress_state_after_diagnostic: ${progress_state:-unknown}"
  if [[ -n "$latest_log" ]]; then
    printf '%s\n\n' "- latest_log: $latest_log"
    printf '## Latest Diagnostic Log Tail\n\n```text\n'
    tail -n 120 "$latest_log" || true
    printf '\n```\n'
  else
    printf '%s\n' '- latest_log: none found'
  fi
  printf '\n## Git Status Before Fast Goal\n\n```text\n'
  git -C "$REPO" status --short
  printf '```\n'
} > "$HANDOFF_FILE"

printf 'wrote diagnostic handoff: %s\n' "$HANDOFF_FILE"

exec "$REPO/scripts/codex_run_until_fast_no_fn.sh" "${FAST_ARGS[@]}"
