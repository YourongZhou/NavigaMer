#!/usr/bin/env bash
set -euo pipefail

RUN_AT="2026-06-29 04:40:00"
SESSION_ID="019eee59-afba-7fc2-920e-b9dc1c84f89e"
WORKTREE="/home/luting/projects/AnchorMapping/NavigaMer/.worktrees/ecoli-comparison"
OUT_ROOT="/home/luting/projects/AnchorMapping/NavigaMer/.tmp_experiments/ecoli_1p1m_formal/deferred_codex_resume_0440"
PROMPT="继续当前目标：补需要的 Results，先用 1.1M 的来。请从当前 worktree 和现有结果状态检查开始，继续推进 Result 1/2/3/4/5/6；优先刷新验证日志、扩展 Result 4 path/corner 样本、推进 Result 2/3 最小证明实验，并把结果写回 results_summary_v1。"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: schedule_codex_resume.sh [options]

Options:
  --dry-run              Print the planned command without sleeping or writing state.
  --run-at <time|now>    Local time accepted by date -d, default 2026-06-29 04:40:00.
  --session-id <id>      Codex session id to resume.
  --worktree <path>      Working directory to cd into before resuming.
  --out-root <path>      Directory for logs and STATUS.
  --prompt <text>        Prompt sent to codex exec resume.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run-at)
      RUN_AT=${2:?missing value for --run-at}
      shift 2
      ;;
    --session-id)
      SESSION_ID=${2:?missing value for --session-id}
      shift 2
      ;;
    --worktree)
      WORKTREE=${2:?missing value for --worktree}
      shift 2
      ;;
    --out-root)
      OUT_ROOT=${2:?missing value for --out-root}
      shift 2
      ;;
    --prompt)
      PROMPT=${2:?missing value for --prompt}
      shift 2
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

if [[ "$RUN_AT" == "now" ]]; then
  target_epoch=$(date +%s)
else
  target_epoch=$(date -d "$RUN_AT" +%s)
fi
now_epoch=$(date +%s)
delay_seconds=$(( target_epoch - now_epoch ))
if (( delay_seconds < 0 )); then
  delay_seconds=0
fi

command_preview=(
  codex exec resume
  --dangerously-bypass-approvals-and-sandbox
  "$SESSION_ID"
  "$PROMPT"
)

if (( DRY_RUN )); then
  echo "mode=dry-run"
  echo "run_at=$RUN_AT"
  echo "delay_seconds=$delay_seconds"
  echo "session_id=$SESSION_ID"
  echo "worktree=$WORKTREE"
  echo "out_root=$OUT_ROOT"
  echo "prompt=$PROMPT"
  printf 'command='
  printf '%q ' "${command_preview[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUT_ROOT"
status="$OUT_ROOT/STATUS"
log="$OUT_ROOT/codex_resume.log"
last_message="$OUT_ROOT/last_message.txt"

{
  echo "scheduled_at=$(date --iso-8601=seconds)"
  echo "run_at=$RUN_AT"
  echo "delay_seconds=$delay_seconds"
  echo "session_id=$SESSION_ID"
  echo "worktree=$WORKTREE"
  echo "out_root=$OUT_ROOT"
} > "$status"

if (( delay_seconds > 0 )); then
  sleep "$delay_seconds"
fi

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "command=${command_preview[*]}"
} >> "$status"

cd "$WORKTREE"
set +e
codex exec resume \
  --dangerously-bypass-approvals-and-sandbox \
  -o "$last_message" \
  "$SESSION_ID" \
  "$PROMPT" > "$log" 2>&1
exit_code=$?
set -e

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_code=$exit_code"
  echo "log=$log"
  echo "last_message=$last_message"
} >> "$status"

exit "$exit_code"
