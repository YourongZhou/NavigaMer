#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/codex_autorun.sh [options]

Run Codex against a repo-local goal file and automatically resume after
quota-like interruptions.

Options:
  --repo PATH          Repository path. Default: current working directory.
  --prompt-file PATH   Goal prompt file. Default: CODEX_GOAL.md in the repo.
  --sleep DURATION     Sleep duration after quota/rate limit. Default: 5h10m.
  --log-dir PATH       Log directory. Default: .codex/logs in the repo.
  -h, --help           Show this help.

Examples:
  scripts/codex_autorun.sh
  scripts/codex_autorun.sh --repo /home/luting/projects/AnchorMapping/NavigaMer
  scripts/codex_autorun.sh --sleep 5h
USAGE
}

REPO="$(pwd -P)"
PROMPT_FILE=""
SLEEP_DURATION="5h10m"
LOG_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      if [[ $# -lt 2 ]]; then
        printf 'error: --repo requires a path\n' >&2
        exit 64
      fi
      REPO="$2"
      shift 2
      ;;
    --prompt-file)
      if [[ $# -lt 2 ]]; then
        printf 'error: --prompt-file requires a path\n' >&2
        exit 64
      fi
      PROMPT_FILE="$2"
      shift 2
      ;;
    --sleep)
      if [[ $# -lt 2 ]]; then
        printf 'error: --sleep requires a duration\n' >&2
        exit 64
      fi
      SLEEP_DURATION="$2"
      shift 2
      ;;
    --log-dir)
      if [[ $# -lt 2 ]]; then
        printf 'error: --log-dir requires a path\n' >&2
        exit 64
      fi
      LOG_DIR="$2"
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

if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$REPO/CODEX_GOAL.md"
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$REPO/.codex/logs"
fi

CONTINUATION_PROMPT="继续当前目标。先读 CODEX_GOAL.md、CODEX_PROGRESS.md、CODEX_TEST_LOG.md、git status、git diff、最近测试输出，然后继续未完成任务。"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

contains_limit_message() {
  grep -qiE 'usage limit|rate limit|try again|resets' "$1"
}

validate_inputs() {
  if [[ ! -d "$REPO" ]]; then
    printf 'error: repository path does not exist: %s\n' "$REPO" >&2
    exit 66
  fi

  if [[ ! -f "$PROMPT_FILE" ]]; then
    printf 'error: prompt file does not exist: %s\n' "$PROMPT_FILE" >&2
    exit 66
  fi

  mkdir -p "$LOG_DIR"
}

new_log_file() {
  printf '%s/codex_run_%s_%03d.log' "$LOG_DIR" "$(timestamp)" "$ATTEMPT_COUNTER"
}

update_latest_log() {
  local log_file="$1"
  local latest="$LOG_DIR/latest.log"

  rm -f "$latest"
  if ! ln -s "$(basename "$log_file")" "$latest" 2>/dev/null; then
    printf '%s\n' "$log_file" > "$latest"
  fi
}

run_codex_exec() {
  local log_file="$1"
  local prompt

  prompt="$(cat "$PROMPT_FILE")"
  codex exec --cd "$REPO" "$prompt" 2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

run_codex_resume() {
  local log_file="$1"

  codex exec resume --last --cd "$REPO" "$CONTINUATION_PROMPT" 2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

run_codex_fresh_continuation() {
  local log_file="$1"

  codex exec --cd "$REPO" "$CONTINUATION_PROMPT" 2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

run_attempt() {
  local mode="$1"
  local log_file
  local code

  log_file="$(new_log_file)"
  update_latest_log "$log_file"
  log_line "starting Codex $mode run in $REPO" | tee -a "$log_file"

  case "$mode" in
    exec)
      run_codex_exec "$log_file"
      code=$?
      ;;
    resume)
      run_codex_resume "$log_file"
      code=$?
      ;;
    fresh-continuation)
      run_codex_fresh_continuation "$log_file"
      code=$?
      ;;
    *)
      log_line "internal error: unknown run mode: $mode" | tee -a "$log_file"
      return 70
      ;;
  esac

  log_line "Codex $mode run exited with status $code" | tee -a "$log_file"
  LAST_LOG_FILE="$log_file"
  return "$code"
}

validate_inputs

LAST_LOG_FILE=""
ATTEMPT_COUNTER=0
mode="exec"

while true; do
  ATTEMPT_COUNTER=$((ATTEMPT_COUNTER + 1))
  run_attempt "$mode"
  code=$?

  if [[ -n "$LAST_LOG_FILE" ]] && contains_limit_message "$LAST_LOG_FILE"; then
    log_line "Codex appears to have hit a usage/rate limit. Sleeping $SLEEP_DURATION before resume." | tee -a "$LAST_LOG_FILE"
    sleep "$SLEEP_DURATION"
    mode="resume"
    continue
  fi

  if [[ "$mode" == "resume" && "$code" -ne 0 ]]; then
    log_line "resume failed without a detected usage/rate limit; trying one fresh continuation." | tee -a "$LAST_LOG_FILE"
    run_attempt "fresh-continuation"
    code=$?

    if [[ -n "$LAST_LOG_FILE" ]] && contains_limit_message "$LAST_LOG_FILE"; then
      log_line "fresh continuation also hit a usage/rate limit. Sleeping $SLEEP_DURATION before resume." | tee -a "$LAST_LOG_FILE"
      sleep "$SLEEP_DURATION"
      mode="resume"
      continue
    fi
  fi

  exit "$code"
done
