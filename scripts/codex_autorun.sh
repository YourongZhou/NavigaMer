#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/codex_autorun.sh [options]

Run Codex against a repo-local goal file and automatically continue until the
repo-local progress file marks the goal complete. Quota-like interruptions are
resumed after sleeping.

Options:
  --repo PATH          Repository path. Default: current working directory.
  --prompt-file PATH   Goal prompt file. Default: CODEX_GOAL.md in the repo.
  --progress-file PATH Progress file checked for completion. Default:
                       CODEX_PROGRESS.md in the repo.
  --sleep DURATION     Sleep duration after quota/rate limit. Default: 5h10m.
  --log-dir PATH       Log directory. Default: .codex/logs in the repo, or
                       .codex_logs when .codex is a file.
  --max-attempts N     Stop after N Codex attempts. Default: 0, unlimited.
  -h, --help           Show this help.

Environment overrides:
  CODEX_MODEL_OVERRIDE          Passed as: --model VALUE
  CODEX_FALLBACK_MODEL          Model to use after primary usage limits.
                                Default: gpt-5.3-codex-spark. Set to an empty
                                string to disable model fallback.
  CODEX_MODEL_REASONING_EFFORT  Passed as: -c model_reasoning_effort="VALUE"
  CODEX_SERVICE_TIER            Passed as: -c service_tier="VALUE"

History note:
  This script uses non-interactive Codex exec sessions. To view or resume these
  sessions in the interactive history picker, run:
    codex resume --include-non-interactive

Examples:
  scripts/codex_autorun.sh
  scripts/codex_autorun.sh --repo /home/luting/projects/AnchorMapping/NavigaMer
  scripts/codex_autorun.sh --sleep 5h
USAGE
}

REPO="$(pwd -P)"
PROMPT_FILE=""
PROGRESS_FILE=""
SLEEP_DURATION="5h10m"
LOG_DIR=""
LOG_DIR_EXPLICIT=0
MAX_ATTEMPTS=0

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
    --progress-file)
      if [[ $# -lt 2 ]]; then
        printf 'error: --progress-file requires a path\n' >&2
        exit 64
      fi
      PROGRESS_FILE="$2"
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
      LOG_DIR_EXPLICIT=1
      shift 2
      ;;
    --max-attempts)
      if [[ $# -lt 2 ]]; then
        printf 'error: --max-attempts requires a number\n' >&2
        exit 64
      fi
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        printf 'error: --max-attempts must be a non-negative integer\n' >&2
        exit 64
      fi
      MAX_ATTEMPTS="$2"
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

if [[ -z "$PROGRESS_FILE" ]]; then
  PROGRESS_FILE="$REPO/CODEX_PROGRESS.md"
fi

if [[ -z "$LOG_DIR" ]]; then
  if [[ -e "$REPO/.codex" && ! -d "$REPO/.codex" ]]; then
    LOG_DIR="$REPO/.codex_logs"
  else
    LOG_DIR="$REPO/.codex/logs"
  fi
fi

CONTINUATION_PROMPT="继续当前目标。先读 CODEX_GOAL.md、CODEX_PROGRESS.md、CODEX_TEST_LOG.md、git status、git diff、最近测试输出，然后继续未完成任务。只有当 CODEX_GOAL.md 的全部目标都完成时，才把 CODEX_PROGRESS.md 的 State 精确写成 complete。"
PRIMARY_MODEL="${CODEX_MODEL_OVERRIDE:-}"
FALLBACK_MODEL="${CODEX_FALLBACK_MODEL-gpt-5.3-codex-spark}"
ACTIVE_MODEL="$PRIMARY_MODEL"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

log_line() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

contains_limit_message() {
  grep -qiE 'usage limit|rate limit|try again|resets' "$1"
}

codex_exec_args() {
  local args=(exec --cd "$REPO")

  if [[ -n "$ACTIVE_MODEL" ]]; then
    args+=(--model "$ACTIVE_MODEL")
  fi
  if [[ -n "${CODEX_MODEL_REASONING_EFFORT:-}" ]]; then
    args+=(-c "model_reasoning_effort=\"${CODEX_MODEL_REASONING_EFFORT}\"")
  fi
  if [[ -n "${CODEX_SERVICE_TIER:-}" ]]; then
    args+=(-c "service_tier=\"${CODEX_SERVICE_TIER}\"")
  fi

  printf '%s\0' "${args[@]}"
}

run_codex() {
  local -a args=()
  local arg

  while IFS= read -r -d '' arg; do
    args+=("$arg")
  done < <(codex_exec_args)

  codex "${args[@]}" "$@"
}

goal_is_complete() {
  [[ -f "$PROGRESS_FILE" ]] &&
    grep -qiE '^[[:space:]]*-[[:space:]]*State:[[:space:]]*(complete|completed|done|finished)[[:space:]]*$' "$PROGRESS_FILE"
}

active_model_label() {
  if [[ -n "$ACTIVE_MODEL" ]]; then
    printf '%s' "$ACTIVE_MODEL"
  else
    printf 'configured default'
  fi
}

can_switch_to_fallback_model() {
  [[ -n "$FALLBACK_MODEL" && "$ACTIVE_MODEL" != "$FALLBACK_MODEL" ]]
}

switch_to_fallback_model() {
  ACTIVE_MODEL="$FALLBACK_MODEL"
}

switch_to_primary_model() {
  ACTIVE_MODEL="$PRIMARY_MODEL"
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

  if [[ ! -f "$PROGRESS_FILE" ]]; then
    printf 'error: progress file does not exist: %s\n' "$PROGRESS_FILE" >&2
    exit 66
  fi

  if ! mkdir -p "$LOG_DIR"; then
    if [[ "$LOG_DIR_EXPLICIT" -eq 0 &&
          "$LOG_DIR" == "$REPO/.codex/logs" &&
          -e "$REPO/.codex" &&
          ! -d "$REPO/.codex" ]]; then
      LOG_DIR="$REPO/.codex_logs"
      mkdir -p "$LOG_DIR" || {
        printf 'error: failed to create fallback log directory: %s\n' "$LOG_DIR" >&2
        exit 73
      }
    else
      printf 'error: failed to create log directory: %s\n' "$LOG_DIR" >&2
      exit 73
    fi
  fi
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
  run_codex "$prompt" 2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

run_codex_resume() {
  local log_file="$1"

  run_codex resume --last "$CONTINUATION_PROMPT" 2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

run_codex_fresh_continuation() {
  local log_file="$1"

  run_codex "$CONTINUATION_PROMPT" 2>&1 | tee -a "$log_file"
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

if goal_is_complete; then
  log_line "progress file already marks goal complete; nothing to run."
  exit 0
fi

while true; do
  if [[ "$MAX_ATTEMPTS" -gt 0 && "$ATTEMPT_COUNTER" -ge "$MAX_ATTEMPTS" ]]; then
    log_line "max attempts reached before goal was marked complete; exiting."
    exit 0
  fi

  ATTEMPT_COUNTER=$((ATTEMPT_COUNTER + 1))
  run_attempt "$mode"
  code=$?

  if [[ -n "$LAST_LOG_FILE" ]] && contains_limit_message "$LAST_LOG_FILE"; then
    if can_switch_to_fallback_model; then
      log_line "Codex appears to have hit a usage/rate limit on $(active_model_label). Switching to fallback model $FALLBACK_MODEL before resume." | tee -a "$LAST_LOG_FILE"
      switch_to_fallback_model
      mode="resume"
      continue
    fi

    log_line "Codex appears to have hit a usage/rate limit on $(active_model_label). Sleeping $SLEEP_DURATION before resume." | tee -a "$LAST_LOG_FILE"
    sleep "$SLEEP_DURATION"
    switch_to_primary_model
    mode="resume"
    continue
  fi

  if [[ "$mode" == "resume" && "$code" -ne 0 ]]; then
    log_line "resume failed without a detected usage/rate limit; trying one fresh continuation." | tee -a "$LAST_LOG_FILE"
    run_attempt "fresh-continuation"
    code=$?

    if [[ -n "$LAST_LOG_FILE" ]] && contains_limit_message "$LAST_LOG_FILE"; then
      if can_switch_to_fallback_model; then
        log_line "fresh continuation hit a usage/rate limit on $(active_model_label). Switching to fallback model $FALLBACK_MODEL before resume." | tee -a "$LAST_LOG_FILE"
        switch_to_fallback_model
        mode="resume"
        continue
      fi

      log_line "fresh continuation also hit a usage/rate limit on $(active_model_label). Sleeping $SLEEP_DURATION before resume." | tee -a "$LAST_LOG_FILE"
      sleep "$SLEEP_DURATION"
      switch_to_primary_model
      mode="resume"
      continue
    fi
  fi

  if [[ "$code" -ne 0 ]]; then
    exit "$code"
  fi

  if goal_is_complete; then
    log_line "progress file marks goal complete; exiting." | tee -a "$LAST_LOG_FILE"
    exit 0
  fi

  log_line "Codex exited successfully, but progress is not complete; starting a fresh continuation." | tee -a "$LAST_LOG_FILE"
  mode="fresh-continuation"
done
