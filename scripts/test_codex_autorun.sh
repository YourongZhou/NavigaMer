#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
AUTORUN="$SCRIPT_DIR/codex_autorun.sh"

make_repo() {
  local root="$1"
  mkdir -p "$root"
  printf '# goal\n' > "$root/CODEX_GOAL.md"
  printf '# progress\n\n## Current Status\n\n- State: %s\n' "$2" > "$root/CODEX_PROGRESS.md"
}

make_fake_codex() {
  local bin_dir="$1"
  cat > "$bin_dir/codex" <<'FAKE_CODEX'
#!/usr/bin/env bash
count="$(cat "$FAKE_CODEX_COUNT" 2>/dev/null || printf '0')"
count=$((count + 1))
printf '%s' "$count" > "$FAKE_CODEX_COUNT"
printf '%s:%s\n' "$count" "$*" >> "$FAKE_CODEX_LOG"

if [[ "${FAKE_LIMIT_ON_ATTEMPT:-0}" -eq "$count" ]]; then
  printf 'usage limit reached\n'
fi

if [[ "${FAKE_COMPLETE_ON_ATTEMPT:-0}" -eq "$count" ]]; then
  printf '# progress\n\n## Current Status\n\n- State: complete\n' > "$FAKE_PROGRESS_FILE"
fi

exit 0
FAKE_CODEX
  chmod +x "$bin_dir/codex"
}

assert_equals() {
  local expected="$1"
  local actual="$2"
  local message="$3"

  if [[ "$expected" != "$actual" ]]; then
    printf 'FAIL: %s\nexpected: %s\nactual: %s\n' "$message" "$expected" "$actual" >&2
    exit 1
  fi
}

tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT

bash -n "$AUTORUN"

test_continues_after_clean_exit_until_progress_complete() {
  local repo="$tmp_root/repo_continue"
  local fakebin="$tmp_root/fakebin_continue"
  local log_dir="$tmp_root/logs_continue"
  local count_file="$tmp_root/count_continue"
  local codex_log="$tmp_root/codex_continue.log"

  mkdir -p "$fakebin" "$log_dir"
  make_repo "$repo" "PR1 local-router complete, ready for PR2"
  make_fake_codex "$fakebin"

  FAKE_CODEX_COUNT="$count_file" \
  FAKE_CODEX_LOG="$codex_log" \
  FAKE_PROGRESS_FILE="$repo/CODEX_PROGRESS.md" \
  FAKE_COMPLETE_ON_ATTEMPT=2 \
  PATH="$fakebin:$PATH" \
    bash "$AUTORUN" --repo "$repo" --log-dir "$log_dir" --sleep 0s --max-attempts 3

  assert_equals "2" "$(cat "$count_file")" "autorun should continue after a clean exit until progress is complete"
}

test_already_complete_skips_codex() {
  local repo="$tmp_root/repo_complete"
  local fakebin="$tmp_root/fakebin_complete"
  local log_dir="$tmp_root/logs_complete"
  local count_file="$tmp_root/count_complete"
  local codex_log="$tmp_root/codex_complete.log"

  mkdir -p "$fakebin" "$log_dir"
  make_repo "$repo" "complete"
  make_fake_codex "$fakebin"

  FAKE_CODEX_COUNT="$count_file" \
  FAKE_CODEX_LOG="$codex_log" \
  FAKE_PROGRESS_FILE="$repo/CODEX_PROGRESS.md" \
  PATH="$fakebin:$PATH" \
    bash "$AUTORUN" --repo "$repo" --log-dir "$log_dir" --sleep 0s --max-attempts 1

  if [[ -f "$count_file" ]]; then
    printf 'FAIL: autorun should not call codex when progress is already complete\n' >&2
    exit 1
  fi
}

test_stage_complete_phrase_is_not_goal_complete() {
  local repo="$tmp_root/repo_stage"
  local fakebin="$tmp_root/fakebin_stage"
  local log_dir="$tmp_root/logs_stage"
  local count_file="$tmp_root/count_stage"
  local codex_log="$tmp_root/codex_stage.log"

  mkdir -p "$fakebin" "$log_dir"
  make_repo "$repo" "PR1 local-router complete, ready for PR2"
  make_fake_codex "$fakebin"

  FAKE_CODEX_COUNT="$count_file" \
  FAKE_CODEX_LOG="$codex_log" \
  FAKE_PROGRESS_FILE="$repo/CODEX_PROGRESS.md" \
  PATH="$fakebin:$PATH" \
    bash "$AUTORUN" --repo "$repo" --log-dir "$log_dir" --sleep 0s --max-attempts 1

  assert_equals "1" "$(cat "$count_file")" "stage-complete progress text should not be treated as goal completion"
}

test_default_log_dir_falls_back_when_dot_codex_is_file() {
  local repo="$tmp_root/repo_default_log_fallback"
  local fakebin="$tmp_root/fakebin_default_log_fallback"
  local count_file="$tmp_root/count_default_log_fallback"
  local codex_log="$tmp_root/codex_default_log_fallback.log"
  local fallback_log_dir="$repo/.codex_logs"

  mkdir -p "$fakebin"
  make_repo "$repo" "in_progress"
  : > "$repo/.codex"
  make_fake_codex "$fakebin"

  FAKE_CODEX_COUNT="$count_file" \
  FAKE_CODEX_LOG="$codex_log" \
  FAKE_PROGRESS_FILE="$repo/CODEX_PROGRESS.md" \
  FAKE_COMPLETE_ON_ATTEMPT=2 \
  PATH="$fakebin:$PATH" \
    bash "$AUTORUN" --repo "$repo" --sleep 0s --max-attempts 3

  assert_equals "2" "$(cat "$count_file")" "autorun should keep running when default .codex/logs is blocked by a .codex file"
  if [[ ! -d "$fallback_log_dir" ]]; then
    printf 'FAIL: autorun should create fallback log dir when .codex is a file\n' >&2
    exit 1
  fi
  if [[ ! -e "$fallback_log_dir/latest.log" ]]; then
    printf 'FAIL: autorun should update latest.log in fallback log dir\n' >&2
    exit 1
  fi
}

test_resume_keeps_global_cd_before_resume_subcommand() {
  local repo="$tmp_root/repo_resume_cd"
  local fakebin="$tmp_root/fakebin_resume_cd"
  local log_dir="$tmp_root/logs_resume_cd"
  local count_file="$tmp_root/count_resume_cd"
  local codex_log="$tmp_root/codex_resume_cd.log"

  mkdir -p "$fakebin" "$log_dir"
  make_repo "$repo" "in_progress"
  make_fake_codex "$fakebin"

  FAKE_CODEX_COUNT="$count_file" \
  FAKE_CODEX_LOG="$codex_log" \
  FAKE_PROGRESS_FILE="$repo/CODEX_PROGRESS.md" \
  FAKE_LIMIT_ON_ATTEMPT=1 \
  FAKE_COMPLETE_ON_ATTEMPT=2 \
  CODEX_FALLBACK_MODEL="" \
  PATH="$fakebin:$PATH" \
    bash "$AUTORUN" --repo "$repo" --log-dir "$log_dir" --sleep 0s --max-attempts 3

  local second_call
  second_call="$(sed -n '2p' "$codex_log")"
  assert_equals "2:exec --cd $repo resume --last 继续当前目标。先读 CODEX_GOAL.md、CODEX_PROGRESS.md、CODEX_TEST_LOG.md、git status、git diff、最近测试输出，然后继续未完成任务。只有当 CODEX_GOAL.md 的全部目标都完成时，才把 CODEX_PROGRESS.md 的 State 精确写成 complete。" "$second_call" "resume should keep --cd as a codex exec global option before the resume subcommand"
}

test_codex_model_and_service_tier_overrides_are_passed_to_exec() {
  local repo="$tmp_root/repo_exec_overrides"
  local fakebin="$tmp_root/fakebin_exec_overrides"
  local log_dir="$tmp_root/logs_exec_overrides"
  local count_file="$tmp_root/count_exec_overrides"
  local codex_log="$tmp_root/codex_exec_overrides.log"

  mkdir -p "$fakebin" "$log_dir"
  make_repo "$repo" "in_progress"
  make_fake_codex "$fakebin"

  CODEX_MODEL_OVERRIDE="gpt-5.5" \
  CODEX_MODEL_REASONING_EFFORT="high" \
  CODEX_SERVICE_TIER="fast" \
  FAKE_CODEX_COUNT="$count_file" \
  FAKE_CODEX_LOG="$codex_log" \
  FAKE_PROGRESS_FILE="$repo/CODEX_PROGRESS.md" \
  FAKE_COMPLETE_ON_ATTEMPT=1 \
  PATH="$fakebin:$PATH" \
    bash "$AUTORUN" --repo "$repo" --log-dir "$log_dir" --sleep 0s --max-attempts 1

  local first_call
  first_call="$(sed -n '1p' "$codex_log")"
  assert_equals "1:exec --cd $repo --model gpt-5.5 -c model_reasoning_effort=\"high\" -c service_tier=\"fast\" # goal" "$first_call" "autorun should pass explicit model, reasoning effort, and service tier overrides to codex exec"
}

test_usage_limit_switches_from_primary_to_fallback_model() {
  local repo="$tmp_root/repo_model_fallback"
  local fakebin="$tmp_root/fakebin_model_fallback"
  local log_dir="$tmp_root/logs_model_fallback"
  local count_file="$tmp_root/count_model_fallback"
  local codex_log="$tmp_root/codex_model_fallback.log"

  mkdir -p "$fakebin" "$log_dir"
  make_repo "$repo" "in_progress"
  make_fake_codex "$fakebin"

  CODEX_MODEL_OVERRIDE="gpt-5.5" \
  CODEX_FALLBACK_MODEL="gpt-5.3-codex-spark" \
  FAKE_CODEX_COUNT="$count_file" \
  FAKE_CODEX_LOG="$codex_log" \
  FAKE_PROGRESS_FILE="$repo/CODEX_PROGRESS.md" \
  FAKE_LIMIT_ON_ATTEMPT=1 \
  FAKE_COMPLETE_ON_ATTEMPT=2 \
  PATH="$fakebin:$PATH" \
    bash "$AUTORUN" --repo "$repo" --log-dir "$log_dir" --sleep 0s --max-attempts 3

  local second_call
  second_call="$(sed -n '2p' "$codex_log")"
  assert_equals "2:exec --cd $repo --model gpt-5.3-codex-spark resume --last 继续当前目标。先读 CODEX_GOAL.md、CODEX_PROGRESS.md、CODEX_TEST_LOG.md、git status、git diff、最近测试输出，然后继续未完成任务。只有当 CODEX_GOAL.md 的全部目标都完成时，才把 CODEX_PROGRESS.md 的 State 精确写成 complete。" "$second_call" "usage limit on the primary model should continue with the fallback model"
}

test_continues_after_clean_exit_until_progress_complete
test_already_complete_skips_codex
test_stage_complete_phrase_is_not_goal_complete
test_default_log_dir_falls_back_when_dot_codex_is_file
test_resume_keeps_global_cd_before_resume_subcommand
test_codex_model_and_service_tier_overrides_are_passed_to_exec
test_usage_limit_switches_from_primary_to_fallback_model

printf 'codex_autorun tests passed\n'
