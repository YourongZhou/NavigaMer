# Codex Autorun Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a robust repo-local Codex autorun wrapper and durable state-file templates for recoverable long-running objectives.

**Architecture:** Keep the workflow as plain repository files: Markdown state files in the repo root, generated logs under `.codex/logs/`, and one Bash runner under `scripts/`. The runner wraps `codex exec`, detects quota-like output, sleeps, attempts `resume --last`, and falls back to a fresh continuation prompt if resume is unavailable.

**Tech Stack:** Bash, Markdown, existing Codex CLI, standard Unix tools.

---

### Task 1: State File Templates

**Files:**
- Create: `CODEX_GOAL.md`
- Create: `CODEX_PROGRESS.md`
- Create: `CODEX_TEST_LOG.md`

- [ ] **Step 1: Create `CODEX_GOAL.md`**

Write a root-level goal template with this structure:

```markdown
# Current Codex Goal

## Objective

Describe the current long-running Codex objective here before starting
`scripts/codex_autorun.sh`.

## Repository Contract

- Read and follow `AGENTS.md` before making changes.
- Treat `navigamer_cpp/` as the production NavigaMer implementation path.
- Treat `methods/` as experiment orchestration, plotting, or baseline tooling.
- Do not commit generated binaries, object files, notebook outputs, images, or
  ad hoc datasets unless the task explicitly asks for them.
- Preserve recall-safety for changes that affect indexing, pruning, layer
  wiring, or leaf attachment.

## Constraints

- Prefer the smallest change that preserves existing CLI behavior.
- If CLI flags, output columns, or command behavior change, update the relevant
  documentation in the same patch.
- If indexing or pruning logic changes, run `test_recall` and
  `test_distance_bound` before handoff.
- If C++ search behavior changes, compare against exhaustive or brute-force
  behavior where practical.

## Acceptance Criteria

- Record every completed step in `CODEX_PROGRESS.md`.
- Record validation commands and outcomes in `CODEX_TEST_LOG.md`.
- Leave the worktree in a handoff-ready state with known gaps called out.

## Continuation Startup Checklist

Every Codex continuation must start by running or reading:

1. `git status --short`
2. `git diff --stat`
3. `CODEX_GOAL.md`
4. `CODEX_PROGRESS.md`
5. `CODEX_TEST_LOG.md`
6. The most relevant touched source or documentation files

Then continue the first unfinished item from `CODEX_PROGRESS.md`.
```

- [ ] **Step 2: Create `CODEX_PROGRESS.md`**

Write a root-level progress ledger template with this structure:

```markdown
# Codex Progress

## Current Status

- State: Not started
- Last updated: 2026-06-29
- Active branch: record with `git branch --show-current`

## Completed

- Add completed work items here, newest last.

## In Progress

- Add the current task here before making changes.

## Next Steps

1. Replace this line with the next concrete action.

## Blockers

- None recorded.

## Handoff Notes

- Before stopping, summarize what changed, what was validated, and what remains.
```

- [ ] **Step 3: Create `CODEX_TEST_LOG.md`**

Write a root-level test log template with this structure:

```markdown
# Codex Test Log

Record concise validation evidence here. Keep full raw logs in
`.codex/logs/` only when they are useful for debugging.

## Entries

### 2026-06-29

- Command: not run yet
- Exit status: n/a
- Result: initialize this file before the first autorun task.
- Notes: Add exact commands, short outcomes, and any known gaps after each run.
```

- [ ] **Step 4: Inspect templates**

Run: `sed -n '1,220p' CODEX_GOAL.md CODEX_PROGRESS.md CODEX_TEST_LOG.md`

Expected: all three files are readable, specific to this repository, and contain no ambiguous empty sections beyond intentional user-editable prompts.

### Task 2: Autorun Script

**Files:**
- Create: `scripts/codex_autorun.sh`

- [ ] **Step 1: Write script help and argument parser**

Create a Bash script with:

```bash
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
```

Parse the options using a `while [[ $# -gt 0 ]]` loop and reject unknown options
with exit status `64`.

- [ ] **Step 2: Add validation and helper functions**

Add functions:

```bash
timestamp() { date +"%Y%m%d_%H%M%S"; }
log_line() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
contains_limit_message() { grep -qiE 'usage limit|rate limit|try again|resets' "$1"; }
```

Validate that the repo directory exists and the prompt file exists. Create the
log directory before running Codex.

- [ ] **Step 3: Implement initial exec, resume, and fallback loop**

Implement:

```bash
run_codex_exec() {
  local log_file="$1"
  local prompt
  prompt="$(cat "$PROMPT_FILE")"
  codex exec --cd "$REPO" "$prompt" 2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

run_codex_resume() {
  local log_file="$1"
  codex exec resume --last --cd "$REPO" \
    "继续当前目标。先读 CODEX_GOAL.md、CODEX_PROGRESS.md、CODEX_TEST_LOG.md、git status、git diff、最近测试输出，然后继续未完成任务。" \
    2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

run_codex_fresh_continuation() {
  local log_file="$1"
  codex exec --cd "$REPO" \
    "继续当前目标。先读 CODEX_GOAL.md、CODEX_PROGRESS.md、CODEX_TEST_LOG.md、git status、git diff、最近测试输出，然后继续未完成任务。" \
    2>&1 | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}
```

Use a loop that runs the initial exec once, then on detected quota messages
sleeps and tries resume. If resume exits non-zero without a detected quota
message, log the failure and try the fresh continuation once for that cycle.

- [ ] **Step 4: Keep logs discoverable**

For each attempt, create a timestamped log file under the log directory and
update `latest.log` as a symlink when possible. If symlink creation fails,
write a small text file named `latest.log` containing the current log path.

- [ ] **Step 5: Make the script executable**

Run: `chmod +x scripts/codex_autorun.sh`

Expected: file mode allows direct execution.

### Task 3: Generated Log Ignore Rule

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Inspect ignore file**

Run: `sed -n '1,220p' .gitignore`

Expected: identify whether `.codex/logs/` is already ignored.

- [ ] **Step 2: Add ignore entry if missing**

If absent, add:

```gitignore
# Local Codex autorun logs
.codex/logs/
```

Keep existing ignore rules unchanged.

### Task 4: Validation

**Files:**
- Test: `scripts/codex_autorun.sh`
- Test: `CODEX_GOAL.md`
- Test: `CODEX_PROGRESS.md`
- Test: `CODEX_TEST_LOG.md`

- [ ] **Step 1: Shell syntax check**

Run: `bash -n scripts/codex_autorun.sh`

Expected: exit status `0`.

- [ ] **Step 2: Help output check**

Run: `scripts/codex_autorun.sh --help`

Expected: usage text prints and exit status `0`.

- [ ] **Step 3: Template sanity check**

Run: `grep -nEi 'TBD|TODO|PLACEHOLDER|xxx|\\?\\?' CODEX_GOAL.md CODEX_PROGRESS.md CODEX_TEST_LOG.md || true`

Expected: no matches.

- [ ] **Step 4: Git review**

Run: `git status --short CODEX_GOAL.md CODEX_PROGRESS.md CODEX_TEST_LOG.md scripts/codex_autorun.sh .gitignore`

Expected: only intended workflow files are listed.

Run: `git diff -- CODEX_GOAL.md CODEX_PROGRESS.md CODEX_TEST_LOG.md scripts/codex_autorun.sh .gitignore`

Expected: diff matches this plan and does not alter C++ implementation files.

### Task 5: Commit

**Files:**
- Stage: `CODEX_GOAL.md`
- Stage: `CODEX_PROGRESS.md`
- Stage: `CODEX_TEST_LOG.md`
- Stage: `scripts/codex_autorun.sh`
- Stage: `.gitignore` only if changed

- [ ] **Step 1: Commit implementation**

Run:

```bash
git add CODEX_GOAL.md CODEX_PROGRESS.md CODEX_TEST_LOG.md scripts/codex_autorun.sh
git add .gitignore
git commit -m "chore: add codex autorun workflow"
```

If `.gitignore` did not change, omit it from the commit.

Expected: one commit containing only the workflow templates and runner.
