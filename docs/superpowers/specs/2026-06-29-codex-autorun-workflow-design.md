# Codex Autorun Workflow Design

## Goal

Add a repository-local workflow for long-running Codex objectives that can
recover after quota or rate-limit interruptions without relying only on chat
memory. The workflow should make the current goal, progress, and test evidence
explicit in files that the next Codex run can read before continuing.

## Scope

This is a workflow/tooling change only. It must not modify NavigaMer C++
indexing, search semantics, CLI behavior, or experiment logic.

## Files

- `CODEX_GOAL.md`: durable objective, constraints, acceptance criteria, and
  required startup checklist for every continuation.
- `CODEX_PROGRESS.md`: mutable progress ledger for completed work, next steps,
  blockers, and handoff notes.
- `CODEX_TEST_LOG.md`: compact record of validation commands and outcomes.
- `scripts/codex_autorun.sh`: robust wrapper around `codex exec` and
  `codex exec resume --last`.
- `.codex/logs/`: generated run logs, ignored by git if a gitignore update is
  needed.

## Runner Behavior

The runner should:

- Default to this repository when run from the repo, while allowing `--repo`.
- Read the prompt from `CODEX_GOAL.md`, while allowing `--prompt-file`.
- Sleep for `5h10m` after detected usage/rate limits, while allowing
  `--sleep`.
- Write one timestamped log per Codex attempt and update
  `.codex/logs/latest.log`.
- Detect quota-like output with a conservative case-insensitive pattern such as
  `usage limit`, `rate limit`, `try again`, or `resets`.
- Resume with `codex exec resume --last --cd "$REPO"` after sleeping.
- Fall back to a fresh `codex exec` if resume fails, using a continuation
  prompt that tells Codex to read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`,
  `CODEX_TEST_LOG.md`, `git status`, and `git diff`.
- Exit with the original non-quota exit code when the failure is not a detected
  quota/rate-limit interruption.

## State File Content

The templates should be useful immediately for this repository:

- The goal file should remind Codex to follow `AGENTS.md`, avoid committing
  generated binaries or ad hoc datasets, and treat `navigamer_cpp/` as the
  production implementation path.
- The progress file should provide a concise ledger format that can be updated
  at the end of each run.
- The test log should record command, date, exit status, and short result
  summary rather than storing full raw logs.

## Validation

Validation should avoid launching a real long-running Codex job by default.
The implementation is acceptable when:

- `bash -n scripts/codex_autorun.sh` passes.
- `scripts/codex_autorun.sh --help` prints usage successfully.
- The generated state files are readable and contain no placeholder ambiguity.
- No C++ build or search tests are required because this change is tooling/docs
  only.

## Risks

- Usage-limit messages may change. The script should keep detection
  conservative and allow manual reruns rather than masking unknown failures.
- `codex exec resume --last` can fail if the previous session is unavailable.
  The fallback fresh `exec` path reduces drift by forcing Codex to read the
  repo-local state files.
- Fully unattended automation can continue a bad goal if the goal file is stale.
  The templates should emphasize updating `CODEX_PROGRESS.md` and
  `CODEX_TEST_LOG.md` before each handoff.
