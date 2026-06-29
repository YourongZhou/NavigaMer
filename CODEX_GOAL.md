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
