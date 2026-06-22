# Phase 2 CUDA Rollback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all experimental CUDA support while preserving the CPU Phase 2 verifier, OpenMP construction, persistence, progress reporting, and unrelated worktree changes.

**Architecture:** Keep `Phase2DistanceVerifier` as a CPU-only batch interface and remove backend selection from its API. Remove CUDA wiring from build systems and CLI/statistics surfaces, then delete only CUDA-specific source and design artifacts.

**Tech Stack:** C++17, OpenMP, GNU Make, CMake, Edlib

---

### Task 1: Define the CPU-only verifier contract

**Files:**
- Modify: `navigamer_cpp/include/phase2_distance_verifier.hpp`
- Modify: `navigamer_cpp/src/phase2_distance_verifier.cpp`
- Modify: `navigamer_cpp/src/test_phase2_distance_verifier.cpp`

- [ ] Remove CUDA/backend assertions from the verifier test while preserving CPU batch acceptance and distance-mode checks.
- [ ] Run `make test_phase2_distance_verifier` before implementation and confirm it fails against the old backend API expectation.
- [ ] Replace backend selection with `make_phase2_distance_verifier(DistanceMode)` and remove CUDA-only result counters.
- [ ] Run `make test_phase2_distance_verifier` and confirm it passes.

### Task 2: Remove CUDA from construction and CLI surfaces

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/test_build_timing_stats.cpp`
- Modify: `navigamer_cpp/src/test_build_scale_smoke.cpp`
- Modify: `navigamer_cpp/src/test_index_persistence.cpp`

- [ ] Update tests to require CPU-only output schemas and reject the removed `--phase2-distance-backend` option.
- [ ] Run the focused tests and confirm failure while CUDA/backend fields still exist.
- [ ] Remove backend config, parsing, dispatch, statistics, and CSV columns without changing CPU candidate generation or OpenMP loops.
- [ ] Run focused build, timing, persistence, and smoke tests.

### Task 3: Remove CUDA build and documentation artifacts

**Files:**
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`
- Delete: `navigamer_cpp/src/phase2_distance_verifier_cuda.cu`
- Delete: `docs/superpowers/plans/2026-06-19-phase2-cuda-distance-verifier.md`
- Delete: `docs/superpowers/specs/2026-06-19-phase2-cuda-distance-verifier-design.md`

- [ ] Remove CUDA compiler options, link flags, targets, and user-facing documentation.
- [ ] Delete only CUDA implementation/design files; retain the rollback design and plan.
- [ ] Run `rg -n -i 'cuda|phase2-distance-backend'` over tracked source/docs and confirm no production CUDA references remain.

### Task 4: Verify and publish

**Files:**
- Commit only tracked source, tests, and documentation relevant to the retained branch work.
- Exclude: object files, binaries, CSV outputs, `.tmp_experiments/`, and ad hoc datasets.

- [ ] Run `make clean && make -j`.
- [ ] Run Phase 2, build timing, build range, persistence, build-scale, recall, and distance-bound tests.
- [ ] Configure and build `navigamer`, `test_phase2_distance_verifier`, and `test_index_persistence` with CMake.
- [ ] Run `git diff --check` and audit staged files.
- [ ] Commit the intended retained source/docs and push the current branch to `origin`.
