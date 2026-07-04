# Results 1-6 Benchmark Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible script-driven pipeline that runs and summarizes the benchmark evidence for Result 1-6.

**Architecture:** Add one shell runner that invokes existing C++ and Python benchmark entrypoints, one Python summarizer that normalizes heterogeneous TSV outputs into status/report files, and one shell checker that fails on correctness regressions. Keep algorithm logic in existing C++/experiment code; the new files only orchestrate, summarize, and validate.

**Tech Stack:** Bash, Python standard library `csv/json/pathlib`, existing `navigamer_cpp` CLI/tests, existing `experiments/ecoli_1p1m` scripts.

---

### Task 1: Pipeline Contract Test

**Files:**
- Create: `scripts/test_results_1to6_pipeline.sh`

- [ ] **Step 1: Write the failing test**

Create a shell test that builds a synthetic run directory, calls `scripts/summarize_results_1to6.py`, calls `scripts/check_results_1to6.sh`, and verifies that `scripts/run_results_1to6_pipeline.sh --dry-run --quick` records commands without executing expensive work.

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/test_results_1to6_pipeline.sh`

Expected: FAIL because `scripts/summarize_results_1to6.py`, `scripts/check_results_1to6.sh`, and `scripts/run_results_1to6_pipeline.sh` are not implemented yet.

### Task 2: Result Summarizer

**Files:**
- Create: `scripts/summarize_results_1to6.py`

- [ ] **Step 1: Implement minimal summarizer**

Read known result subdirectories under a run directory and write:
- `summary_all.tsv`
- `result_status.tsv`
- `report.md`

Statuses should distinguish `supported`, `preliminary`, `mixed`, and `missing`.

- [ ] **Step 2: Run test**

Run: `bash scripts/test_results_1to6_pipeline.sh`

Expected: FAIL until checker and runner are added.

### Task 3: Correctness Checker

**Files:**
- Create: `scripts/check_results_1to6.sh`

- [ ] **Step 1: Implement checker**

Require the three summarizer outputs and scan TSV files for nonzero `fn_count`, `mismatch_count`, and `false_negative_count_total`.

- [ ] **Step 2: Run test**

Run: `bash scripts/test_results_1to6_pipeline.sh`

Expected: FAIL until runner dry-run is added.

### Task 4: Pipeline Runner

**Files:**
- Create: `scripts/run_results_1to6_pipeline.sh`

- [ ] **Step 1: Implement runner**

Support `--quick`, `--full`, `--out-dir`, `--label`, `--ref`, `--index`, `--threads`, `--dry-run`, `--skip-build`, and `--run-external-baselines`. Record all commands to `commands.tsv`.

- [ ] **Step 2: Run test**

Run: `bash scripts/test_results_1to6_pipeline.sh`

Expected: PASS.

### Task 5: User Documentation

**Files:**
- Create: `docs/benchmarks/result_1to6_pipeline.md`

- [ ] **Step 1: Document run modes and result interpretation**

Explain quick/full modes, required inputs, output directory layout, Result 1-6 mapping, and how to compare future methods.

- [ ] **Step 2: Verify syntax and smoke checks**

Run:

```bash
bash -n scripts/run_results_1to6_pipeline.sh
bash -n scripts/check_results_1to6.sh
python3 -m py_compile scripts/summarize_results_1to6.py
bash scripts/test_results_1to6_pipeline.sh
```

Expected: PASS.
