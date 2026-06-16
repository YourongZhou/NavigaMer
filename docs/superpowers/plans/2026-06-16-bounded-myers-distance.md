# Bounded Myers Distance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional bounded Myers edit-distance backend for adaptive query search while keeping DP as the default and preserving exact result-set equivalence.

**Architecture:** `tools.hpp/cpp` owns distance mode parsing and bounded distance implementations. `SearchConfig` carries `distance_mode`, and adaptive search uses the mode-aware wrapper only for bounded child-center checks. CLI and query-benchmark expose/report the mode while baseline remains DP.

**Tech Stack:** C++17, existing Makefile/CMake tests, existing query benchmark no-FN gate.

---

### Task 1: Distance API And Myers Tests

**Files:**
- Modify: `navigamer_cpp/include/tools.hpp`
- Modify: `navigamer_cpp/src/tools.cpp`
- Create: `navigamer_cpp/src/test_bounded_myers.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing Myers differential test**

Create `navigamer_cpp/src/test_bounded_myers.cpp` with random ACGT checks, known edit examples, non-ACGT fallback checks, and parse/name checks. The test must call `compute_distance_bounded_myers()`, `compute_distance_bounded_dp()`, `parse_distance_mode()`, and `distance_mode_name()` before those APIs exist.

- [ ] **Step 2: Wire the test target**

Add `test_bounded_myers` to the Makefile and CMake build, and include it in `test_all`.

- [ ] **Step 3: Verify RED**

Run:

```bash
cd navigamer_cpp && make test_bounded_myers
```

Expected: compile failure because the new distance APIs are not defined yet.

- [ ] **Step 4: Implement minimal API and single-word Myers**

Add `DistanceMode`, parse/name helpers, `compute_distance_bounded_dp()`, `compute_distance_bounded_myers()`, and `compute_distance_bounded_with_mode()`. Move the existing bounded DP body into `compute_distance_bounded_dp()` and keep `compute_distance_bounded()` as a DP wrapper.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
cd navigamer_cpp && make test_bounded_myers
```

Expected: `bounded Myers edit distance tests passed`.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-06-16-bounded-myers-distance-design.md docs/superpowers/plans/2026-06-16-bounded-myers-distance.md navigamer_cpp/include/tools.hpp navigamer_cpp/src/tools.cpp navigamer_cpp/src/test_bounded_myers.cpp navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "add bounded myers distance mode"
```

### Task 2: Adaptive Search Distance Mode

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Create: `navigamer_cpp/src/test_search_distance_mode.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write failing search equivalence test**

Create `test_search_distance_mode.cpp` that builds one index, runs adaptive search with `distance_mode=dp` and `distance_mode=myers`, and asserts exact result ID equality for multiple queries and tolerances.

- [ ] **Step 2: Wire the test target**

Add `test_search_distance_mode` to Makefile/CMake and `test_all`.

- [ ] **Step 3: Verify RED**

Run:

```bash
cd navigamer_cpp && make test_search_distance_mode
```

Expected: compile failure because `SearchConfig::distance_mode` is not wired yet.

- [ ] **Step 4: Add search mode plumbing**

Add `DistanceMode distance_mode = DistanceMode::DP` to `SearchConfig`. Replace adaptive bounded child-center calls with `compute_distance_bounded_with_mode(query, center, tau, config_.distance_mode)`. Leave index construction and exact leaf verification unchanged.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
cd navigamer_cpp && make test_search_distance_mode
```

Expected: `search distance mode tests passed`.

- [ ] **Step 6: Commit**

```bash
git add navigamer_cpp/include/search_engine.hpp navigamer_cpp/src/search_engine.cpp navigamer_cpp/src/test_search_distance_mode.cpp navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "wire distance mode into adaptive search"
```

### Task 3: CLI, Benchmark Reporting, Docs

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] **Step 1: Write failing benchmark/CLI coverage**

Extend `test_query_benchmark_gate.cpp` to require `distance_mode` fields in baseline and optimized JSON profiles.

- [ ] **Step 2: Verify RED**

Run:

```bash
cd navigamer_cpp && make test_query_benchmark
```

Expected: test failure because JSON does not include distance mode yet.

- [ ] **Step 3: Add CLI flag and JSON reporting**

Parse `--distance-mode dp|myers|auto`, set `search_config.distance_mode`, include it in usage text, and emit profile distance modes in query benchmark JSON. Baseline profile must stay `dp`.

- [ ] **Step 4: Update docs**

Document the new flag in `README.md`, `navigamer_cpp/README.md`, and `navigamer_cpp/CLI_REFERENCE.md`, including that `dp` remains default and `auto` is conservative in this PR.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
cd navigamer_cpp && make test_query_benchmark
```

Expected: query benchmark gate passes.

- [ ] **Step 6: Commit**

```bash
git add navigamer_cpp/src/main.cpp navigamer_cpp/src/query_benchmark.cpp navigamer_cpp/src/test_query_benchmark_gate.cpp README.md navigamer_cpp/README.md navigamer_cpp/CLI_REFERENCE.md
git commit -m "expose distance mode for query benchmark"
```

### Task 4: Final Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Build**

Run:

```bash
cd navigamer_cpp && make -j
```

Expected: build succeeds.

- [ ] **Step 2: Full test suite**

Run:

```bash
cd navigamer_cpp && make test_all
```

Expected: all tests pass.

- [ ] **Step 3: Demo**

Run:

```bash
cd navigamer_cpp && ./navigamer demo --size 200 --distance-mode myers
```

Expected: demo completes and prints search results.

- [ ] **Step 4: Query benchmark gate**

Run a small deterministic query benchmark with optimized `--distance-mode myers` and verify the JSON reports zero equality failures and zero false negatives.

- [ ] **Step 5: Status check**

Run:

```bash
git status --short
```

Expected: only generated build outputs and temporary benchmark files are untracked/modified.
