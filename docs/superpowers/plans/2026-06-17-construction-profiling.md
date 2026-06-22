# Construction Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-overhead construction profiling and a `build-scale` CSV command for C++ index-build bottleneck analysis.

**Architecture:** Add aggregate `std::chrono::steady_clock` timers to builder and range-join code. Store all timing on existing statistics/result structs, print it in builder summary, and export it through `build-scale`.

**Tech Stack:** C++17, OpenMP, Makefile, CMake, existing `assert`-style C++ test binaries, shell smoke test.

---

## File Structure

- Modify `navigamer_cpp/include/index_builder.hpp`: add builder statistics timing fields.
- Modify `navigamer_cpp/include/range_join.hpp`: add range-join timing fields to query results.
- Modify `navigamer_cpp/src/index_builder.cpp`: add `ScopedTimer`, phase/substep timers, summary timing output.
- Modify `navigamer_cpp/src/range_join.cpp`: populate range-join timing fields.
- Modify `navigamer_cpp/src/main.cpp`: add `build-scale`, CSV columns, bottleneck stderr summary, and `--prefix-lengths` parsing.
- Create `navigamer_cpp/src/test_build_timing_stats.cpp`: direct builder timing regression.
- Create `navigamer_cpp/src/test_build_scale_smoke.cpp`: CLI smoke test via `std::system`.
- Modify `navigamer_cpp/Makefile`: add test targets.
- Modify `navigamer_cpp/CMakeLists.txt`: add test executables.
- Modify `README.md`, `navigamer_cpp/README.md`, `navigamer_cpp/CLI_REFERENCE.md`: document profiling and `build-scale`.

## Task 1: Builder Timing Test

**Files:**
- Create: `navigamer_cpp/src/test_build_timing_stats.cpp`
- Modify: `navigamer_cpp/Makefile`

- [ ] **Step 1: Write the failing test**

Create `test_build_timing_stats.cpp` with a small synthetic build. Check:

```cpp
auto stats = builder.get_statistics();
assert(stats.total_build_ms > 0.0);
assert(stats.phase0_dedup_ms >= 0.0);
assert(stats.phase1_sketch_ms >= 0.0);
assert(stats.phase2_rebinding_ms >= 0.0);
assert(stats.phase3_mbb_ms >= 0.0);
assert(stats.phase4_attach_ms >= 0.0);
assert(stats.assign_ids_ms >= 0.0);
assert(stats.graph_view_ms >= 0.0);
double measured_sum = stats.phase0_dedup_ms + stats.phase1_sketch_ms +
    stats.phase2_rebinding_ms + stats.phase3_mbb_ms +
    stats.phase4_attach_ms + stats.assign_ids_ms + stats.graph_view_ms;
assert(measured_sum <= stats.total_build_ms * 1.2);
assert(stats.phase2_candidate_query_ms >= 0.0);
assert(stats.phase3_child_mbb_distance_ms >= 0.0);
assert(stats.leaf_beacon_distance_ms >= 0.0);
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd navigamer_cpp && make test_build_timing_stats
```

Expected: compile failure because timing fields do not exist.

- [ ] **Step 3: Add the Makefile target**

Add variables and target for `test_build_timing_stats`, linking with `$(LIB_OBJ)`.

- [ ] **Step 4: Keep RED meaningful**

Run the same command again. Expected: still fails on missing timing fields, not missing Make target.

## Task 2: Builder and Range-Join Timing Fields

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/include/range_join.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/range_join.cpp`

- [ ] **Step 1: Add timing fields**

Add all fields listed in the design to `BioGeometryIndexBuilder::Statistics` and `RangeJoinQueryResult`, initialized to `0.0`.

- [ ] **Step 2: Add scoped timer helpers**

Add internal `ScopedTimer`, `elapsed_ms_since`, and range-timing merge helpers in `.cpp` files using `std::chrono::steady_clock`.

- [ ] **Step 3: Time builder phases**

Wrap dedup, sketch, rebinding, MBB, leaf attach, assign IDs, graph view, summary print, and total build wall time.

- [ ] **Step 4: Time Phase2 substeps**

Time index build, candidate query, exact verify loop, and edge insertion.

- [ ] **Step 5: Time Phase3 substeps**

Time beacon collection, child collapse/dedup, child-beacon distance rows, and rectangle-index build.

- [ ] **Step 6: Time Phase4 substeps**

Time index build, candidate query, exact verify, tuple emit, merge/sort, populate, and leaf beacon distances. For current direct insertion, tuple merge/sort may remain zero unless a tuple buffer is used.

- [ ] **Step 7: Time range-join internals**

Time full scan length filtering, posting lookup, seed union, q-gram query, and hybrid intersection in `RangeJoinQueryResult`.

- [ ] **Step 8: Verify GREEN**

Run:

```bash
cd navigamer_cpp && make test_build_timing_stats && ./test_build_timing_stats
```

Expected: binary builds and prints `build timing stats tests passed`.

## Task 3: Build Summary Output

**Files:**
- Modify: `navigamer_cpp/src/index_builder.cpp`

- [ ] **Step 1: Extend summary test coverage**

Ensure `test_build_timing_stats` calls `builder.build(...)` normally so `print_summary()` runs.

- [ ] **Step 2: Print timing section**

Add a `Build timing:` section to `print_summary()` with the requested phase and substep layout.

- [ ] **Step 3: Verify**

Run:

```bash
cd navigamer_cpp && make test_build_timing_stats && ./test_build_timing_stats
```

Expected: no crash and timing section appears on stderr.

## Task 4: `build-scale` Smoke Test

**Files:**
- Create: `navigamer_cpp/src/test_build_scale_smoke.cpp`
- Modify: `navigamer_cpp/Makefile`

- [ ] **Step 1: Write the failing test**

Create a test that removes `/tmp/navigamer_build_scale_smoke.csv`, runs:

```bash
./navigamer build-scale --ref ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT --window 12 --stride 4 --prefix-lengths 24,36 --primary-radii 12,6,2 --out /tmp/navigamer_build_scale_smoke.csv
```

Then parse the CSV header and rows. Assert the header contains `prefix_len`, `total_build_ms`, `phase0_dedup_ms`, `phase2_candidate_query_ms`, `phase3_child_mbb_distance_ms`, `leaf_beacon_distance_ms`, `range_candidate_mode`, and `qgram_q`; assert there are two data rows and every `total_build_ms` is greater than zero.

- [ ] **Step 2: Verify RED**

Run:

```bash
cd navigamer_cpp && make test_build_scale_smoke
```

Expected: command builds test but test fails because `./navigamer build-scale` is unknown.

## Task 5: `build-scale` CLI

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`

- [ ] **Step 1: Add parsing**

Add `--prefix-lengths csv` parsing via existing integer CSV helper.

- [ ] **Step 2: Add runner**

Implement `run_build_scale()` that loads reference, truncates to each prefix, builds windows through `build_reference_windows`, builds an index, retrieves statistics, writes requested CSV columns, and prints bottleneck summaries.

- [ ] **Step 3: Add command dispatch**

Add `build-scale` to usage and `main()` dispatch. Require `--ref` and `--out`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cd navigamer_cpp && make -j && make test_build_scale_smoke && ./test_build_scale_smoke
```

Expected: smoke test passes and CSV contains two rows.

## Task 6: CMake and Docs

**Files:**
- Modify: `navigamer_cpp/CMakeLists.txt`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] **Step 1: Add CMake tests**

Add `test_build_timing_stats` and `test_build_scale_smoke` executables with the same source dependencies and include/link settings as neighboring builder tests.

- [ ] **Step 2: Document command and output**

Add `build-scale` usage, timing summary notes, and CSV column list to both README files and the CLI reference.

- [ ] **Step 3: Verify docs mention changed CLI**

Run:

```bash
rg -n "build-scale|Build timing|total_build_ms" README.md navigamer_cpp/README.md navigamer_cpp/CLI_REFERENCE.md
```

Expected: all three docs mention the new command or timing output.

## Task 7: Required Validation

**Files:** none

- [ ] **Step 1: Run requested tests**

```bash
cd navigamer_cpp && make test_build_timing_stats && ./test_build_timing_stats
cd navigamer_cpp && make test_build_scale_smoke && ./test_build_scale_smoke
```

- [ ] **Step 2: Run index-builder guards**

```bash
cd navigamer_cpp && make test_recall test_distance_bound && ./test_recall && ./test_distance_bound
```

- [ ] **Step 3: Run small scaling**

```bash
cd navigamer_cpp && ./navigamer build-scale --ref ../data/human/chr1_subset --window 250 --stride 1 --prefix-lengths 10000,50000,100000 --primary-radii 30,15,5 --range-candidate-mode auto --out /tmp/navigamer_build_scale.csv
```

- [ ] **Step 4: Inspect scaling output**

Read `/tmp/navigamer_build_scale.csv` and report per prefix: total build time, largest phase, largest substep, phase2 candidate/exact reduction, and leaf candidate/exact reduction.

## Self-Review

Spec coverage: all requested timing fields, summary output, build-scale CSV columns, bottleneck stderr summary, and two tests are covered.

Placeholder scan: no task depends on an unspecified function or incomplete requirement.

Type consistency: timing fields use `double` milliseconds; counters stay `size_t`; CSV output uses existing string formatting helpers.
