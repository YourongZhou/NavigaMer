# SIMD Leaf-Beacon Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SIMD acceleration for flat adaptive leaf-beacon filtering while preserving exact result sets.

**Architecture:** Reuse PR7 `SimdMode` and AVX2 runtime detection. Add a leaf-beacon helper that operates on the existing `SearchGraphView::leaf_beacon_dists` SoA layout, then call it only from the flat leaf verification path before unchanged exact edit-distance verification.

**Tech Stack:** C++17, Makefile/CMake, OpenMP-linked test binaries, AVX2 intrinsics guarded by runtime and compiler checks.

---

### Task 1: Design Checkpoint

**Files:**
- Create: `docs/superpowers/specs/2026-06-16-simd-leaf-beacon-filter-design.md`
- Create: `docs/superpowers/plans/2026-06-16-simd-leaf-beacon-filter.md`

- [ ] **Step 1: Commit the PR8 design and plan**

Run:

```bash
git add docs/superpowers/specs/2026-06-16-simd-leaf-beacon-filter-design.md \
  docs/superpowers/plans/2026-06-16-simd-leaf-beacon-filter.md
git commit -m "docs: plan simd leaf beacon filter"
```

Expected: commit succeeds and source files are unchanged.

### Task 2: Add Leaf SIMD Helper Test

**Files:**
- Create: `navigamer_cpp/src/test_simd_leaf_beacon_filter.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create a test that builds random SoA leaf distance arrays and compares a local
scalar reference against:

```cpp
navigamer::filter_leaf_beacon_survivors(..., navigamer::SimdMode::Scalar, ...)
navigamer::filter_leaf_beacon_survivors(..., navigamer::SimdMode::Auto, ...)
navigamer::filter_leaf_beacon_survivors(..., navigamer::SimdMode::AVX2, ...)
```

The reference condition is:

```cpp
if (std::abs(query[dim] - leaf_dist[dim * leaf_count + leaf_idx]) > tolerance)
  prune;
```

- [ ] **Step 2: Add build-system targets**

Add `test_simd_leaf_beacon_filter` to Makefile and CMake. Link it with the
same SIMD helper object used by production code.

- [ ] **Step 3: Verify RED**

Run:

```bash
cd navigamer_cpp && make test_simd_leaf_beacon_filter
```

Expected: fail because `filter_leaf_beacon_survivors` does not exist.

### Task 3: Implement Leaf SIMD Helper

**Files:**
- Modify: `navigamer_cpp/include/simd_mbb_filter.hpp`
- Modify: `navigamer_cpp/src/simd_mbb_filter.cpp`

- [ ] **Step 1: Add public API**

Add `LeafBeaconFilterSimdStats` and `filter_leaf_beacon_survivors()` beside the
MBB helper.

- [ ] **Step 2: Add scalar implementation**

Implement the scalar survivor filter as the reference path. It returns survivor
offsets in ascending leaf order and increments `scalar_checks` by `leaf_count`.

- [ ] **Step 3: Add AVX2 implementation**

For each batch of 8 leaves, keep an `alive` mask. For every beacon dimension:

```text
diff = abs(query_dist - leaf_dist)
alive &= diff <= tolerance
```

Use scalar tail handling for non-multiple-of-8 counts. Unsupported `auto`,
forced `avx2`, and `avx512` paths fall back scalar.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cd navigamer_cpp && make test_simd_leaf_beacon_filter
```

Expected: `SIMD leaf beacon filter tests passed`.

- [ ] **Step 5: Commit**

```bash
git add navigamer_cpp/include/simd_mbb_filter.hpp \
  navigamer_cpp/src/simd_mbb_filter.cpp \
  navigamer_cpp/src/test_simd_leaf_beacon_filter.cpp \
  navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "add simd leaf beacon filter helper"
```

### Task 4: Integrate Flat Leaf Verification

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Modify: `navigamer_cpp/src/test_search_graph_view.cpp`

- [ ] **Step 1: Extend stats**

Add:

```cpp
size_t leaf_beacon_scalar_checks = 0;
size_t leaf_beacon_simd_batches = 0;
size_t leaf_beacon_simd_fallbacks = 0;
```

- [ ] **Step 2: Write/extend equivalence test first**

Ensure `test_search_graph_view` compares original scalar search against flat
search under `Scalar`, `Auto`, and `AVX2` modes.

- [ ] **Step 3: Verify RED/GREEN boundary**

Run:

```bash
cd navigamer_cpp && make test_search_graph_view
```

Expected before integration: compiles but does not exercise new leaf counters.
Expected after integration: passes with identical result IDs.

- [ ] **Step 4: Integrate helper**

In `verify_leaf_candidates_view()`, when `has_leaf_sieve` is true, convert
`V_Q` to `int32_t`, call `filter_leaf_beacon_survivors()`, update counters, and
run exact verification only for returned offsets. Keep the no-sieve path as a
straight all-leaf exact verification loop.

- [ ] **Step 5: Commit**

```bash
git add navigamer_cpp/include/search_engine.hpp navigamer_cpp/src/search_engine.cpp \
  navigamer_cpp/src/test_search_graph_view.cpp
git commit -m "use simd leaf beacon filter in flat search"
```

### Task 5: Output Counters and Validate

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] **Step 1: Add benchmark output counters**

Add the three leaf SIMD counters to regular `benchmark` TSV and
`query-benchmark` detail/summary rows.

- [ ] **Step 2: Update docs**

Document that `--simd-mode` applies to flat child MBB and flat leaf-beacon
filters, with unsupported modes falling back scalar.

- [ ] **Step 3: Verify targeted tests**

Run:

```bash
cd navigamer_cpp && make test_simd_leaf_beacon_filter test_search_graph_view test_query_benchmark
```

Expected: all pass.

- [ ] **Step 4: Full validation**

Run:

```bash
cd navigamer_cpp && make -j && make test_all && ./navigamer demo --size 200 --simd-mode auto
```

Expected: all pass.

- [ ] **Step 5: Benchmark gate**

Run a small deterministic `query-benchmark` with `--graph-view flat
--simd-mode auto` and confirm:

```text
gate_passed=true
mismatch_count=0
false_negative_count=0
leaf_beacon_simd_batches is reported
```

- [ ] **Step 6: Commit**

```bash
git add README.md navigamer_cpp/README.md navigamer_cpp/CLI_REFERENCE.md \
  navigamer_cpp/src/main.cpp navigamer_cpp/src/query_benchmark.cpp \
  navigamer_cpp/src/test_query_benchmark_gate.cpp
git commit -m "report simd leaf beacon counters"
```
