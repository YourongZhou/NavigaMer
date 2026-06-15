# Exact MBB Rectangle Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional exact SoA rectangle lookup for adaptive-search child MBB filtering while preserving scan-equivalent results and exact fallback behavior.

**Architecture:** Each non-finest parent world optionally owns an immutable `MBBRectIndex` built from its existing child MBB rows. `BioGeometrySearchEngine` selects scan or rectangle survivor enumeration through `SearchConfig`; rectangle failures fall back to the original scan path before unchanged center-distance and traversal logic.

**Tech Stack:** C++17, STL, OpenMP, Make, CMake, assert-based standalone C++ tests

---

### Task 1: Exact Rectangle Index

**Files:**
- Create: `navigamer_cpp/include/mbb_rect_index.hpp`
- Create: `navigamer_cpp/src/mbb_rect_index.cpp`
- Create: `navigamer_cpp/src/test_mbb_rect_index.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] Write `test_mbb_rect_index.cpp` with handwritten 2D/3D intersections, random naive-scan equivalence for dimensions `1,2,4,8,16`, and invalid/empty input cases.
- [ ] Add the new source and test target to Make and CMake, then run `make test_mbb_rect_index`; verify it fails because the index API is missing.
- [ ] Implement `MBBRectIndex` with validated SoA storage and exact all-dimension intersection.
- [ ] Run `make test_mbb_rect_index && ./test_mbb_rect_index`; verify all rectangle-index tests pass.

### Task 2: Builder And Adaptive Search Integration

**Files:**
- Modify: `navigamer_cpp/include/structure.hpp`
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Create: `navigamer_cpp/src/test_mbb_filter_equivalence.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] Write end-to-end tests comparing scan and rect result sets, rect versus brute-force recall, low-fanout fallback, missing-index fallback, dimension-mismatch fallback, and mode-specific counters.
- [ ] Add the test target and run `make test_mbb_filter_equivalence`; verify it fails because search configuration and counters are missing.
- [ ] Add optional per-world rectangle indexes and build them from existing phase-3 MBB rows when fanout and dimensions are valid.
- [ ] Add `MBBFilterMode`, parsing/name helpers, `SearchConfig`, and the requested `SearchStats` counters.
- [ ] Refactor adaptive child survivor generation into exact scan and rectangle paths with exception-safe scan fallback; keep downstream traversal unchanged.
- [ ] Run `make test_mbb_filter_equivalence && ./test_mbb_filter_equivalence`; verify all equivalence, recall, fallback, and counter assertions pass.
- [ ] Run `make test_recall test_distance_bound && ./test_recall && ./test_distance_bound`.

### Task 3: CLI, Benchmark Instrumentation, And Documentation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/include/map150.hpp`
- Modify: `navigamer_cpp/src/map150.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] Add CLI parsing and validation for `--mbb-filter-mode scan|rect` and `--min-rect-index-fanout N`.
- [ ] Pass the build threshold and search mode through every adaptive-search path: demo, query, run, map150, benchmark, boundary, and layer-radius-experiment.
- [ ] Append benchmark TSV columns for mode, scan checks, rect queries/candidates/fallbacks, center-distance calls after MBB, and per-parent/per-query averages.
- [ ] Update all three documentation files with flags, defaults, exact semantics, fallback, and benchmark columns.
- [ ] Run query and benchmark smoke commands in both modes and compare benchmark result sets independently of row order.

### Task 4: Full Verification And Small Benchmark

**Files:**
- Modify only if verification exposes a defect.

- [ ] Run `make -j`.
- [ ] Run `make test_all`.
- [ ] Run `./test_mbb_rect_index` and `./test_mbb_filter_equivalence`.
- [ ] Run `./navigamer demo --size 200 --mbb-filter-mode scan`.
- [ ] Run `./navigamer demo --size 200 --mbb-filter-mode rect --min-rect-index-fanout 2`.
- [ ] Run a same-input scan versus rect benchmark, measure elapsed query time, aggregate MBB candidate/check and center-distance counters, and verify identical result rows.
- [ ] Inspect `git diff --check`, confirm `phase2_inter_tier_rebinding()` and `attach_leaves()` semantics were not modified, and report validation evidence and benchmark numbers.
