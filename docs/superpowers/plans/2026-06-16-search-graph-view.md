# SearchGraphView Continuous Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a continuous query-side graph view and route adaptive search through it while preserving exact result sets.

**Architecture:** Build `SearchGraphView` after integer ID assignment, expose it from `BioGeometryIndexBuilder`, and add a view-backed adaptive path selected by `SearchConfig::graph_view_mode`. Keep original `WorldNode` storage intact and use it as the regression baseline.

**Tech Stack:** C++17, existing NavigaMer builder/search classes, Make/CMake tests, query-benchmark no-FN gate.

---

### Task 1: SearchGraphView Data and Builder

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Create: `navigamer_cpp/src/test_search_graph_view.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] Write failing tests that build a small hierarchy and compare view child lists, leaf lists, MBB values, and leaf-beacon distances against the original graph.
- [ ] Register `test_search_graph_view` in Make/CMake and run `cd navigamer_cpp && make test_search_graph_view`; expect compile failure for missing `SearchGraphView`.
- [ ] Add `SearchGraphView` with node/leaf pointer tables, child ranges, leaf ranges, MBB SoA arrays, beacon ranges, and leaf-beacon SoA arrays.
- [ ] Implement `build_search_graph_view()` after `assign_integer_ids()` and expose `search_graph_view()`.
- [ ] Implement `validate_search_graph_view()` for tests and defensive checks.
- [ ] Run `cd navigamer_cpp && make test_search_graph_view`; expect view equivalence tests to pass.
- [ ] Commit with `git commit -m "add search graph view layout"`.

### Task 2: Adaptive Flat View Search Path

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Modify: `navigamer_cpp/src/test_search_graph_view.cpp`

- [ ] Add failing adaptive equivalence tests for `GraphViewMode::Original` vs `GraphViewMode::Flat` across MBB scan/rect, q-gram off/on, and visited string/epoch.
- [ ] Run `cd navigamer_cpp && make test_search_graph_view`; expect compile failure for missing `GraphViewMode`.
- [ ] Add `GraphViewMode`, parser/name helpers, and `SearchConfig::graph_view_mode = GraphViewMode::Flat`.
- [ ] Keep current adaptive helpers for `Original`.
- [ ] Add view-backed adaptive helpers that traverse `NodeId` child candidates from `SearchGraphView` and verify leaf IDs through the view leaf table.
- [ ] Preserve MBB, q-gram, visited, center-distance, and exact leaf verification counters at the same logical points.
- [ ] Run `cd navigamer_cpp && make test_search_graph_view test_epoch_visited test_search_stats test_mbb_filter test_search_qgram`.
- [ ] Commit with `git commit -m "use search graph view for adaptive search"`.

### Task 3: CLI, Benchmark Metadata, and Full Validation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] Add `--graph-view original|flat` and wire it to `SearchConfig`.
- [ ] Make query-benchmark baseline use `graph_view=original` and optimized use the CLI-selected mode.
- [ ] Add JSON profile assertions for baseline original and optimized flat.
- [ ] Document the flag and benchmark profile behavior.
- [ ] Run `cd navigamer_cpp && make test_query_benchmark`.
- [ ] Run `cd navigamer_cpp && make test_all`.
- [ ] Run `cd navigamer_cpp && ./navigamer demo --size 200`.
- [ ] Run a small query-benchmark with `--graph-view flat` and verify zero mismatches/FNs.
- [ ] Run `git diff --check` and `git status --short`.
- [ ] Commit with `git commit -m "validate search graph view mode"`.

