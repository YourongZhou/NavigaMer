# Integer IDs and Epoch Visited Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace adaptive-search optimized visited tracking with contiguous integer node IDs and a reusable epoch visited array while preserving exact result sets.

**Architecture:** Add finalized integer IDs to `WorldNode` and `BioSequence`, assign them after build, introduce `SearchScratch` and `VisitedMode`, and route adaptive search through either the legacy string-set path or the new epoch path. Keep construction, pruning, q-gram filtering, and exact verification unchanged.

**Tech Stack:** C++17, OpenMP thread-local storage, Make, CMake, existing NavigaMer test binaries.

---

### Task 1: Integer ID Assignment

**Files:**
- Modify: `navigamer_cpp/include/structure.hpp`
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Create: `navigamer_cpp/src/test_epoch_visited.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] Write failing tests that build a small hierarchy and assert world IDs and sequence IDs are unique, contiguous, and below builder counts.
- [ ] Register `test_epoch_visited` in Make/CMake and run `cd navigamer_cpp && make test_epoch_visited`; expect compile failure for missing ID fields/APIs.
- [ ] Add `NodeId`, `LeafId`, invalid constants, `WorldNode::integer_id`, and `BioSequence::sequence_id`.
- [ ] Add builder counters/getters, `assign_integer_ids()`, and `validate_integer_ids()`.
- [ ] Call `assign_integer_ids()` immediately after `attach_leaves()`.
- [ ] Run `cd navigamer_cpp && make test_epoch_visited && ./test_epoch_visited`; expect ID tests to pass.
- [ ] Commit with `git commit -m "add stable integer ids to built index"`.

### Task 2: SearchScratch Epoch Semantics

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Modify: `navigamer_cpp/src/test_epoch_visited.cpp`

- [ ] Add failing tests for `SearchScratch::begin_query()` and `mark_visited()` same-epoch duplicate, new-epoch reset, and overflow fallback.
- [ ] Run `cd navigamer_cpp && make test_epoch_visited`; expect compile failure for missing `SearchScratch`.
- [ ] Implement `SearchScratch` with reusable `visited_epoch`, `current_epoch`, and reusable frontier/candidate vectors.
- [ ] Implement overflow by filling `visited_epoch` with zero and setting `current_epoch` to `1`.
- [ ] Run `cd navigamer_cpp && make test_epoch_visited && ./test_epoch_visited`; expect scratch tests to pass.
- [ ] Commit with `git commit -m "add epoch visited search scratch"`.

### Task 3: Adaptive Epoch Visited Path

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Modify: `navigamer_cpp/src/test_epoch_visited.cpp`
- Modify: `navigamer_cpp/src/test_search_stats.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`

- [ ] Add failing equivalence tests that run string-set and epoch modes over the same index/queries for scan/off, rect/off, scan/q-gram-on, and rect/q-gram-on.
- [ ] Run `cd navigamer_cpp && make test_epoch_visited`; expect compile failure for missing `VisitedMode`.
- [ ] Add `VisitedMode` and `SearchConfig::visited_mode`, defaulting to `Epoch`.
- [ ] Keep the legacy string-set adaptive helper available for baseline mode.
- [ ] Add epoch adaptive helpers using `thread_local SearchScratch`, `begin_query(index_.num_world_nodes())`, and `mark_visited(node->integer_id)`.
- [ ] Preserve visited counter increments at the same logical check points.
- [ ] Reuse scratch overlap vector for the epoch path; keep child-survivor vector behavior otherwise unchanged.
- [ ] Run `cd navigamer_cpp && make test_epoch_visited test_search_stats test_mbb_filter test_search_qgram test_query_benchmark`.
- [ ] Commit with `git commit -m "switch adaptive search to epoch visited mode"`.

### Task 4: Full Validation and PR5 Benchmark Gate

**Files:**
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md` only if user-visible behavior changes; otherwise leave docs unchanged.

- [ ] Run `cd navigamer_cpp && make test_all`.
- [ ] Run `cd navigamer_cpp && ./navigamer demo --size 200`.
- [ ] Run the mixed synthetic `query-benchmark` command from PR4 and parse JSON.
- [ ] Compare benchmark summary counters before/after where available; result equality must pass.
- [ ] Run `git diff --check` and `git status --short`.
- [ ] If no user-visible CLI/doc behavior changed, do not edit README/CLI docs for PR5.
- [ ] Commit any required docs or benchmark notes with `git commit -m "validate epoch visited search path"`.
