# SIMD MBB Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scalar/AVX2 MBB rectangle filtering over the flat SearchGraphView SoA layout while preserving exact survivor sets and query results.

**Architecture:** Implement a small `simd_mbb_filter` helper with scalar and optional AVX2 backends, route flat-view MBB scan through it, expose `--simd-mode`, and report SIMD counters through existing benchmark output.

**Tech Stack:** C++17, GCC/Clang x86 AVX2 target attributes when available, existing Make/CMake test targets.

---

### Task 1: Standalone SIMD MBB Helper

**Files:**
- Create: `navigamer_cpp/include/simd_mbb_filter.hpp`
- Create: `navigamer_cpp/src/simd_mbb_filter.cpp`
- Create: `navigamer_cpp/src/test_simd_mbb_filter.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] Write failing randomized tests comparing scalar, auto, and AVX2 survivor index outputs.
- [ ] Register `test_simd_mbb_filter` in Make/CMake and run `cd navigamer_cpp && make test_simd_mbb_filter`; expect compile failure for missing helper.
- [ ] Implement `SimdMode`, parser/name helpers, scalar filter, runtime AVX2 support check, and AVX2 fallback.
- [ ] Run `cd navigamer_cpp && make test_simd_mbb_filter`; expect pass.
- [ ] Commit with `git commit -m "add simd mbb filter helper"`.

### Task 2: Adaptive Search Integration

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Modify: `navigamer_cpp/src/test_search_graph_view.cpp`

- [ ] Add failing search equivalence tests comparing `SimdMode::Scalar`, `Auto`, and `AVX2`.
- [ ] Add `SearchConfig::simd_mode = SimdMode::Auto`.
- [ ] Add `SearchStats` counters for scalar checks, SIMD batches, and SIMD fallbacks.
- [ ] Route flat graph-view MBB scan through `filter_mbb_survivors`.
- [ ] Preserve existing MBB counters and survivor ordering.
- [ ] Run `cd navigamer_cpp && make test_search_graph_view test_simd_mbb_filter`.
- [ ] Commit with `git commit -m "use simd mbb filter in flat search"`.

### Task 3: CLI, Benchmark, and Full Validation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] Add `--simd-mode auto|scalar|avx2|avx512`.
- [ ] Add benchmark detail/summary columns for SIMD MBB counters.
- [ ] Record `simd_mode` in benchmark JSON profile metadata.
- [ ] Add query benchmark JSON assertions for optimized `simd_mode`.
- [ ] Run `cd navigamer_cpp && make test_query_benchmark`.
- [ ] Run `cd navigamer_cpp && make test_all`.
- [ ] Run `cd navigamer_cpp && ./navigamer demo --size 200`.
- [ ] Run a small query benchmark with `--simd-mode auto` and verify zero mismatches/FNs.
- [ ] Run `git diff --check` and `git status --short`.
- [ ] Commit with `git commit -m "validate simd mbb filter mode"`.

