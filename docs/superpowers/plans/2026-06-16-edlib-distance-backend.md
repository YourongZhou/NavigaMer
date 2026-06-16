# Edlib Distance Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Edlib as an optional query and build edit-distance backend without changing default search or construction semantics.

**Architecture:** Vendor Edlib source into `navigamer_cpp/third_party/edlib`, wrap it behind NavigaMer distance helpers, and expose it through `--distance-mode edlib` plus separate `--build-distance-mode dp|edlib|auto`. Query defaults remain Myers; build defaults remain DP.

**Tech Stack:** C++17, Makefile, CMake, OpenMP, vendored Edlib C++ source.

---

### Task 1: Bounded Edlib Wrapper

**Files:**
- Modify: `navigamer_cpp/include/tools.hpp`
- Modify: `navigamer_cpp/src/tools.cpp`
- Modify: `navigamer_cpp/src/test_bounded_myers.cpp`

- [ ] Add failing tests that parse `edlib`, compare Edlib bounded output against full DP for random DNA strings, indels, and `N` inputs, and assert `DistanceMode::Auto` still routes to DP.
- [ ] Add `DistanceMode::Edlib`, `compute_distance_bounded_edlib()`, and route `compute_distance_bounded_with_mode(..., Edlib)` to the wrapper.
- [ ] Keep `compute_distance_bounded()` as DP reference.

### Task 2: Build Integration

**Files:**
- Create: `navigamer_cpp/third_party/edlib/include/edlib.h`
- Create: `navigamer_cpp/third_party/edlib/src/edlib.cpp`
- Create: `navigamer_cpp/third_party/edlib/LICENSE`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] Vendor Edlib header/source/license from `Martinsos/edlib`.
- [ ] Add the third-party include path and source file to CLI and test builds.
- [ ] Verify `make test_bounded_myers` compiles and links Edlib.

### Task 3: Query Search Mode

**Files:**
- Modify: `navigamer_cpp/src/test_search_distance_mode.cpp`
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`

- [ ] Extend search distance-mode equivalence tests to include Edlib.
- [ ] Accept `--distance-mode edlib`.
- [ ] Ensure query benchmark JSON/TSV reports `edlib` when selected.

### Task 4: Build Distance Mode

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/main.cpp`
- Create or modify: `navigamer_cpp/src/test_build_distance_mode.cpp`

- [ ] Add `BuildDistanceMode dp|edlib|auto`, default `dp`.
- [ ] Route exact and bounded construction distance calls through build-distance helpers.
- [ ] Add build equivalence tests comparing graph structure and search results for DP vs Edlib builds.

### Task 5: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] Document `--distance-mode edlib` and `--build-distance-mode`.
- [ ] Run `make -j`, `make test_all`, and targeted query/build benchmarks.
- [ ] Commit and push only source, tests, vendored Edlib files, and docs.
