# Selectivity-Aware Auto Range Join Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make auto construction range joins detect large pigeonhole candidate sets and safely switch to q-gram or hybrid candidates.

**Architecture:** `ExactRangeJoinIndex` owns the two-stage selectivity decision and exposes per-query decision statistics. `BioGeometryIndexBuilder` only aggregates those statistics and continues exact bounded verification before materializing edges or leaf attachments.

**Tech Stack:** C++17, STL, GNU Make, CMake, existing NavigaMer range-join and construction tests.

---

### Task 1: Selectivity-Aware Auto Policy

**Files:**
- Modify: `navigamer_cpp/include/range_join.hpp`
- Modify: `navigamer_cpp/src/range_join.cpp`
- Modify: `navigamer_cpp/src/test_range_join.cpp`

- [ ] Add a deterministic overlap-heavy test whose valid pigeonhole seeds return
  most targets while q-gram prunes targets. Assert default auto returns forced
  hybrid candidates and records rejection/q-gram/hybrid counters.
- [ ] Run `cd navigamer_cpp && make test_range_join_check`; expect compilation
  failure because auto config/stat fields do not exist.
- [ ] Add config defaults `4096`, `0.25`, and `true`, plus query-result auto
  decision fields.
- [ ] Implement auto: short seed returns q-gram; otherwise accept pigeonhole
  when count threshold OR ratio threshold passes; rejected large results invoke
  q-gram and return intersection or direct q-gram according to the flag.
- [ ] Add tests for hybrid disabled, permissive old-auto thresholds, inclusive
  threshold boundaries, and exact-verified equality with full matches.
- [ ] Run `make test_range_join_check`; expect PASS.

### Task 2: Builder Statistics and Construction Equivalence

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/test_build_range_equivalence.cpp`

- [ ] Add failing assertions that overlap-heavy auto construction invokes
  hybrid and remains exactly equal to full edges, leaf attachments, and search
  results.
- [ ] Run `make test_build_range`; expect compilation failure for missing
  builder auto statistics.
- [ ] Add phase-2 and leaf auto counters, aggregate query-result fields, compute
  average measured pigeonhole ratio, and print them in builder summaries.
- [ ] Run `make test_build_range`; expect PASS.

### Task 3: CLI and Documentation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] Run a CLI smoke command with new flags and confirm the current summary
  does not expose configured thresholds.
- [ ] Add parsing and validation for
  `--auto-pigeonhole-max-candidates`,
  `--auto-pigeonhole-max-ratio`, and
  `--auto-hybrid-on-large-candidates`.
- [ ] Document defaults, OR acceptance semantics, safe hybrid behavior, auto
  counters, and old-auto reproduction flags.
- [ ] Run qgram/hybrid/new-auto/old-auto CLI smoke commands and invalid
  ratio/boolean checks.

### Task 4: Full Validation and 2 kb Benchmark

**Files:**
- Create: `docs/benchmarks/2026-06-14-selectivity-aware-auto-range-join.md`

- [ ] Run `cd navigamer_cpp && make -j && make test_all`.
- [ ] Run a Release CMake build in `/home/tmp/navigamer-auto-build`.
- [ ] Benchmark first 2,000 bp of `data/human/chr1_subset`, 250 bp windows,
  stride 1, comparing full, pigeonhole, qgram, hybrid, old-auto, and new-auto.
- [ ] Compare sorted search-result keys across all modes and record exact calls,
  accepted results, wall time, auto decisions, and equality evidence.
- [ ] Run `git diff --check`, commit only source/tests/docs, and leave generated
  binaries and object files uncommitted.
