# Search Q-Gram Safe Prefilter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional no-false-negative q-gram prefilter between adaptive child MBB filtering and exact child-center edit-distance verification.

**Architecture:** Extend the existing q-gram module with compact immutable signatures, cache primary-world signatures inside each search engine, and pass one per-query signature through adaptive traversal. Preserve all construction, leaf-refinement, containment, and overlap semantics while exposing search-specific counters and CLI flags.

**Tech Stack:** C++17, OpenMP, Make, CMake, existing NavigaMer edit-distance and test harnesses.

---

### Task 1: Compact Q-Gram Signature API

**Files:**
- Modify: `navigamer_cpp/include/qgram_filter.hpp`
- Modify: `navigamer_cpp/src/qgram_filter.cpp`
- Modify: `navigamer_cpp/src/test_qgram_filter.cpp`

- [ ] Add failing tests that require `QGramSignature`, validate q=2 counts for `ACGT` and `ACGTAC`, check randomized no-false-negative pruning, and verify non-ACGT/invalid-q conservative fallback.
- [ ] Run `cd navigamer_cpp && make test_qgram && ./test_qgram_filter`; expect compilation failure because the signature API is absent.
- [ ] Add `QGramEntry`, `QGramSignature`, `compute_qgram_signature`, `qgram_l1_distance`, and `qgram_can_prune_edit_distance`. Encode A/C/G/T in two bits, sort/count entries, and return no-prune for unsafe or incompatible signatures.
- [ ] Run `cd navigamer_cpp && make test_qgram && ./test_qgram_filter`; expect all q-gram tests to pass.
- [ ] Commit only the signature API and tests with `git commit -m "add compact qgram signatures"`.

### Task 2: Adaptive Search Integration And Equivalence Tests

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Create: `navigamer_cpp/src/test_search_qgram_prefilter.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] Add a failing `test_search_qgram_prefilter` that compares scan/rect crossed with q-gram on/off, covers ambiguous-N fallback and strict containment, and asserts q-gram/call-count invariants.
- [ ] Register the test in Make and CMake, then run `cd navigamer_cpp && make test_search_qgram`; expect compilation failure because search q-gram config and counters are absent.
- [ ] Add `search_qgram_prefilter` and `search_qgram_q` to `SearchConfig`, add required counters and prune-ratio helper to `SearchStats`, and add an engine-owned node-ID signature cache built from finalized primary layers.
- [ ] Compute the query signature once in `search_adaptive`, pass it through adaptive helpers, and apply safe pruning only to MBB-surviving child worlds immediately before bounded exact center verification.
- [ ] Preserve strict containment and overlap handling; leave coarsest candidates, leaf refinement, construction joins, greedy, exhaustive, and brute-force paths unchanged.
- [ ] Run `cd navigamer_cpp && make test_search_qgram && ./test_search_qgram_prefilter`; expect all search q-gram tests to pass.
- [ ] Run `cd navigamer_cpp && make test_mbb_filter test_recall test_dist`; expect all existing adaptive-search guards to pass.
- [ ] Commit the adaptive search integration and test with `git commit -m "add adaptive search qgram prefilter"`.

### Task 3: CLI Flags And Instrumentation Output

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/test_search_stats.cpp`

- [ ] Extend the search-stats test to assert default-off behavior, enabled q/q-build counters, before/after call invariants, and result counts.
- [ ] Run `cd navigamer_cpp && make test_search_stats`; expect failure until the CLI-facing statistics behavior exists.
- [ ] Parse `--search-qgram-prefilter off|on` and `--search-qgram-q N`, populate `SearchConfig`, and include the flags in usage text.
- [ ] Extend adaptive query summary and benchmark TSV with enabled/q/build/missing/check/pruned/passed/before/after/prune-ratio/result-count fields while retaining `center_distance_calls_after_mbb`.
- [ ] Run `cd navigamer_cpp && make test_search_stats && ./test_search_stats_bin`; expect the statistics test to pass.
- [ ] Build the CLI and run one adaptive query with the prefilter off and on; expect identical hit IDs and visible q-gram counters.
- [ ] Commit CLI and instrumentation changes with `git commit -m "expose search qgram instrumentation"`.

### Task 4: Documentation, Full Validation, And Benchmark

**Files:**
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`
- Create: `navigamer_cpp/SEARCH_QGRAM_PREFILTER_BENCHMARK.md`

- [ ] Document safety semantics, defaults, search/construction q separation, fallback behavior, scaling boundary, and benchmark columns.
- [ ] Prepare deterministic 250 bp query FASTQ files under `.tmp_experiments/` from the first 2 kb and first 10 kb of `data/human/chr1_subset`; do not commit generated inputs or outputs.
- [ ] Run scan/off, scan/on, rect/off, and rect/on benchmarks for 2 kb; compare result tuples and summarize timing/counters.
- [ ] Run the same four modes for 10 kb if runtime is practical; otherwise record the measured blocker.
- [ ] Write `SEARCH_QGRAM_PREFILTER_BENCHMARK.md` with commands, equality result, average query time, MBB survivors, q-gram checks/prunes, before/after calls, prune ratio, and result count.
- [ ] Run `cd navigamer_cpp && make test_all`; expect every existing and new test to pass.
- [ ] Run `cd navigamer_cpp && make -j`; expect the CLI build to pass.
- [ ] Run `cd navigamer_cpp && ./navigamer demo --size 200 --search-qgram-prefilter on --search-qgram-q 5`; expect the smoke run to complete without recall loss.
- [ ] Run `git diff --check`; expect no whitespace errors.
- [ ] Commit docs and benchmark summary with `git commit -m "document search qgram prefilter benchmark"`.
