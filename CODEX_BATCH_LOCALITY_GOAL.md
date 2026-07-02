# Codex Goal: Batch-Locality Competitive NavigaMer Query Case

## Objective

Find and, if needed, implement a correctness-preserving NavigaMer C++ query
path that is faster in a clearly scoped similar-query batch-locality case.

This is not a claim that NavigaMer is generally faster than q-gram,
pigeonhole, minimap2, strobealign, or every baseline. The target claim is
narrower and more defensible:

> When many similar queries are searched as a batch, NavigaMer can exploit
> traversal locality and safe reuse to beat the current internal adaptive
> baseline, and should be tested against existing q-gram/pigeonhole candidate
> retrieval baselines on that same special case.

Do not mark this goal complete based only on a single low-fanout or noisy
microbenchmark.

## Non-Negotiable Safety Contract

- Exact edit-distance verification remains the final authority for returned
  hits.
- Optimized results must have no false negatives relative to baseline adaptive
  and the repository brute-force/exhaustive regression protocol.
- Query ordering, router hints, cached shortlists, anchor caches, safe-child
  candidate caches, and productive-world reuse may never suppress exact
  verification or a required no-FN fallback.
- Any reused candidate set must be keyed by exact query-derived signatures and
  guarded so it is a safe superset for the query being answered. If that safety
  cannot be established, fall back to the normal safe path.

## Required Context To Read First

1. `AGENTS.md`
2. `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, and `CODEX_TEST_LOG.md`
3. `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/RESULTS_1P1M_SUMMARY.md`
4. Current query scheduling/reuse code in `navigamer_cpp/src/search_engine.cpp`
   and `navigamer_cpp/src/query_benchmark.cpp`
5. Current baseline/candidate comparison scripts or tables under
   `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/` when available

## Implementation Direction

1. Build or reuse a deterministic similar-query workload:
   - repeated queries,
   - adjacent sliding-window queries,
   - source-sorted batches,
   - q-gram/minimizer/router-signature sorted batches.

2. Make batch-locality reuse real and measurable:
   - preserve user-visible output order after internal scheduling,
   - reuse safe anchor distances, child shortlists, safe-child candidate sets,
     and productive worlds where valid,
   - expose counters including `path_reuse_hit_ratio`,
     `anchor_cache_hit_count`, `child_shortlist_cache_hit_count`,
     `safe_child_candidate_cache_hit_count`, and
     `productive_world_reuse_hit_count`.

3. Compare in layers:
   - first against the current internal adaptive baseline in the same C++
     benchmark harness,
   - then against the existing q-gram q5 and pigeonhole tau5 candidate
     retrieval baselines from the 1.1M workflow if those scripts/data are
     available and practical,
   - optionally compare minimap2/strobealign only if already installed and
     easy to run; do not block completion on external mapper setup.

4. Keep safe-child-router work only if it helps the batch-locality case:
   - high fallback or candidate ratio near 1.0 means it is not the main lever,
   - low-fanout workloads should not pay router overhead,
   - a selective safe-child cache is valuable only when it reduces repeated
     child enumeration without false negatives.

## Required Benchmarks

Use deterministic commands and record exact outputs in `CODEX_TEST_LOG.md`.

Minimum evidence:

- Build and correctness:
  - `cd navigamer_cpp && make -j`
  - `cd navigamer_cpp && make test_recall && ./test_recall`
  - `cd navigamer_cpp && make test_distance_bound && ./test_distance_bound`
  - focused no-FN regressions for safe-child/router/path-reuse/query-planner
    if present in the Makefile or build outputs.

- Internal batch-locality benchmark:
  - at least one repeated-query workload,
  - at least one adjacent/source-local workload,
  - at least one sorted signature workload,
  - `mismatch_count=0`,
  - p95 or wall-clock speedup over internal adaptive baseline reproduced at
    least three times for the claimed scenario.

- External/candidate comparison attempt:
  - q-gram q5 and pigeonhole tau5 if local 1.1M scripts/data are available,
  - same or clearly matched similar-query workload,
  - report candidate counts, total/p95 time, and source recovery/no-FN status.

## Completion Criteria

Do not set `CODEX_PROGRESS.md` to `State: complete` unless all are true:

1. C++ build and no-FN regression tests pass.
2. The claimed batch-locality workload reports `mismatch_count=0`.
3. The optimized path beats the internal adaptive baseline on a deterministic
   similar-query workload with stable evidence:
   - target: `p95_speedup_vs_baseline >= 1.2` or clear wall-clock speedup,
   - reproduced at least three times, or one larger deterministic run whose
     wall-clock timing dominates noise.
4. Reuse counters show actual reused work, not just sorting:
   - `path_reuse_hit_ratio >= 0.8`,
   - at least one of `anchor_cache_hit_count`,
     `child_shortlist_cache_hit_count`,
     `safe_child_candidate_cache_hit_count`, or
     `productive_world_reuse_hit_count` is positive.
5. A q-gram/pigeonhole comparison has been attempted on the closest practical
   matched workload and recorded with numbers.
6. `CODEX_TEST_LOG.md` records exact commands, exit statuses, benchmark rows,
   and the final interpretation. If the external q-gram/pigeonhole baseline
   still wins, complete only if the internal batch-locality win is real and the
   blocker is documented quantitatively.

## Preferred Final Interpretation

If the experiments pass, write the result as a scoped case:

> NavigaMer does not currently dominate simple safe q-gram/pigeonhole filters
> on the general 1.1M retrieval benchmark. However, in a constructed
> similar-query batch-locality workload, correctness-preserving scheduling and
> reuse reduce repeated traversal work and beat the internal adaptive baseline;
> external candidate baselines were then tested separately under the same
> locality condition.

