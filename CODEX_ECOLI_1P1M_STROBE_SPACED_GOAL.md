# Codex Goal: E. coli 1.1M No-FN Faster Than Strobemer And Spaced Seed

## Objective

Optimize the NavigaMer C++ query path until it is faster than strobemer and
spaced-seed baselines on an E. coli 1.1M benchmark, while preserving exact
edit-distance verification and no false negatives.

This is a scoped competitive-performance goal. Do not complete it by comparing
against low-recall seed hits. The final comparison must use the same E. coli
1.1M reference/query workload and must report exact-verified recovery/no-FN
status for NavigaMer and for the seed-based baselines.

## Primary Claim To Earn

> On E. coli 1.1M, NavigaMer is faster than strobemer and spaced seed for a
> clearly defined repetitive or batch-locality query workload, while preserving
> exact edit-distance verification and no false negatives.

Do not claim broad speed superiority over all workloads unless the data
actually support it.

## Non-Negotiable Safety Contract

- Exact edit distance is the final authority for every returned hit.
- NavigaMer optimized output must have no false negatives relative to the
  repository exhaustive/brute-force protocol on sampled oracle subsets.
- Any cache, query scheduling, router hint, child shortlist, anchor-distance
  reuse, safe-child candidate reuse, or productive-world reuse must be
  correctness-preserving.
- Cached final hits are valid only if they are exact-verified for the query and
  tolerance being answered, or if a stored exact-distance certificate proves
  they still satisfy the current query/tolerance.
- If a reused candidate/path/shortlist is not provably safe, fall back to the
  normal adaptive search path.
- Strobemer and spaced-seed comparisons must not be treated as no-FN unless
  exact verification and source/oracle recovery checks are run and recorded.

## Required Context To Read First

1. `AGENTS.md`
2. `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, and `CODEX_TEST_LOG.md`
3. `CODEX_ECOLI_INDEX_HINT.md` if present. Prefer the existing NFS
   `w150_s1.navidx` index documented there before attempting to rebuild a
   1.1M/full E. coli index.
4. `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/RESULTS_1P1M_SUMMARY.md`
5. The local candidate baseline implementation under
   `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/src/`, especially:
   - `candidate_tool.cpp`
   - `candidate_indexes.hpp`
   - `randstrobe_index.cpp`
   - `spaced_seed_index.cpp`
6. Current NavigaMer query scheduling/reuse code in:
   - `navigamer_cpp/src/search_engine.cpp`
   - `navigamer_cpp/src/query_benchmark.cpp`
   - `navigamer_cpp/src/main.cpp`

## Workload Design

Use E. coli 1.1M as the final benchmark scale. If a smaller prefix is needed
for rapid debugging, it is allowed only as an intermediate step and must not be
the final completion evidence.

Required final workload family:

1. Exact repeated or duplicated-real-read batch:
   - many repeated query sequences,
   - tolerance 5 or 8,
   - length 150,
   - sufficient query count to make wall-clock timing meaningful,
   - exact output order preserved after any internal scheduling.

2. At least one broader locality coverage workload:
   - source-sorted nearby reads, adjacent/sliding-window reads, or mixed
     repeat + near-repeat + random queries,
   - used to show whether the speedup is narrow or generalizing.

The first accepted success case may be repetitive, but it must be explicit.

## Optimization Direction

Prioritize changes that can plausibly beat strobemer/spaced seed under exact
verification:

1. Batch-level exact-query and duplicate-read reuse:
   - group identical query sequence + tolerance pairs,
   - run adaptive search once per unique query,
   - exact-verify/certify returned hits,
   - broadcast verified results back to all duplicate query IDs.

2. Persisted-index 1.1M query execution:
   - avoid rebuilding the index inside repeated benchmark loops,
   - reuse the loaded/persisted index across schedules and repetitions.

3. Signature-sorted locality execution:
   - q-gram/minimizer/router-signature sorting,
   - preserve output order,
   - record neighbor similarity and reuse counters.

4. Safe near-query reuse only if there is time:
   - reuse candidate supersets, child shortlists, or productive worlds between
     similar but non-identical queries,
   - never prune solely from an unsafe similarity heuristic.

5. Planner gating:
   - low duplicate/locality workloads should avoid cache/scheduler overhead,
   - high duplicate/locality workloads should batch unique queries first.

## Baseline Requirements

Run or repair local 1.1M baseline comparisons against:

- strobemer/randstrobe candidate retrieval,
- spaced-seed candidate retrieval.

Use local implementations where practical:

- `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/src/randstrobe_index.cpp`
- `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/src/spaced_seed_index.cpp`
- `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/candidate_tool`

For each baseline row report:

- build time if applicable,
- query wall-clock time,
- mean and p95 candidate count,
- source recovery rate,
- exact-verified recall/no-FN on the sampled oracle subset if available,
- whether the baseline is safe under the requested edit-distance tolerance.

If strobemer/spaced seed are faster only by missing true hits, say that clearly.

## Required Benchmarks And Validation

Record exact commands, exit status, and key rows in `CODEX_TEST_LOG.md`.

Correctness gates:

- `cd navigamer_cpp && make -j`
- `cd navigamer_cpp && make test_recall && ./test_recall`
- `cd navigamer_cpp && make test_distance_bound && ./test_distance_bound`
- focused no-FN tests for safe-child/router/path-reuse/query-planner/query-
  benchmark if they exist
- a final E. coli 1.1M sampled oracle/no-FN check with `false_negative_count=0`
  or `mismatch_count=0`

Performance gates:

- E. coli 1.1M NavigaMer optimized repeated/duplicated workload:
  - `mismatch_count=0`,
  - `path_reuse_hit_ratio >= 0.8` or equivalent duplicate-group hit ratio,
  - positive reused-work counter such as `productive_world_reuse_hit_count`,
    `verified_result_cache_hit_count`, `child_shortlist_cache_hit_count`, or
    `safe_child_candidate_cache_hit_count`.

- E. coli 1.1M strobemer and spaced-seed matched workload:
  - same reads or clearly matched query family,
  - exact-verified source/oracle recovery recorded,
  - wall-clock measured from the same machine/session where possible.

Completion speed target:

- NavigaMer optimized wall-clock must beat strobemer by at least `1.2x`.
- NavigaMer optimized wall-clock must beat spaced seed by at least `1.2x`.
- The winning result must be reproduced at least three times, or one larger
  deterministic run must be long enough that the difference is not timing
  noise.

## Completion Criteria

Do not set `CODEX_PROGRESS.md` to `State: complete` unless all are true:

1. All required C++ correctness tests pass.
2. Final benchmark scale is E. coli 1.1M, not only a small prefix.
3. NavigaMer optimized query has no false negatives on the final benchmark
   oracle protocol.
4. Strobemer and spaced-seed baselines are run or a hard blocker is documented
   with exact missing command/dependency/data.
5. NavigaMer optimized query beats both strobemer and spaced seed on the final
   matched workload under exact-verified/no-FN accounting, or the goal remains
   `in_progress`.
6. `CODEX_TEST_LOG.md` contains the commands, rows, speedups, and final
   interpretation.

If correctness passes but speed does not beat both baselines, keep working.
Possible next levers are duplicate grouping before search, persisted-index
reuse, near-query safe candidate reuse, and planner gating.

## Final Interpretation Template

If successful, use scoped wording:

> On E. coli 1.1M, for a repetitive batch-locality workload, NavigaMer's
> exact-verified query reuse outperforms strobemer and spaced seed while
> preserving no false negatives.

Also report the boundary:

> This does not establish that NavigaMer is faster on random non-local query
> workloads; it establishes a correctness-preserving advantage in the measured
> repetitive/locality case.
