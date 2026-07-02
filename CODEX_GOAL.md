# Codex Goal: E. coli 1.1M Real-Read Batch Locality V2

## Objective

Optimize and evaluate NavigaMer C++ query performance on E. coli 1.1M using
more realistic batch-locality workloads:

1. duplicated real reads, and
2. source-sorted near-repeat reads.

The goal is to preserve exact edit-distance verification and no false
negatives while demonstrating that real query locality, not only synthetic
exact-repeat queries, can reduce traversal work. The duplicated-real-read
workload is the primary speed target. The source-sorted near-repeat workload is
the primary generalization test.

## Primary Claim To Earn

> On E. coli 1.1M, NavigaMer exploits real-read batch locality through exact
> duplicate grouping and correctness-preserving near-query reuse. In duplicated
> real-read workloads, verified-result reuse reduces repeated traversal while
> preserving no false negatives. Source-sorted near-repeat workloads expose
> traversal locality separately from exact duplicate reuse.

If the data support it, also claim speed wins against strobemer/randstrobe and
spaced seed under exact-verified accounting. Do not claim random-workload speed
superiority unless directly measured.

## Non-Negotiable Safety Contract

- Exact edit distance is the final authority for every returned hit.
- NavigaMer optimized output must have no false negatives relative to the
  repository exhaustive/brute-force protocol on sampled oracle subsets.
- Duplicate grouping may reuse final hits only for identical query sequence +
  tolerance pairs, and those hits must be exact-verified or carry exact
  distance certificates.
- Near-repeat reuse may reuse hints, caches, candidate supersets, child
  shortlists, anchor distances, and productive worlds only when doing so cannot
  suppress a possible true hit.
- Similarity signatures, source sorting, q-gram signatures, minimizers, router
  signatures, and previous productive worlds may improve ordering or cache
  lookup, but may not become unsafe pruning reasons.
- If any reused near-query state is uncertain, stale, too narrow, or not
  provably safe, fall back to the normal adaptive path.

## Required Context To Read First

1. `AGENTS.md`
2. `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, and `CODEX_TEST_LOG.md`
3. `CODEX_ECOLI_1P1M_STROBE_SPACED_GOAL.md`
4. `CODEX_ECOLI_INDEX_HINT.md` if present. Prefer the existing NFS
   `w150_s1.navidx` index documented there before attempting to rebuild a
   1.1M/full E. coli index.
5. `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/RESULTS_1P1M_SUMMARY.md`
6. Current NavigaMer query scheduling/reuse code in:
   - `navigamer_cpp/src/search_engine.cpp`
   - `navigamer_cpp/src/query_benchmark.cpp`
   - `navigamer_cpp/src/main.cpp`
7. Local candidate baseline code under
   `.worktrees/ecoli-comparison/experiments/ecoli_1p1m/src/`, especially:
   - `candidate_tool.cpp`
   - `randstrobe_index.cpp`
   - `spaced_seed_index.cpp`

If `CODEX_ECOLI_STROBE_TO_REAL_LOCALITY_HANDOFF.md` exists, read it before
choosing the first implementation step.

## Workload Design

Final evidence must use E. coli 1.1M scale. Smaller prefixes are allowed only
for debugging or red/green regression tests.

### Workload A: Duplicated Real Reads

Generate or reuse real mutated reads from E. coli 1.1M, then duplicate the
same real read sequences.

Required variants:

- `real_dup_1x`: unique real reads, no deliberate duplicates; baseline
  overhead check.
- `real_dup_4x`: each real read duplicated four times.
- `real_dup_16x`: each real read duplicated sixteen times.

At least one duplicated variant must be source-sorted and at least one must be
shuffled, so duplicate grouping is measured separately from source-order
locality.

Required metrics:

- `query_count`
- `unique_query_count`
- `duplicate_group_count`
- `duplicate_ratio`
- `verified_result_cache_hit_count`
- `mismatch_count`
- `false_negative_count` or sampled oracle equivalent
- total wall-clock time
- mean and p95 query time
- speedup versus internal adaptive baseline
- speedup versus strobemer/randstrobe and spaced seed if matched baselines run

### Workload B: Source-Sorted Near-Repeat Reads

Generate or reuse reads from adjacent or nearby E. coli 1.1M source positions,
sorted by source position or by a source-local proxy. Reads should be similar
but not all identical.

Required variants:

- `source_sorted_stride1`: adjacent windows or adjacent source positions.
- `source_sorted_mutated_tau5`: nearby reads with edits/mutations under
  tolerance 5.
- `source_sorted_mutated_tau8` or a harder near-repeat setting if practical.

Required metrics:

- `mean_neighbor_edit_distance`
- `p95_neighbor_edit_distance`
- `mean_neighbor_qgram_jaccard` or equivalent signature similarity
- `near_query_reuse_hit_count`
- `anchor_cache_hit_count`
- `child_shortlist_cache_hit_count`
- `safe_child_candidate_cache_hit_count`
- `productive_world_reuse_hit_count`
- `mismatch_count`
- `false_negative_count` or sampled oracle equivalent
- total wall-clock time
- mean and p95 query time

For near-repeat rows, do not reuse final hits unless the query is exactly
identical. The goal is safe traversal/candidate reuse, not unsafe answer reuse.

### Optional Workload C: Mixed Locality

If time permits, run:

- 50% duplicated real reads,
- 30% source-sorted near-repeat reads,
- 20% random real reads.

Use this as a reality check for how narrow the speedup is.

## Optimization Direction

1. Duplicate grouping before search:
   - group by exact query sequence + tolerance,
   - search each unique query once,
   - exact-verify hits once,
   - broadcast verified hits back to duplicate query IDs,
   - preserve user-visible output order.

2. Verified-result cache:
   - store exact verified hits or exact distance certificates,
   - expose `verified_result_cache_hit_count`,
   - ensure cache keys include tolerance and any relevant verification mode.

3. Near-query safe reuse:
   - use source/order/q-gram/minimizer/router-signature sorting,
   - reuse anchor distances only under exact matching keys or safe bounds,
   - reuse child shortlists and safe-child candidate sets only as supersets,
   - reuse productive worlds only as hints unless a safety proof exists.

4. Planner gating:
   - high duplicate ratio: enable duplicate grouping and verified-result cache,
   - high neighbor similarity: enable near-query reuse,
   - low locality: disable expensive locality machinery.

5. Persisted-index execution:
   - avoid rebuilding the 1.1M index inside repeated timing loops,
   - reuse loaded/persisted index for all schedule/baseline repetitions.

## Baseline Requirements

Compare against matched strobemer/randstrobe and spaced-seed rows when local
baseline tooling is practical.

For each baseline row report:

- same reads or clearly matched workload,
- build time if applicable,
- query wall-clock time,
- mean and p95 candidate count,
- source recovery rate,
- exact-verified recall/no-FN on sampled oracle subset when available,
- whether the baseline is safe under the requested tolerance.

Do not count a baseline as correctness-equivalent if it skips exact
verification or misses source/oracle hits.

## Required Validation

Record exact commands, exit status, and key rows in `CODEX_TEST_LOG.md`.

Required correctness gates:

- `cd navigamer_cpp && make -j`
- `cd navigamer_cpp && make test_recall && ./test_recall`
- `cd navigamer_cpp && make test_distance_bound && ./test_distance_bound`
- focused no-FN tests for safe-child/router/path-reuse/query-planner/query-
  benchmark if present
- final E. coli 1.1M sampled oracle/no-FN check with `false_negative_count=0`
  or `mismatch_count=0`

Required performance gates:

- duplicated real-read workload:
  - `mismatch_count=0`,
  - positive `verified_result_cache_hit_count`,
  - duplicate grouping reduces searched unique queries,
  - target speedup over internal adaptive baseline: `>= 1.2x`,
  - target speedup over strobemer/randstrobe and spaced seed: `>= 1.2x` if
    exact-verified matched baselines run.

- source-sorted near-repeat workload:
  - `mismatch_count=0`,
  - positive near-query reuse counter or a measured blocker explaining why
    safe near-query reuse did not activate,
  - report whether wall-clock improves; speed win is a stretch goal, not the
    only useful outcome for this workload.

Repeat the claimed speed result at least three times, or run one deterministic
large enough benchmark where timing noise is clearly dominated.

## Completion Criteria

Do not set `CODEX_PROGRESS.md` to `State: complete` unless all are true:

1. Final evidence uses E. coli 1.1M scale.
2. Required C++ correctness tests pass.
3. Duplicated real-read workload has exact verification/no-FN evidence.
4. Source-sorted near-repeat workload has exact verification/no-FN evidence.
5. Duplicate grouping and verified-result cache counters show real reused
   work on duplicated real reads.
6. Near-repeat rows either show safe reuse and timings, or document a measured
   blocker with enough diagnostics to decide the next optimization.
7. Strobemer/randstrobe and spaced-seed baselines are run on matched workloads
   when practical, or hard blockers are documented.
8. `CODEX_TEST_LOG.md` records commands, rows, speedups, and the final
   interpretation.

If duplicated real reads pass but source-sorted near-repeat does not show
speedup, completion is allowed only if no-FN is proven and the near-repeat
blocker is quantitatively documented. If duplicated real reads do not pass
no-FN or do not show real reused work, keep the goal `in_progress`.

## Final Interpretation Template

If successful:

> On E. coli 1.1M, NavigaMer's V2 real-read locality path groups duplicated
> real reads and reuses exact-verified results without false negatives. In
> source-sorted near-repeat reads, NavigaMer preserves no-FN and reports safe
> reuse diagnostics separately from exact duplicate reuse.

Boundary:

> This is a batch-locality result, not a random-query speed claim.
