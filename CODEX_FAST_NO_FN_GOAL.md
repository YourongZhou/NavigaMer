# Codex Goal: Faster NavigaMer Query With No False Negatives

## Objective

Repair the NavigaMer C++ query path until the optimized mapper is faster than
the current baseline adaptive algorithm on a focused high-fanout/ambiguous
benchmark, while preserving no-false-negative behavior.

This goal supersedes the earlier 2026-07-01 diagnostic-only focus. Diagnostics
are still useful, but they are not sufficient. Do not mark this goal complete
until the optimized path beats baseline under the acceptance gates below.

## Non-Negotiable Safety Contract

- Exact edit-distance verification remains the final authority for returned
  hits.
- Safe-child candidate routing may prune child enumeration only when the
  candidate set is a mathematically safe superset of all children that could
  contain a hit.
- Router hints, local-router ordering, best-first ordering, q-gram/minimizer
  hints, and path reuse may never become sole unsafe pruning reasons.
- Optimized results must have no false negatives relative to baseline adaptive
  and the repository brute-force/exhaustive regression protocol.
- If a candidate generator is uncertain, too broad, inconsistent, or fails its
  safety checks, it must fall back to full safe enumeration.

## Required Implementation Direction

1. Prioritize query-similarity scheduling and real traversal reuse:
   - Implement or improve deterministic batch ordering so similar query
     sequences are searched adjacent to each other.
   - Prefer signatures that correlate with shared traversal: minimizer-sorted,
     q-gram-signature sorted, router-signature sorted, or a hybrid of these.
   - Preserve user-visible output order even when internal query execution is
     reordered.
   - Reuse real work across similar queries where safe: anchor distances,
     child shortlists, safe-child candidate sets, productive worlds, and
     traversal warm-starts. Reuse must be keyed by exact query-derived
     signatures and must never suppress exact verification or no-FN fallback.
   - Add counters for the reuse path if missing:
     `query_similarity_schedule_enabled`, `query_similarity_cluster_count`,
     `query_similarity_mean_neighbor_distance`,
     `anchor_cache_hit_count`, `child_shortlist_cache_hit_count`,
     `safe_child_candidate_cache_hit_count`, and
     `productive_world_reuse_hit_count`.

2. Add the missing fine-grained safe-child-router diagnostics:
   - `child_count_before_router`
   - `post_mbb_survivor_count`
   - `safe_router_candidate_count`
   - `candidate_ratio_to_all_children`
   - `candidate_ratio_to_post_mbb_survivors`
   - `children_actually_processed`
   - `center_checks_saved`
   These must be visible in query/query-index profiling output and in
   query-benchmark/locality benchmark summaries.

3. Verify and enforce accepted-candidate enumeration:
   - Accepted safe-child router candidate set: enumerate only candidate
     children.
   - Fallback: enumerate all children.
   - Do not silently do candidate-first plus remaining fallback in the safe
     router path. That is only valid for hint ordering, not for work reduction.

4. Make safe-child routing selective:
   - First implement or fix radius-bucketed safe child routing so children are
     queried with `tau = tolerance + bucket_radius`, not a parent-wide maximum
     child radius.
   - Prefer child-specific or radius-bucket-specific postfilters before
     accepting the candidate set.
   - If q-gram child-center routing remains non-selective, add an
     anchor/MBB-based safe child router mode using local anchor distances and
     child MBB intervals:
     `d(query, anchor) < child.min_anchor_dist - tolerance` or
     `d(query, anchor) > child.max_anchor_dist + tolerance` safely rejects a
     child.

5. Fix or clarify proximal-oracle diagnostics only insofar as needed to guide
   performance work:
   - Per-query detail should make global/frontier/true-path anchor sets
     auditable.
   - If an envelope metric is an aggregate distance score that can grow with
     `k`, label it clearly and add a true tightness metric such as interval
     width, candidate envelope size, or max lower-bound exposure.

6. Add or update tests before production code where behavior changes:
   - Regression must fail before the fix when checking the new diagnostics or
     accepted-candidate enumeration behavior.
   - Existing no-FN tests must continue to pass.

## Required Benchmarks

Use deterministic benchmark inputs. Keep build sizes practical enough for
automation, but the workload must actually exercise high-fanout candidate
reduction.

Minimum benchmark evidence:

- Safe-child sweep:
  `--safe-child-router-min-fanout 1,16,32,64`
  and `--safe-child-router-max-ratio 0.1,0.25,0.5,1.0`.
- Query-similarity scheduling benchmark:
  compare original order vs minimizer/q-gram/router-signature sorted order on
  repeat, batch-locality, and ambiguous high-fanout workloads. Report
  path/cache reuse hit ratios, p95 speedup, and no-FN/mismatch gates.
- High-fanout benchmark reporting:
  `mean_fanout`, `p95_fanout`, `max_fanout`,
  `safe_child_router_invoked_ratio`, `router_invoked_ratio`,
  world-access reduction, center-distance reduction, p95 speedup.
- No-FN/correctness gates:
  `test_recall`, `test_distance_bound`, safe-router regression tests, and at
  least one CLI benchmark gate with `mismatch_count=0`.

## Completion Criteria

Do not set `CODEX_PROGRESS.md` to `State: complete` unless all are true:

1. `make -j` succeeds in `navigamer_cpp/`.
2. `test_recall` passes with zero failures.
3. `test_distance_bound` passes with zero failures.
4. Safe-child/router no-FN regression tests pass.
5. A deterministic high-fanout benchmark reports `mismatch_count=0`.
6. On that benchmark, optimized query performance beats baseline:
   - `p95_speedup_vs_baseline > 1.0`, and
   - either `center_distance_reduction > 0`, `world_access_reduction > 0`,
     or query-similarity scheduling/reuse counters show real work reuse
     (`anchor_cache_hit_count`, `child_shortlist_cache_hit_count`,
     `safe_child_candidate_cache_hit_count`, or
     `productive_world_reuse_hit_count` increases over baseline), and
   - if the speedup is attributed to safe-child routing, fine-grained
     diagnostics show accepted safe-router candidate sets are actually smaller
     than the processed child set or post-MBB survivor set.
7. `CODEX_TEST_LOG.md` records exact commands, exit statuses, and the key
   benchmark numbers.

If correctness passes but speed does not beat baseline, do not mark complete.
Continue improving selectivity, gating overhead with the query planner, or
record a blocker only after three consecutive attempts reach the same
unresolved condition.

## Startup Checklist For Every Continuation

1. Read `AGENTS.md`.
2. Read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, and `CODEX_TEST_LOG.md`.
3. Run `git status --short` and `git diff --stat`.
4. If `CODEX_DIAGNOSTIC_HANDOFF.md` exists, read it and use the diagnostic
   results to choose the first implementation step.
5. Inspect query scheduling/reuse code, the current safe-child-router
   implementation in `navigamer_cpp/src/search_engine.cpp`, and the
   benchmark/reporting code in `navigamer_cpp/src/query_benchmark.cpp`.
6. Continue from the first unfinished item.
