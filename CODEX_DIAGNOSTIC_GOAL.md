# Codex Goal: Safe Router Selectivity And Oracle Audit

## Current Focus: 2026-07-01 Safe Router Selectivity And Oracle Audit

Continue the NavigaMer C++ query-performance work from the 2026-06-30
experiments. The current stack is intended to be correctness-preserving, but
the next step is diagnostic evidence before another development autorun starts.

This diagnostic run must answer three questions:

1. Does the safe child router actually reduce child enumeration?
2. Do high-fanout benchmark settings exercise router advantage?
3. Does proximal-oracle output show that nearby anchors are unavailable, or only
   that query routing/anchor selection fails to find them?

Do not mark this diagnostic goal complete until the requested experiments have
been run and the conclusions are written to `CODEX_PROGRESS.md` and
`CODEX_TEST_LOG.md`.

## Safety Contract

- Preserve no-false-negative behavior.
- Exact edit-distance verification remains the final authority for returned
  hits.
- Router hints may affect ordering only unless a mathematically safe candidate
  set is accepted.
- Safe-child routing may restrict enumeration only when the candidate set is a
  safe superset of all children that could contain a hit.
- If any diagnostic exposes a correctness risk, stop optimization work and run
  the correctness guards first.

## Required Experiments

### Experiment 1: Safe Child Router Sweep

Run a sweep with:

- `--safe-child-router 1`
- `--safe-child-router-min-fanout 1,16,32,64`
- `--safe-child-router-mode qgram`
- `--safe-child-router-max-ratio 0.1,0.25,0.5,1.0`
- `--query-planner 1`
- `--query-profile 1`

Record at least:

- `safe_child_router_invoked_count`
- `safe_child_router_candidate_count`
- `safe_child_router_pruned_by_not_candidate_count`
- `safe_child_router_fallback_count`
- `center_distance_count`
- `world_access_count`
- `p95_query_ms`

Interpretation rules:

- If invoked is high, pruned-by-not-candidate is high, fallback is low, and
  center-distance work drops, safe child routing is effective.
- If invoked is high but fallback is high, the candidate ratio is too broad and
  the router is not pruning useful work.
- If invoked is low, the benchmark is still low-fanout and does not test the
  intended advantage.

### Experiment 2: High-Fanout Benchmark

Create deliberately high-fanout workloads. Start with:

- prefix sizes: `100k`, `250k`, `500k`, `1M` where practical for automation
- window: `150` or `250`
- stride: `1`
- radii: `40,20,8` or `30,15,5`
- tolerance: `3`, `5`, `8`

The output must include:

- `mean_fanout`
- `p95_fanout`
- `max_fanout`
- `safe_child_router_invoked_ratio`
- `router_invoked_ratio`
- world-access reduction
- center-distance reduction
- p95 speedup
- `mismatch_count`

If fanout remains low, adjust radii/layer settings. If fanout is high but there
is no speedup, conclude that the router candidate set is not selective enough.

### Experiment 3: Proximal Oracle Decomposition

Run with:

- `--proximal-oracle 1`
- `--proximal-oracle-k 1,2,4`

Compare:

- `actual_envelope`
- `frontier_oracle_envelope`
- `true_path_oracle_envelope`
- `global_oracle_envelope`
- `random_envelope`

Interpretation rules:

- global oracle small, actual large: query routing or anchor selection problem.
- frontier oracle small, actual large: local anchor ranking problem.
- true-path oracle small, frontier oracle large: traversal is not entering the
  correct region.
- global oracle also large: construction or anchor supply problem.

## Required Validation

Run the narrow correctness checks needed for any code touched by the diagnostic
work. If diagnostics are read-only, still run at least one no-FN benchmark gate
with `mismatch_count=0`. If pruning/search logic changed, run:

1. `cd navigamer_cpp && make -j`
2. `cd navigamer_cpp && make test_recall test_distance_bound`
3. `cd navigamer_cpp && ./test_recall && ./test_distance_bound`
4. safe-child-router regression tests if available

## Completion Criteria

- `CODEX_TEST_LOG.md` contains exact diagnostic commands and exit statuses.
- `CODEX_PROGRESS.md` contains a short conclusion for each experiment.
- The handoff makes clear which development direction should run next:
  selective safe-child routing, more aggressive query planner gating, proximal
  anchor selection, or batch scheduling/reuse.
- No diagnostic result may be presented as performance success unless
  `mismatch_count=0` is recorded for the relevant benchmark.

## Startup Checklist

1. Read `AGENTS.md`.
2. Read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, and `CODEX_TEST_LOG.md`.
3. Run `git status --short` and `git diff --stat`.
4. Inspect the safe-child-router implementation and benchmark/reporting code.
5. Run the three experiments above, then write the diagnostic conclusion.
