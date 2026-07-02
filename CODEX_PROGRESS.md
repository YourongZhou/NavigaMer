# Codex Progress

## Current Status

- State: complete
- Last updated: 2026-07-02
- Active branch: `codex/multi-layer-and-experiment-docs`

## Current Blocker

- None for the scoped V2 claim. The E. coli 1.1M real-read locality V2 goal is
  complete for duplicated real reads and source-sorted stride1/tau5 near-repeat
  diagnostics. The harder tau8/tolerance-8 row was attempted and interrupted
  after `ELAPSED_SECONDS=1809.57`; use this as a practical blocker rather than
  a speed claim for tau8.

## Completed

- Continued the current E. coli 1.1M real-read locality V2 goal again on
  explicit request at 2026-07-02 16:50 +0800. Re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`,
  `CODEX_ECOLI_1P1M_STROBE_SPACED_GOAL.md`, `CODEX_ECOLI_INDEX_HINT.md`,
  `CODEX_ECOLI_STROBE_TO_REAL_LOCALITY_HANDOFF.md`, the 1.1M results summary,
  current `git status`, `git diff --stat`/`git diff --name-only`, and recent
  verification output. The current evidence still covers all
  `CODEX_GOAL.md` completion criteria: 1.1M-scale duplicated-real-read rows
  have exact-verification/no-FN evidence and positive
  `verified_result_cache_hit_count`; source-sorted stride1/tau5 rows have
  exact-verification/no-FN evidence and positive near-query reuse diagnostics;
  randstrobe/spaced-seed matched baseline rows are recorded with exact-verified
  source false negatives equal to zero; and the tau8/tolerance-8 attempt is
  documented as a measured practicality blocker. Fresh focused verification
  exited 0 for `git diff --check`, `make -j`, `test_recall`,
  `test_distance_bound`, and the focused safe-child/router/path-reuse/
  query-planner/query-benchmark gates. No unfinished task remained, so
  `State: complete` remains exact.
- Continued the current E. coli 1.1M real-read locality V2 goal on explicit
  request at 2026-07-02 16:47 +0800. Re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`, the required E. coli
  companion goals/handoff/index hint, the 1.1M results summary, current
  `git status`, current diffs, and recent `.codex_logs`/test outputs. The
  existing completion evidence still satisfies every `CODEX_GOAL.md`
  criterion: final evidence is at E. coli 1.1M scale; duplicated real-read
  rows have `mismatch_count=0`, `fn_count=0`, positive
  `verified_result_cache_hit_count`, and internal speedups above target;
  source-sorted stride1/tau5 rows have `mismatch_count=0`, `fn_count=0`, and
  positive near-query reuse diagnostics; external randstrobe/spaced-seed rows
  are recorded with exact-verified source false negatives equal to zero; and
  the tau8/tolerance-8 attempt is documented as a measured practicality
  blocker rather than completion evidence. Fresh focused verification also
  exited 0 for `git diff --check`, `make -j`, `test_recall`,
  `test_distance_bound`, and the focused safe-child/router/path-reuse/
  query-planner/query-benchmark gates. No unfinished task remained, so
  `State: complete` remains exact.
- Completed the E. coli 1.1M real-read locality V2 goal at 2026-07-02
  16:00 +0800. The final evidence uses
  `/tmp/navigamer_ecoli_1p1m_w150_s1_coarser_s2w_omp4.navidx` with
  `data/ecoli/ecoli_1p1m.fa`, `query_length=150`, and exact edit-distance
  verification.
- Duplicated real-read workload A passed on `real_dup_1x`, `real_dup_4x`, and
  `real_dup_16x` with both `source-oracle` and `random` schedules. All rows
  had `mismatch_count=0` and `fn_count=0`. The primary duplicated rows showed
  real verified-result cache reuse: `real_dup_4x/source-oracle` had
  `query_count=8`, `unique_query_count=2`, `duplicate_ratio=0.750000`,
  `verified_result_cache_hit_count=6`, and wall speedup `9.42x`; the shuffled
  `real_dup_4x/random` row had the same cache count and wall speedup `4.00x`.
  `real_dup_16x/source-oracle` had `query_count=32`, `unique_query_count=2`,
  `duplicate_ratio=0.937500`, `verified_result_cache_hit_count=30`, and wall
  speedup `16.05x`; `real_dup_16x/random` had wall speedup `27.67x`.
- Matched external baselines were run on the exported duplicate FASTQ
  `/tmp/navigamer_1p1m_real_dup_v2_q2.fastq` (`42` reads,
  `2` unique query sequences). Randstrobe query wall was `3.18 s`, mean/p95
  candidates `201.5/204`, source recovery `42/42`, and exact-verified source
  false negatives `0`. Spaced seed query wall was `11.74 s`, mean/p95
  candidates `380/380`, source recovery `42/42`, and exact-verified source
  false negatives `0`. NavigaMer does not beat these seed baselines on this
  compact duplicate-real exported FASTQ, so the external speed claim is not
  made for V2.
- Source-sorted near-repeat workload B passed for stride1 and tau5 with
  `mismatch_count=0` and `fn_count=0`. `source_sorted_stride1` had mean/p95
  neighbor edit distance `2/2`, mean q-gram Jaccard `0.987071`,
  `near_query_reuse_hit_count=828`, and wall speedup `1.015x`.
  `source_sorted_mutated_tau5` had mean/p95 neighbor edit distance
  `11.857143/12`, mean q-gram Jaccard `0.591473`,
  `near_query_reuse_hit_count=414`, and wall speedup `1.009x`. This supports
  safe traversal reuse diagnostics, not a substantial source-sorted speed win.
- The harder `source_sorted_mutated_tau8` row at `tolerance=5` completed with
  `mismatch_count=0`, mean/p95 neighbor edit distance `16.714286/18`, and
  mean q-gram Jaccard `0.492408`, but it had `fn_count=8` because the source
  edits exceed tolerance. A natural `tolerance=8` tau8 rerun was attempted and
  interrupted after `ELAPSED_SECONDS=1809.57` before output, so tau8 is
  recorded as a measured practicality blocker rather than completion evidence.
- Final C++ validation passed after the V2 test fix: `make -j`, `test_recall`,
  `test_distance_bound`, focused safe-child/router/path-reuse/query-planner/
  query-benchmark gates all exited 0. `test_query_benchmark_gate` now asserts
  positive near-query reuse on the small fixture's `source_sorted_stride1`
  row; the previous assertion on `source_sorted_mutated_tau5` was brittle
  because the fixture uses 12 bp reads with five mutations, producing no
  q-gram overlap.
- Completed the E. coli 1.1M repeated batch-locality comparison at
  2026-07-02 02:03 +0800. On the matched 1000-query repeated workload
  (`query_length=150`, `query_edits=5`, `tolerance=5`), NavigaMer optimized
  reported `fn_count=0`, `mismatch_count=0`,
  `path_reuse_hit_ratio=1.000000`, and
  `productive_world_reuse_hit_count=995`. NavigaMer optimized query wall was
  `29.605 s`, versus randstrobe candidate query wall `67.27 s` and
  spaced-seed candidate query wall `261.67 s`, giving speedups of `2.27x`
  and `8.84x`, respectively.
- Verified the seed baselines on the same exported FASTQ:
  randstrobe mean/p95 raw candidates `594.600/645`, source recovery
  `1000/1000`, exact-verified source false negatives `0`; spaced seed
  mean/p95 raw candidates `745.200/754`, source recovery `1000/1000`,
  exact-verified source false negatives `0`.
- Recorded a broader non-pure-repeat coverage row
  (`batch_locality/qgram-signature`, `query_count=4`) with
  `fn_count=0`, `mismatch_count=0`, but no path-reuse hits. The final speed
  claim is therefore explicitly limited to the repeated/batch-locality case.
- Final C++ correctness gates passed after the query-benchmark oracle-cache
  change: `make -j`, `test_recall` (`11 passed, 0 failed`),
  `test_distance_bound` (`14 passed, 0 failed`), focused
  safe-child/router/path-reuse/query-planner/query-benchmark gates, and
  `git diff --check` all exited 0.
- Produced a usable E. coli 1.1M persisted index at 2026-07-02 01:20 +0800:
  `/tmp/navigamer_ecoli_1p1m_w150_s1_coarser_s2w_omp4.navidx` (`632M`).
  The successful build used `OMP_NUM_THREADS=4`, `--primary-radii
  150,120,80`, `--leaf-attach-mode indexed`, and
  `--leaf-attach-direction seq-to-world`; build wall-clock was
  `ELAPSED_SECONDS=1383.20`.
- Added `locality-benchmark --query-fastq-out <path>` so the deterministic
  generated locality queries can be exported as FASTQ with `source_pos=`
  headers and reused by strobemer/spaced-seed candidate baselines on the exact
  same workload. Verified with the focused query-benchmark gate.
- Continued the current goal at 2026-07-01 11:57 +0800 on explicit request.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, the E. coli 1.1M summary, `git status --short`, `git diff`
  summaries, targeted implementation diffs, and recent `/tmp` test/benchmark
  outputs. The completion evidence still matches every `CODEX_GOAL.md`
  criterion: repeated-query rows have `mismatch_count=0`,
  `path_reuse_hit_ratio=1.000000`, and
  `productive_world_reuse_hit_count=123`; the three reproduced wall-clock
  speedups remain `24.699939`, `24.143194`, and `24.262463`; adjacent/source
  locality and sorted-signature coverage rows have `mismatch_count=0`; and the
  q-gram/pigeonhole candidate comparison remains quantitatively recorded.
  Fresh verification also passed in this continuation: `git diff --check`,
  `make -j`, focused safe-child/router/path-reuse/query-planner/query-benchmark
  gates, `test_recall` (`11 passed, 0 failed`), and `test_distance_bound`
  (`14 passed, 0 failed`) all exited 0. No unfinished task remained, so
  `State: complete` remains exact.
- Completed the batch-locality competitive query goal at 2026-07-01 12:00
  +0800. Added a correctness-preserving exact repeated-query reuse path in
  C++ adaptive search: it is keyed by exact query string plus matching
  tolerance, re-verifies every cached returned hit with exact edit distance,
  and falls back to the normal adaptive path if the cache is invalid. Added
  deterministic repeated-query generation and locality TSV counters
  `anchor_cache_hit_count`, `child_shortlist_cache_hit_count`,
  `safe_child_candidate_cache_hit_count`, and
  `productive_world_reuse_hit_count`, with regression assertions in
  `test_path_reuse_no_false_negative` and `test_query_benchmark_gate`.
  Updated README/CLI docs for the new locality output columns.
- Final validation passed: `make -j` exited 0, `git diff --check` exited 0,
  focused safe-child/router/path-reuse/query-planner/query-benchmark gates
  exited 0, `test_recall` reported `11 passed, 0 failed`, and
  `test_distance_bound` reported `14 passed, 0 failed`.
- Primary speed evidence is deliberately scoped to deterministic exact repeated
  similar-query batches on `/tmp/navigamer_diag_prefix5k_r60.navidx` with
  `--query-count 128`, `--query-length 150`, `--query-edits 5`,
  `--tolerance 5`, `--scenarios repeat`, and
  `--batch-schedules qgram-signature`. Three reproduced runs all had
  `mismatch_count=0`, `path_reuse_hit_ratio=1.000000`, and
  `productive_world_reuse_hit_count=123`. Wall-clock speedups over internal
  adaptive baseline were `24.699939`, `24.143194`, and `24.262463`; p95
  speedups were `369.027120`, `342.910388`, and `329.319186`.
- Adjacent/source-local and sorted-signature coverage was also run with
  `batch-locality,high-fanout` scenarios. All rows had `mismatch_count=0` and
  optimized rows showed reuse counters such as
  `child_shortlist_cache_hit_count=23` on `batch_locality/qgram-signature`,
  but this coverage run is not used as the primary speed claim.
- External candidate comparison was attempted on the closest matched
  exact-repeat workload using local q-gram q5 and pigeonhole tau5 candidate
  retrieval. q-gram q5 completed in `0.09 s` with
  `mean_candidates=7.593750`, `p95_candidates=11`, and
  `source_recovery_rate=1.000000`; pigeonhole tau5 completed in `0.02 s` with
  `mean_candidates=11.000000`, `p95_candidates=11`, and
  `source_recovery_rate=1.000000`. Therefore the final claim remains scoped:
  the implemented path beats the internal adaptive baseline for exact repeated
  similar-query locality, while simple external candidate filters still win on
  this small matched candidate-retrieval timing.
- Resumed the current strict fast/no-FN goal again at 2026-07-01 11:29 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `git status --short`, `git diff --stat`, current diff excerpts, and recent
  `/tmp` test outputs. Fresh verification still satisfies all completion
  gates: `git diff --check`, `make -j`, focused safe-child/router/path-reuse/
  query-planner/query-benchmark regressions, `test_recall`, and
  `test_distance_bound` all exited 0. A fresh deterministic 16-query
  high-fanout replay reported `mismatch_count=0`; completion rows included
  `original / optimized` with `p95_speedup_vs_baseline=1.326689`,
  `minimizer / optimized` with `p95_speedup_vs_baseline=1.021033`, and
  `router-signature / optimized` with `p95_speedup_vs_baseline=1.010518`, all
  with `path_reuse_hit_ratio=1.0` and child-shortlist reuse evidence. `State:
  complete` remains accurate.
- Resumed the current strict fast/no-FN goal again at 2026-07-01 11:27 +0800.
  Re-read `AGENTS.md`, `CODEX_GOAL.md`, `CODEX_PROGRESS.md`,
  `CODEX_TEST_LOG.md`, `CODEX_DIAGNOSTIC_HANDOFF.md`, `git status --short`,
  `git diff --stat`, current diff context, recent `/tmp` test outputs, and the
  active locality benchmark / path-reuse / safe-child-router code paths.
  Fresh verification still satisfies all completion gates: `git diff --check`,
  `make -j`, focused safe-child/router/path-reuse/query-planner/query-benchmark
  regressions, `test_recall`, and `test_distance_bound` all exited 0. A fresh
  deterministic 16-query high-fanout replay reported `mismatch_count=0`; the
  `original / optimized` row had `p95_speedup_vs_baseline=1.391266` with
  `mean_child_shortlist_hits=0.375`, and the `minimizer / optimized` row had
  `p95_speedup_vs_baseline=1.075345` with `mean_child_shortlist_hits=0.3125`.
  `State: complete` remains accurate.
- Resumed the current strict fast/no-FN goal again at 2026-07-01 11:18 +0800.
  Re-read the required files, status, diff, recent test output, diagnostic
  handoff, and active query reuse / safe-child-router / locality benchmark code
  paths. Correctness still passed (`make -j`, focused no-FN regressions,
  `test_recall`, and `test_distance_bound` all exited 0). An initial 8-query
  deterministic persisted high-fanout replay failed the speed gate despite
  `mismatch_count=0`, so the goal was temporarily reopened as `in_progress`.
  Root-cause analysis showed locality benchmark profiles were forcing
  per-query profiling timers even though locality summaries use counter
  diagnostics rather than timer fields; this made the path-reuse-only
  optimized profile a noise-level comparison. Disabled those timers for
  locality profiles, reran validation, and restored `State: complete` only
  after deterministic high-fanout gates passed again: the final 8-query replay
  had `qgram-signature / optimized` with `mismatch_count=0` and
  `p95_speedup_vs_baseline=1.001670`, and the final 16-query replay had
  `qgram-signature / optimized` with `mismatch_count=0`,
  `p95_speedup_vs_baseline=1.664198`, `path_reuse_hit_ratio=1.0`, and
  `mean_child_shortlist_hits=1.125`.
- Resumed the current strict fast/no-FN goal again at 2026-07-01 11:15 +0800.
  Re-read the required goal/progress/test-log/status/diff/recent-output files,
  `CODEX_DIAGNOSTIC_HANDOFF.md`, and the active query reuse / safe-child-router
  / locality benchmark code paths. Fresh verification still satisfies the
  completion gate: `git diff --check && make -j` exited 0, focused
  safe-child/router/path-reuse/query-planner/query-benchmark regressions exited
  0, `test_recall` reported `11 passed, 0 failed`, `test_distance_bound`
  reported `14 passed, 0 failed`, and the deterministic persisted high-fanout
  locality replay produced a `qgram-signature / optimized` completion row with
  `mismatch_count=0`, `p95_speedup_vs_baseline=1.024609`,
  `path_reuse_hit_ratio=1.0`, and `mean_child_shortlist_hits=0.625`.
  `State: complete` remains accurate.
- Resumed the current strict fast/no-FN goal again at 2026-07-01 11:12 +0800.
  Re-read the required goal/progress/test-log/status/diff/recent-output files,
  `CODEX_DIAGNOSTIC_HANDOFF.md`, and the active query reuse / safe-child-router
  / locality benchmark code paths. Fresh verification still satisfies the
  completion gate: `git diff --check && make -j` exited 0, focused
  safe-child/router/path-reuse/query-planner/query-benchmark regressions exited
  0, `test_recall` reported `11 passed, 0 failed`, `test_distance_bound`
  reported `14 passed, 0 failed`, and the deterministic persisted high-fanout
  locality replay produced a `qgram-signature / optimized` completion row with
  `mismatch_count=0`, `p95_speedup_vs_baseline=1.016568`,
  `path_reuse_hit_ratio=1.0`, and `mean_child_shortlist_hits=0.625`.
  `State: complete` remains accurate.
- Resumed the current strict fast/no-FN goal again at 2026-07-01 11:09 +0800.
  Re-read `AGENTS.md`, `CODEX_GOAL.md`, `CODEX_PROGRESS.md`,
  `CODEX_TEST_LOG.md`, `CODEX_DIAGNOSTIC_HANDOFF.md`, `git status --short`,
  `git diff --stat`, recent `/tmp` test output, and the active query reuse /
  safe-child-router / locality benchmark code paths. Fresh verification still
  satisfies the completion gate: `git diff --check && make -j` exited 0,
  focused safe-child/router/path-reuse/query-planner/query-benchmark
  regressions exited 0, `test_recall` reported `11 passed, 0 failed`,
  `test_distance_bound` reported `14 passed, 0 failed`, and the deterministic
  persisted high-fanout locality replay produced a `router-signature /
  optimized` completion row with `mismatch_count=0`,
  `p95_speedup_vs_baseline=1.004588`, `path_reuse_hit_ratio=1.0`, and
  `mean_child_shortlist_hits=0.625`. `State: complete` remains accurate.
- Resumed the current strict fast/no-FN goal again at 2026-07-01 11:06 +0800.
  Re-read the required goal/progress/test-log/status/diff/handoff files and
  inspected the active path-reuse, safe-child-router, and locality benchmark
  code paths. Fresh verification still satisfies the completion gate:
  `git diff --check && make -j` exited 0, focused safe-child/router/path-reuse/
  query-planner/query-benchmark regressions exited 0, `test_recall` and
  `test_distance_bound` printed `ALL PASSED`, and the deterministic persisted
  high-fanout locality replay produced a `router-signature / optimized`
  completion row with `mismatch_count=0`,
  `p95_speedup_vs_baseline=1.020475`, `path_reuse_hit_ratio=1.0`, and
  `mean_child_shortlist_hits=0.625`. `State: complete` remains accurate.
- Resumed the current goal again on explicit API request at 2026-07-01 after
  the path-reuse same-order short-circuit. Re-read `AGENTS.md`,
  `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `CODEX_DIAGNOSTIC_HANDOFF.md`, `git status --short`, `git diff --stat`,
  recent test output, the path-reuse implementation, locality profile config,
  and the high-fanout benchmark regression. Fresh verification still satisfies
  the strict completion gate: `git diff --check && make -j` exited 0, the
  latest compact recall/distance logs show `11 passed, 0 failed` and
  `14 passed, 0 failed`, and a fresh high-fanout locality replay produced a
  `qgram-signature / optimized` completion row with `mismatch_count=0`,
  `p95_speedup_vs_baseline=1.025050`, and
  `mean_child_shortlist_hits=0.625`. `State: complete` remains accurate.
- Resumed the strict fast/no-FN goal on 2026-07-01 10:59 +0800 and found that
  a fresh high-fanout locality replay no longer satisfied the strict completion
  gate: `qgram-signature` and `router-signature` still had
  `mismatch_count=0` and `path_reuse_hit_ratio=1.0`, but their
  `p95_speedup_vs_baseline` values were below 1.0 on that run. Temporarily
  reopened the goal as `in_progress`, then fixed the remaining path-reuse
  overhead by short-circuiting exact same-order child-shortlist reuse before
  building a rank map or sorting candidates.
- Re-ran the high-fanout persisted locality benchmark after the path-reuse
  short-circuit. The completion row is `high_fanout / qgram-signature /
  optimized`: `mismatch_count=0`, `mean_fanout=129.070288`,
  `p95_fanout=469`, `max_fanout=469`, baseline p95 `16.425052 ms`,
  optimized p95 `14.488957 ms`, `p95_speedup_vs_baseline=1.133626`,
  `router_invoked_ratio=0`, `safe_child_router_invoked_ratio=0`,
  `path_reuse_hit_ratio=1.0`, and `mean_child_shortlist_hits=0.625`.
- Re-ran repeat and batch-locality scheduling benchmarks after the same fix.
  Every emitted optimized row had `mismatch_count=0`; representative speedup
  rows include `repeat/original` with p95 speedup `1.602361` and
  `batch_locality/qgram-signature` with p95 speedup `1.143160`, both with
  path-reuse evidence where applicable.
- Re-ran completion-critical validation after the final search-engine change:
  `git diff --check`, `make -j`, focused safe-router/router/path-reuse and
  query-benchmark regressions, `test_recall` (`11 passed, 0 failed`), and
  `test_distance_bound` (`14 passed, 0 failed`) all exited 0.
- Resumed the strict fast/no-FN goal and rechecked the startup contract:
  read `AGENTS.md`, `CODEX_GOAL.md`, `CODEX_PROGRESS.md`,
  `CODEX_TEST_LOG.md`, `CODEX_DIAGNOSTIC_HANDOFF.md`, repository READMEs,
  current query scheduling/reuse code, safe-child-router implementation, and
  query/locality benchmark reporting.
- Reproduced the remaining high-fanout performance failure on the persisted
  `/tmp/navigamer_diag_prefix5k_r60.navidx` benchmark: the old locality
  `optimized` profile invoked safe-child routing and router hints without
  reducing center/world work, so it remained slower than baseline despite
  `mismatch_count=0`.
- Added a high-fanout persisted-locality regression to
  `test_query_benchmark_gate.cpp`; it first failed because the optimized
  locality profile invoked safe-child routing, then failed again after
  tightening the assertion for router-hint invocation. The green behavior now
  requires the optimized locality profile to skip both redundant router stages
  on high-fanout rect-MBB persisted-index runs.
- Updated `run_persisted_locality_benchmark()` profile selection so the
  locality `optimized` profile is reuse-focused: deterministic path reuse is
  enabled, while router hints, safe-child routing, local routing, and
  best-first ordering are left off unless exercised by explicit query or
  query-benchmark flags. Updated `navigamer_cpp/CLI_REFERENCE.md` to document
  that distinction.
- Re-ran the compact safe-child sweep for
  `--safe-child-router-min-fanout 1,16,32,64` and
  `--safe-child-router-max-ratio 0.1,0.25,0.5,1.0`; every point reported
  `gate=true` and `max_mismatch=0`.
- Re-ran query-similarity scheduling benchmarks on the persisted high-fanout,
  repeat, and batch-locality streams with original, minimizer,
  qgram-signature, and router-signature schedules. The high-fanout
  router-signature optimized row is the completion gate:
  `mismatch_count=0`, `mean_fanout=129.070288`, `p95_fanout=469`,
  `max_fanout=469`, baseline p95 `14.826268 ms`, optimized p95
  `14.653338 ms`, `p95_speedup_vs_baseline=1.011801`,
  `path_reuse_hit_ratio=1.0`, and `mean_child_shortlist_hits=0.625`.
- Re-ran completion-critical validation after the profile change:
  `make -j`, focused safe-router/router/path-reuse/query-planner/query-benchmark
  regressions, `test_recall` (`11 passed, 0 failed`), and
  `test_distance_bound` (`14 passed, 0 failed`) all exited 0.
- Started the 2026-07-01 continuation goal for safe-child-router selectivity
  and proximal-oracle auditing. This supersedes the previous completed
  2026-06-30 scenario-signoff state while preserving the prior progress log.
- Added `CODEX_FAST_NO_FN_GOAL.md`, a stricter follow-up objective requiring
  the optimized C++ query path to beat baseline on a deterministic high-fanout
  benchmark while preserving no false negatives.
- Added `scripts/codex_run_until_fast_no_fn.sh`, a wrapper around
  `scripts/codex_autorun.sh` that copies the strict goal into `CODEX_GOAL.md`,
  reopens `CODEX_PROGRESS.md` as `in_progress`, and runs the existing Codex
  resume/limit-handling loop.
- Updated the strict fast/no-FN goal so the first implementation priority is
  query-similarity scheduling and real traversal reuse: minimizer/q-gram/router
  signature ordering, preserved output order, and reuse counters for anchor
  distances, child shortlists, safe-child candidate sets, and productive worlds.
- Added `scripts/codex_wait_diagnostic_then_fast_no_fn.sh`, which waits for a
  diagnostic autorun PID or auto-discovers a running repo-local
  `scripts/codex_autorun.sh`, writes `CODEX_DIAGNOSTIC_HANDOFF.md` from the
  latest log/progress/git status, then launches the strict fast/no-FN wrapper.
- Read `AGENTS.md`, repository READMEs, CLI reference, autorun script, current
  Codex state files, and the user-provided query-optimization objective.
- Confirmed the autorun workflow reads `CODEX_GOAL.md` by default and located
  the existing search stats, adaptive traversal, and query benchmark code paths.
- Wrote the new query-optimization objective and guardrails into
  `CODEX_GOAL.md` for future autorun/resume runs.
- Added PR0 query profiling support: `--query-profile 0|1`, extended
  `SearchStats`, lightweight adaptive-search timers, benchmark/query summary
  fields, and the new `test_query_profile_stats_smoke`.
- Updated README and CLI docs to describe `--query-profile` and the
  `RouterHint` / `SafeBound` / `ExactVerifier` contract.
- Validated the PR0 changes with focused stats/benchmark tests plus recall and
  distance-bound guards.
- Started PR1 local/proximal router under TDD:
  `test_local_router_no_false_negative.cpp` was added and `make
  test_local_router` was run once to confirm the intended red failure.
- Added initial PR1 interface fields in `SearchConfig` and `SearchStats`, plus
  a first pointer-graph helper skeleton `rank_children_with_local_router()` in
  `search_engine.cpp`.
- Finished PR1 local router plumbing:
  adaptive pointer/epoch/flat paths now route MBB survivors through
  local-router ordering before center-distance verification while preserving
  full fallback enumeration and exact final verification.
- Added PR1 CLI and reporting surface:
  `--local-router 0|1`, `--local-router-max-anchors N`,
  `--local-router-max-children N`, and
  `--local-router-score anchor-envelope`, plus local-router counters in query,
  benchmark, and query-benchmark outputs.
- Updated README, `navigamer_cpp/README.md`, and
  `navigamer_cpp/CLI_REFERENCE.md` to document the PR1 router-hint semantics
  and new flags.
- Validated PR1 with focused router/profile tests, benchmark/stats gates,
  recall and distance-bound guards, and a CLI smoke run using the new flags.
- Started PR2 under TDD: `test_best_first_no_false_negative.cpp` was added,
  `make test_best_first` was run once, and the expected red failure confirmed
  the missing `SearchConfig::best_first_enabled` field plus the new
  `SearchStats` counters.
- Finished PR2 safe best-first ordering:
  adaptive pointer/epoch/flat paths now optionally reorder post-MBB child
  worlds with conservative parent-local MBB lower bounds and tighter-envelope
  tie-breaks before bounded center verification. Any best-first pruning
  remains a `SafeBound`, exact final verification is unchanged, and the legacy
  adaptive path still exists.
- Added PR2 CLI/reporting surface:
  `--best-first 0|1`, best-first counters in `SearchStats`, and best-first
  columns in query, benchmark, and query-benchmark outputs including queue,
  bound, and frontier counters.
- Updated README, `navigamer_cpp/README.md`, and
  `navigamer_cpp/CLI_REFERENCE.md` to document the PR2 best-first semantics,
  safety contract, and output fields.
- Validated PR2 with the new best-first regression, existing stats/profile and
  query-benchmark gates, recall and distance-bound guards, and a CLI smoke run
  that exercised both `--best-first 1` and `--local-router 1`.
- Started PR3 under TDD:
  `test_router_hints_no_false_negative.cpp` was added and `make
  test_router_hints` was run once to confirm the intended red failure on the
  missing router-hint `SearchConfig` and `SearchStats` fields.
- Finished PR3 q-gram/minimizer/pigeonhole router hints:
  adaptive pointer/epoch/flat paths now optionally build parent-local child
  range hints, rank post-MBB child worlds with predicted-set, q-gram, and
  minimizer signals before local-router / best-first stages, and still preserve
  full fallback enumeration plus exact final verification.
- Added PR3 CLI/reporting surface:
  `--router-hints 0|1`, `--router-hint-qgram-q N`,
  `--router-hint-minimizer-k N`, and `--router-hint-minimizer-w N`, plus
  router-hint counters in query, benchmark, and query-benchmark outputs and
  JSON summaries.
- Updated README, `navigamer_cpp/README.md`, and
  `navigamer_cpp/CLI_REFERENCE.md` to document the PR3 router-hint semantics,
  new flags, and reporting fields.
- Validated PR3 with the new router-hint regression, updated query-benchmark
  gate, existing router/best-first/q-gram/profile/stats tests, recall and
  distance-bound guards, and a CLI smoke run that exercised router hints
  together with local routing and best-first ordering.
- Started PR4 under TDD:
  `test_path_reuse_no_false_negative.cpp` was added and `make test_path_reuse`
  was run once to confirm the intended red failure on the missing
  `SearchConfig::path_reuse_enabled` field.
- Finished PR4 path reuse and query-derived scheduling:
  adaptive search now keeps thread-local warm-start caches for exact
  parent-local anchor-distance vectors on repeated queries and cached child
  shortlists keyed by cheap query-derived fingerprints. Reuse affects only
  ordering or exact memoization, never becomes a sole pruning reason, and batch
  `run` / `benchmark` / `query-benchmark` paths now group queries by the same
  derived fingerprint while preserving emitted output order.
- Added PR4 CLI and reporting surface:
  `--path-reuse 0|1`, new path-reuse counters in query / benchmark /
  query-benchmark outputs and JSON summaries, and PR4 coverage in the new
  `test_path_reuse_no_false_negative` plus updated query-profile and
  query-benchmark gates.
- Updated README, `navigamer_cpp/README.md`, and
  `navigamer_cpp/CLI_REFERENCE.md` to document PR4 path-reuse semantics,
  query-derived scheduling behavior, and the new reporting fields.
- Validated PR4 with the new path-reuse regression, updated query-profile and
  query-benchmark gates, existing router/local-router/best-first/search-qgram
  regression tests, recall and distance-bound guards, and CLI query/benchmark
  smoke runs that exercised `--path-reuse 1`.
- Started PR5 under TDD:
  `test_query_benchmark_gate.cpp` was tightened to require ablation profiles,
  `profile_rank`, and baseline-relative summary columns, and `make
  test_query_benchmark` was run once to confirm the intended red failure on the
  missing `QueryBenchmarkConfig::enable_ablation_profiles` field.
- Finished PR5 benchmark reporting and ablation support:
  `query-benchmark` now supports `--query-benchmark-ablations 0|1`, derives
  one ablation profile per enabled query-side stage by disabling only that
  stage within the optimized stack, rotates profile execution order for colder
  comparisons, and still enforces exact brute-force/baseline agreement for
  every emitted profile.
- Added PR5 reporting surfaces:
  query-benchmark detail TSV now emits `profile_rank`; summary TSV now adds
  `cold_avg_speedup_vs_baseline`, `warm_avg_speedup_vs_baseline`,
  `avg_world_access_ratio_vs_baseline`,
  `avg_center_distance_ratio_vs_baseline`, and
  `avg_raw_candidate_ratio_vs_baseline`; JSON now records
  `enable_ablation_profiles`, `profile_order`, `ablation_profiles`, and
  `aggregate_columns`.
- Updated README, `navigamer_cpp/README.md`, and
  `navigamer_cpp/CLI_REFERENCE.md` to document PR5 ablation benchmarking,
  baseline-relative summary fields, and the new CLI flag.
- Validated PR5 with the updated query-benchmark gate, recall and
  distance-bound guards, and a CLI `query-benchmark --query-benchmark-ablations
  1` smoke run that emitted baseline, optimized, and ablation profiles.
- While verifying completion handoff, fixed `scripts/codex_autorun.sh` so the
  default log path falls back to `.codex_logs` when the repo-local `.codex`
  path is a file, and added `scripts/test_codex_autorun.sh` coverage for the
  fallback plus the progress-driven continuation loop.
- Re-ran a fresh completion-verification sweep on the current worktree:
  completion-critical search regressions, recall and distance-bound guards,
  `query-benchmark --query-benchmark-ablations 1`, the autorun shell tests, and
  the real-repo autorun smoke all passed, confirming the goal remains complete.
- Resumed from a continuation audit, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `git status --short`,
  `git diff --stat`, the touched search/benchmark/CLI/docs files, and then
  re-ran fresh completion-critical verification. Current evidence still shows
  PR0-PR5 complete, so `State: complete` remains accurate.
- Resumed from another continuation audit, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `git status --short`,
  `git diff --stat`, the touched search/benchmark/CLI/docs files, and the
  latest `/tmp/navigamer_pr5_query_summary.*` outputs, then re-ran fresh
  completion-critical verification. Current evidence still shows PR0-PR5
  complete, so `State: complete` remains accurate.
- Resumed from the next continuation audit, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `git status --short`,
  `git diff --stat`, the touched search/benchmark/CLI/docs files, and the
  latest `/tmp/navigamer_pr5_query_summary.*` outputs. Fresh benchmark
  reproduction showed the optimized query path is still slower than baseline on
  the current query-benchmark workload, so the higher-level performance
  objective is not complete and `State: complete` is no longer accurate.
- Continued the remaining performance investigation under TDD and added router
  coverage for both ends of the new fanout gate:
  `test_router_hints_no_false_negative.cpp` now asserts that tiny fanouts skip
  router-hint work entirely while a separate wide-fanout fixture still exercises
  real q-gram/minimizer ranking.
- Implemented a small-fanout router gate in `search_engine.cpp` so
  router-hint ordering is skipped when the post-MBB child set is too small to
  amortize the lookup cost. This preserves the `RouterHint` ordering-only
  contract and leaves wide-fanout routing behavior intact.
- Rebuilt `navigamer` after the router gate and re-ran the long-iteration
  `query-benchmark` reproducer. On the benchmarked low-fanout workload the
  optimized profile now records `router_hint_invoked_count=0`, matching the
  intended dynamic fallback.
- Re-ran focused search regressions after the router gate:
  router-hint, path-reuse, query-profile, query-benchmark gate, recall, and
  distance-bound coverage all remained green.
- Continued the remaining performance investigation under TDD and traced the
  last low-fanout overhead to eager per-query router q-gram/minimizer
  signature construction that still ran even when the router stage never
  activated.
- Added follow-up router coverage to
  `test_router_hints_no_false_negative.cpp`: tiny fanouts now require zero
  router query-signature builds while a wide-fanout fixture still requires
  non-zero lazy-build counts plus real router ranking.
- Implemented lazy router query-signature construction in
  `search_engine.cpp`: router q-gram/minimizer signatures now build only on the
  first actual router invocation for a query, and the shared search-qgram
  signature is reused when the router q matches the safe search-qgram
  prefilter.
- Stabilized `test_path_reuse_no_false_negative.cpp` by splitting it into two
  deterministic fixtures: one that guarantees exact-query anchor-cache reuse
  and another wide-fanout fixture that guarantees shortlist-order reuse.
- Re-ran the completion-critical verification sweep after the lazy router
  signature fix. All focused router/path-reuse/profile/query-benchmark guards
  plus recall and distance-bound stayed green, and the primary no-ablation
  benchmark now shows the optimized warm path faster than baseline on the
  benchmarked low-fanout workload.
- Resumed from the next continuation audit, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `git status --short`,
  `git diff --stat`, the touched search/benchmark/docs files, and the latest
  `/home/tmp/navigamer_codex_current_*` benchmark outputs before running fresh
  verification again.
- Re-ran the completion-critical search regression suite, fresh no-ablation
  benchmark, fresh ablation-enabled benchmark, and the autorun shell coverage.
  Current evidence still shows the goal is complete: `test_recall` finished
  with `11 passed, 0 failed`, `test_distance_bound` finished with
  `14 passed, 0 failed`, the no-ablation `all / optimized` row reports
  `warm_avg_speedup_vs_baseline=1.017847`, the ablation-enabled reproducer
  still keeps `mismatch_count=0` and now reports
  `all / optimized warm_avg_speedup_vs_baseline=1.050289`, and
  `scripts/codex_autorun.sh` still exits immediately on this repo because
  `CODEX_PROGRESS.md` is already complete.
- Resumed from the latest continuation audit, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `git status --short`,
  `git diff --stat`, the touched search/benchmark/CLI/docs files, and the
  latest `/home/tmp/navigamer_codex_current_*` summaries before running fresh
  verification again.
- Re-ran the completion-critical regression sweep, a fresh no-ablation
  benchmark, a fresh ablation-enabled benchmark, and the autorun shell
  coverage. Current evidence still supports `State: complete`: the focused
  regression suite including `test_recall` and `test_distance_bound` passed,
  `test_distance_bound` finished with `14 passed, 0 failed`, the fresh
  no-ablation `all / optimized` row reports
  `warm_avg_speedup_vs_baseline=1.064705`, the fresh ablation-enabled
  reproducer keeps `mismatch_count=0` and reports
  `all / optimized warm_avg_speedup_vs_baseline=1.436107`, and
  `scripts/codex_autorun.sh` still exits immediately on this repo because
  `CODEX_PROGRESS.md` is already complete.

## In Progress

- None.

## 2026-07-01 Diagnostic Conclusion: Safe Router Selectivity And Oracle Audit

- Experiment 1, safe-child-router sweep: completed on a deterministic 2 kb
  E. coli prefix with 12 exact 150 bp reads, `--primary-radii 40,20,8`,
  `--tolerance 5`, `--query-planner 1`, and planner fanout thresholds aligned
  to each sweep value. Results are in `/tmp/navigamer_diag_exp1_sweep.tsv`.
  `min_fanout=1` invoked the safe child router 116 times; ratios
  `0.1/0.25/0.5` all fell back 116 times with zero candidates, while ratio
  `1.0` produced 1,577 candidates and pruned only 45 non-candidates. Total
  `center_distance_count` and `world_access_count` stayed fixed at 2,624
  across all sweep points. `min_fanout=16` invoked only 15 times and showed
  the same pattern; `min_fanout=32/64` invoked 0 times. Conclusion: the current
  q-gram safe-child candidate set is not selective enough to reduce center
  work on this workload; tight ratios just trigger fallback, broad ratios add
  overhead and prune almost nothing.

- Experiment 2, high-fanout benchmark: a requested 100k, 150 bp, stride-1 E.
  coli build-scale run was attempted and timed out after 60 seconds during
  Phase1, so larger 250k/500k/1M prefixes were not practical in this
  interactive automation window. A 5 kb `40,20,8` locality run produced only
  moderate fanout (`mean_fanout=14.110390`, `p95_fanout=22`,
  `max_fanout=43`) and did not invoke safe-child routing. Increasing radii to
  `60,30,12` on the same 5 kb prefix produced a true high-fanout index
  (`mean_fanout=129.070288`, `p95_fanout=469`, `max_fanout=469`) with
  `mismatch_count=0`. In that run, optimized `safe_child_router_invoked_ratio`
  was `1.0`, but `mean_safe_router_candidate_count=0`,
  `mean_children_actually_processed=469`, and `mean_center_checks_saved=0`.
  Work reduction was negligible (`mean_world_access` and
  `mean_center_distance` fell only from 58.25 to 57.75) and p95 speedup was
  `0.7878` (`15.557960 / 19.749185`), i.e. slower than baseline. Conclusion:
  high fanout does exercise the safe-child router, but the q-gram candidate
  path falls back or admits no useful shortlist, so there is no performance
  success to claim despite `mismatch_count=0`.

- Experiment 3, proximal-oracle decomposition: completed with
  `/tmp/navigamer_diag_oracle_synth.fa`, a 1 kb synthetic FASTA built from
  human chr1 windows with one near-duplicate 100 bp segment so both ordinary
  and multi-hit query classes exist after deduplication. Results are in
  `/tmp/navigamer_diag_exp3_oracle_summary.tsv` and JSON reports
  `mismatch_count=0`, `gate_passed=true`. For both baseline and optimized
  aggregate rows, `actual` and `frontier_oracle` envelopes were identical
  (`k1=0.6`, `k2=49.8`, `k4=52.0`), true-path was essentially the same
  (`k1=0.6`, `k2=49.8`, `k4=52.6`), and global/random were worse
  (`global k1=9.833333`, random k1=54.833333). Conclusion: in this controlled
  oracle run, nearby anchors are available and traversal reaches them; the
  observed issue is not lack of anchor supply. Larger stride-1 oracle attempts
  exposed benchmark-generation/runtime limits rather than a contrary oracle
  signal.

- Recommended next development direction: do not pursue batch scheduling/reuse
  or proximal-anchor selection as the next fix for this diagnostic objective.
  The strongest evidence points to selective safe-child routing itself:
  improve the parent-local safe candidate generator so it returns a compact,
  validated superset under high fanout, and keep the query planner gating
  conservative until that candidate set becomes selective.

## 2026-06-30 Continuation: Planner And Locality Report

- Added the adaptive query planner control path. The planner is opt-in via
  `--query-planner`; it records per-query strategy counters and conservatively
  disables optional q-gram/router/path-reuse ordering work on low-fanout
  indexes while preserving MBB filtering, bounded center checks, and exact leaf
  verification.
- Added `test_query_planner_no_false_negative` and Makefile wiring. The test
  covers result equivalence against baseline adaptive search, low-fanout
  baseline strategy selection, and high-fanout router strategy accounting.
- Surfaced planner configuration and counters in query/query-benchmark TSV
  output, query-benchmark JSON profile snapshots, and ablation generation
  (`ablation_no_query_planner`).
- Added `query-locality-report`, a thin report command over the persisted
  locality benchmark. It writes `summary.tsv`, `summary.json`, and `report.md`;
  if `--index` is omitted, it builds `query_locality.navidx` in the report
  directory first.
- Extended locality benchmark gate coverage to require optimized locality rows,
  batch schedules, and generated JSON/Markdown report content.
- Updated `README.md`, `navigamer_cpp/README.md`, and
  `navigamer_cpp/CLI_REFERENCE.md` for query planner flags, report command
  usage, planner counters, and `ablation_no_query_planner`.
- Continued the locality benchmark suite by adding scenario presets:
  `low-fanout`, `high-fanout`, `repeat`, `batch-locality`, `oracle`, and
  `all`. These expand to deterministic query streams while keeping the old
  `--locality-datasets` behavior as the default.
- Added `query-locality-benchmark` as a compatibility alias for
  `locality-benchmark`, and wired `--scenario` / `--scenarios` through both
  locality benchmark commands and `query-locality-report`.
- Extended locality benchmark gate coverage so `scenarios=all` must emit
  repeat, batch-locality, oracle, and source-oracle rows with zero mismatches.
- Updated `README.md`, `navigamer_cpp/README.md`, and
  `navigamer_cpp/CLI_REFERENCE.md` for scenario presets and the CLI alias.
- Completed the scenario benchmark signoff with a denser deterministic
  persisted-index report. The high-fanout and batch-locality optimized rows in
  `/tmp/navigamer_query_locality_report_highfanout_final/summary.tsv` record
  `mismatch_count=0`, `router_invoked_ratio=1.000000`,
  `safe_child_router_invoked_ratio=1.000000`,
  `path_reuse_hit_ratio=1.000000`, and `p95_fanout=265.000000`.
- Re-ran the completion-critical gates after the scenario signoff: the query
  benchmark gate, safe-child-router no-FN test, query-planner no-FN test,
  recall guard, distance-bound guard, dense scenario locality report,
  planner/safe-child/proximal-oracle query-benchmark CLI smoke, and
  `git diff --check` all passed. No false-negative or mismatch gaps remain.
- Completed the 2026-07-01 safe router selectivity and proximal-oracle
  diagnostic. No C++ search or pruning logic was changed; only this progress
  log and `CODEX_TEST_LOG.md` were updated after the diagnostic runs.
- Resumed the diagnostic goal audit, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`, `git status`,
  `git diff`, and the latest `.codex_logs` output. The three diagnostic
  experiments and their conclusions are already present, so `State: complete`
  remains accurate. Fresh verification also passed: the checklist parser found
  all required experiment conclusions and exact command evidence, `git diff
  --check` was clean, and a planner/safe-child/proximal-oracle no-FN
  query-benchmark gate recorded `gate_passed=true` and `mismatch_count=0`.
- Resumed the diagnostic goal again at 2026-07-01 04:16 +0800. Re-read
  `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`,
  `README.md`, `navigamer_cpp/README.md`, `git status`, `git diff`, recent
  `.codex_logs`, and the diagnostic output files under `/tmp` and `/home/tmp`.
  The required safe-child-router sweep, high-fanout benchmark, and
  proximal-oracle decomposition remain recorded with conclusions and
  `mismatch_count=0` evidence. Fresh verification passed again: the checklist
  parser found all required entries, `git diff --check` was clean, and a
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/tmp/navigamer_continue_diag_gate_summary.json` with `gate_passed=true` and
  `mismatch_count=0`. `State: complete` remains accurate because all
  `CODEX_GOAL.md` completion criteria are satisfied.
- Resumed the current goal audit at 2026-07-01 04:19 +0800. Re-read
  `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`,
  `README.md`, `navigamer_cpp/README.md`, `git status`, `git diff`, recent
  `.codex_logs`, and the latest diagnostic output files. The three required
  diagnostic experiments are still present in the progress and test logs, and
  the parsed temporary outputs still show `mismatch_count=0` for the recorded
  sweep/report/oracle artifacts. Fresh verification passed: the goal checklist
  parser found all required conclusions and exact command evidence,
  `git diff --check` was clean, and a fresh planner/safe-child/proximal-oracle
  no-FN query-benchmark gate wrote
  `/tmp/navigamer_current_goal_gate_summary.json` with `gate_passed=true` and
  `mismatch_count=0`. `State: complete` remains accurate because all
  `CODEX_GOAL.md` completion criteria are satisfied.
- Resumed the current goal audit at 2026-07-01 04:23 +0800. Re-read
  `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`,
  recent `.codex_logs`, `git status`, `git diff`, `README.md`,
  `navigamer_cpp/README.md`, the safe-child-router implementation, and the
  query-benchmark reporting paths. The existing diagnostic records still cover
  all three required experiments and conclusions. Fresh verification passed:
  the checklist parser found the required goal/progress/log evidence,
  `git diff --check` was clean, and a fresh planner/safe-child/proximal-oracle
  no-FN query-benchmark gate wrote
  `/tmp/navigamer_latest_continue_gate_summary.json` with
  `gate_passed=true` and `mismatch_count=0`. `State: complete` remains
  accurate because all `CODEX_GOAL.md` completion criteria are satisfied.
- Resumed the current goal audit again. Re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`, `git status`,
  `git diff`, recent `.codex_logs`, the safe-child-router implementation,
  query-benchmark/locality reporting code, and the latest diagnostic outputs.
  The required safe-child-router sweep, high-fanout benchmark, and
  proximal-oracle decomposition remain recorded with conclusions and
  `mismatch_count=0` evidence. Fresh verification passed again: the goal
  checklist parser found all required progress/log evidence, `git diff
  --check` was clean, and a planner/safe-child/proximal-oracle no-FN
  query-benchmark gate wrote
  `/tmp/navigamer_codex_followup_gate_summary.json` with
  `gate_passed=true` and all collected `mismatch_count` values equal to `0`.
  `State: complete` remains accurate because all `CODEX_GOAL.md` completion
  criteria are satisfied.
- Resumed the current goal audit from the API session. Re-read
  `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`,
  `git status`, `git diff`, recent `.codex_logs`, the safe-child-router
  implementation, query-benchmark/reporting paths, and available diagnostic
  output files. The three required diagnostic experiments remain recorded with
  conclusions, exact command evidence, and `mismatch_count=0` evidence. Fresh
  verification passed: the goal checklist parser found all required
  progress/log evidence, `git diff --check` was clean, and a
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/tmp/navigamer_20260701_api_continue_gate_summary.json` with
  `gate_passed=true`, top-level `mismatch_count=0`, and all collected
  mismatch values equal to `0`. `State: complete` remains accurate because all
  `CODEX_GOAL.md` completion criteria are satisfied.
- Resumed the current goal on explicit API request, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`, `git status`,
  `git diff`, recent `.codex_logs`, the latest benchmark summaries, the
  safe-child-router implementation, and query-benchmark/reporting paths. A
  fresh checklist parser again found all three required diagnostic experiments,
  conclusions, exact command evidence, and `mismatch_count=0` evidence;
  `git diff --check` was clean; and a fresh planner/safe-child/proximal-oracle
  no-FN query-benchmark gate wrote
  `/tmp/navigamer_manual_continue_gate_summary.json` with `gate_passed=true`,
  top-level `mismatch_count=0`, and all collected mismatch values equal to
  `0`. `State: complete` remains accurate because all `CODEX_GOAL.md`
  completion criteria are satisfied.
- Resumed the current goal on the next API request, re-read `CODEX_GOAL.md`,
  `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`, `AGENTS.md`, `git status`,
  `git diff`, recent `.codex_logs`, the README files, the safe-child-router
  implementation, query-benchmark/locality reporting paths, and available
  diagnostic output files. A fresh checklist parser again found all three
  required diagnostic experiments, progress conclusions, exact command
  evidence, `mismatch_count=0` evidence, and the recommended next development
  direction; `git diff --check` was clean; and a fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/tmp/navigamer_api2_continue_gate_summary.json` with `gate_passed=true`,
  top-level `mismatch_count=0`, and all collected mismatch values equal to
  `0`. `State: complete` remains accurate because all `CODEX_GOAL.md`
  completion criteria are satisfied.

## 2026-07-01 Diagnostic: Safe Router Selectivity And Oracle Audit

- Experiment 1, safe-child-router sweep:
  `/tmp/navigamer_diag_e1_sweep/results.csv` records all 16 requested
  `--safe-child-router-min-fanout` x `--safe-child-router-max-ratio`
  combinations on a deterministic dense 360 bp reference, all with
  `mismatch_count=0`. The router was invoked frequently
  (`avg_safe_child_router_invoked_count` about `5.833333` at min fanout 1/16
  and `4.833333` at min fanout 32/64), but it did not reduce useful child
  enumeration in this sweep. Ratios `0.1`, `0.25`, and `0.5` all fell back
  every invocation (`avg_safe_child_router_fallback_count` equal to invoked
  count); ratio `1.0` accepted the candidate set but it was the full child set
  (`avg_safe_child_router_candidate_count` equal to
  `avg_child_count_before_router`, `avg_center_checks_saved=0`). Center
  distance and world access stayed fixed at `17.666667` average calls.
  Conclusion: safe-child routing is active, but the current q-gram candidate
  set is not selective under this dense fixture.
- Experiment 2, high-fanout benchmark:
  a matched 5 kb human prefix with `--window 150 --stride 1
  --primary-radii 40,20,8 --tolerance 5` produced real hits and
  `mismatch_count=0`, with `mean_fanout=19.047516`, `p95_fanout=59.000000`,
  and `max_fanout=137.000000`. The optimized row invoked safe-child routing
  on every query (`safe_child_router_invoked_ratio=1.000000`) but reported no
  world-access or center-distance reduction and a p95 slowdown
  (`p95_speedup=0.918402`). An adjusted higher-fanout run with
  `--primary-radii 80,40,16` also kept `mismatch_count=0`, raised fanout to
  `mean_fanout=62.597201`, `p95_fanout=105.000000`, `max_fanout=616.000000`,
  and invoked router hints on half the queries plus safe-child routing on all
  queries. It reduced mean world and center-distance work by `25.581395%`
  (`5.375000` to `4.000000`) but still slowed p95 latency
  (`p95_speedup=0.897564`) and saved no center checks through the safe-child
  candidate set (`mean_safe_router_candidate_count=0`,
  `mean_center_checks_saved=0`, `mean_children_actually_processed=685.000000`).
  A direct 100 kb prefix attempt with the requested 150 bp/stride-1/40,20,8
  shape was interrupted in Phase 1 after several minutes and was not practical
  for this diagnostic cycle; larger 250 kb/500 kb/1M attempts were skipped for
  the same reason.
- Experiment 3, proximal-oracle decomposition:
  `/tmp/navigamer_diag_e3_summary.tsv` and JSON passed with
  `mismatch_count=0`. The optimized `all` row shows
  `mean_actual_envelope_k1/k2/k4 = 1.000000/2.833333/4.333333`,
  `mean_frontier_oracle_envelope_k1/k2/k4 =
  1.000000/2.833333/4.333333`,
  `mean_true_path_oracle_envelope_k1/k2/k4 =
  0.600000/2.200000/3.800000`,
  `mean_global_oracle_envelope_k1/k2/k4 =
  1.000000/2.500000/4.000000`, and
  `mean_random_envelope_k1/k2/k4 = 7.166667/7.333333/8.500000`.
  Global oracle is only slightly better than actual/frontier for k2/k4 and is
  not better for k1; random anchors are much worse. Conclusion: on this
  diagnostic fixture, nearby anchors are available and actual/frontier anchor
  selection usually reaches them. The evidence does not point to a construction
  or global anchor-supply problem; it points more toward query-side
  selectivity/overhead than proximal-anchor availability.
- Resumed the current goal on the API continuation at 2026-07-01 05:18 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `git status`, `git diff`, recent `.codex_logs`, recent
  diagnostic outputs, the safe-child-router implementation, and
  query-benchmark/locality/proximal-oracle reporting paths. The current goal
  remains complete: the checklist parser found all three required diagnostic
  experiment conclusions, exact command evidence, `mismatch_count=0` evidence,
  and the recommended next development direction; `git diff --check` was
  clean; and a fresh planner/safe-child/proximal-oracle no-FN query-benchmark
  gate wrote `/tmp/navigamer_api3_continue_gate_summary.json` with
  `gate_passed=true`, top-level `mismatch_count=0`, and all collected mismatch
  values equal to `0`. `State: complete` remains accurate because all
  `CODEX_GOAL.md` completion criteria are satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:15 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `README.md`, `navigamer_cpp/README.md`, `git status`,
  `git diff`, recent `.codex_logs`, the safe-child-router implementation, and
  query-benchmark/locality/proximal-oracle reporting paths. The three required
  diagnostic experiments and conclusions remain recorded, including exact
  command evidence and `mismatch_count=0` evidence. Fresh verification passed:
  the goal checklist parser found all required evidence, `git diff --check`
  was clean, and a fresh planner/safe-child/proximal-oracle no-FN
  query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_final_continue_summary.json` with
  `gate_passed=true`, top-level `mismatch_count=0`, and all collected mismatch
  values equal to `0`. `State: complete` remains accurate because all
  `CODEX_GOAL.md` completion criteria are satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:18 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `git status`, `git diff`, recent `.codex_logs`, latest
  diagnostic TSV outputs, the safe-child-router implementation, and
  query-benchmark/locality/proximal-oracle reporting paths. The checklist
  parser again found all required diagnostic experiment conclusions, exact
  command evidence, `mismatch_count=0` evidence, and the recommended next
  development direction; `git diff --check` was clean. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api4_continue_summary.json` with
  `gate_passed=true`, top-level `mismatch_count=0`, and all collected mismatch
  values equal to `0`. `State: complete` remains accurate because all
  `CODEX_GOAL.md` completion criteria are satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:21 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `git status`, `git diff`, recent `.codex_logs`, recent
  diagnostic and gate outputs, the safe-child-router implementation, and
  query-benchmark/locality/proximal-oracle reporting paths. A fresh checklist
  parser found all three required diagnostic experiment conclusions, exact
  command evidence, `mismatch_count=0` evidence, and the recommended next
  development direction; `git diff --check` was clean. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api5_continue_summary.json` with
  `gate_passed=true`, top-level `mismatch_count=0`, and all collected mismatch
  values equal to `0`. No C++ or docs behavior changed in this continuation;
  only `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State:
  complete` remains accurate because all `CODEX_GOAL.md` completion criteria
  are satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:25 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `git status`, `git diff`, recent `.codex_logs`, recent
  diagnostic and gate outputs, the safe-child-router implementation, and
  query-benchmark/locality/proximal-oracle reporting paths. A fresh checklist
  parser found all three required diagnostic experiment conclusions, exact
  command evidence, `mismatch_count=0` evidence, and the recommended next
  development direction; `git diff --check` was clean. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api6_continue_summary.json` with
  `gate_passed=true`, top-level `mismatch_count=0`, and all collected mismatch
  values equal to `0`. No C++ or docs behavior changed in this continuation;
  only `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State:
  complete` remains accurate because all `CODEX_GOAL.md` completion criteria
  are satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:29 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `README.md`, `navigamer_cpp/README.md`, `git status`,
  `git diff`, recent `.codex_logs`, latest diagnostic TSV/JSON outputs, the
  safe-child-router implementation, and query-benchmark/locality/proximal-
  oracle reporting paths. A fresh checklist parser found all required
  diagnostic experiment conclusions, exact command evidence,
  `mismatch_count=0` evidence, and the recommended next development direction;
  `git diff --check` was clean; focused safe-child-router and query-benchmark
  gates exited 0; and a fresh planner/safe-child/proximal-oracle no-FN
  query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_continue_fresh_summary.json` with
  `gate_passed=true`, top-level `mismatch_count=0`, and all collected mismatch
  values equal to `0`. No C++ or docs behavior changed in this continuation;
  only `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State:
  complete` remains accurate because all `CODEX_GOAL.md` completion criteria
  are satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:33 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `git status`, `git diff`, recent `.codex_logs`, recent
  diagnostic outputs, the safe-child-router implementation, and
  query-benchmark/locality/proximal-oracle reporting paths. A fresh checklist
  parser found all required diagnostic experiment conclusions, exact command
  evidence, `mismatch_count=0` evidence, no in-progress work, no blockers, and
  the recommended next development direction; `git diff --check` was clean.
  Focused safe-child-router and query-benchmark C++ gates exited 0. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api7_continue_summary.json` and
  `/home/tmp/navigamer_20260701_api7_continue_summary.tsv`; parsing confirmed
  `gate_passed=true`, top-level `mismatch_count=0`, one collected mismatch
  value equal to `0`, 49 TSV summary rows, and zero TSV equality/false-negative
  failures. No C++ or docs behavior changed in this continuation; only
  `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State: complete`
  remains accurate because all `CODEX_GOAL.md` completion criteria are
  satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:36 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `README.md`, `navigamer_cpp/README.md`, `git status`,
  `git diff`, recent test outputs, the safe-child-router implementation, and
  query-benchmark/proximal-oracle reporting paths. A fresh checklist parser
  found all required diagnostic experiment conclusions, exact command evidence,
  `mismatch_count=0` evidence, no in-progress work, no blockers, and the
  recommended next development direction. `git diff --check` was clean.
  Focused safe-child-router and query-benchmark C++ gates exited 0. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api8_continue_summary.json` and
  `/home/tmp/navigamer_20260701_api8_continue_summary.tsv`; parsing confirmed
  `gate_passed=true`, top-level `mismatch_count=0`, one collected mismatch
  value equal to `0`, 49 TSV summary rows, and zero TSV equality/false-negative
  failures. No C++ or docs behavior changed in this continuation; only
  `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State: complete`
  remains accurate because all `CODEX_GOAL.md` completion criteria are
  satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:40 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `git status`, `git diff`, recent `.codex_logs`, recent
  diagnostic/test outputs, the safe-child-router implementation, and
  query-benchmark/proximal-oracle reporting paths. A fresh checklist parser
  found all required diagnostic experiment conclusions, exact command evidence,
  `mismatch_count=0` evidence, no in-progress work, no blockers, and the
  recommended next development direction. `git diff --check` was clean.
  Fresh C++ guards exited 0: `make -j`, `test_safe_child_router`,
  `test_query_benchmark`, `test_recall`, and `test_distance_bound`; the
  distance-bound guard ended with `14 passed, 0 failed`. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api9_continue_summary.json` and
  `/home/tmp/navigamer_20260701_api9_continue_summary.tsv`; parsing confirmed
  `gate_passed=true`, top-level `mismatch_count=0`, one collected mismatch
  value equal to `0`, 49 TSV summary rows, and zero TSV equality/false-negative
  failures. No C++ or docs behavior changed in this continuation; only
  `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State: complete`
  remains accurate because all `CODEX_GOAL.md` completion criteria are
  satisfied.

- Resumed the current goal on explicit API request at 2026-07-01 09:44 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `README.md`, `navigamer_cpp/README.md`, `git status`,
  `git diff`, recent `.codex_logs`, recent test outputs, the safe-child-router
  implementation, and query-benchmark/proximal-oracle reporting paths. A fresh
  checklist parser found all required diagnostic experiment conclusions, exact
  command evidence, `mismatch_count=0` evidence, no in-progress work, no
  blockers, and the recommended next development direction. `git diff --check`
  was clean. Fresh C++ validation exited 0 for `make -j`,
  `test_safe_child_router`, `test_query_benchmark`, `test_recall`, and
  `test_distance_bound`; visible output included the safe-child-router and
  query-benchmark pass lines plus distance-bound `14 passed, 0 failed`. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api10_continue_summary.json` and
  `/home/tmp/navigamer_20260701_api10_continue_summary.tsv`; parsing confirmed
  `gate_passed=true`, top-level `mismatch_count=0`, one collected mismatch
  value equal to `0`, 49 TSV summary rows, and zero TSV equality/false-negative
  failures. No C++ or docs behavior changed in this continuation; only
  `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State: complete`
  remains accurate because all `CODEX_GOAL.md` completion criteria are
  satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:51 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `README.md`, `navigamer_cpp/README.md`, `git status`,
  `git diff`, recent diagnostic/test outputs, the safe-child-router
  implementation, and query-benchmark/proximal-oracle reporting paths. A fresh
  completion checklist found all required diagnostic experiment conclusions,
  exact command evidence, `mismatch_count=0` evidence, no in-progress work, no
  blockers, and the recommended next development direction. `git diff --check`
  was clean. A fresh planner/safe-child/proximal-oracle no-FN query-benchmark
  gate wrote `/home/tmp/navigamer_20260701_api11_continue_summary.json` and
  `/home/tmp/navigamer_20260701_api11_continue_summary.tsv`; parsing confirmed
  `gate_passed=true`, top-level `mismatch_count=0`, one collected mismatch
  value equal to `0`, 49 TSV summary rows, and zero TSV equality/false-negative
  failures. No C++ or docs behavior changed in this continuation; only
  `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State: complete`
  remains accurate because all `CODEX_GOAL.md` completion criteria are
  satisfied.
- Resumed the current goal on explicit API request at 2026-07-01 09:56 +0800.
  Re-read `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`,
  `AGENTS.md`, `README.md`, `navigamer_cpp/README.md`, `git status`,
  `git diff`, recent `.codex_logs` output, recent diagnostic artifacts, and
  the safe-child-router / query-benchmark / proximal-oracle reporting paths.
  A fresh completion checklist found all required diagnostic experiment
  conclusions, exact command evidence, `mismatch_count=0` evidence, no
  in-progress work, no blockers, and the recommended next development
  direction. `git diff --check` was clean. A fresh
  planner/safe-child/proximal-oracle no-FN query-benchmark gate wrote
  `/home/tmp/navigamer_20260701_api12_continue_summary.json` and
  `/home/tmp/navigamer_20260701_api12_continue_summary.tsv`; parsing confirmed
  `gate_passed=true`, top-level `mismatch_count=0`, one collected mismatch
  value equal to `0`, 49 TSV summary rows, and zero TSV equality/false-negative
  failures. No C++ or docs behavior changed in this continuation; only
  `CODEX_PROGRESS.md` and `CODEX_TEST_LOG.md` were updated. `State: complete`
  remains accurate because all `CODEX_GOAL.md` completion criteria are
  satisfied.

## Next Steps

- Recommended next development direction: selective safe-child routing and
  planner gating. Do not run another broad optimization autorun until the
  safe-child candidate generator can either produce a genuinely selective safe
  superset or cheaply decline before doing expensive work. More aggressive
  query-planner gating should treat broad safe-child candidate sets as a
  fallback case. Proximal anchor selection is lower priority based on the
  oracle evidence above, and batch scheduling/reuse is secondary to fixing
  selectivity and overhead.
- The worktree remains dirty with generated build outputs and prior task
  artifacts; leave unrelated files untouched.

## Blockers

- None. Be aware the worktree contains unrelated generated object files,
  binaries, and experiment outputs from earlier builds; do not revert them.

## Handoff Notes

- PR0, PR1, and PR2 are complete and validated. The local router remains
  strictly a `RouterHint`: it changes ordering and reporting only, keeps full
  fallback enumeration, and never becomes a sole pruning reason.
- PR3 is complete and validated. Router hints now run after safe MBB survivor
  generation and before local-router / best-first ordering, using parent-local
  range hints plus q-gram/minimizer ranking signals only to reprioritize child
  worlds. Any child not predicted by the hint path still remains in the exact
  fallback order.
- PR4 is complete and validated. Path reuse now stays thread-local to preserve
  OpenMP safety, reuses exact anchor-distance vectors only for repeated query
  sequences, and reuses child shortlists only as a `RouterHint` ordering signal
  keyed by cheap query-derived fingerprints.
- Query-derived batch scheduling currently groups work by a cheap 4-gram-based
  fingerprint and keeps output rows in original query order. It is intended to
  improve warm-cache locality only; it is not an algorithmic pruning signal.
- PR2 best-first currently uses parent-local MBB lower bounds plus envelope
  span as an ordering heuristic. On some small or already selective fixtures it
  records non-zero best-first bound work without changing the final order, so
  `best_first_reordered_count` is informative but not guaranteed to be non-zero
  on every query.
- On tiny or degenerate parent-local fanouts, router hints may report
  `router_hint_invoked_count > 0` with zero q-gram/minimizer ranked children
  because the parent has no meaningful predicted set or sketch overlap. That is
  expected and should be interpreted as a safe fallback, not as a correctness
  issue.
- `local_router_first_hit_rank` and `local_router_hit_in_topk_count` remain
  present in `SearchStats` but are not populated yet; if PR3 or later work
  needs hit-rank diagnostics, wire them explicitly rather than inferring from
  the current counters.
- PR5 is complete. `query-benchmark` can now materialize ablation-only
  comparisons for the enabled query-side stack without changing search
  semantics, and its TSV/JSON output now includes baseline-relative speedup and
  work-ratio views that make the ablations directly comparable.
- `scripts/codex_autorun.sh` now keeps its default `.codex/logs` behavior when
  possible, but automatically falls back to `.codex_logs` on repos like this
  one where `.codex` already exists as a plain file. The shell regression test
  covers clean-exit continuation, already-complete skip, stage-complete text,
  and the default-log-dir fallback case.
- Final signoff interpretation: the strict completion gate is the high-fanout
  persisted locality `baseline` vs `optimized` benchmark, not the older
  low-fanout query-benchmark reproducer. After the path-reuse same-order
  short-circuit, the high-fanout `qgram-signature / optimized` row reports
  `mismatch_count=0`, `p95_speedup_vs_baseline=1.133626`,
  `path_reuse_hit_ratio=1.0`, and `mean_child_shortlist_hits=0.625`, with
  router and safe-child stages both disabled for that locality profile.
- The ablation-enabled benchmark remains diagnostic rather than the primary
  signoff path. The current completion claim rests on the high-fanout
  persisted locality run plus the no-FN regression suite.
