# Codex Goal: E. coli 1.1M Geom Final Benchmark

## Objective

Run the final E. coli 1.1M benchmark matrix for the current NavigaMer C++
method after contained path reuse, using the existing `geom_L4_leaf5_a0p5`
persisted index. The goal is to produce stable, source-grounded result files
that separate hot query performance from cold index load cost and compare the
hot query result against randstrobe/strobemer and spaced-seed baselines under
exact-verified no-false-negative accounting.

Do not rebuild the geom index. Use the existing NFS index.

## Current Starting Point

Read these files first:

1. `AGENTS.md`
2. `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, and `CODEX_TEST_LOG.md`
3. `navigamer_cpp/src/search_engine.cpp`
4. `navigamer_cpp/src/query_benchmark.cpp`
5. Existing q64 result files:
   - `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/geom_source_sorted_stride1_q64.tsv`
   - `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_contained_reuse_rerun_20260703/geom_source_sorted_stride1_q64.tsv`
   - `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/randstrobe_verified_source_summary.tsv`
   - `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/spaced_verified_source_summary.tsv`

Important paths:

- Geom index:
  `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/uniform_hierarchy_sweep_20260703_fullcores_run2/schedules/geom_L4_leaf5_a0p5/geom_L4_leaf5_a0p5.navidx`
- E. coli 1.1M reference:
  `/home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa`
- Suggested output root:
  `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/geom_final_benchmark_contained_reuse_20260703`

## Non-Negotiable Rules

- Exact edit distance remains the final authority.
- Final NavigaMer rows must have `fn_count=0` and `mismatch_count=0`.
- Randstrobe/strobemer and spaced-seed comparisons must include exact verify
  time and exact-verified source FN/FP accounting.
- Do not count NavigaMer cold index load as hot query time. Report both.
- Do not claim broad random-query speed superiority unless random workload
  rows are actually run and support that claim.
- Do not set `CODEX_PROGRESS.md` to `State: complete` until the completion
  criteria below are satisfied and recorded in `CODEX_TEST_LOG.md`.

## Required Validation

Run and record:

```bash
cd navigamer_cpp && make -j
cd navigamer_cpp && ./test_path_reuse_no_false_negative
cd navigamer_cpp && ./test_query_benchmark_gate
cd navigamer_cpp && ./test_recall
cd navigamer_cpp && ./test_distance_bound
git diff --check
```

If a long correctness test was already run in this exact continuation and the
output is available, it may be referenced, but at least
`test_path_reuse_no_false_negative`, `test_query_benchmark_gate`, and
`git diff --check` must be fresh.

## Required NavigaMer Benchmark Matrix

Use `OMP_NUM_THREADS=64` unless the machine is under memory pressure. All runs
must use the absolute reference path above, not a relative path.

Primary hot-query row:

```bash
/home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/navigamer locality-benchmark \
  --index /home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/uniform_hierarchy_sweep_20260703_fullcores_run2/schedules/geom_L4_leaf5_a0p5/geom_L4_leaf5_a0p5.navidx \
  --ref /home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa \
  --out "$OUT/geom_source_sorted_stride1_q64.tsv" \
  --query-count 64 \
  --query-length 150 \
  --query-edits 5 \
  --tolerance 5 \
  --seed 20260703 \
  --locality-profiles optimized \
  --locality-datasets source_sorted_stride1 \
  --batch-schedules source-oracle
```

Additional rows to run if memory/time permit:

- `source_sorted_stride1` with `query-count=128`
- `source_sorted_stride1` with `query-count=256`
- `source_sorted_mutated_tau5` with `query-count=64`
- `real_dup_4x` with `query-count=64`, schedule `source-oracle`
- `random_windows` with `query-count=64`, schedule `source-oracle`, as a
  boundary/negative-control row

Because this index takes several minutes and roughly 80GB+ RSS to load, prefer
one command that includes multiple datasets for the same query count when that
does not change the interpretation. If a larger query-count run is killed or
too slow, record the exact blocker and keep the completed q64 rows.

## Baseline Comparison

Use the existing q64 randstrobe/spaced outputs if they match the q64
source-sorted FASTQ. If re-running is practical, rerun in the same final output
root. In either case, compute and record:

- candidate retrieval wall time,
- exact verify wall time,
- total baseline time,
- raw candidate count,
- verified match count,
- truth/source match count,
- `tp_count`, `fp_count`, `fn_count`.

Use these known existing files as the baseline reference:

- `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/randstrobe_q64.time`
- `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/randstrobe_verify_source.time`
- `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/randstrobe_verified_source_summary.tsv`
- `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/spaced_q64.time`
- `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/spaced_verify_source.time`
- `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703/spaced_verified_source_summary.tsv`

## Required Summary Artifacts

Create or update these files in the output root:

- `commands.tsv`: exact commands, start/end time, exit status.
- `geom_final_summary.tsv`: one row per NavigaMer/baseline result with hot
  query time, verify time, cold load time, FN/FP, and notes.
- `geom_final_report.md`: short human-readable interpretation.
- `logs/`: stdout/stderr/time files for each run.

The report must include:

1. NavigaMer q64 contained-reuse result.
2. Speedup versus pre-contained q64 geom result.
3. Speedup versus randstrobe + verify.
4. Speedup versus spaced seed + verify.
5. Cold-load caveat.
6. Boundary rows or documented blockers for q128/q256, tau5, real-dup, and
   random controls.

## Completion Criteria

Set `CODEX_PROGRESS.md` to `State: complete` only when all are true:

1. Required fresh validation commands are recorded.
2. At least the q64 `source_sorted_stride1` geom row is present and has
   `fn_count=0`, `mismatch_count=0`.
3. A comparison table reports hot-query speedups versus pre-contained geom,
   randstrobe+verify, and spaced-seed+verify.
4. Cold load time and hot query time are explicitly separated.
5. Any skipped larger/extra rows have measured blockers or a clear reason.
6. `CODEX_TEST_LOG.md` points to the final output root and key rows.

## Final Wording Boundary

Use scoped language:

> On E. coli 1.1M `source_sorted_stride1` q64, the current geom index with
> contained path reuse has no false negatives and substantially improves hot
> query time versus the previous geom result and the exact-verified
> randstrobe/spaced-seed baselines. This is a hot-query/service-mode result;
> cold one-shot runs remain dominated by loading the 26GB geom index.
