# Codex Goal: Overnight Result 1-6 Gap-Fill Benchmark

## Objective

Run an overnight benchmark pass that fills the current Result 1-6 evidence
gaps without overloading the machine. Result 3 must use an E. coli 10K prefix
instead of 1.1M. Other experiments should use the 1.1M reference and existing
persisted 1.1M geom index when practical, with explicit blockers when a row is
too slow or unsafe to claim.

The output should be a single run directory with raw TSV/CSV/logs, a compact
summary, and an updated Result 1-6 mapping figure.

## Starting Context To Read First

Read these before running anything expensive:

1. `AGENTS.md`
2. `CODEX_GOAL.md`, `CODEX_PROGRESS.md`, `CODEX_TEST_LOG.md`
3. `docs/benchmarks/result_1to6_pipeline.md`
4. `scripts/summarize_results_1to6.py`
5. `scripts/plot_geom_final_result_map.py`
6. `navigamer_cpp/CLI_REFERENCE.md`
7. `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/geom_final_benchmark_contained_reuse_20260703/geom_final_summary.tsv`
8. `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/geom_final_benchmark_contained_reuse_20260703/geom_final_report.md`

## Important Paths

- 1.1M reference:
  `/home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa`
- Existing 1.1M geom index, do not rebuild:
  `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/uniform_hierarchy_sweep_20260703_fullcores_run2/schedules/geom_L4_leaf5_a0p5/geom_L4_leaf5_a0p5.navidx`
- Previous final geom result root:
  `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/geom_final_benchmark_contained_reuse_20260703`
- New overnight output root:
  `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/result1to6_overnight_20260704`
- 10K prefix FASTA to create if absent:
  `$OUT/inputs/ecoli_10k.fa`

## Non-Negotiable Rules

- Do not rebuild the existing 1.1M geom index.
- Do not run Result 3 hierarchy/layer sweeps on 1.1M. Result 3 is 10K only.
- Do not run 1.1M anchor ablation with stride 1. The anchor script precomputes
  all pairwise window distances, so stride 1 at 1.1M is impractical. Use sparse
  `stride=150` for the 1.1M anchor ablation unless you implement and validate a
  more scalable method.
- Every claimed row must have exact edit-distance verification or an explicit
  verifier summary. Any nonzero `fn_count`, `mismatch_count`,
  `false_negative_count_total`, or equivalent makes that claim failed.
- Keep cold index load separate from hot query time.
- Keep random query rows as boundary/negative-control evidence unless they
  actually support a random-query speed claim.
- If a command is killed, times out, or would exceed memory, record the exact
  command, elapsed time if available, and reason in the report.

## Required Fresh Validation

Run and record in `commands.tsv`/`logs/`:

```bash
cd navigamer_cpp && make -j
cd navigamer_cpp && ./test_path_reuse_no_false_negative
cd navigamer_cpp && ./test_query_benchmark_gate
cd navigamer_cpp && ./test_recall
cd navigamer_cpp && ./test_distance_bound
git diff --check
```

If the full tests are too slow, at minimum run `make -j`,
`test_path_reuse_no_false_negative`, `test_query_benchmark_gate`, and
`git diff --check`, then record the skipped tests as blockers. Prefer running
all tests overnight.

## Setup

Create the output layout:

```bash
OUT=/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/result1to6_overnight_20260704
mkdir -p "$OUT"/{logs,inputs,result1_correctness,result2_anchor,result3_hierarchy,result4_corner,result5_candidates,result6_locality,figures}
```

Create the 10K prefix FASTA for Result 3:

```bash
python3 - <<'PY'
from pathlib import Path
src = Path('/home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa')
dst = Path('/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/result1to6_overnight_20260704/inputs/ecoli_10k.fa')
seq = ''.join(line.strip().upper() for line in src.read_text().splitlines() if line and not line.startswith('>'))
dst.parent.mkdir(parents=True, exist_ok=True)
wrapped = '\n'.join(seq[:10000][i:i+80] for i in range(0, min(10000, len(seq)), 80))
dst.write_text('>ecoli_10k_from_1p1m\n' + wrapped + '\n')
print(dst, len(seq[:10000]))
PY
```

Use a command runner that records label, command, start/end, exit status,
stdout/stderr, and `/usr/bin/time` output to `commands.tsv`.

## Result 1: Correctness / No False Negatives

Run a 1.1M correctness smoke with proximal-oracle diagnostics. This is not the
entire paper-scale oracle sweep, but it should strengthen Result 1 and provide
anchor-related columns for Result 2 interpretation.

Suggested command:

```bash
OMP_NUM_THREADS=64 /home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/navigamer query-benchmark \
  --ref /home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa \
  --reference-subset-length 1100000 \
  --window 150 \
  --stride 1 \
  --query-length 150 \
  --tolerance 5 \
  --threads 64 \
  --queries-per-class 2 \
  --warmup-iterations 1 \
  --measured-iterations 3 \
  --cold-cache-bytes 0 \
  --query-benchmark-ablations 1 \
  --proximal-oracle 1 \
  --proximal-oracle-k 1,2,4 \
  --out "$OUT/result1_correctness/query_benchmark_detail.tsv" \
  --summary-out "$OUT/result1_correctness/query_benchmark_summary.tsv" \
  --json-out "$OUT/result1_correctness/query_benchmark_summary.json"
```

If this is too slow, retry with `--queries-per-class 1` and document the
blocker for the larger row.

Completion condition: summary has no nonzero false-negative/mismatch counters.

## Result 2: Anchor Selection on 1.1M Sparse Windows

Run anchor ablation on the 1.1M reference using sparse windows. Do not use
stride 1 here.

Primary command:

```bash
python3 /home/luting/projects/AnchorMapping/NavigaMer/experiments/ecoli_1p1m/scripts/anchor_ablation.py \
  --reference /home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa \
  --out "$OUT/result2_anchor/anchor_selection_1p1m_stride150.tsv" \
  --window-length 150 \
  --stride 150 \
  --query-count 64 \
  --query-edits 5 \
  --tolerance 5 \
  --anchor-counts 1,2,4,8 \
  --strategies random,far,proximal,actual \
  --seed 20260704
```

If the full strategy/count matrix is too slow, retry with
`--query-count 32 --anchor-counts 1,2,4` and record the reason. Summarize
proximal/actual versus random envelope size, exact calls, pruning ratio, source
recovery, and false negatives.

Completion condition: output exists and each claimed strategy row has
`false_negative_count_total=0`.

## Result 3: Hierarchy / World Ablation on 10K Only

This result must be 10K, not 1.1M. Use the 10K prefix FASTA created above.

Primary command:

```bash
OMP_NUM_THREADS=64 /home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/navigamer layer-radius-experiment \
  --ref "$OUT/inputs/ecoli_10k.fa" \
  --length 150 \
  --tolerance 5 \
  --query-edits 5 \
  --queries-per-cell 32 \
  --stride 1 \
  --seed 20260704 \
  --L-values 1,2,3,4 \
  --r-leaf-values 5,8,12 \
  --alpha-values 0.5,0.7 \
  --out "$OUT/result3_hierarchy/layer_radius_10k_dense.csv"
```

If the full matrix is too slow, retry with
`--queries-per-cell 16 --L-values 1,2,3,4 --r-leaf-values 5,8 --alpha-values 0.5`.

Completion condition: CSV exists, has rows for multiple `L` values, and any
available correctness/mismatch counters are zero. The interpretation must say
"10K hierarchy ablation", not 1.1M.

## Result 4: Contained / Overlap / Uncovered Path Classes

Use existing 1.1M geom index and generate a query FASTQ, then run
`query-index-batch` with path classification. Prefer reusing one cold load per
dataset where possible.

Step A: generate boundary query FASTQ and locality summary:

```bash
OMP_NUM_THREADS=64 /home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/navigamer locality-benchmark \
  --index /home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/uniform_hierarchy_sweep_20260703_fullcores_run2/schedules/geom_L4_leaf5_a0p5/geom_L4_leaf5_a0p5.navidx \
  --ref /home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa \
  --out "$OUT/result4_corner/locality_for_path_classes.tsv" \
  --query-fastq-out "$OUT/result4_corner/path_class_queries.fastq" \
  --query-count 64 \
  --query-length 150 \
  --query-edits 5 \
  --tolerance 5 \
  --seed 20260704 \
  --locality-profiles optimized \
  --locality-datasets source_sorted_stride1,source_sorted_mutated_tau5,real_dup_4x,random_windows \
  --batch-schedules source-oracle
```

Step B: classify the generated queries:

```bash
OMP_NUM_THREADS=64 /home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/navigamer query-index-batch \
  --index /home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/uniform_hierarchy_sweep_20260703_fullcores_run2/schedules/geom_L4_leaf5_a0p5/geom_L4_leaf5_a0p5.navidx \
  --reads "$OUT/result4_corner/path_class_queries.fastq" \
  --out "$OUT/result4_corner/path_class_detail.tsv" \
  --path-trace-out "$OUT/result4_corner/path_class_trace.tsv" \
  --tolerance 5 \
  --mode adaptive
```

Then summarize by `query_path_class` into:

- `$OUT/result4_corner/path_class_by_class.tsv`
- `$OUT/result4_corner/path_class_report.md`

Completion condition: classes and counters are reported; any FN/mismatch row in
the paired locality summary must be zero.

## Result 5: Candidate Baseline Expansion

Use existing q64 source-sorted baseline artifacts from the previous final run
as the starting point, then try to add safe q-gram / pigeonhole candidate
baselines. Build `candidate_tool` if possible:

```bash
make -C /home/luting/projects/AnchorMapping/NavigaMer/experiments/ecoli_1p1m candidate_tool TENSOR_SKETCH_STRICT=0
```

Use the query FASTQ generated in Result 4 if it exists. If it mixes datasets
and is too broad, generate a source-sorted q64 FASTQ with `locality-benchmark
--query-fastq-out` and use that.

Try these candidate indexes on the 1.1M reference:

```bash
# q-gram safe
/home/luting/projects/AnchorMapping/NavigaMer/experiments/ecoli_1p1m/candidate_tool build \
  --method qgram-safe --q 5 \
  --ref /home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa \
  --window 150 --stride 1 \
  --out-dir "$OUT/result5_candidates/qgram_safe_index"

/home/luting/projects/AnchorMapping/NavigaMer/experiments/ecoli_1p1m/candidate_tool query \
  --index "$OUT/result5_candidates/qgram_safe_index/index.bin" \
  --reads "$OUT/result4_corner/path_class_queries.fastq" \
  --tau 5 \
  --out "$OUT/result5_candidates/qgram_safe_candidates.tsv"

/home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/navigamer candidate-verify \
  --ref /home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa \
  --reads "$OUT/result4_corner/path_class_queries.fastq" \
  --candidates "$OUT/result5_candidates/qgram_safe_candidates.tsv" \
  --out "$OUT/result5_candidates/qgram_safe_verify_detail.tsv" \
  --summary-out "$OUT/result5_candidates/qgram_safe_verify_summary.tsv" \
  --window 150 --stride 1 --tolerance 5 --truth source
```

Repeat analogously for pigeonhole if q-gram succeeds:

```bash
candidate_tool build --method pigeonhole --tau 5 --nominal-read-length 150 ...
candidate_tool query --index ... --tau 5 ...
navigamer candidate-verify ...
```

Also copy or normalize the existing randstrobe/spaced seed summary rows from:

`/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/reproduce_source_sorted_stride1_geom_q64_20260703`

Completion condition: at least one new safe baseline attempt is recorded, or a
clear blocker is recorded. Existing randstrobe/spaced exact-verified rows must
be included in the final Result 5 summary.

## Result 6: Service-Mode / Locality Throughput

Run one additional source-sorted larger batch if time permits, using the
existing 1.1M geom index:

```bash
OMP_NUM_THREADS=64 /home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/navigamer locality-benchmark \
  --index /home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/uniform_hierarchy_sweep_20260703_fullcores_run2/schedules/geom_L4_leaf5_a0p5/geom_L4_leaf5_a0p5.navidx \
  --ref /home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa \
  --out "$OUT/result6_locality/source_sorted_stride1_q512.tsv" \
  --query-count 512 \
  --query-length 150 \
  --query-edits 5 \
  --tolerance 5 \
  --seed 20260704 \
  --locality-profiles optimized \
  --locality-datasets source_sorted_stride1 \
  --batch-schedules source-oracle
```

If q512 is too slow, retry q384 or q256 with a fresh seed. If `perf` is
available and allowed, run one `perf stat` wrapper around a smaller q64
locality benchmark and record LLC/cache counters; if unavailable, record that
perf counters are missing.

Completion condition: a Result 6 row is present with cold load, hot query time,
RSS, `fn_count=0`, and `mismatch_count=0`, or a measured blocker.

## Final Summary And Figures

Create:

- `$OUT/commands.tsv`
- `$OUT/result_status.tsv`
- `$OUT/summary_all.tsv`
- `$OUT/report.md`
- `$OUT/result1to6_current_summary.tsv`
- `$OUT/figures/result1to6_overnight_map.svg`
- `$OUT/figures/result1to6_overnight_map.pdf`
- `$OUT/figures/result1to6_overnight_map.png`
- `$OUT/figures/result1to6_overnight_map.tiff`

Reuse and adapt `scripts/summarize_results_1to6.py` and
`scripts/plot_geom_final_result_map.py` where possible. The final report must
map every completed/blocked row to Result 1-6:

- Result 1: correctness / no-FN evidence.
- Result 2: 1.1M sparse anchor ablation.
- Result 3: 10K hierarchy/layer sweep only.
- Result 4: path class/corner/fallback evidence.
- Result 5: candidate baseline evidence, including exact verify.
- Result 6: locality/service-mode/cold-load evidence.

## Completion Criteria

Set `CODEX_PROGRESS.md` to `State: complete` only when:

1. The output root exists and contains `commands.tsv`, `report.md`, and summary
   tables.
2. Result 3 is clearly labeled as 10K and no 1.1M hierarchy sweep was run.
3. All successful claimed rows have zero FN/mismatch counters.
4. Any skipped or failed experiment has a measured blocker and command log.
5. `CODEX_TEST_LOG.md` includes the output root, key rows, validation commands,
   and final `git diff --check` status.
6. A final Result 1-6 mapping figure is generated or a plotting blocker is
   recorded.

## Final Wording Boundary

Use scoped language:

> This overnight run fills several Result 1-6 gaps. Result 3 is a 10K
> hierarchy ablation by design; it should not be described as 1.1M hierarchy
> evidence. The 1.1M rows support correctness, anchor-selection diagnostics,
> path-class/fallback diagnostics, candidate-baseline accounting, and
> service-mode locality evidence where their exact FN/mismatch gates pass.
