# NavigaMer Result 1-6 Benchmark Pipeline

This pipeline is a reproducible wrapper around the existing C++ CLI and
experiment scripts. It does not reimplement NavigaMer logic.

## Quick Start

From the repository root:

```bash
bash scripts/run_results_1to6_pipeline.sh \
  --quick \
  --label smoke \
  --out-dir /tmp/navigamer_result_1to6_smoke \
  --ref ACGTACGTACGTACGTACGTACGTACGTACGT
```

For a persisted-index locality run:

```bash
bash scripts/run_results_1to6_pipeline.sh \
  --full \
  --label ecoli_1p1m_current \
  --out-dir /path/to/run_dir \
  --ref /path/to/reference.fa \
  --index /path/to/current.navidx \
  --threads 64
```

Use `--dry-run` first when changing parameters. It writes `commands.tsv` and
summary placeholders without running expensive benchmarks:

```bash
bash scripts/run_results_1to6_pipeline.sh --full --dry-run --ref ref.fa --index current.navidx
```

## Output Layout

Each run directory contains:

- `commands.tsv`: every command the pipeline recorded or executed.
- `result_status.tsv`: one row per Result 1-6 with `supported`, `preliminary`,
  `mixed`, `missing`, or `failed`.
- `summary_all.tsv`: normalized machine-readable metrics extracted from raw
  TSV/CSV files.
- `report.md`: short human-readable status report.
- `result1_correctness/`: `query-benchmark` detail, summary, and JSON.
- `result2_anchor/`: anchor-selection ablation TSV.
- `result3_hierarchy/`: layer-radius or uniform-hierarchy sweep TSV/CSV.
- `result4_corner/`: corner/fallback notes or dedicated corner-case TSVs.
- `result5_candidates/`: external candidate/baseline comparison summaries.
- `result6_locality/`: persisted-index locality benchmark output.

## Result Mapping

| Result | Pipeline Evidence | Current Interpretation |
| --- | --- | --- |
| Result 1 | `test_recall`, `test_distance_bound`, `test_query_benchmark_gate`, and `query-benchmark` | Main reliability/no-false-negative evidence. This must stay first. |
| Result 2 | `experiments/ecoli_1p1m/scripts/anchor_ablation.py` | Anchor envelope size, pruning ratio, source recovery, and FN count. |
| Result 3 | `layer-radius-experiment` and optional imported uniform-hierarchy sweep summaries | Hierarchy/layer-radius search-cost evidence; full paper claim needs q8/q32/real-dup coverage. |
| Result 4 | Dedicated `result4_corner/*.tsv` files, plus current query-benchmark class coverage note | Still preliminary unless a deterministic corner-case runner is added. |
| Result 5 | `candidate_tool`/external candidate summaries plus `candidate-verify` summaries | Usually a mixed comparison: separate recall, raw candidates, verification time, and generation time. |
| Result 6 | `locality-benchmark` or `query-locality-report` | Persisted-index query-only locality/reuse evidence with `fn_count` and `mismatch_count` gates. |

## Checking A Run

After a run, or after copying in old result files:

```bash
python3 scripts/summarize_results_1to6.py /path/to/run_dir
bash scripts/check_results_1to6.sh /path/to/run_dir
```

`check_results_1to6.sh` fails if any TSV/CSV under the run directory reports a
nonzero `fn_count`, `mismatch_count`, `false_negative_count_total`, or related
correctness counter. Missing results are reported in `result_status.tsv` but do
not make the checker fail; a missing result means “not claimed by this run.”

## Comparing Future Methods

Keep one run directory per method/parameter set. The stable comparison files are:

- `commands.tsv`: provenance and exact command line.
- `result_status.tsv`: whether each Result is supported by that run.
- `summary_all.tsv`: normalized metrics suitable for joining across methods.

A practical comparison workflow is:

```bash
python3 scripts/summarize_results_1to6.py /path/to/run_a
python3 scripts/summarize_results_1to6.py /path/to/run_b
bash scripts/check_results_1to6.sh /path/to/run_a
bash scripts/check_results_1to6.sh /path/to/run_b
```

Then compare `summary_all.tsv` by `result_id`, `source`, and `metric`.

## Current Gaps To Fill

- Result 1 is the strongest reliability path and should remain the first section
  of any result report.
- Result 3 needs complete uniform-hierarchy coverage for q8, q32, and real-dup
  schedules before it should be treated as final.
- Result 4 needs a real deterministic corner-case runner instead of a note file.
- Result 5 needs dataset-specific external candidate generation commands and
  exact `candidate-verify` summaries before speed/size claims are final.
- Result 6 is strongest when run with a persisted `.navidx` and all schedules:
  `original,random,minimizer,qgram-signature,router-signature,source-oracle`.
