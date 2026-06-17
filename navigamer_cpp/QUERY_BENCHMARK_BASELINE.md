# Query Benchmark Baseline

Date: 2026-06-15

## Configuration

The completed Step 0 comparison uses one thread, seed `42`, one query per
class, one untimed warmup, three measured warm iterations, and a 1 MiB
best-effort eviction buffer. The baseline is fixed to MBB scan with search
q-gram off. The optimized profile uses MBB rect with search q-gram on (`q=3`).
Candidate-set comparison and per-query allocation counting are unavailable.

Successful mixed synthetic command:

```bash
REF=$(python3 -c 'import random; r=random.Random(7); b=["".join(r.choice("ACGT") for _ in range(24)) for _ in range(10)]; b += ["A"*24, "CCCCGGGGCCCCGGGGCCCCGGGG", "CCCCGGGGCCCCGGGGCCCCGGGA"]; print("".join(b))')
./navigamer query-benchmark --ref "$REF" --window 24 --stride 24 \
  --query-length 24 --tolerance 2 --primary-radii 24,12,4 \
  --min-rect-index-fanout 1 --mbb-filter-mode rect \
  --search-qgram-prefilter on --search-qgram-q 3 --seed 42 --threads 1 \
  --warmup-iterations 1 --measured-iterations 3 --queries-per-class 1 \
  --cold-cache-bytes 1048576 \
  --out ../.tmp_experiments/query_benchmark/synthetic_mixed_detail.tsv \
  --summary-out ../.tmp_experiments/query_benchmark/synthetic_mixed_summary.tsv \
  --json-out ../.tmp_experiments/query_benchmark/synthetic_mixed.json
```

## Dataset Availability

| Dataset | Result |
| --- | --- |
| Mixed synthetic unique/repetitive/low-complexity | Completed; all six classes generated |
| Synthetic random DNA, 500 bp, stride 1 | Stopped: unable to generate `ordinary_region`; dense adjacent windows were multi-hit at tolerance 2 |
| Synthetic repetitive/low-complexity | Stopped: unable to generate `ordinary_region` |
| E. coli first 2 kb (`data/ecoli/fasta/ecoli.fa`) | Available; stopped because sparse windows had no `multi_hit` class |
| chr1 subset first 2 kb (`data/human/chr1_subset`) | Available; stopped because sparse windows had no `multi_hit` class |
| chr1 subset first 10 kb | Runtime practical; stopped because sparse windows had no `multi_hit` class |

The failed class-generation attempts are expected gate setup failures, not
search mismatches. Outputs are retained only under `.tmp_experiments/`.

## Results

Build duration was `3.065538 ms` for 13 indexed and 13 unique sequences.
All 6 generated queries passed repeated-execution, baseline/optimized equality,
and brute-force no-FN checks.

| Profile | Cold avg | Cold p50 | Cold p95/p99 | Warm avg | Warm p50 | Warm p95/p99 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline scan/q-gram-off | 0.055252 ms | 0.037300 ms | 0.086341 ms | 0.052524 ms | 0.036380 ms | 0.086680 ms | pass |
| optimized rect/q-gram-on | 0.061859 ms | 0.043720 ms | 0.103091 ms | 0.055695 ms | 0.039250 ms | 0.093690 ms | pass |

| Average logical counter | Baseline | Optimized |
| --- | ---: | ---: |
| world access | 7.000000 | 7.000000 |
| node access | 1.000000 | 1.000000 |
| edge access / MBB checks | 39.000000 | 39.000000 |
| MBB survivors | 7.000000 | 7.000000 |
| q-gram checks | 0.000000 | 6.000000 |
| center exact calls | 7.000000 | 7.000000 |
| leaf beacon checks | 1.000000 | 1.000000 |
| leaf exact calls | 1.000000 | 1.000000 |
| visited checks / hits | 10.166667 / 1.000000 | 10.166667 / 1.000000 |
| candidates / verified candidates | 1.000000 / 1.000000 | 1.000000 / 1.000000 |

The optimized profile was about 5.7% slower on warm average
(`0.052524 / 0.055695 = 0.943x`). On this small index the rectangle lookup and
q-gram checks did not reduce MBB survivors or exact center calls, so their
overhead outweighed any benefit. This is a correctness baseline, not evidence
that the optimization helps larger selective workloads.

## Memory

| Snapshot | Current RSS | Peak RSS |
| --- | ---: | ---: |
| before build | 1,756 KiB | 1,756 KiB |
| after build | 4,468 KiB | 4,468 KiB |
| after benchmark | 5,644 KiB | 5,644 KiB |

Peak RSS includes earlier process activity and is reported best effort by
`getrusage`. Current RSS comes from `/proc/self/status`. Per-query allocation
measurement is unavailable.
