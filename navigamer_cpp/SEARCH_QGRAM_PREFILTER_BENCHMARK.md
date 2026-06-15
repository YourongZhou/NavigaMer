# Search Q-Gram Prefilter Benchmark

## Configuration

- Date: 2026-06-15
- Reference: first 2,000 bases of `data/human/chr1_subset`
- Indexed windows: 1,751 windows, length 250, stride 1
- Query: reference bases `[500, 750)`, length 250
- Tolerance: 2
- Primary radii: `30,15,5`
- Construction candidate mode: `auto`
- Rectangle minimum fanout: 1
- Search q-gram q: 5

Each mode was run in a separate process, so each timing is one query after a
fresh index build. The TSV `query_time_ms` excludes construction time.

```bash
./navigamer benchmark \
  --ref "$REF" --reads "$QUERY" --window 250 --stride 1 --tolerance 2 \
  --mbb-filter-mode scan|rect --min-rect-index-fanout 1 \
  --search-qgram-prefilter off|on --search-qgram-q 5 --out <mode>.tsv
```

## Results

| MBB mode | Q-gram | Query ms | MBB survivors | Q-gram checks | Q-gram pruned | Center calls before | Center calls after | Prune ratio | Results |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scan | off | 5.931644 | 4 | 0 | 0 | 2 | 2 | 0.000000 | 3 |
| scan | on | 17.367015 | 4 | 2 | 0 | 2 | 2 | 0.000000 | 3 |
| rect | off | 6.951012 | 4 | 0 | 0 | 2 | 2 | 0.000000 | 3 |
| rect | on | 6.104667 | 4 | 2 | 0 | 2 | 2 | 0.000000 | 3 |

All four result tuple sets `(query_id, hit_id, distance)` were identical, with
SHA-256:

`6044ac1659c3221c833ef008f6d16f616cd1e1cec91f6d096c974f241985274c`

MBB filtering left only four child survivors, and visited-node deduplication
reduced them to two child-center verification opportunities. At q=5 both
centers passed the safe q-gram necessary condition, so the prefilter reduced no
exact calls. The counters explain why this dataset does not benefit, and the
single-query timings are too noisy to claim a speed difference.

## Medium Dataset Attempt

The same four-mode benchmark was attempted on the first 10,000 bases
(9,751 stride-1 windows). The first build did not complete after approximately
three minutes, so the four-process comparison was stopped. The current
`benchmark` command rebuilds the in-memory index for every mode; a useful
medium-scale timing comparison should build once and run a mode matrix in one
process, which is outside this PR's search-prefilter scope.

The compact search signature cache stores only finalized primary-world centers,
not leaves. This is suitable for E. coli-scale or partitioned indexes, but does
not by itself make a full human-genome stride-1 in-memory index practical.
