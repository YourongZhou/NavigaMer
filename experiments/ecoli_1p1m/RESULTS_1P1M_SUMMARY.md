# E. coli 1.1M Results Summary

This summary captures the current 1.1M E. coli evidence set used for the
NavigaMer comparison workflow. The full generated tables live under
`.tmp_experiments/ecoli_1p1m_formal/results_summary_v1/` in the local
experiment workspace; this file records the manuscript-facing conclusions and
the boundaries that should not be overstated.

## Current Status

| Result | Status | Summary |
|---|---|---|
| Result 1 | supported | NavigaMer produced zero false negatives in the tested oracle subsets. |
| Result 2 | supported | Proximal anchors gave much tighter candidate envelopes than random anchors while preserving no-FN. |
| Result 3 | preliminary | L=2/3/4 hierarchy ablations preserved no-FN, but current sparse-window evidence does not prove hierarchy necessity. |
| Result 4 | supported | Contained, overlap, and uncovered path classes were observed and summarized. |
| Result 5a | supported | Persisted-index NavigaMer retrieval recovered the source locus for all tested 1.1M read sets. |
| Result 5b | mixed | q-gram and pigeonhole baselines were faster and smaller than NavigaMer under the current 1.1M implementation. |
| Result 5c | supported | Native mappers mapped all reads, but primary source-locus recovery declined under harder errors. |
| Result 6 | preliminary | Source-sorted query order increased path reuse, but current prefetch/order settings did not improve wall-clock time. |
| Build/Persistence | supported | 1.1M persisted-index build metadata and index size are recorded. |

## Claim-Level Evidence

### Result 1. Correctness / no false negatives

Across the tested E. coli 1.1M oracle subsets, NavigaMer produced zero false
negatives relative to exhaustive edit-distance range search.

Key numbers:

- NavigaMer oracle rows: 8
- Total false negatives: 0
- Minimum recall: 1
- Module checks passing: 5/5

Evidence tables:

- `result1_no_fn_oracle.tsv`
- `result1_module_equivalence.tsv`

Boundary: the oracle evidence is based on sampled prefixes and query sets, not
every possible 150-mer in the 1.1M reference.

### Result 2. Anchor selection

With two anchors, proximal-anchor envelopes were substantially smaller than
random-anchor envelopes in the tested 1.1M ablation settings, with zero false
negatives.

Key numbers:

- Anchor-selection rows: 40
- All false-negative totals: 0
- Random/proximal envelope ratio at two anchors: 146.719 and 412.391
- Far/proximal envelope ratio at two anchors: 26.3594 and 68.9897

Evidence table:

- `result2_anchor_selection.tsv`

Boundary: the current actual-anchor rows are still a proxy. Far anchors were
also competitive at high anchor counts and should not be described as failing.

### Result 3. Hierarchy / world ablation

Across stride100/50/25 sparse-window hierarchy ablations, L=2/3/4 all preserved
no-FN. L=4 increased edge-access overhead without a clear p95 latency win.

Key numbers:

- Hierarchy rows: 9
- All no-FN rate: 1
- L4/L3 edge-access ratios: 5.7656, 10.2652, and 62.8288

Evidence table:

- `result3_hierarchy_ablation.tsv`

Boundary: this result does not yet prove hierarchy necessity versus a true
flat/all-beacon baseline.

### Result 4. Corner and boundary paths

The tested query sets include contained, overlap, and uncovered cases;
uncovered queries were present but relatively rare in the summarized path-class
counts.

Key numbers:

- Path-class rows: 30
- Contained queries: 836
- Overlap queries: 1218
- Uncovered queries: 122
- Fallback count in summarized rows: 0

Evidence tables:

- `result4_corner_paths.tsv`
- `result4_corner_paths_by_class.tsv`

Boundary: the current data support path classification more than
fallback-overhead stress testing.

### Result 5. Candidate retrieval and mapper baselines

Using the persisted 1.1M NavigaMer index, every tested read set had
`source_recovery_rate = 1.0`.

Key numbers:

- Persisted NavigaMer retrieval rows: 11
- Source recovery rate: 1.0 for all persisted-index NavigaMer rows
- Mixed tau5 raw candidate means:
  - NavigaMer: 48.6005
  - q-gram q5: 16.4749
  - pigeonhole tau5: 12.2897
  - contiguous k23: 248.478
- Mixed tau5 p95 total milliseconds:
  - NavigaMer: 2610.78
  - q-gram q5: 28.3142
  - pigeonhole tau5: 6.27835
- Mapper rows: 12
- Mixed tau5 primary source recovery:
  - minimap2: 0.9781
  - strobealign: 0.9435

Evidence tables:

- `result5_navigamer_persisted_retrieval.tsv`
- `result5_candidate_retrieval.tsv`
- `result5_mapper_end_to_end.tsv`

Boundary: this comparison should not be interpreted as a throughput advantage
for NavigaMer under the current 1.1M implementation. Safe q-gram and
pigeonhole filters dominate NavigaMer on candidate count and p95 time in the
current benchmark.

### Result 6. Locality / prefetch / system behavior

Source-sorted queries showed much higher world-path reuse than random order,
but current prefetch and ordering did not produce a measurable wall-clock
speedup on 1.1M.

Key numbers:

- Single-thread 128-read order experiment:
  - Random mean previous-world Jaccard: 0.481549
  - Source-sorted mean previous-world Jaccard: 0.934639
  - Random wall time: 166.31 s
  - Source-sorted prefetch off wall time: 167.03 s
  - Source-sorted prefetch on wall time: 166.35 s
- Locality rows: 18
- System A/B rows: 6
- Perf-stat rows: 4

Evidence tables:

- `result6_locality_prefetch.tsv`
- `result6_system_ab.tsv`
- `result6_perf_stat.tsv`

Boundary: hardware cache counters were unavailable or unreliable on accessible
machines, so cache-miss reduction and memory-bound throughput claims remain
unsupported.

## Build and Persistence

The 1.1M comparison uses a persisted `.navidx` index and records build/index
metadata.

Key numbers:

- Build rows: 17
- NavigaMer persisted-index rows: 1
- 1.1M `.navidx` size: 667,212,411 bytes

Evidence tables:

- `index_build_summary.tsv`
- `result5_navigamer_persisted_retrieval.tsv`

Boundary: full E. coli persisted-index evidence remains separate from this
1.1M results set.

## Claims To Avoid

- Do not claim a general full-reference no-FN guarantee from the 1.1M sampled
  oracle rows alone.
- Do not claim that the current hierarchy ablation proves hierarchy necessity
  versus a true flat/all-beacon index.
- Do not claim that NavigaMer is faster than safe q-gram or pigeonhole filters
  on the current 1.1M benchmark.
- Do not claim cache-miss reduction from the current perf-stat rows.
