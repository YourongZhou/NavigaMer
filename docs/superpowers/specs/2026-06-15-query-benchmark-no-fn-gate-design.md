# Query Benchmark and No-FN Gate Design

## Scope

This design covers Step 0 of the NavigaMer query-performance optimization
series. It adds a repeatable adaptive-query benchmark and a result-equivalence
gate before changing visited tracking, graph layout, SIMD filters, or bounded
edit distance.

The implementation must not change index construction, search traversal,
pruning, leaf attachment, exact verification, or result semantics.

## Command Boundary

Add a new `query-benchmark` command. Keep the existing `benchmark` command and
its hit-oriented TSV output unchanged.

`query-benchmark` builds one in-memory index, then creates two search engines
over that same builder:

- `baseline`: fixed `mbb-filter-mode=scan` and search q-gram prefilter disabled.
- `optimized`: uses the adaptive-search CLI configuration, initially
  `--mbb-filter-mode scan|rect`, `--search-qgram-prefilter off|on`, and
  `--search-qgram-q`.

Future optimization steps may add optimized-profile fields without changing
the baseline definition or benchmark output model.

Both engines share the same built index. Therefore construction edges, MBBs,
and leaf attachments are identical by construction.

## Inputs and Reproducibility

The command accepts and records:

- `--ref`: reference FASTA or literal sequence.
- `--reference-subset-length`: prefix length used from the reference; `0`
  means the full reference.
- `--window`: indexed reference-window length.
- `--stride`: indexed reference-window stride.
- `--query-length`: generated query length.
- `--tolerance`: adaptive-search threshold.
- hierarchy radius flags already accepted by the CLI.
- build range-candidate flags already accepted by the CLI.
- `--seed`: deterministic query-generation seed.
- `--threads`: OpenMP thread count recorded and applied for the command.
- `--mbb-filter-mode scan|rect`: optimized profile setting.
- `--warmup-iterations`: untimed preparation executions per query and mode.
- `--measured-iterations`: timed executions per query and mode.
- `--cold-cache-bytes`: bytes in a reusable eviction buffer touched before
  each cold measurement; `0` disables best-effort cache eviction.
- `--queries-per-class`: generated query count per class.
- `--out`: per-execution TSV path.
- `--summary-out`: aggregate TSV path.
- `--json-out`: JSON summary path.

All generated queries and class assignments must be deterministic for the same
reference subset and seed.

## Query Classes

The driver generates six classes:

1. `random_region`: uniformly sampled valid reference regions.
2. `ordinary_region`: sampled reference regions excluding low-complexity
   regions and regions already classified as high-repeat.
3. `low_complexity_region`: sampled regions with the lowest distinct-base or
   entropy score among valid reference regions.
4. `no_hit`: deterministic random DNA strings whose brute-force result set is
   empty at the configured tolerance.
5. `single_hit`: sampled or deterministically mutated reference regions whose
   brute-force result set contains exactly one indexed sequence ID.
6. `multi_hit`: sampled or deterministically mutated reference regions whose
   brute-force result set contains at least two indexed sequence IDs.

Classification uses brute-force search against the builder's deduplicated
indexed sequences. Query generation retries deterministically up to a fixed
attempt limit. If a requested class cannot be produced, the command fails with
a clear error instead of silently relabeling a query.

The benchmark records the brute-force result count used to validate each
class. Brute-force classification is outside measured adaptive-query latency.

## Measurement Model

Execution is serial in Step 0 even when `--threads` is greater than one. The
thread count is applied and recorded for reproducibility, but parallel
throughput benchmarking is deferred until the search engine has per-worker
scratch in Step 1.

For each query and mode:

1. Touch the reusable cold-cache eviction buffer, then run one separately
   labeled cold measurement.
2. Run `warmup-iterations` untimed adaptive searches.
3. Run `measured-iterations` timed warm searches.

The cold-cache measurement is best effort: touching a buffer larger than the
expected last-level cache reduces hot index data without claiming to flush OS
page cache or every CPU cache level. The eviction-buffer size and method are
recorded in the outputs. The buffer is allocated once outside timed regions.

Latency uses `std::chrono::steady_clock`. Aggregate latency reports
`avg`, `p50`, `p95`, and `p99`. Percentiles use the nearest-rank method on
sorted observed samples and document that method in the JSON metadata.

Baseline and optimized measurements alternate their first-run order by query
index to reduce systematic ordering bias. The order is recorded in detailed
output.

## No-FN and Equivalence Gate

For every generated query:

- Sort and deduplicate baseline result sequence IDs.
- Sort and deduplicate optimized result sequence IDs.
- Compare both sets for exact equality.
- Compare each adaptive result set with the brute-force result set.

Any mismatch fails the gate and causes a nonzero process exit after writing
diagnostic output. Diagnostics include query ID, class, query sequence,
baseline-only IDs, optimized-only IDs, and brute-force-only IDs.

Step 0 does not expose a stable candidate-ID set from adaptive search.
Candidate-level comparison is therefore recorded as `unavailable`. Existing
candidate counters remain reported, but they are not treated as set
equivalence.

Recall is reported as recovered brute-force hits divided by brute-force hits,
with empty-ground-truth queries reported separately. Acceptance requires exact
result equality and no brute-force false negatives for both profiles.

## Counters

Extend `SearchStats` only where an unambiguous existing event can be counted
without changing search behavior. Step 0 reports:

- result count
- world access count
- node access count
- edge access count
- MBB checks
- MBB survivors
- q-gram checks
- center exact-distance calls
- leaf beacon checks
- leaf exact-distance calls
- visited checks
- visited hits
- candidate count
- verified candidate count

Existing fields are mapped directly where possible:

- MBB checks: `mbb_scan_child_checks` for scan, or the logical checked-child
  count represented by existing MBB filter counters for rect.
- MBB survivors: `mbb_surviving_child_count`.
- q-gram checks: `search_qgram_checks`.
- center exact-distance calls: existing center-distance counters.
- leaf exact-distance calls: `leaf_verify_count`.
- candidate count: `candidate_count`.
- verified candidate count: `candidate_verify_count`.

Add explicit leaf-beacon, visited-check, and visited-hit counters to
`SearchStats`. Incrementing these counters must not alter traversal or
allocation behavior. Later steps may refine SIMD-specific counters while
preserving these logical totals.

## Outputs

### Detailed TSV

One row per query, profile, and measured execution, plus separately labeled
cold rows. Columns include configuration identity, query class, execution
order, latency, result count, equivalence status, recall status, and all search
counters.

Warmup executions are not emitted as measured rows.

### Summary TSV

One row per query class and profile, plus an `all` aggregate row. It includes
query count, sample count, cold latency statistics, warm latency statistics,
result totals, recall totals, gate failures, and average/sum counter values.

### JSON Summary

The JSON file contains:

- schema version
- full benchmark configuration
- build duration and builder statistics
- query-generation counts
- per-class/profile latency aggregates
- per-class/profile counter aggregates
- equivalence and no-FN gate status
- mismatch diagnostics
- candidate-set comparison availability

JSON is written with a small repository-local serializer sufficient for this
fixed schema; no new third-party dependency is introduced.

## Code Organization

Keep `main.cpp` responsible for argument parsing and command dispatch. Put the
benchmark model, deterministic query generation, comparison logic,
aggregation, and output writing in focused files:

- `navigamer_cpp/include/query_benchmark.hpp`
- `navigamer_cpp/src/query_benchmark.cpp`
- `navigamer_cpp/src/test_query_benchmark_gate.cpp`

Reuse existing FASTA loading, reference-window generation, TSV helpers,
`BioGeometryIndexBuilder`, and `BioGeometrySearchEngine`.

## Error Handling

Invalid numeric options, impossible query lengths, missing output paths, empty
references, and unproducible query classes fail with descriptive exceptions or
nonzero command status.

Output files are still written when a search-equivalence mismatch occurs so
the failure is diagnosable. Configuration or query-generation failures occur
before measured output is written.

## Tests

Add `test_query_benchmark_gate` and include it in Make and CMake test targets.
It verifies:

- deterministic query generation for a fixed seed
- all six requested query classes on a purpose-built mixed synthetic reference
- exact baseline/optimized/brute-force result equality
- no-FN gate success
- deliberate mismatch detection in the comparison helper
- cold and warm sample counts
- percentile calculation
- required TSV and JSON fields
- newly added logical search counters

Existing `test_recall`, `test_distance_bound`, `test_mbb_filter_equivalence`,
and `test_search_qgram_prefilter` remain unchanged correctness references.

## Documentation and Validation

Update:

- root `README.md`
- `navigamer_cpp/README.md`
- `navigamer_cpp/CLI_REFERENCE.md`

Validation for Step 0:

```bash
cd navigamer_cpp
make test_all
./navigamer query-benchmark \
  --ref ../data/human/chr1_subset \
  --reference-subset-length 2000 \
  --window 150 \
  --stride 1 \
  --query-length 150 \
  --tolerance 2 \
  --seed 42 \
  --threads 1 \
  --mbb-filter-mode rect \
  --cold-cache-bytes 268435456 \
  --warmup-iterations 2 \
  --measured-iterations 5 \
  --queries-per-class 2 \
  --out /tmp/navigamer_query_benchmark.tsv \
  --summary-out /tmp/navigamer_query_benchmark_summary.tsv \
  --json-out /tmp/navigamer_query_benchmark_summary.json
```

Acceptance requires `make test_all` to pass, the benchmark command to exit
zero, exact baseline/optimized/brute-force result equality, and a no-FN gate
pass.

## Deferred Work

Step 0 does not add integer IDs, epoch visited arrays, scratch reuse,
`SearchGraphView`, SIMD filtering, Myers distance, allocation measurement,
RSS measurement, cache flushing, or parallel throughput measurement. Those
belong to later independently reviewed optimization steps.
