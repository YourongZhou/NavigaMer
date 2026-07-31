# NavigaMer — C++ reference implementation

This directory contains the **C++17 v8** reference indexer and CLI (`navigamer`) used for the paper implementation. The build pipeline follows a **top-down extended hierarchy**, **inter-tier DAG wiring**, **auxiliary-tier collapse with beacon sequences and MBBs**, and **leaf attachment** to the finest primary layer. Construction stores nodes in a `BuildWorldNodeRecord` array and all relationships as integer IDs, without allocating a `WorldNode` pointer graph. The finalized index is stored as `WorldNodeRecord` and `BioSequence` arrays plus flat child, leaf, beacon, MBB, and leaf-beacon arrays, which adaptive search uses directly.

## Build

Requires **g++** (or Clang) with **OpenMP**.

```bash
make -j
# or
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Output: `./navigamer` (Makefile) or `build/navigamer` (CMake).

## CLI (summary)

```bash
./navigamer demo   [--size N] [--primary-radii 30,15,5 | --r-sw 5 --r-mw 15 --r-lw 30]
./navigamer build  --ref <fasta|sequence> --reads <fastq|sequence> [--index index.navidx] [same primary-layer flags]
./navigamer build-scale --ref <fasta|sequence> --window 250 --stride 1 --prefix-lengths 50000 --out build_scale.csv [--index index.navidx] [same primary-layer flags]
./navigamer query  --reads <fastq|sequence> --query <sequence> [--index index.navidx] [--tolerance 2] [--mode adaptive|greedy|exhaustive]
./navigamer query-index --index index.navidx --query <sequence> [--tolerance 2] [--mode adaptive|greedy|exhaustive]
./navigamer query-index-batch --index index.navidx --reads <fastq> [--tolerance 2] [--out out.tsv] [--path-trace-out trace.tsv]
./navigamer run    --ref <fasta|sequence> --reads <fastq|sequence> [--tolerance 2] [--out out.tsv]
./navigamer map150 --ref <fasta|sequence> --reads <fastq|sequence> --tolerance <N> --out out.tsv [--locator refpos|seqan]
./navigamer benchmark --ref <fasta> --reads <fastq> [--tolerance 2] [--window 200] [--stride 1] [--out out.tsv]
./navigamer query-benchmark --ref <fasta|sequence> --out detail.tsv --summary-out summary.tsv --json-out summary.json [--window 200] [--query-length 200] [--query-benchmark-ablations 0|1] [--proximal-oracle 0|1] [--proximal-oracle-k 1,2,4]
./navigamer candidate-verify --ref <fasta|sequence> --reads <fastq> --candidates candidates.tsv --out detail.tsv --summary-out summary.tsv [--window 150] [--stride 1] [--tolerance 5] [--truth source|exhaustive]
./navigamer locality-benchmark --index index.navidx --ref <fasta|sequence> --out summary.tsv [--scenarios low-fanout,high-fanout,repeat,batch-locality,oracle,all]
./navigamer query-locality-benchmark --index index.navidx --ref <fasta|sequence> --out summary.tsv [same flags as locality-benchmark]
./navigamer query-locality-report --ref <fasta|sequence> --out-dir report_dir [--index index.navidx] [--scenarios all]
./navigamer boundary --ref <fasta> [--length 250] [--error-rates csv] [--tolerance-rates csv] [--queries-per-cell 200] [--stride-mode sparse|dense] [--seed 42] [--out out.tsv]
./navigamer layer-radius-experiment --ref <fasta> [--length 250] [--tolerance 2] [--query-edits 2] [--queries-per-cell 200] [--stride 1 | --stride-mode sparse|dense] [--seed 42] [--L-values 2,3,4,5] [--r-leaf-values 4,8,12] [--alpha-values 0.5,0.7] [--out out.csv]
```

**Full syntax and defaults:** [`CLI_REFERENCE.md`](CLI_REFERENCE.md).

All adaptive-search commands accept `--mbb-filter-mode scan|rect` and
`--min-rect-index-fanout N` (default `64`). Both modes apply exact
all-dimension filtering over the flat MBB arrays; `rect` retains its fanout
threshold and reporting counters and falls back for small fanout.
They also accept `--visited-mode string|epoch` (default `epoch`),
`--graph-view original|flat` (default `flat`),
`--simd-mode auto|scalar|avx2|avx512` (default `auto`),
`--distance-mode dp|myers|edlib|auto` (default `myers`),
`--build-distance-mode dp|edlib|auto` (default `edlib`),
`--search-prefetch off|on` (default `off`),
`--search-qgram-prefilter off|on` (default `off`), and `--search-qgram-q N`
(default `5`). `string` keeps the legacy per-query string visited set for
regression comparisons; `epoch` uses integer node IDs and a reused epoch array.
`original` remains accepted as a compatibility label, but both values traverse
the canonical array index. SIMD mode applies to flat child-MBB and leaf-beacon
filtering; unsupported modes
conservatively fall back to scalar and keep the same survivor set. The search q is independent from
construction `--qgram-q`. Enabled search-side filtering runs only on
MBB-surviving child-world centers before bounded exact center verification;
unsafe or missing signatures fall back to no pruning. Distance mode affects
only adaptive bounded child-center checks after MBB/q-gram filtering. `myers`
is the default mode and uses the optional Myers backend through 256bp ACGT
shorter-input length, falling back to DP otherwise. `edlib` uses the vendored
Edlib bounded distance backend. `dp` remains the reference mode; `auto` is
conservative and currently uses DP. Build distance mode is separate and affects
only index construction exact/bounded distance calls; its default is `edlib`.
Adaptive profiling additionally accepts `--query-profile 0|1` (default `0`) and
records per-query timing/counter buckets in `SearchStats`, `benchmark`, and
`query-benchmark` output without changing search results.

Adaptive path reuse additionally accepts `--path-reuse 0|1` (default `1`).
When enabled, adaptive search keeps thread-local warm-start caches for exact
parent-local anchor-distance vectors on repeated queries and cached child
shortlists keyed by cheap query-derived fingerprints. This remains an
ordering/cache hint only: it never becomes the sole pruning reason, preserves
exact verification, and records `path_reuse_attempt_count`,
`path_reuse_hit_count`, `anchor_cache_hit_count`, and
`child_shortlist_reuse_hit_count`, plus locality-summary counters such as
	`child_shortlist_cache_hit_count`, `safe_child_candidate_cache_hit_count`, and
	`productive_world_reuse_hit_count`. Near-query reuse appends triangle-bound
	center, leaf, and direct-verify counters plus `center_distance_reduction`,
	`world_access_reduction`, and `p95_speedup` to locality/query-benchmark TSVs.
	Leaf verification uses bounded exact edit distance; when path reuse is enabled,
	it caches leaf distances up to the configured near-query neighbor bound so the
	next nearby query can safely prune leaves by triangle inequality without
	rebuilding the index.
	Batch-oriented commands group queries by the same query-derived fingerprint
while keeping emitted output rows in original query order.

Adaptive router hints additionally accept `--router-hints 0|1`,
`--router-hint-qgram-q N`, `--router-hint-minimizer-k N`, and
`--router-hint-minimizer-w N`. The current implementation builds parent-local
child-center range hints from q-gram/pigeonhole candidate supersets and uses
q-gram plus minimizer scores only to reprioritize post-MBB child traversal.
This remains a `RouterHint` only: it never becomes the sole pruning reason and
always preserves full fallback enumeration and exact verification.

Adaptive local routing additionally accepts `--local-router 0|1`,
`--local-router-max-anchors N`, `--local-router-max-children N`, and
`--local-router-score anchor-envelope`. The current router uses parent-local
beacon-envelope scoring only to reorder post-MBB child traversal; it never
becomes the sole pruning reason and always preserves full fallback enumeration
and exact verification.

Adaptive safe child routing additionally accepts `--safe-child-router 0|1`,
`--safe-child-router-min-fanout N`,
`--safe-child-router-max-candidates N`,
`--safe-child-router-max-ratio R`,
`--safe-child-router-min-seed-len N`,
`--safe-child-router-mode auto|pigeonhole|qgram|mbb|full-fallback`, and
`--safe-child-router-validate 0|1`. This is a `SafeCandidateRouter`: it may
reduce child enumeration only when a parent-local radius-bucketed child-center
range query or parent-local MBB interval query returns a safe candidate
superset. Radius buckets use `tolerance + child.radius`; MBB mode rejects a
child only when the query-to-anchor distance is outside that child's stored
MBB interval by more than `tolerance`. Candidate sets that are too broad or
cannot be proven safe fall back to full enumeration, and every survivor still
goes through bounded center verification and final exact leaf verification.

Adaptive query planning additionally accepts `--query-planner 0|1`,
`--planner-router-min-fanout N`, and
`--planner-safe-child-router-min-fanout N`. The current planner is
conservative: it records a per-query strategy and can skip optional
q-gram/router ordering work on low-fanout indexes, but it never skips MBB
filtering, bounded center verification, or final exact leaf verification.
TSV/JSON outputs include planner strategy counters and `planner_decision_ms`.

`query-benchmark` can enable proximal-anchor oracle diagnostics with
`--proximal-oracle 1` and `--proximal-oracle-k 1,2,4`. The extra output compares
actual anchor sources, traversed frontier anchors, true-path anchors, global
nearest anchors, and deterministic random anchors using exact edit-distance
envelopes. It records diagnostics only and does not change search results.

Adaptive safe best-first ordering additionally accepts `--best-first 0|1`
(default `0`). The current implementation uses conservative parent-local MBB
lower bounds and tighter envelope spans to reprioritize post-MBB child worlds
before bounded center verification. It records queue/bound counters in
`SearchStats` and may prune only when that lower bound itself is a conservative
`SafeBound`.

Query-side optimization safety contract:

- `RouterHint` may affect only ordering, warm-starts, or candidate priority.
- `SafeCandidateRouter` may reduce enumeration only with a safe candidate
  superset and must full-fallback otherwise.
- `SafeBound` may prune only when it is conservative and no-false-negative.
- `ExactVerifier` remains the final authority for returned hits.

Build commands also expose Phase1 helper thresholds for tuning the extended
sketch step: `--phase1-metric-min-fanout N` (default `12`),
`--phase1-qgram-min-fanout N` (default `12`), and
`--phase1-qgram-max-touched N` (default `250000`). These switch parent-local
candidate groups from direct scan to the metric helper and then to the q-gram
helper; every surviving candidate is still bounded-exact verified before a
world link is accepted.

Long builds emit timestamped phase progress to stderr every 600 seconds by
default. `--progress-interval-seconds N` changes the interval; zero disables
periodic heartbeats but retains phase start/finish reports. `build-scale` can
persist a reference-window index with `--index <file>` when exactly one prefix
length is requested. Multiple prefixes with one output index are rejected.

Indexed leaf attachment directly verifies range candidates by default. Use
`--leaf-qgram-postfilter on` to apply a safe q-gram L1 necessary condition
before bounded exact verification. This can reduce exact calls for unusually
broad candidate sets, but its signature-building overhead is slower on the
default prepared-DNA distance path. Either setting is no-false-negative:
accepted links are still determined by bounded exact edit distance.

`query-benchmark` fixes the baseline profile to MBB scan, legacy string
visited mode, the `original` compatibility label (which uses the same canonical
array traversal), `dp` distance mode, and search q-gram disabled. It compares
that with the profile selected by `--mbb-filter-mode`,
`--visited-mode`, `--graph-view`, `--simd-mode`, `--distance-mode`,
`--search-qgram-prefilter`, `--search-qgram-q`, `--router-hints`,
`--local-router`, `--best-first`, `--safe-child-router`, `--path-reuse`, and
`--query-planner`.
With
`--query-benchmark-ablations 1`, it also derives one ablation profile per
enabled query-side optimization stage by disabling only that stage within the
optimized stack.
It deterministically generates random-region, ordinary-region,
low-complexity-region, no-hit, single-hit, and multi-hit queries. Step 0 runs
queries serially even though `--threads` is recorded and applied to OpenMP.
One best-effort eviction-buffer cold sample and configured warm samples are
reported per query/profile. The summary TSV and JSON add baseline-relative
speedup/work-ratio columns so optimized and ablation profiles can be compared
without post-processing. With `--proximal-oracle 1`, detail rows add
actual/frontier/true-path/global/random envelope fields for k1/k2/k4,
nearest-anchor distances, and global-oracle gap fields; summary rows add mean
envelopes and fractions where the global oracle is materially better than the
observed actual/frontier anchors. Any repeated-result, cross-profile, or
brute-force no-FN mismatch makes the command return `2`.

`candidate-verify` is the exact verifier used for external seed baselines such
as randstrobe, strobemer, or spaced-seed candidate TSVs. It reads query FASTQ
records, candidate window IDs, and the same reference/window geometry, then
exactly verifies each candidate with bounded edit distance before reporting final
matches. `--truth source` uses `source_pos=` FASTQ annotations as the expected
origin window, while `--truth exhaustive` scans all reference windows for a
small-run no-FN audit. Candidate generation time remains the external tool's
time; `verify_ms` is the exact extension/verification time, and `truth_ms` is
quality-audit time only.

## Module map

| Header / source | Purpose |
| --------------- | ------- |
| `include/structure.hpp`, `src/structure.cpp` | `BioSequence`, `MBB`, legacy `WorldNode` declaration, and default radii |
| `include/tools.hpp`, `src/tools.cpp` | Levenshtein `compute_distance`, helpers |
| `include/qgram_filter.hpp`, `src/qgram_filter.cpp` | Exact q-gram multiset count filter and inverted index |
| `include/range_join.hpp`, `src/range_join.cpp` | Exact full, adaptive-pigeonhole, q-gram, hybrid, and auto candidate generation |
| `include/phase2_distance_verifier.hpp`, `src/phase2_distance_verifier.cpp` | CPU batch exact verifier used by Phase2 rebinding |
| `include/mbb_rect_index.hpp`, `src/mbb_rect_index.cpp` | Exact SoA rectangle lookup for parent-local child MBB filtering |
| `include/index_builder.hpp`, `src/index_builder.cpp` | ID-array construction plus packing into `SequenceStore`, `WorldNodeRecord`, and flat relationship arrays |
| `include/index_persistence.hpp`, `src/index_persistence.cpp` | Array-format v3 binary persistence and manifest signatures |
| `include/candidate_verifier.hpp`, `src/candidate_verifier.cpp` | Exact edit-distance verifier and TP/FP/FN accounting for external seed candidate TSVs |
| `include/search_engine.hpp`, `src/search_engine.cpp` | `search_adaptive`, `verify_leaf_candidates`, `search_greedy`, `search_exhaustive`, `search_brute_force` |
| `include/io_utils.hpp`, `src/io_utils.cpp` | FASTA/FASTQ load, TSV output |
| `include/map150.hpp`, `src/map150.cpp` | Fixed-150bp mapper pipeline: stride-1 reference windows, `2t` candidate search, occurrence location, final exact verifier |
| `src/main.cpp` | CLI entry points |
| `include/experiment_utils.hpp`, `src/experiment_utils.cpp` | Radius-schedule helpers for layer/radius search-cost experiments |

**Note:** Genomic coordinates in existing TSV paths are emitted from `BioSequence::ref_positions`. `map150 --locator refpos` uses the same scaffold locator, while `--locator seqan` is an optional build-time backend that fills `BioSequence::bwt_interval` as a suffix-array interval and uses that stored interval as the occurrence lookup handle during mapping.

**Note:** `build` and `query` can persist and reuse an index with
`--index <file>`. The binary file stores a manifest signature derived from input
fingerprints and construction parameters, followed by the sequence store, node
records, layer ranges, child/leaf/beacon IDs, MBB rows, and leaf-beacon rows.
Format v3 loads those arrays directly; v1/v2 files must be rebuilt.
`query-index` is the
pure load-and-search command for one query, so repeated invocations include
index load time each time. Use `locality-benchmark --index <navidx> --ref
<fasta> --out <tsv>` or the alias `query-locality-benchmark` to load once and
separate persisted-index load time, search-engine initialization time, and
query-only latency. `--scenarios low-fanout,high-fanout,repeat,batch-locality,
oracle,all` selects deterministic query streams for router gating, nearby
window routing, repeat stress, batch locality, and source-oracle diagnostics.
The locality summary reports the actual loaded-index fanout distribution
(`mean_fanout`, `p95_fanout`, `max_fanout`) plus router/path-reuse ratios so a
run can distinguish low-fanout gating from high-fanout router usage.
`--batch-schedules` defaults to source-sorted oracle query ordering for locality
benchmarks and can compare original, random, minimizer, q-gram signature, router
signature, and source-sorted oracle query ordering; the source oracle schedule is
diagnostic only.
Use `--query-fastq-out <path>` to export the deterministic generated queries
with `source_pos=` read-header annotations for matched external baseline
candidate-recovery checks.
`query-locality-report --ref <fasta|sequence> --out-dir <dir>` wraps that
persisted benchmark and writes `summary.tsv`, `summary.json`, and `report.md`;
if `--index` is omitted, it reuses a manifest-compatible
`query_locality.navidx` in the report directory or builds it when missing or
stale. `run`, `benchmark`,
`map150`, and `boundary` still
build in-memory indexes for their current workflows. `boundary` avoids repeated
rebuilds within a parameter sweep by building once per stride mode and reusing
that in-memory index across the full rate grid. The Phase2 distance backend is
not part of the manifest signature because it changes only how exact checks are
executed during construction.

**Note:** The legacy three-primary-layer configuration remains the default CLI path, but the generalized implementation accepts arbitrary primary-layer lists such as `--primary-radii 40,28,18,10`. One auxiliary tier is generated automatically between each adjacent pair of primary layers and collapsed before search.

**Note:** Phase-2 rebinding and leaf attachment default to exact indexed range
joins. Use `--link-mode full` and/or `--leaf-attach-mode full` for the original
full-pairwise construction. `--range-candidate-mode auto` uses adaptive
pigeonhole seeds when they are at least 8 bp, accepts them by actual candidate
count, and early-aborts to the q-gram safe fallback when the seed union exceeds
the configured candidate threshold. The legacy ratio flag is ignored, so normal
pigeonhole queries do not full-scan all length-compatible targets just to
compute a denominator. `--leaf-attach-direction auto` chooses world-to-sequence
leaf attachment when there are fewer finest worlds than unique sequences;
explicit `seq-to-world` and `world-to-seq` are also supported. Forced `qgram`
and `hybrid` modes are still available. All modes exact verify every surviving
candidate before adding a link.

**Note:** Every build prints an aggregate `Build timing` section to stderr.
The timing fields are wall-clock milliseconds collected with
`std::chrono::steady_clock`; high-frequency loops are timed in aggregate to keep
profiling overhead low. In parallel Phase 2 indexed rebinding,
`phase2_rebinding_ms`, `phase2_index_build_ms`, and
`phase2_edge_insert_ms` remain wall-clock timings, while the Phase 2
candidate-query and exact-verify worker fields are accumulated per-thread time.
The `build-scale` command rebuilds in memory for each requested reference prefix
and writes phase timing, substep timing, construction counters, range candidate
mode, and q-gram length to CSV. With one prefix and `--index`, it also writes a
loadable persisted index whose manifest includes the reference fingerprint,
actual prefix length, window length, and stride.

## Parameter sweeps

For long-sequence boundary studies, `boundary` outputs one aggregated TSV row per `(error_rate, tolerance_rate)` cell with source-recovery and pruning metrics for fixed-length `L=250` windows derived from a reference FASTA such as `chr1_subset`. Broader experiment orchestration and comparative baseline workflows live under the repository-level `methods/` directory.

## Tests

| Target | Command |
| ------ | ------- |
| Recall (adaptive vs brute force, 0 FN under test protocol) | `make test_recall && ./test_recall` |
| Distance bounds (violations report) | `make test_distance_bound && ./test_distance_bound` |
| 150bp mapper recall and verifier checks | `make test_map150 && ./test_map150_recall` |
| Bounded edit distance | `make test_bounded && ./test_bounded_edit_distance` |
| Bounded Myers edit distance | `make test_bounded_myers && ./test_bounded_myers_bin` |
| Exact range join | `make test_range_join && ./test_range_join` |
| Q-gram count filter | `make test_qgram && ./test_qgram_filter` |
| Full/indexed construction equivalence | `make test_build_range && ./test_build_range_equivalence` |
| Build timing statistics | `make test_build_timing_stats && ./test_build_timing_stats` |
| Build-scale CSV smoke | `make test_build_scale_smoke && ./test_build_scale_smoke` |
| Exact MBB rectangle lookup | `make test_mbb_rect && ./test_mbb_rect_index` |
| Scan/rect adaptive equivalence and fallback | `make test_mbb_filter && ./test_mbb_filter_equivalence` |
| Search q-gram on/off and scan/rect equivalence | `make test_search_qgram && ./test_search_qgram_prefilter` |
| Safe child router no-FN / candidate superset / fallback | `make test_safe_child_router && ./test_safe_child_router_no_false_negative` |
| Persisted index round-trip and manifest matching | `make test_index_persistence && ./test_index_persistence_bin` |
| Phase2 CPU verifier behavior | `make test_phase2_distance_verifier && ./test_phase2_distance_verifier_bin` |
| Build heartbeat formatting and timer | `make test_build_progress` |

Search q-gram benchmark results and commands are recorded in
[`SEARCH_QGRAM_PREFILTER_BENCHMARK.md`](SEARCH_QGRAM_PREFILTER_BENCHMARK.md).
