# NavigaMer — C++ reference implementation

This directory contains the **C++17 v8** reference indexer and CLI (`navigamer`) used for the paper implementation. The build pipeline follows a **top-down extended hierarchy**, **inter-tier DAG wiring**, **auxiliary-tier collapse with beacon sequences and MBBs**, and **leaf attachment** to the finest primary layer. Adaptive search uses **precomputed per-edge MBB rows** (`WorldNode::child_beacon_mbbs`) for hierarchy pruning and **finest-layer leaf beacon rows** (`WorldNode::leaf_beacon_dists`) for the final local refinement step before exact verification.

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
./navigamer run    --ref <fasta|sequence> --reads <fastq|sequence> [--tolerance 2] [--out out.tsv]
./navigamer map150 --ref <fasta|sequence> --reads <fastq|sequence> --tolerance <N> --out out.tsv [--locator refpos|seqan]
./navigamer benchmark --ref <fasta> --reads <fastq> [--tolerance 2] [--window 200] [--stride 1] [--out out.tsv]
./navigamer query-benchmark --ref <fasta|sequence> --out detail.tsv --summary-out summary.tsv --json-out summary.json [--window 200] [--query-length 200]
./navigamer boundary --ref <fasta> [--length 250] [--error-rates csv] [--tolerance-rates csv] [--queries-per-cell 200] [--stride-mode sparse|dense] [--seed 42] [--out out.tsv]
./navigamer layer-radius-experiment --ref <fasta> [--length 250] [--tolerance 2] [--query-edits 2] [--queries-per-cell 200] [--stride 1 | --stride-mode sparse|dense] [--seed 42] [--L-values 2,3,4,5] [--r-leaf-values 4,8,12] [--alpha-values 0.5,0.7] [--out out.csv]
```

**Full syntax and defaults:** [`CLI_REFERENCE.md`](CLI_REFERENCE.md).

All adaptive-search commands accept `--mbb-filter-mode scan|rect` and
`--min-rect-index-fanout N` (default `64`). Rect mode performs exact
all-dimension MBB rectangle intersection and falls back to the original scan
path for small fanout, missing indexes, dimension mismatches, or exceptions.
They also accept `--visited-mode string|epoch` (default `epoch`),
`--graph-view original|flat` (default `flat`),
`--simd-mode auto|scalar|avx2|avx512` (default `auto`),
`--distance-mode dp|myers|edlib|auto` (default `myers`),
`--build-distance-mode dp|edlib|auto` (default `edlib`),
`--search-qgram-prefilter off|on` (default `off`), and `--search-qgram-q N`
(default `5`). `string` keeps the legacy per-query string visited set for
regression comparisons; `epoch` uses integer node IDs and a reused epoch array.
`original` traverses the existing `WorldNode` pointer vectors; `flat` traverses
the generated continuous query view. SIMD mode applies to flat child-MBB
rectangle filtering and flat leaf-beacon filtering; unsupported modes
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

Build commands also expose Phase1 helper thresholds for tuning the extended
sketch step: `--phase1-metric-min-fanout N` (default `64`),
`--phase1-qgram-min-fanout N` (default `64`), and
`--phase1-qgram-max-touched N` (default `250000`). These switch parent-local
candidate groups from direct scan to the metric helper and then to the q-gram
helper; every surviving candidate is still bounded-exact verified before a
world link is accepted.

Long builds emit timestamped phase progress to stderr every 600 seconds by
default. `--progress-interval-seconds N` changes the interval; zero disables
periodic heartbeats but retains phase start/finish reports. `build-scale` can
persist a reference-window index with `--index <file>` when exactly one prefix
length is requested. Multiple prefixes with one output index are rejected.

Indexed leaf attachment applies `--leaf-qgram-postfilter on` by default. The
filter runs after range candidate generation and before bounded exact
verification, using a safe q-gram L1 necessary condition to reduce leaf exact
distance calls without changing accepted links. Use
`--leaf-qgram-postfilter off` for direct verify-after-candidate comparisons.

`query-benchmark` fixes the baseline profile to MBB scan, legacy string
visited mode, original graph traversal, `dp` distance mode, and search q-gram
disabled. It compares that with the profile selected by `--mbb-filter-mode`,
`--visited-mode`, `--graph-view`, `--simd-mode`, `--distance-mode`,
`--search-qgram-prefilter`, and `--search-qgram-q`.
It deterministically generates random-region, ordinary-region,
low-complexity-region, no-hit, single-hit, and multi-hit queries. Step 0 runs
queries serially even though `--threads` is recorded and applied to OpenMP.
One best-effort eviction-buffer cold sample and configured warm samples are
reported per query/profile. Any repeated-result, cross-profile, or brute-force
no-FN mismatch makes the command return `2`.

## Module map

| Header / source | Purpose |
| --------------- | ------- |
| `include/structure.hpp`, `src/structure.cpp` | `BioSequence`, `WorldNode`, `MBB`, finest-layer leaf beacon caches, default radii `R_SW` / `R_MW` / `R_LW` |
| `include/tools.hpp`, `src/tools.cpp` | Levenshtein `compute_distance`, helpers |
| `include/qgram_filter.hpp`, `src/qgram_filter.cpp` | Exact q-gram multiset count filter and inverted index |
| `include/range_join.hpp`, `src/range_join.cpp` | Exact full, adaptive-pigeonhole, q-gram, hybrid, and auto candidate generation |
| `include/phase2_distance_verifier.hpp`, `src/phase2_distance_verifier.cpp` | CPU batch exact verifier used by Phase2 rebinding |
| `include/mbb_rect_index.hpp`, `src/mbb_rect_index.cpp` | Exact SoA rectangle lookup for parent-local child MBB filtering |
| `include/index_builder.hpp`, `src/index_builder.cpp` | `BioGeometryIndexBuilder`: dedup → phase1 extended sketch → phase2 rebinding → phase3 auxiliary collapse + MBB → leaves |
| `include/index_persistence.hpp`, `src/index_persistence.cpp` | Binary index persistence, manifest signatures, save/load, and load-time graph reconstruction |
| `include/search_engine.hpp`, `src/search_engine.cpp` | `search_adaptive`, `verify_leaf_candidates`, `search_greedy`, `search_exhaustive`, `search_brute_force` |
| `include/io_utils.hpp`, `src/io_utils.cpp` | FASTA/FASTQ load, TSV output |
| `include/map150.hpp`, `src/map150.cpp` | Fixed-150bp mapper pipeline: stride-1 reference windows, `2t` candidate search, occurrence location, final exact verifier |
| `src/main.cpp` | CLI entry points |
| `include/experiment_utils.hpp`, `src/experiment_utils.cpp` | Radius-schedule helpers for layer/radius search-cost experiments |

**Note:** Genomic coordinates in existing TSV paths are emitted from `BioSequence::ref_positions`. `map150 --locator refpos` uses the same scaffold locator, while `--locator seqan` is an optional build-time backend that fills `BioSequence::bwt_interval` as a suffix-array interval and uses that stored interval as the occurrence lookup handle during mapping.

**Note:** `build` and `query` can persist and reuse an index with
`--index <file>`. The binary file stores a manifest signature derived from input
fingerprints and construction parameters, followed by the collapsed primary DAG,
unique sequences, `ref_positions`, optional BWT/SA intervals, beacons, MBB rows,
leaf links, and leaf-beacon rows. Load reconstructs pointer links,
`SearchGraphView`, and any eligible MBB rectangle indexes. `query-index` is the
pure load-and-search command. `run`, `benchmark`, `map150`, and `boundary` still
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
| Persisted index round-trip and manifest matching | `make test_index_persistence && ./test_index_persistence_bin` |
| Phase2 CPU verifier behavior | `make test_phase2_distance_verifier && ./test_phase2_distance_verifier_bin` |
| Build heartbeat formatting and timer | `make test_build_progress` |

Search q-gram benchmark results and commands are recorded in
[`SEARCH_QGRAM_PREFILTER_BENCHMARK.md`](SEARCH_QGRAM_PREFILTER_BENCHMARK.md).
