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
./navigamer build  --ref <fasta|sequence> --reads <fastq|sequence>  [same primary-layer flags]
./navigamer query  --reads <fastq|sequence> --query <sequence> [--tolerance 2] [--mode adaptive|greedy|exhaustive]
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
`--distance-mode dp|myers|auto` (default `myers`),
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
shorter-input length, falling back to DP otherwise. `dp` remains the reference
mode; `auto` is conservative and currently uses DP.

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
| `include/mbb_rect_index.hpp`, `src/mbb_rect_index.cpp` | Exact SoA rectangle lookup for parent-local child MBB filtering |
| `include/index_builder.hpp`, `src/index_builder.cpp` | `BioGeometryIndexBuilder`: dedup → phase1 extended sketch → phase2 rebinding → phase3 auxiliary collapse + MBB → leaves |
| `include/search_engine.hpp`, `src/search_engine.cpp` | `search_adaptive`, `verify_leaf_candidates`, `search_greedy`, `search_exhaustive`, `search_brute_force` |
| `include/io_utils.hpp`, `src/io_utils.cpp` | FASTA/FASTQ load, TSV output |
| `include/map150.hpp`, `src/map150.cpp` | Fixed-150bp mapper pipeline: stride-1 reference windows, `2t` candidate search, occurrence location, final exact verifier |
| `src/main.cpp` | CLI entry points |
| `include/experiment_utils.hpp`, `src/experiment_utils.cpp` | Radius-schedule helpers for layer/radius search-cost experiments |

**Note:** Genomic coordinates in existing TSV paths are emitted from `BioSequence::ref_positions`. `map150 --locator refpos` uses the same scaffold locator, while `--locator seqan` is an optional build-time backend that fills `BioSequence::bwt_interval` as a suffix-array interval and uses that stored interval as the occurrence lookup handle during mapping.

**Note:** The current C++ implementation still does **not** serialize the index to disk. `build`, `query`, `run`, `benchmark`, `map150`, and `boundary` all rebuild the index in memory for each invocation. `boundary` avoids repeated rebuilds within a parameter sweep by building once per stride mode and reusing that in-memory index across the full rate grid.

**Note:** The legacy three-primary-layer configuration remains the default CLI path, but the generalized implementation accepts arbitrary primary-layer lists such as `--primary-radii 40,28,18,10`. One auxiliary tier is generated automatically between each adjacent pair of primary layers and collapsed before search.

**Note:** Phase-2 rebinding and leaf attachment default to exact indexed range
joins. Use `--link-mode full` and/or `--leaf-attach-mode full` for the original
full-pairwise construction. `--range-candidate-mode auto` uses adaptive
pigeonhole seeds when they are at least 8 bp, then checks actual candidate
count and candidate ratio. Large candidate sets invoke q-gram and use the safe
hybrid intersection by default. Forced `qgram` and `hybrid` modes are also
available. All modes exact verify every surviving candidate before adding a
link.

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
| Exact MBB rectangle lookup | `make test_mbb_rect && ./test_mbb_rect_index` |
| Scan/rect adaptive equivalence and fallback | `make test_mbb_filter && ./test_mbb_filter_equivalence` |
| Search q-gram on/off and scan/rect equivalence | `make test_search_qgram && ./test_search_qgram_prefilter` |

Search q-gram benchmark results and commands are recorded in
[`SEARCH_QGRAM_PREFILTER_BENCHMARK.md`](SEARCH_QGRAM_PREFILTER_BENCHMARK.md).
