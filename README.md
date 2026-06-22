# NavigaMer

[![Build](https://img.shields.io/badge/build-passing-success)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)

## Overview

**NavigaMer** (*Multilateration-Based Indexing and Navigation for Error-Tolerant Read Mapping*) is a multi-tiered indexer that formulates read mapping as **geometric localization in a coordinate-free metric space** under **edit distance**. Rather than embedding sequences in a continuous sketch space (with embedding distortion) or relying only on fixed seeds under high mutation rates, NavigaMer uses **beacon-mediated multilateration** and **triangle-inequality pruning** over a hierarchy of metric “worlds.” The **adaptive** search aims for **zero false negatives** (perfect recall relative to the indexed sequence set) within a user-specified edit-distance threshold, while pruning candidates that cannot contain a match.

## Methodology ↔ Code (for better code readability)

| Concept (paper) | Implementation |
| --------------- | -------------- |
| **Extended world hierarchy (sketch)** | `BioGeometryIndexBuilder::phase1_build_extended_sketch()` — `navigamer_cpp/src/index_builder.cpp` |
| **DAG topology & overlap binding** | `BioGeometryIndexBuilder::phase2_inter_tier_rebinding()` — same file |
| **Beacon extraction & tier collapse + MBBs** | `BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb()` — one auxiliary tier between each adjacent pair of primary layers is collapsed into **Metric Bounding Boxes (MBBs)** for consistency checks |
| **Leaf beacon refinement** | `BioGeometryIndexBuilder::attach_leaves()` precomputes finest-layer leaf-to-beacon distances; `BioGeometrySearchEngine::verify_leaf_candidates()` applies the final local beacon sieve before exact verification |
| **Hierarchical multilateration search** | `BioGeometrySearchEngine::search_adaptive()` — `navigamer_cpp/src/search_engine.cpp` (MBB-based pruning plus finest-layer leaf refinement via triangle inequality) |

Data structures (`WorldNode`, `MBB`, `BioSequence`) are in `navigamer_cpp/include/structure.hpp`. Edit distance is in `navigamer_cpp/src/tools.cpp`; FASTA/FASTQ/TSV I/O is in `navigamer_cpp/src/io_utils.cpp`.

## Installation

**Requirements:** Linux, **C++17**, **OpenMP**, optional **CMake >= 3.14**.

```bash
cd navigamer_cpp
make -j
# or:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Binary: `navigamer_cpp/navigamer` (or `navigamer_cpp/build/navigamer` with CMake).

**Python** is only needed for notebooks and baseline/reproducibility scripts; there is no separate Python implementation path for the paper algorithm:

```bash
pip install -r reproducibility/requirements.txt
```

## Quick start

```bash
cd navigamer_cpp
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mode adaptive
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mbb-filter-mode rect --min-rect-index-fanout 2
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --search-qgram-prefilter on --search-qgram-q 5
./navigamer build --ref ref --reads ACGTACGTACGTACGT --index /tmp/navigamer.navidx
./navigamer query-index --index /tmp/navigamer.navidx --query ACGTACGTACGTACGT --tolerance 2
./navigamer demo --size 200 --range-candidate-mode hybrid --qgram-q 5
./navigamer demo --size 200
./navigamer demo --primary-radii 30,15,5
./navigamer build-scale --ref ../data/human/chr1_subset --window 250 --stride 1 --prefix-lengths 10000,50000 --out /tmp/build_scale.csv
./navigamer build-scale --ref ../data/human/chr1_subset --window 250 --stride 1 --prefix-lengths 50000 --index /tmp/reference.navidx --out /tmp/build_scale.csv
./navigamer query-benchmark --ref ACGTGCACTGATTGCATAGCTACGAAAAAAAAAAAACCCCGGGGCCCCCCCCCGGGCCCC --window 12 --stride 12 --query-length 12 --tolerance 1 --primary-radii 12,6,2 --min-rect-index-fanout 1 --mbb-filter-mode rect --search-qgram-prefilter on --search-qgram-q 3 --warmup-iterations 1 --measured-iterations 2 --cold-cache-bytes 0 --out /tmp/query_detail.tsv --summary-out /tmp/query_summary.tsv --json-out /tmp/query_summary.json
```

Fixed-length 150 bp mapper path with final gap-aware verification:

```bash
cd navigamer_cpp
READ=$(printf 'ACGT%.0s' {1..37})AC
REF=TTTT${READ}GGGG
./navigamer map150 --ref "$REF" --reads "$READ" --tolerance 1 --out /tmp/navigamer_map150.tsv
```

Optional boundary sweep using the bundled small reference:

```bash
cd navigamer_cpp
./navigamer boundary --ref ../data/human/chr1_subset --length 250 --stride-mode sparse --queries-per-cell 1 --error-rates 0 --tolerance-rates 0 --out /tmp/navigamer_boundary.tsv
```

Layer/radius sweep for search-cost instrumentation:

```bash
cd navigamer_cpp
./navigamer layer-radius-experiment \
  --ref ../data/human/chr1_subset \
  --length 250 \
  --stride 1 \
  --tolerance 2 \
  --query-edits 2 \
  --queries-per-cell 50 \
  --L-values 2,3,4,5 \
  --r-leaf-values 4,8,12 \
  --alpha-values 0.5,0.7 \
  --out /tmp/layer_radius_search_stats.csv
```

Full CLI reference: [`navigamer_cpp/CLI_REFERENCE.md`](navigamer_cpp/CLI_REFERENCE.md). C++ layout and tests: [`navigamer_cpp/README.md`](navigamer_cpp/README.md).

The C++ `build` and `query` commands support explicit persisted indexes with
`--index <file>`. The index file stores a manifest with input fingerprints and
construction parameters plus the collapsed primary DAG, unique sequences,
reference positions, optional BWT/SA intervals, beacons, MBB rows, leaf links,
and leaf-beacon distances. `query --index <file> --query <seq>` and
`query-index --index <file> --query <seq>` load the index directly. When
`query` is given both `--reads` and `--index`, it compares the requested build
manifest with the stored manifest and reuses the file only on an exact
signature match; otherwise it rebuilds and overwrites the index. Experiment
commands such as `run`, `benchmark`, `map150`, and `boundary` still build
in-memory indexes for their current workflows. The `boundary` command reuses a
single in-memory index across the full `error_rate × tolerance_rate` grid
inside one run.

`build-scale` can also persist its reference-window index with `--index <file>`
when exactly one prefix length is requested. Multiple prefixes with one index
path are rejected instead of overwriting the file. Builds emit timestamped
phase progress to stderr every 600 seconds by default; use
`--progress-interval-seconds N` to change the interval or `0` to disable
periodic heartbeats while retaining phase-boundary reports.

Indexed construction supports exact `auto`, `pigeonhole`, `qgram`, `hybrid`,
and `full` candidate modes. Q-gram filtering uses the necessary condition
`qgram_l1(a,b) <= 2*q*tau`; hybrid intersects the pigeonhole and q-gram safe
candidate supersets. Every surviving pair is still verified by bounded exact
edit distance before an edge or leaf attachment is added.

Phase1 sketch construction has separate helper thresholds for experiments:
`--phase1-metric-min-fanout` switches parent-local candidate groups from direct
scan to the metric helper, `--phase1-qgram-min-fanout` switches larger groups to
the q-gram helper, and `--phase1-qgram-max-touched` controls conservative
fallback when q-gram candidate expansion is too broad. These knobs do not skip
bounded exact verification.

Indexed leaf attachment also uses `--leaf-qgram-postfilter on` by default. It
runs a safe q-gram L1 postfilter after range candidate generation and before
bounded exact verification, reducing leaf exact distance calls without changing
accepted links.

Auto construction accepts pigeonhole candidates when their count is at most
`4096`; if the seed union grows beyond that threshold, it early-aborts the
pigeonhole collection and invokes the q-gram safe fallback. The legacy
`--auto-pigeonhole-max-ratio` flag is parsed for command compatibility but no
longer drives auto selection, so normal pigeonhole queries do not full-scan all
length-compatible targets just to compute a ratio.

Index construction now reports aggregate build timing to stderr. The timing
breakdown covers Phase0 deduplication, Phase1 sketch construction, Phase2
rebinding, Phase3 MBB computation, Phase4 leaf attachment, ID assignment,
graph-view flattening, and selected range-join, MBB, and leaf-attachment
substeps. Phase2 rebinding, index build, and edge insert timings are wall-clock
milliseconds; Phase2 candidate-query and exact-verify worker fields are
accumulated per-thread time. The `build-scale` command writes the same timing
breakdown and construction counters to CSV for multiple reference prefix
lengths.

Adaptive search supports `--mbb-filter-mode scan|rect` (default `scan`). The
`rect` mode uses an exact in-memory rectangle index over the existing per-child
MBB rows and falls back to the original scan whenever the index is unavailable
or inconsistent. `--min-rect-index-fanout` controls the build threshold and
defaults to `64`.

Flat adaptive traversal supports `--simd-mode auto|scalar|avx2|avx512`
(default `auto`) for child MBB rectangle filtering and leaf-beacon filtering.
Unsupported SIMD paths fall back to the scalar filter and preserve the same
survivor set.

Adaptive bounded child-center distance supports `--distance-mode dp|myers|edlib|auto`
(default `myers`). Myers supports ACGT inputs through 256bp shorter input
length and falls back to DP for unsupported inputs. `edlib` uses the vendored
Edlib bounded distance backend. `dp` remains the reference mode; `auto`
currently remains DP. Index construction separately supports
`--build-distance-mode dp|edlib|auto` and defaults to `edlib`.

Adaptive child-world traversal also supports the optional
`--search-qgram-prefilter on` with independent `--search-qgram-q` (default
`5`). After MBB filtering, it safely rejects a child center only when
`qgram_l1(query, center) > 2*q*(child.radius+tolerance)`. Every passing child
still receives bounded exact edit-distance verification. The default is
`off`; unsupported q values or non-ACGT sequences conservatively fall back to
no pruning.

`query-benchmark` is a deterministic correctness and latency gate for adaptive
search. It compares a fixed baseline (`scan`, scalar MBB filtering, search
q-gram off, `dp` distance mode), the optimized profile selected by
adaptive-search flags, and exact brute-force IDs across six query classes. It writes detailed TSV, aggregate
TSV, and JSON output and returns exit status `2` on any result mismatch or
false negative.

The default CLI path still uses the legacy three primary layers (`LW/MW/SW`) via `--r-lw`, `--r-mw`, and `--r-sw`, but the C++ implementation now also supports any number of primary layers `K >= 2` through `--primary-radii coarse,...,fine`. One auxiliary tier is inserted automatically between each adjacent pair of primary layers during index construction and collapsed away before query-time navigation.

## Tests

```bash
cd navigamer_cpp
make test_recall test_distance_bound
./test_recall
./test_distance_bound
make test_build_timing_stats test_build_scale_smoke
./test_build_timing_stats
./test_build_scale_smoke
```

## Repository layout

| Path | Role |
| ---- | ---- |
| `navigamer_cpp/` | C++ v8 reference implementation and `navigamer` CLI |
| `data/human/chr1_subset` | Small reference sequence used by README examples |
| `methods/` | Comparative baselines, experiment notebooks, and plotting/evaluation workflows |
| `reproducibility/` | Optional Python dependencies (`requirements.txt`) for notebooks and baseline workflows |
