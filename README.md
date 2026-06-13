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

**Requirements:** Linux, **C++17**, **OpenMP**, optional **CMake ≥ 3.14**.

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
./navigamer demo --size 200 --range-candidate-mode hybrid --qgram-q 5
./navigamer demo --size 200
./navigamer demo --primary-radii 30,15,5
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

The current C++ index is in-memory only: `build`, `query`, `run`, `benchmark`, `map150`, and `boundary` rebuild the index for each process invocation. The `boundary` command is intended for long-sequence capability sweeps and reuses a single in-memory index across the full `error_rate × tolerance_rate` grid inside one run.

Indexed construction supports exact `auto`, `pigeonhole`, `qgram`, `hybrid`,
and `full` candidate modes. Q-gram filtering uses the necessary condition
`qgram_l1(a,b) <= 2*q*tau`; hybrid intersects the pigeonhole and q-gram safe
candidate supersets. Every surviving pair is still verified by bounded exact
edit distance before an edge or leaf attachment is added.

Adaptive search supports `--mbb-filter-mode scan|rect` (default `scan`). The
`rect` mode uses an exact in-memory rectangle index over the existing per-child
MBB rows and falls back to the original scan whenever the index is unavailable
or inconsistent. `--min-rect-index-fanout` controls the build threshold and
defaults to `64`.

The default CLI path still uses the legacy three primary layers (`LW/MW/SW`) via `--r-lw`, `--r-mw`, and `--r-sw`, but the C++ implementation now also supports any number of primary layers `K >= 2` through `--primary-radii coarse,...,fine`. One auxiliary tier is inserted automatically between each adjacent pair of primary layers during index construction and collapsed away before query-time navigation.

## Tests

```bash
cd navigamer_cpp
make test_recall test_distance_bound
./test_recall
./test_distance_bound
```

## Repository layout

| Path | Role |
| ---- | ---- |
| `navigamer_cpp/` | C++ v8 reference implementation and `navigamer` CLI |
| `data/human/chr1_subset` | Small reference sequence used by README examples |
| `methods/` | Comparative baselines, experiment notebooks, and plotting/evaluation workflows |
| `reproducibility/` | Optional Python dependencies (`requirements.txt`) for notebooks and baseline workflows |
