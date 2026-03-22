# NavigaMer

[![Build](https://img.shields.io/badge/build-passing-success)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)

## Overview

**NavigaMer** (*Multilateration-Based Indexing and Navigation for Error-Tolerant Read Mapping*) is a multi-tiered indexer that formulates read mapping as **geometric localization in a coordinate-free metric space** under **edit distance**. Rather than embedding sequences in a continuous sketch space (with embedding distortion) or relying only on fixed seeds under high mutation rates, NavigaMer uses **beacon-mediated multilateration** and **triangle-inequality pruning** over a hierarchy of metric “worlds.” The **adaptive** search aims for **zero false negatives** (perfect recall relative to the indexed sequence set) within a user-specified edit-distance threshold, while pruning candidates that cannot contain a match.

## Methodology ↔ Code

| Concept (paper) | Implementation |
| --------------- | -------------- |
| **Extended world hierarchy (sketch)** | `BioGeometryIndexBuilder::phase1_build_extended_sketch()` — `navigamer_cpp/src/index_builder.cpp` |
| **DAG topology & overlap binding** | `BioGeometryIndexBuilder::phase2_inter_tier_rebinding()` — same file |
| **Beacon extraction & tier collapse + MBBs** | `BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb()` — intermediate tiers collapsed; **Metric Bounding Boxes (MBBs)** for consistency checks |
| **Hierarchical multilateration search** | `BioGeometrySearchEngine::search_adaptive()` — `navigamer_cpp/src/search_engine.cpp` (MBB-based pruning via triangle inequality) |

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

**Python** (notebooks / reproduction):

```bash
pip install -r reproducibility/requirements.txt
```

## Quick start

```bash
cd navigamer_cpp
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mode adaptive
./navigamer demo --size 200
```

Full CLI reference: [`navigamer_cpp/CLI_REFERENCE.md`](navigamer_cpp/CLI_REFERENCE.md). C++ layout and tests: [`navigamer_cpp/README.md`](navigamer_cpp/README.md).

## Tests

```bash
cd navigamer_cpp
make test_recall test_distance_bound
./test_recall
./test_distance_bound
```

## Reproducing paper results (Figure 2)

```bash
cd reproducibility
bash run_figure_2_reproduction.sh
```

(Provided in the submission artifact when available.)

## Repository layout

| Path | Role |
| ---- | ---- |
| `navigamer_cpp/` | C++ reference implementation and `navigamer` CLI |
| `reproducibility/` | Scripts and dependencies for paper figures |
| `methods/` | Comparative baselines and experiment notebooks (historical) |
