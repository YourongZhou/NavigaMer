# NavigaMer

[![Build](https://img.shields.io/badge/build-passing-success)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)

## Overview

**NavigaMer** (Multilateration-Based Indexing and Navigation for Error-Tolerant Read Mapping) is a multi-tiered indexer that treats read mapping as **geometric localization in a coordinate-free metric space** defined by the **edit distance**. Instead of embedding sequences in a continuous space (with unavoidable distortion) or relying solely on fixed seeds at high mutation rates, NavigaMer uses **beacon-mediated multilateration** and **triangle-inequality pruning** across a hierarchy of metric “worlds.” The adaptive search is designed so that, within a user-specified edit-distance threshold, **retrieval has zero false negatives (perfect recall)** with respect to the indexed sequence set, while aggressively pruning unrelated candidates.

## Methodology Alignment

The C++ implementation is structured so that each major theoretical object in the manuscript maps to a concrete module and entry point:

| Manuscript concept | Code location |
| ------------------ | ------------- |
| **The World Hierarchy (Extended Sketch)** | `BioGeometryIndexBuilder::phase1_build_extended_sketch()` in `navigamer_cpp/src/index_builder.cpp` |
| **DAG Topology & Overlap Binding** | `BioGeometryIndexBuilder::phase2_inter_tier_rebinding()` in `navigamer_cpp/src/index_builder.cpp` |
| **Beacon Extraction & Tier Collapse** | `BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb()` in `navigamer_cpp/src/index_builder.cpp` — intermediate tiers are collapsed and **Metric Minimum Bounding Boxes (MBBs)** are attached for \(O(1)\) consistency checks |
| **Hierarchical Multilateration Search** | `BioGeometrySearchEngine::search_adaptive()` in `navigamer_cpp/src/search_engine.cpp` — **MBB-based pruning** using the triangle inequality (constant-time per beacon dimension, subject to the number of beacons) |

Supporting definitions (metric balls, `WorldNode`, `MBB`, sequence records) live in `navigamer_cpp/include/structure.hpp`. Edit distance and I/O are in `navigamer_cpp/src/tools.cpp` and `navigamer_cpp/src/io_utils.cpp`.

## Installation

**Environment:** Linux with a C++17 compiler, **OpenMP**, and optionally **CMake 3.14+**.

**C++ backend (reference implementation):**

```bash
cd navigamer_cpp
make -j
# or:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

The main binary is `navigamer_cpp/navigamer` (or `navigamer_cpp/build/navigamer` when using CMake).

**Python (notebooks / reproduction scripts):**

```bash
pip install -r reproducibility/requirements.txt
```

## Quick Start (Toy Example)

From the repository root, after building:

```bash
cd navigamer_cpp && ./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mode adaptive
```

One-line built-in stress test (synthetic reference and reads):

```bash
./navigamer demo --size 200
```

The CLI accepts FASTA/FASTQ paths or raw sequence strings; see `navigamer_cpp/CLI参数说明.md` for parameters (`--r-sw`, `--r-mw`, `--r-lw`, `run`, `benchmark`, etc.).

## Reproducing Paper Results (Figure 2)

To reproduce the recall and precision benchmarks shown in **Figure 2** of the manuscript, navigate to the `reproducibility/` directory and run:

```bash
bash run_figure_2_reproduction.sh
```

This script is provided as part of the submission artifact; it regenerates the figures and summary statistics from the paper’s experimental protocol.

---

*For implementation-focused build and module notes specific to the C++ tree, see [`navigamer_cpp/README.md`](navigamer_cpp/README.md).*
