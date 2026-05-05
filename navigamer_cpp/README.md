# NavigaMer — C++ reference implementation

This directory contains the **C++17 v8** reference indexer and CLI (`navigamer`) used for the paper implementation. The build pipeline follows a **top-down extended hierarchy**, **inter-tier DAG wiring**, **intermediate-tier collapse with beacon sequences and MBBs**, and **leaf attachment** to small-world (SW) nodes. Adaptive search uses **precomputed per-edge MBB rows** (`WorldNode::child_beacon_mbbs`) for hierarchy pruning and **SW leaf beacon rows** (`WorldNode::leaf_beacon_dists`) for the final local refinement step before exact verification.

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
./navigamer demo   [--size N] [--r-sw 5] [--r-mw 15] [--r-lw 30]
./navigamer build  --ref <fasta|sequence> --reads <fastq|sequence>  [same radius flags]
./navigamer query  --reads <fastq|sequence> --query <sequence> [--tolerance 2] [--mode adaptive|greedy|exhaustive]
./navigamer run    --ref <fasta|sequence> --reads <fastq|sequence> [--tolerance 2] [--out out.tsv]
./navigamer benchmark --ref <fasta> --reads <fastq> [--tolerance 2] [--window 200] [--stride 1] [--out out.tsv]
./navigamer boundary --ref <fasta> [--length 250] [--error-rates csv] [--tolerance-rates csv] [--queries-per-cell 200] [--stride-mode sparse|dense] [--seed 42] [--out out.tsv]
```

**Full syntax and defaults:** [`CLI_REFERENCE.md`](CLI_REFERENCE.md).

## Module map

| Header / source | Purpose |
| --------------- | ------- |
| `include/structure.hpp`, `src/structure.cpp` | `BioSequence`, `WorldNode`, `MBB`, SW leaf beacon caches, radii `R_SW` / `R_MW` / `R_LW` |
| `include/tools.hpp`, `src/tools.cpp` | Levenshtein `compute_distance`, helpers |
| `include/index_builder.hpp`, `src/index_builder.cpp` | `BioGeometryIndexBuilder`: dedup → phase1 extended sketch → phase2 rebinding → phase3 collapse + MBB → leaves |
| `include/search_engine.hpp`, `src/search_engine.cpp` | `search_adaptive`, `verify_leaf_candidates`, `search_greedy`, `search_exhaustive`, `search_brute_force` |
| `include/io_utils.hpp`, `src/io_utils.cpp` | FASTA/FASTQ load, TSV output |
| `src/main.cpp` | CLI entry points |

**Note:** Genomic coordinates in TSV are emitted from `BioSequence::ref_positions`; no separate FM-index lookup path is part of the current paper implementation.

**Note:** The current C++ implementation still does **not** serialize the index to disk. `build`, `query`, `run`, `benchmark`, and `boundary` all rebuild the index in memory for each invocation. `boundary` avoids repeated rebuilds within a parameter sweep by building once per stride mode and reusing that in-memory index across the full rate grid.

## Parameter sweeps

For long-sequence boundary studies, `boundary` outputs one aggregated TSV row per `(error_rate, tolerance_rate)` cell with source-recovery and pruning metrics for fixed-length `L=250` windows derived from a reference FASTA such as `chr1_subset`. Broader experiment orchestration and comparative baseline workflows live under the repository-level `methods/` directory.

## Tests

| Target | Command |
| ------ | ------- |
| Recall (adaptive vs brute force, 0 FN under test protocol) | `make test_recall && ./test_recall` |
| Distance bounds (violations report) | `make test_distance_bound && ./test_distance_bound` |
