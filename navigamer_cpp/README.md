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
./navigamer boundary --ref <fasta> [--length 250] [--error-rates csv] [--tolerance-rates csv] [--queries-per-cell 200] [--stride-mode sparse|dense] [--seed 42] [--out out.tsv]
./navigamer layer-radius-experiment --ref <fasta> [--length 250] [--tolerance 2] [--query-edits 2] [--queries-per-cell 200] [--stride 1 | --stride-mode sparse|dense] [--seed 42] [--L-values 2,3,4,5] [--r-leaf-values 4,8,12] [--alpha-values 0.5,0.7] [--out out.csv]
```

**Full syntax and defaults:** [`CLI_REFERENCE.md`](CLI_REFERENCE.md).

## Module map

| Header / source | Purpose |
| --------------- | ------- |
| `include/structure.hpp`, `src/structure.cpp` | `BioSequence`, `WorldNode`, `MBB`, finest-layer leaf beacon caches, default radii `R_SW` / `R_MW` / `R_LW` |
| `include/tools.hpp`, `src/tools.cpp` | Levenshtein `compute_distance`, helpers |
| `include/range_join.hpp`, `src/range_join.cpp` | Exact adaptive-pigeonhole candidate generation with explicit full-scan fallback |
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
full-pairwise construction. Indexed mode caps adaptive pigeonhole seeds at 20
bp, visibly falls back when seeds would be shorter than 8 bp, and exact verifies
every candidate before adding a link.

## Parameter sweeps

For long-sequence boundary studies, `boundary` outputs one aggregated TSV row per `(error_rate, tolerance_rate)` cell with source-recovery and pruning metrics for fixed-length `L=250` windows derived from a reference FASTA such as `chr1_subset`. Broader experiment orchestration and comparative baseline workflows live under the repository-level `methods/` directory.

## Tests

| Target | Command |
| ------ | ------- |
| Recall (adaptive vs brute force, 0 FN under test protocol) | `make test_recall && ./test_recall` |
| Distance bounds (violations report) | `make test_distance_bound && ./test_distance_bound` |
| 150bp mapper recall and verifier checks | `make test_map150 && ./test_map150_recall` |
| Bounded edit distance | `make test_bounded && ./test_bounded_edit_distance` |
| Exact range join | `make test_range_join && ./test_range_join` |
| Full/indexed construction equivalence | `make test_build_range && ./test_build_range_equivalence` |
