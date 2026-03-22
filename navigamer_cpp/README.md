# NavigaMer — C++ reference implementation

This directory contains the **C++17** reference indexer and CLI (`navigamer`). The build pipeline follows a **top-down extended hierarchy**, **inter-tier DAG wiring**, **intermediate-tier collapse with beacon sequences and MBBs**, and **leaf attachment** to small-world (SW) nodes. Adaptive search uses **precomputed per-edge MBB rows** (see `WorldNode::child_beacon_mbbs`) for pruning.

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
```

**Full syntax and defaults:** [`CLI_REFERENCE.md`](CLI_REFERENCE.md).

## Module map

| Header / source | Purpose |
| --------------- | ------- |
| `include/structure.hpp`, `src/structure.cpp` | `BioSequence`, `WorldNode`, `MBB`, radii `R_SW` / `R_MW` / `R_LW` |
| `include/tools.hpp`, `src/tools.cpp` | Levenshtein `compute_distance`, helpers |
| `include/index_builder.hpp`, `src/index_builder.cpp` | `BioGeometryIndexBuilder`: dedup → phase1 extended sketch → phase2 rebinding → phase3 collapse + MBB → leaves |
| `include/search_engine.hpp`, `src/search_engine.cpp` | `search_adaptive`, `search_greedy`, `search_exhaustive`, `search_brute_force` |
| `include/io_utils.hpp`, `src/io_utils.cpp` | FASTA/FASTQ load, TSV output |
| `src/main.cpp` | CLI entry points |

**Note:** FM-index integration for genomic `ref_positions` in TSV is not implemented; coordinates may be empty unless sequences carry pre-annotated occurrences.

## Parameter sweeps

[`params_test.ipynb`](params_test.ipynb) runs the binary via Python `subprocess` for quick parameter exploration. Set `NAVIGAMER` to the absolute path of `navigamer` if auto-detection fails.

## Tests

| Target | Command |
| ------ | ------- |
| Recall (adaptive vs brute force, 0 FN under test protocol) | `make test_recall && ./test_recall` |
| Distance bounds (violations report) | `make test_distance_bound && ./test_distance_bound` |
