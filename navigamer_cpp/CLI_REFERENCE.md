# NavigaMer CLI reference

The `navigamer` executable (`src/main.cpp`) implements the commands below. Paths are optional: if `--ref` or `--reads` is **not** an existing file, the argument is treated as a **literal DNA string** (see I/O rules).

## Requirements

- **C++17**, **OpenMP**
- Build: `make` or CMake (see [`README.md`](README.md) in this directory)

## Global radius flags

Used by all pipelines that build the index:

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--r-sw` | `5` | Small-world radius (`R_SW` in `structure.hpp`) |
| `--r-mw` | `15` | Mid-world radius |
| `--r-lw` | `30` | Large-world radius |

The implementation also uses **extended** tiers internally (see `index_builder.cpp`); these three knobs set the primary metric balls. SW nodes additionally store local leaf-beacon rows for the final refinement sieve.

## I/O conventions (`io_utils`)

- **`--ref`**: If the value is a **path to a file**, load **FASTA** (`>` header, sequence lines). Otherwise treat the whole argument as one reference sequence (ID `ref`).
- **`--reads`**: If the value is a **file path**, load **FASTQ** (`@` id, sequence, `+`, quality). Otherwise treat it as a **single read** sequence (ID `query_0`).

## Commands

### `demo`

Synthetic reference (~50 kb) and reads (length 20, zero mutation rate). Compares adaptive vs exhaustive vs brute force on a sample of reads.

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--size` | `500` | Number of reads |
| `--r-sw`, `--r-mw`, `--r-lw` | 5 / 15 / 30 | Radii |

### `build`

Deduplicates, builds the index, prints layer sizes. **Index is not serialized to disk**; use `run` for an end-to-end path with optional TSV.

**Required:** `--ref`, `--reads`

### `query`

Builds an index from `--reads`, then searches for `--query`.

**Required:** `--reads`, `--query`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Max edit distance |
| `--mode` | `adaptive` | `adaptive` \| `greedy` \| `exhaustive`; all modes exactly verify returned leaves |
| `--ref` | optional | Placeholder in current flow |

### `run`

Full pipeline: load ref + reads, build, **adaptive** search for every read, optional TSV.

**Required:** `--ref`, `--reads`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Threshold |
| `--out` | *(none)* | If set, write TSV; otherwise only stderr summary |

Uses OpenMP over reads.

### `benchmark`

Slices the reference into windows of length `--window` with stride `--stride`; each window is one indexed sequence with coordinates. Query sequences come from `--reads`. Uses **adaptive** search; TSV includes search statistics.

**Required:** `--ref`, `--reads` (queries)

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Edit threshold |
| `--window` | `200` | Window length on the reference |
| `--stride` | `1` | Step between window starts |
| `--out` | *(none)* | Output TSV path |

If a query has no hit, a placeholder row is still emitted with stats.

### `boundary`

Builds one in-memory index from reference windows of fixed length `--length` and sweeps a full `error_rate × tolerance_rate` grid without rebuilding the index for each cell. This command is intended for capability-boundary exploration on long reference slices such as `chr1_subset`.

**Required:** `--ref`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--length` | `250` | Fixed window/query length; current implementation only accepts `250` |
| `--error-rates` | `0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20` | Comma-separated substitution error rates; each rate becomes `round(rate * 250)` edits |
| `--tolerance-rates` | `0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20` | Comma-separated tolerance rates; each rate becomes `round(rate * 250)` edit distance |
| `--queries-per-cell` | `200` | Number of mutated queries evaluated for each `(error_rate, tolerance_rate)` cell |
| `--stride-mode` | `sparse` | `sparse` uses stride `250`; `dense` uses stride `62` |
| `--seed` | `42` | Random seed used for query sampling and mutation |
| `--out` | *(none)* | Output TSV path for the aggregated boundary table |

`boundary` currently uses substitution-only mutations and, for each cell, additionally samples up to 50 queries for `brute_force` agreement checks. Like the other C++ commands, the index is built in memory only and is not serialized to disk.

## TSV columns

**`run`:**  
`query_id`, `hit_id`, `distance`, `ref_positions`, `read_id`, `read_len`, `ref_id`, `strand`, `query_start`, `reference_start`, `aligned_length`, `score`, `edit_distance`, `query_fragment`, `reference_fragment`, `bwt_start`, `bwt_end`

**`benchmark`** adds:  
`dist_calcs`, `leaf_verify_count`, `candidate_count_for_prune`, `beacon_prune_count`

`candidate_count_for_prune` and `beacon_prune_count` include both hierarchy-level MBB pruning and SW leaf-beacon refinement.

**`boundary`:**  
`length`, `stride_mode`, `num_index_seqs`, `error_rate`, `error_edits`, `tolerance_rate`, `tolerance_edits`, `query_count`, `source_recovery_rate`, `any_hit_rate`, `avg_hit_count`, `avg_dist_calcs`, `avg_leaf_verify_count`, `avg_candidate_count_for_prune`, `avg_beacon_prune_count`, `avg_pruning_rate`, `bf_sample_count`, `bf_source_recovery_rate`, `bf_agreement_rate`, `bf_source_mismatch_count`

## Standalone test binaries

| Binary | Purpose |
| ------ | ------- |
| `test_recall` | Randomized recall check: adaptive vs brute force |
| `test_distance_bound` | Distance-bound checks across search modes |

Build with `make test_recall` / `make test_distance_bound`.
