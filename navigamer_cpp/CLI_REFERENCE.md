# NavigaMer CLI reference

The `navigamer` executable (`src/main.cpp`) implements the commands below. Paths are optional: if `--ref` or `--reads` is **not** an existing file, the argument is treated as a **literal DNA string** (see I/O rules).

## Requirements

- **C++17**, **OpenMP**
- Build: `make` or CMake (see [`README.md`](README.md) in this directory)
- Optional SeqAn3 FM-index locator for `map150`: `make WITH_SEQAN3=1` or CMake with `-DNAVIGAMER_WITH_SEQAN3=ON`

## Global hierarchy flags

Used by all pipelines that build the index:

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--primary-radii` | *(none)* | Comma-separated primary-layer radii from coarsest to finest, e.g. `40,28,18,10` |
| `--r-sw` | `5` | Small-world radius (`R_SW` in `structure.hpp`) |
| `--r-mw` | `15` | Mid-world radius |
| `--r-lw` | `30` | Large-world radius |
| `--link-mode` | `indexed` | Phase-2 rebinding: `full` or exact `indexed` range join |
| `--leaf-attach-mode` | `indexed` | Leaf attachment: `full` or exact `indexed` range join |
| `--range-min-seed-length` | `8` | Full-scan fallback below this adaptive seed length |
| `--range-max-seed-length` | `20` | Maximum adaptive pigeonhole seed length |
| `--range-candidate-mode` | `auto` | Indexed construction candidates: `auto`, `pigeonhole`, `qgram`, `hybrid`, or `full` |
| `--qgram-q` | `5` | Positive q-gram length used by q-gram and hybrid candidate generation |
| `--auto-pigeonhole-max-candidates` | `4096` | Auto accepts pigeonhole when its candidate count is at most this value |
| `--auto-pigeonhole-max-ratio` | `0.25` | Auto accepts pigeonhole when candidates / length-compatible targets is at most this ratio |
| `--auto-hybrid-on-large-candidates` | `true` | Rejected large pigeonhole sets use hybrid when true, direct q-gram when false |
| `--min-rect-index-fanout` | `64` | Minimum child-world fanout required to build an exact MBB rectangle index |
| `--mbb-filter-mode` | `scan` | Adaptive child-MBB filtering: original `scan` or exact `rect` lookup |
| `--search-qgram-prefilter` | `off` | Safe child-world center q-gram prefilter: `off` or `on` |
| `--search-qgram-q` | `5` | Search-only q-gram length; non-positive values disable the prefilter |

If `--primary-radii` is present, it takes precedence and the legacy three-radius flags are ignored. The implementation automatically inserts one auxiliary tier between each adjacent pair of primary layers during build and collapses those auxiliary tiers into beacons + MBB rows before query-time navigation.

Indexed construction is exact. Pigeonhole mode uses
`block_len = floor(L / (tau + 1))` and
`seed_len = min(range_max_seed_length, block_len)`, falling back to the
length-compatible full set when the seed is too short. Q-gram mode safely
prunes only pairs with `qgram_l1(a,b) > 2*q*tau`. Hybrid mode intersects the
pigeonhole and q-gram safe candidate supersets. Auto runs pigeonhole when its
seed is long enough, accepting it when candidate count is below the configured
maximum **or** candidate ratio is below the configured maximum. Otherwise it
invokes q-gram and returns the safe hybrid intersection by default. Full
candidate mode returns every length-compatible item.

Old seed-length-only auto behavior can be reproduced with permissive thresholds
such as `--auto-pigeonhole-max-candidates 18446744073709551615
--auto-pigeonhole-max-ratio 1.0`. Because acceptance uses OR semantics, a
dataset with fewer than the default `4096` length-compatible targets accepts
pigeonhole regardless of candidate ratio.

Every returned candidate is still verified with bounded exact edit distance;
candidate generation never directly adds an edge or leaf attachment. Builder
summaries distinguish possible pairs, returned candidates, exact calls,
accepted results, length pruning, q-gram L1 pruning, per-mode query counts,
fallback counts, and reduction ratios.

Rectangle filtering is also exact. Rect mode returns a child world if and only
if its existing MBB row intersects the query rectangle in every beacon
dimension. It changes only survivor enumeration; center-distance verification,
containment/overlap traversal, and leaf verification remain unchanged. Missing
or inconsistent indexes, small fanout, dimension mismatches, and query
exceptions fall back to scan.

Search-side q-gram filtering is also exact and no-false-negative. It runs only
after child MBB survivor generation and safely prunes when
`qgram_l1(query, child.center) > 2*q*(child.radius+tolerance)`. Passing children
still receive bounded exact center edit-distance verification. It does not
change coarsest-layer search, strict containment, overlap traversal, leaf
refinement, construction rebinding, or leaf attachment. World-center
signatures are cached per search engine. Non-ACGT centers/queries, unsupported
q values, and missing signatures conservatively fall back to no pruning.

## I/O conventions (`io_utils`)

- **`--ref`**: If the value is a **path to a file**, load **FASTA** (`>` header, sequence lines). Otherwise treat the whole argument as one reference sequence (ID `ref`).
- **`--reads`**: If the value is a **file path**, load **FASTQ** (`@` id, sequence, `+`, quality). Otherwise treat it as a **single read** sequence (ID `query_0`).

## Commands

### `demo`

Synthetic reference (~50 kb) and reads (length 20, zero mutation rate). Compares adaptive vs exhaustive vs brute force on a sample of reads.

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--size` | `500` | Number of reads |
| `--primary-radii` or `--r-sw`, `--r-mw`, `--r-lw` | `30,15,5` by default through the legacy flags | Primary-layer radii |

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

### `map150`

Fixed-length mapper path for 150 bp reads. Builds an in-memory index from all forward reference 150-mers at stride 1, searches each read on both strands with `candidate_tolerance = 2 * --tolerance`, then verifies candidate occurrence neighborhoods exactly and emits only alignments with edit distance `<= --tolerance`.

**Required:** `--ref`, `--reads`, `--out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Final edit-distance threshold; the candidate search uses `2 * tolerance` |
| `--mode` | `adaptive` | Currently only `adaptive` is accepted |
| `--locator` | `refpos` | `refpos` uses stored reference-window positions; `seqan` requires optional SeqAn3 support |
| `--out` | *(required)* | Output TSV path; header is written even when there are no hits |

Safety constraints: every read must be exactly 150 bp, reference and reads must contain only A/C/G/T, and the finest primary radius must be greater than `2 * tolerance`.

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

### `layer-radius-experiment`

Builds one index per `(L, r_leaf, alpha)` combination, reuses a fixed query set across the full sweep, and writes one CSV row per query with search-cost instrumentation only.

**Required:** `--ref`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--length` | `250` | Reference window and query length |
| `--tolerance` | `2` | Edit-distance threshold used during search |
| `--query-edits` | `--tolerance` | Number of substitution edits applied when generating the fixed query set |
| `--queries-per-cell` | `200` | Number of fixed queries reused across every `(L, r_leaf, alpha)` combination |
| `--stride` | *(unset)* | Explicit reference-window stride; if set, overrides `--stride-mode` |
| `--stride-mode` | `sparse` | `sparse` uses stride `length`; `dense` uses stride `length / 4` |
| `--seed` | `42` | Seed for fixed query generation |
| `--L-values` | `2,3,4,5` | Comma-separated primary-layer counts |
| `--r-leaf-values` | `4,8,12` | Comma-separated finest-layer radii |
| `--alpha-values` | `0.5,0.7` | Comma-separated geometric decay factors in `(0,1)` |
| `--out` | `layer_radius_search_stats.csv` | Output CSV path |

Each primary-layer radius schedule is generated geometrically from `(L, r_leaf, alpha)` and written to the CSV as a pipe-delimited string such as `64|32|16|8`.

## TSV columns

**`run`:**  
`query_id`, `hit_id`, `distance`, `ref_positions`, `read_id`, `read_len`, `ref_id`, `strand`, `query_start`, `reference_start`, `aligned_length`, `score`, `edit_distance`, `query_fragment`, `reference_fragment`, `bwt_start`, `bwt_end`

**`benchmark`** adds:  
`dist_calcs`, `leaf_verify_count`, `candidate_count_for_prune`, `beacon_prune_count`,
`mbb_filter_mode`, `mbb_scan_child_checks`, `mbb_rect_index_queries`,
`mbb_rect_candidate_children`, `mbb_rect_fallback_count`,
`mbb_surviving_child_count`, `center_distance_calls_after_mbb`,
`search_qgram_prefilter_enabled`, `search_qgram_q`,
`search_qgram_signature_build_count`, `search_qgram_signature_missing_count`,
`search_qgram_checks`, `search_qgram_pruned_children`,
`search_qgram_passed_children`, `center_distance_calls_before_qgram`,
`center_distance_calls_after_qgram`, `qgram_prune_ratio`, `result_count`,
`avg_mbb_candidates_per_parent`,
`avg_center_distance_calls_per_query`, `query_time_ms`

`center_distance_calls_after_mbb` is retained as a compatibility alias for
`center_distance_calls_before_qgram`. The actual bounded center edit-distance
calls are reported by `center_distance_calls_after_qgram`.

`candidate_count_for_prune` and `beacon_prune_count` include both hierarchy-level MBB pruning and finest-layer leaf-beacon refinement.

**`map150`:**
`query_id`, `hit_id`, `distance`, `ref_id`, `strand`, `query_start`, `reference_start`, `aligned_length`, `score`, `edit_distance`, `query_fragment`, `reference_fragment`, `bwt_start`, `bwt_end`, `dist_calcs`, `leaf_verify_count`, `candidate_count_for_prune`, `beacon_prune_count`

For `map150 --locator refpos`, `bwt_start` and `bwt_end` are `-1`. With the optional SeqAn locator they represent the half-open suffix-array interval for the matched 150-mer leaf, not genomic coordinates; mapping uses that stored interval as the occurrence lookup handle.

**`boundary`:**  
`length`, `stride_mode`, `num_index_seqs`, `error_rate`, `error_edits`, `tolerance_rate`, `tolerance_edits`, `query_count`, `source_recovery_rate`, `any_hit_rate`, `avg_hit_count`, `avg_dist_calcs`, `avg_leaf_verify_count`, `avg_candidate_count_for_prune`, `avg_beacon_prune_count`, `avg_pruning_rate`, `bf_sample_count`, `bf_source_recovery_rate`, `bf_agreement_rate`, `bf_source_mismatch_count`

**`layer-radius-experiment`:**  
`dataset`, `query_id`, `query_length`, `L`, `r_leaf`, `alpha`, `radius_schedule`, `query_time_ms`, `world_access_count`, `node_access_count`, `edge_access_count`, `anchor_distance_count`, `bound_check_count`, `candidate_count`, `candidate_verify_count`

## Standalone test binaries

| Binary | Purpose |
| ------ | ------- |
| `test_recall` | Randomized recall check: adaptive vs brute force |
| `test_distance_bound` | Distance-bound checks across search modes |
| `test_hierarchy_config` | Hierarchy-config validation for multi-primary-layer builds |
| `test_search_stats_bin` | Radius-schedule and search-cost instrumentation checks |
| `test_map150_recall` | Fixed-150bp mapper recall, strand, duplicate, and verifier checks |
| `test_bounded_edit_distance` | Banded thresholded distance vs full Levenshtein |
| `test_qgram_filter` | Q-gram counts, L1 bound, ambiguous bases, and index no-false-negative checks |
| `test_range_join` | Pigeonhole/q-gram/hybrid no-false-negative and verified-pair checks |
| `test_build_range_equivalence` | Full vs q-gram/hybrid/auto construction and search-result equivalence |
| `test_mbb_rect_index` | Exact rectangle intersection and randomized naive-scan equivalence |
| `test_mbb_filter_equivalence` | Adaptive scan/rect result equality, recall, counters, and fallback |
| `test_search_qgram_prefilter` | Search q-gram on/off, scan/rect, ambiguous-base fallback, containment, and center-call reduction checks |

Build with `make test_recall` / `make test_distance_bound`.
