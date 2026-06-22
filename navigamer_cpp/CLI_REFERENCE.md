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
| `--leaf-attach-direction` | `auto` | Indexed leaf attachment direction: `auto`, `seq-to-world`, or `world-to-seq`; `auto` uses `world-to-seq` when finest worlds are fewer than unique sequences |
| `--leaf-qgram-postfilter` | `on` | Safe q-gram L1 postfilter for indexed leaf attachment candidates before bounded exact verification |
| `--range-min-seed-length` | `8` | Full-scan fallback below this adaptive seed length |
| `--range-max-seed-length` | `20` | Maximum adaptive pigeonhole seed length |
| `--range-candidate-mode` | `auto` | Indexed construction candidates: `auto`, `pigeonhole`, `qgram`, `hybrid`, or `full` |
| `--qgram-q` | `5` | Positive q-gram length used by q-gram and hybrid candidate generation |
| `--auto-pigeonhole-max-candidates` | `4096` | Auto accepts pigeonhole when its candidate count is at most this value |
| `--auto-pigeonhole-max-ratio` | `0.25` | Compatibility no-op; parsed and validated, but auto no longer computes or uses a candidate ratio |
| `--auto-hybrid-on-large-candidates` | `true` | Compatibility flag; normal auto early-aborts oversized seed unions and uses q-gram safe fallback |
| `--build-distance-mode` | `edlib` | Index construction edit-distance backend: default `edlib`, reference `dp`, or conservative `auto` (currently DP) |
| `--min-rect-index-fanout` | `64` | Minimum child-world fanout required to build an exact MBB rectangle index |
| `--phase1-metric-min-fanout` | `64` | Minimum Phase1 parent-local candidate fanout before building/querying the metric helper instead of scanning |
| `--phase1-qgram-min-fanout` | `64` | Minimum Phase1 parent-local candidate fanout before using the q-gram helper instead of the metric helper |
| `--phase1-qgram-max-touched` | `250000` | Maximum Phase1 q-gram touched/candidate set size before conservatively falling back |
| `--progress-interval-seconds` | `600` | Timestamped build heartbeat interval on stderr; `0` disables periodic heartbeats but keeps phase-boundary reports |
| `--mbb-filter-mode` | `scan` | Adaptive child-MBB filtering: original `scan` or exact `rect` lookup |
| `--visited-mode` | `epoch` | Adaptive visited tracking: legacy per-query `string` set or integer-ID `epoch` array |
| `--graph-view` | `flat` | Adaptive graph traversal storage: existing pointer-vector `original` or continuous query `flat` view |
| `--simd-mode` | `auto` | Flat child-MBB and leaf-beacon filter backend: `auto`, `scalar`, `avx2`, or `avx512`; unsupported SIMD falls back to scalar |
| `--distance-mode` | `myers` | Adaptive bounded child-center distance backend: default Myers through 256bp ACGT shorter-input length, optional `edlib`, reference `dp`, or conservative `auto` (currently DP) |
| `--search-qgram-prefilter` | `off` | Safe child-world center q-gram prefilter: `off` or `on` |
| `--search-qgram-q` | `5` | Search-only q-gram length; non-positive values disable the prefilter |
| `--index` | *(none)* | Persisted NavigaMer index path for `build`, single-prefix `build-scale`, `query`, and `query-index` |

If `--primary-radii` is present, it takes precedence and the legacy three-radius flags are ignored. The implementation automatically inserts one auxiliary tier between each adjacent pair of primary layers during build and collapses those auxiliary tiers into beacons + MBB rows before query-time navigation.

Indexed construction is exact. Pigeonhole mode uses
`block_len = floor(L / (tau + 1))` and
`seed_len = min(range_max_seed_length, block_len)`, falling back to the
length-compatible full set when the seed is too short. Q-gram mode safely
prunes only pairs with `qgram_l1(a,b) > 2*q*tau`. Hybrid mode intersects the
pigeonhole and q-gram safe candidate supersets. Auto runs pigeonhole when its
seed is long enough, accepting it when candidate count is at most the
configured maximum. If the seed union grows beyond the maximum, auto stops
collecting pigeonhole candidates immediately and invokes q-gram as a safe
fallback. It does not full-scan all length-compatible targets to compute a
candidate ratio; `--auto-pigeonhole-max-ratio` is retained only so older
command lines still parse. Full candidate mode returns every
length-compatible item.

Indexed leaf attachment additionally applies `--leaf-qgram-postfilter on` by
default after range candidate generation and before bounded exact verification.
It uses the same no-false-negative q-gram L1 condition and only reduces exact
distance calls; accepted leaf links are still determined by bounded edit
distance. Use `--leaf-qgram-postfilter off` to reproduce the earlier direct
verify-after-candidate behavior.

Old seed-length-only auto behavior can be reproduced with a permissive
candidate-count threshold such as
`--auto-pigeonhole-max-candidates 18446744073709551615`; the ratio flag no
longer affects acceptance.

Every returned candidate is still verified with bounded exact edit distance;
candidate generation never directly adds an edge or leaf attachment. Builder
summaries distinguish possible pairs, returned candidates, exact calls,
accepted results, length pruning, q-gram L1 pruning, per-mode query counts,
fallback counts, seed candidates before length filtering, seed length-pruned
candidates, pigeonhole early-abort counts, final range candidates, and
reduction ratios.

Phase1 sketch construction uses the same bounded exact verifier after any
helper path. For each parent-local candidate group, fanout below
`--phase1-metric-min-fanout` scans directly; fanout between
`--phase1-metric-min-fanout` and `--phase1-qgram-min-fanout` uses the metric
helper; larger fanout uses the q-gram helper unless it exceeds
`--phase1-qgram-max-touched` and falls back conservatively.

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

Deduplicates, builds the index, and prints layer sizes. If `--index <file>` is
provided, writes a persisted binary index with a manifest signature, build
parameters, input fingerprints, unique sequences, `ref_positions`, optional
BWT/SA intervals, collapsed DAG links, beacons, MBB rows, leaf links, and
leaf-beacon distance rows.

**Required:** `--ref`, `--reads`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--index` | *(none)* | Output path for the persisted index |

Every build prints a `Build timing` section to stderr. Timing fields are
aggregate wall-clock milliseconds and include Phase0 deduplication, Phase1
sketch construction, Phase2 rebinding, Phase3 MBB computation, Phase4 leaf
attachment, integer ID assignment, graph-view flattening, and selected
range-join/MBB/leaf substeps. Existing construction counters remain in the
summary.

### `build-scale`

Builds one index per requested reference prefix and writes construction timing
plus construction counters to CSV. With `--index <file>`, exactly one prefix
must be requested and the resulting reference-window index is serialized.

**Required:** `--ref`, `--prefix-lengths`, `--out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--window` | `200` | Reference-window length |
| `--stride` | `1` | Step between window starts |
| `--prefix-lengths` | *(required)* | Comma-separated reference prefix lengths; values larger than the reference use the full reference |
| `--out` | *(required)* | Output CSV path |
| `--index` | *(none)* | Persist the single requested prefix as a loadable binary index |
| `--progress-interval-seconds` | `600` | Periodic progress interval; `0` keeps only phase start/finish reports |

After each prefix, stderr prints the top build phases, top substeps, and
candidate/exact reduction percentages for Phase2 and leaf attachment.
Heartbeat lines include timestamp, phase, completed/total work items,
percentage, elapsed time, observed rate, and ETA. Progress output never changes
the CSV shape or persisted-index signature.

### `query`

Builds an index from `--reads`, then searches for `--query`. With
`--index <file>`, `query` first compares the requested build manifest with the
stored manifest. If the signatures match, it loads the persisted index and skips
construction. If the file is missing, unreadable, or has different inputs or
construction parameters, it rebuilds from `--reads` and writes the new index to
that path. With `--index` and no `--reads`, `query` loads the index directly.

**Required:** `--query` plus either `--reads` or `--index`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Max edit distance |
| `--mode` | `adaptive` | `adaptive` \| `greedy` \| `exhaustive`; all modes exactly verify returned leaves |
| `--ref` | optional | Placeholder in current flow |
| `--index` | *(none)* | Persisted index to reuse, create, or load directly |

### `query-index`

Loads a persisted index and searches `--query`. This command never rebuilds and
does not accept `--reads`; use `query --reads ... --index ...` for automatic
reuse-or-rebuild behavior.

**Required:** `--index`, `--query`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Max edit distance |
| `--mode` | `adaptive` | `adaptive` \| `greedy` \| `exhaustive` |

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

### `query-benchmark`

Builds one shared in-memory index, deterministically generates six query
classes (`random_region`, `ordinary_region`, `low_complexity_region`,
`no_hit`, `single_hit`, and `multi_hit`), and compares:

- baseline: fixed `scan` MBB filtering, legacy `string` visited mode,
  `original` graph traversal, scalar MBB filtering, `dp` distance mode, and
  search q-gram disabled
- optimized: `--mbb-filter-mode`, `--visited-mode`, `--graph-view`,
  `--simd-mode`, `--distance-mode`, `--search-qgram-prefilter`, and
  `--search-qgram-q`
- exact brute-force result IDs computed before timing

**Required:** `--ref`, `--out`, `--summary-out`, `--json-out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--reference-subset-length` | `0` | Reference prefix length; `0` uses the full input |
| `--window` | `200` | Indexed reference-window length |
| `--stride` | `1` | Indexed window stride |
| `--query-length` | `200` | Generated query length |
| `--tolerance` | `2` | Exact edit-distance threshold |
| `--seed` | `42` | Deterministic generation seed |
| `--threads` | `1` | Recorded/applied OpenMP thread count; Step 0 query execution remains serial |
| `--queries-per-class` | `1` | Queries generated for each class |
| `--warmup-iterations` | `2` | Untimed warmups per query/profile |
| `--measured-iterations` | `10` | Timed warm samples per query/profile |
| `--cold-cache-bytes` | `268435456` | Best-effort eviction buffer touched before each cold sample; `0` disables it |
| `--out` | required | Detailed per-sample TSV |
| `--summary-out` | required | Per-class/profile plus `all` aggregate TSV |
| `--json-out` | required | Configuration, build, memory, aggregate, mismatch, and gate JSON |

The timed region contains only `search_adaptive`. Results must be stable across
repeated executions and exactly match between profiles and brute force. The
command writes all outputs before returning and exits `0` when the gate passes,
`2` on a result/no-FN mismatch, and `1` on configuration or runtime errors.
Current/peak RSS telemetry is best effort. Candidate-set comparison and
per-query allocation counting are explicitly reported as `unavailable`.

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

`boundary` currently uses substitution-only mutations and, for each cell,
additionally samples up to 50 queries for `brute_force` agreement checks. This
experiment command builds and reuses an in-memory index within the run; it does
not use the persisted-index path.

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
`mbb_surviving_child_count`, `mbb_scalar_checks`, `mbb_simd_batches`,
`mbb_simd_fallbacks`, `center_distance_calls_after_mbb`,
`leaf_beacon_scalar_checks`, `leaf_beacon_simd_batches`,
`leaf_beacon_simd_fallbacks`,
`search_qgram_prefilter_enabled`, `search_qgram_q`,
`search_qgram_signature_build_count`, `search_qgram_signature_missing_count`,
`search_qgram_checks`, `search_qgram_pruned_children`,
`search_qgram_passed_children`, `center_distance_calls_before_qgram`,
`center_distance_calls_after_qgram`, `qgram_prune_ratio`, `result_count`,
`avg_mbb_candidates_per_parent`,
`avg_center_distance_calls_per_query`, `query_time_ms`

**`query-benchmark` detail:**  
`query_id`, `query_class`, `profile`, `sample_kind`, `iteration`,
`first_profile`, `latency_ms`, `result_count`, `brute_force_result_count`,
`result_equal`, `no_fn`, `world_access_count`, `node_access_count`,
`edge_access_count`, `mbb_checks`, `mbb_survivors`, `mbb_scalar_checks`,
`mbb_simd_batches`, `mbb_simd_fallbacks`, `qgram_checks`,
`center_exact_distance_calls`, `leaf_beacon_checks`,
`leaf_beacon_scalar_checks`, `leaf_beacon_simd_batches`,
`leaf_beacon_simd_fallbacks`,
`leaf_exact_distance_calls`, `visited_checks`, `visited_hits`,
`candidate_count`, `verified_candidate_count`

The summary TSV reports cold/warm average, p50, p95, and p99 latency; query,
sample, result, equality-failure, and false-negative totals; and average
logical counters for each query-class/profile pair plus `all`.

`center_distance_calls_after_mbb` is retained as a compatibility alias for
`center_distance_calls_before_qgram`. The actual bounded center edit-distance
calls are reported by `center_distance_calls_after_qgram`.

`candidate_count_for_prune` and `beacon_prune_count` include both hierarchy-level MBB pruning and finest-layer leaf-beacon refinement.

**`map150`:**
`query_id`, `hit_id`, `distance`, `ref_id`, `strand`, `query_start`, `reference_start`, `aligned_length`, `score`, `edit_distance`, `query_fragment`, `reference_fragment`, `bwt_start`, `bwt_end`, `dist_calcs`, `leaf_verify_count`, `candidate_count_for_prune`, `beacon_prune_count`

For `map150 --locator refpos`, `bwt_start` and `bwt_end` are `-1`. With the optional SeqAn locator they represent the half-open suffix-array interval for the matched 150-mer leaf, not genomic coordinates; mapping uses that stored interval as the occurrence lookup handle.

**`boundary`:**  
`length`, `stride_mode`, `num_index_seqs`, `error_rate`, `error_edits`, `tolerance_rate`, `tolerance_edits`, `query_count`, `source_recovery_rate`, `any_hit_rate`, `avg_hit_count`, `avg_dist_calcs`, `avg_leaf_verify_count`, `avg_candidate_count_for_prune`, `avg_beacon_prune_count`, `avg_pruning_rate`, `bf_sample_count`, `bf_source_recovery_rate`, `bf_agreement_rate`, `bf_source_mismatch_count`

**`build-scale`:**
`prefix_len`, `window_count`, `unique_count`, `world_node_count`,
`finest_world_count`, `total_build_ms`, `phase0_dedup_ms`,
`phase1_sketch_ms`, `phase2_rebinding_ms`, `phase2_index_build_ms`,
`phase2_candidate_query_ms`, `phase2_exact_verify_ms`,
`phase2_distance_batches`,
`phase2_edge_insert_ms`, `phase3_mbb_ms`,
`phase3_collect_beacons_ms`, `phase3_collapse_children_ms`,
`phase3_child_mbb_distance_ms`, `phase3_rect_index_build_ms`,
`phase4_attach_ms`, `leaf_index_build_ms`, `leaf_candidate_query_ms`,
`leaf_exact_verify_ms`, `leaf_tuple_emit_ms`, `leaf_tuple_merge_sort_ms`,
`leaf_populate_ms`, `leaf_beacon_distance_ms`, `assign_ids_ms`,
`graph_view_ms`, `phase2_total_possible_pairs`, `phase2_candidate_pairs`,
`phase2_exact_distance_calls`, `phase2_edges_added`,
`leaf_total_possible_pairs`, `leaf_candidate_pairs`,
`leaf_exact_distance_calls`, `leaf_attachments_added`,
`phase2_full_scan_fallback_count`, `leaf_full_scan_fallback_count`,
`phase2_seed_candidate_pairs_before_length_filter`,
`phase2_seed_length_pruned_candidates`,
`phase2_pigeonhole_early_abort_count`,
`phase2_range_final_candidate_pairs`,
`leaf_seed_candidate_pairs_before_length_filter`,
`leaf_seed_length_pruned_candidates`, `leaf_pigeonhole_early_abort_count`,
`leaf_range_final_candidate_pairs`,
`leaf_attach_direction_used`, `range_candidate_mode`, `qgram_q`,
`phase2_candidate_query_worker_ms`, `phase2_exact_verify_worker_ms`

`phase2_rebinding_ms`, `phase2_index_build_ms`, and
`phase2_edge_insert_ms` are wall-clock timings. The Phase 2 worker fields are
per-thread accumulated query and exact-verification time for parallel indexed
rebinding.

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
| `test_bounded_myers_bin` | Optional bounded Myers backend vs full Levenshtein and DP fallback |
| `test_qgram_filter` | Q-gram counts, L1 bound, ambiguous bases, and index no-false-negative checks |
| `test_range_join` | Pigeonhole/q-gram/hybrid no-false-negative and verified-pair checks |
| `test_build_range_equivalence` | Full vs q-gram/hybrid/auto construction and search-result equivalence |
| `test_build_timing_stats` | Construction timing field and summary smoke checks |
| `test_build_scale_smoke` | `build-scale` CSV timing smoke check |
| `test_build_progress_bin` | Timestamped build progress formatting, forced boundaries, and periodic heartbeat |
| `test_mbb_rect_index` | Exact rectangle intersection and randomized naive-scan equivalence |
| `test_mbb_filter_equivalence` | Adaptive scan/rect result equality, recall, counters, and fallback |
| `test_search_qgram_prefilter` | Search q-gram on/off, scan/rect, ambiguous-base fallback, containment, and center-call reduction checks |
| `test_search_distance_mode_bin` | Adaptive `dp` vs `myers` bounded center-distance equivalence |
| `test_query_benchmark_gate` | Deterministic query classes, dual-profile output, exact result equality, and no-FN gate |
| `test_phase2_distance_verifier_bin` | CPU Phase2 exact verifier batch equivalence |

Build with `make test_recall` / `make test_distance_bound`.
