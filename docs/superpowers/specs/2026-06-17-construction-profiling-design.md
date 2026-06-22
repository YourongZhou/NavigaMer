# Construction Profiling Design

## Goal

Add low-overhead construction profiling to the C++ index builder so build bottlenecks can be attributed to major phases and selected substeps.

## Scope

The C++ implementation records aggregate wall-clock timing in `BioGeometryIndexBuilder::Statistics`. The builder summary prints the timing breakdown, and a new `build-scale` CLI command writes timing and construction counters to CSV for multiple reference prefixes.

The feature does not change index topology, candidate generation safety, edit-distance verification, pruning, or search behavior.

## Architecture

Timing uses `std::chrono::steady_clock` and a small RAII `ScopedTimer` in the builder/range-join implementation. Timers accumulate milliseconds into `double` fields. High-frequency loops are timed at loop or query granularity, not per candidate pair or per edit-distance call.

`ExactRangeJoinIndex::query()` returns internal timing in `RangeJoinQueryResult`, including posting lookup, seed union, length filtering, q-gram query, hybrid intersection, and full-scan fallback time. Phase2 rebinding and Phase4 leaf attachment aggregate those values into builder statistics.

OpenMP regions avoid direct multi-threaded writes to shared timing fields. For the current full leaf attachment path, the builder records the parallel region wall time and aggregate phase wall time.

## Timing Fields

Phase-level fields:

- `total_build_ms`
- `phase0_dedup_ms`
- `phase1_sketch_ms`
- `phase2_rebinding_ms`
- `phase3_mbb_ms`
- `phase4_attach_ms`
- `assign_ids_ms`
- `graph_view_ms`
- `print_summary_ms`

Phase2 fields:

- `phase2_index_build_ms`
- `phase2_candidate_query_ms`
- `phase2_exact_verify_ms`
- `phase2_edge_insert_ms`

Phase3 fields:

- `phase3_collect_beacons_ms`
- `phase3_collapse_children_ms`
- `phase3_child_mbb_distance_ms`
- `phase3_rect_index_build_ms`

Phase4 fields:

- `leaf_index_build_ms`
- `leaf_candidate_query_ms`
- `leaf_exact_verify_ms`
- `leaf_tuple_emit_ms`
- `leaf_tuple_merge_sort_ms`
- `leaf_populate_ms`
- `leaf_beacon_distance_ms`

Range-join aggregate fields:

- `range_posting_lookup_ms`
- `range_seed_union_ms`
- `range_length_filter_ms`
- `range_qgram_query_ms`
- `range_hybrid_intersection_ms`
- `range_full_scan_ms`

## `build-scale`

The new command builds reference windows for each requested reference prefix length:

```bash
./navigamer build-scale --ref data/human/chr1_subset --window 250 --stride 1 --prefix-lengths 10000,50000 --out build_scale.csv
```

For each prefix, the command writes requested CSV columns, including window counts, unique counts, world-node counts, timing breakdown, construction counters, leaf attach direction, range candidate mode, and q-gram q.

After each prefix build, stderr prints a compact bottleneck summary with top phase and substep entries.

## Tests

`test_build_timing_stats` builds a small synthetic index and verifies timing fields are positive or non-negative, phase timing is within a loose total-build budget, and summary printing does not crash.

`test_build_scale_smoke` runs the new CLI command on a tiny literal reference with two prefixes, then verifies timing columns exist and each row has a positive total build time and phase breakdown.
