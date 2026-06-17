# SIMD Leaf-Beacon Filter Design

## Goal

PR8 adds a SIMD backend for adaptive search leaf-beacon filtering without
changing pruning semantics or final result sets. The filter only decides which
leaf candidates proceed to existing exact edit-distance verification.

## Existing Context

The flat `SearchGraphView` already stores leaf beacon distances in a
cache-friendly SoA layout:

```text
leaf_beacon_dists[leaf_beacon_begin[node] + dim * leaf_count + leaf_idx]
```

The current flat search path checks each leaf candidate with scalar logic:

```text
prune iff abs(query_beacon_dist[dim] - leaf_beacon_dist[dim][leaf]) > tolerance
```

The original `WorldNode` path uses `node->leaf_beacon_dists` row vectors and
must remain available for baseline/equivalence runs.

## Approach

Reuse the PR7 `SimdMode` CLI/config and add a leaf-specific helper:

```cpp
struct LeafBeaconFilterSimdStats {
  size_t scalar_checks = 0;
  size_t simd_batches = 0;
  size_t simd_fallbacks = 0;
};

std::vector<uint32_t> filter_leaf_beacon_survivors(
    const int32_t* dist_by_dim,
    size_t leaf_count,
    size_t dim,
    const int32_t* query_beacon_dists,
    int32_t tolerance,
    SimdMode mode,
    LeafBeaconFilterSimdStats* stats = nullptr);
```

The helper returns survivor leaf offsets in ascending order. `Scalar` is the
reference implementation. `Auto` uses AVX2 when compiled and supported at
runtime; otherwise it falls back to scalar and increments fallback counters.
`AVX512` remains a parsed reserved mode and falls back scalar in this PR.

## Search Integration

Only the flat `verify_leaf_candidates_view()` hot path calls the SIMD helper.
The original pointer path keeps its scalar loop, with scalar counter updates
added for reporting.

The flat path updates existing logical counters exactly as before:

- `leaf_beacon_check_count += leaf_count`
- `candidate_count_for_prune += leaf_count`
- `bound_check_count += leaf_count`
- `beacon_prune_count += leaf_count - survivor_count`

After filtering, every survivor still executes:

```text
candidate_count++
candidate_verify_count++
leaf_exact_distance_call_count++
compute_distance(query, leaf)
leaf_verify_count++
```

No exact verification is removed or approximated.

## Counters

Add search stats and benchmark outputs:

- `leaf_beacon_scalar_checks`
- `leaf_beacon_simd_batches`
- `leaf_beacon_simd_fallbacks`

The existing `leaf_exact_distance_call_count` remains the source for exact
verification counts. PR8 does not add separate before/after fields because both
baseline and optimized profiles are already reported as separate rows in
`query-benchmark`.

## Tests

1. `test_simd_leaf_beacon_filter`
   - Random SoA leaf distances.
   - Dimensions: `1,2,4,8,16,32`.
   - Leaf counts: `1,7,8,9,31,64,1000`.
   - Tolerances: `0,1,3,7,15`.
   - Assert scalar, auto, and AVX2 survivor offsets match the reference.

2. `test_search_graph_view`
   - Extend flat/original adaptive search equivalence to exercise
     `SimdMode::Scalar`, `Auto`, and `AVX2`.
   - Result IDs and result counts must remain identical.

3. `test_query_benchmark_gate`
   - Confirm JSON and TSV include leaf SIMD counters.
   - Gate must pass with zero mismatches and zero false negatives.

## Non-Goals

- No routing, shard, sparse graph, construction, or leaf attachment changes.
- No MBB pruning semantics changes.
- No exact distance replacement.
- No AVX512 implementation in PR8.
