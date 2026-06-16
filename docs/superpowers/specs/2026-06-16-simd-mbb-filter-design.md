# SIMD MBB Filtering Design

## Scope

This design covers PR 7 of the NavigaMer query optimization series. It adds a
SIMD backend for child-world MBB rectangle filtering while preserving survivor
sets, survivor ordering, pruning semantics, and final query results.

PR 7 only targets MBB filtering. Leaf-beacon SIMD remains PR 8 and distance
replacement remains PR 9.

## Backend

Add:

```cpp
enum class SimdMode {
  Auto,
  Scalar,
  AVX2,
  AVX512,
};
```

`SearchConfig::simd_mode` defaults to `Auto`. `Auto` uses AVX2 only when the
compiler target and runtime CPU support it; otherwise it falls back to scalar.
`AVX512` is parsed and reported but falls back to scalar in PR 7.

## Filter Contract

The helper takes the PR 6 SoA MBB layout:

```text
lo[dim * child_count + child_idx]
hi[dim * child_count + child_idx]
```

For each child, it returns the child index iff every dimension satisfies:

```text
hi >= query_distance - tolerance
lo <= query_distance + tolerance
```

The returned survivor indices must be in ascending child-index order for both
scalar and SIMD paths.

## Integration

The flat graph-view adaptive path calls the new helper when MBB scan data is
valid. The original pointer-vector path remains available and unchanged as a
baseline. Rect-index mode remains unchanged except for sharing parser/config
metadata.

Counters added to `SearchStats`:

- `mbb_scalar_checks`
- `mbb_simd_batches`
- `mbb_simd_fallbacks`

Existing MBB counters keep their logical meaning.

## Tests

Add `test_simd_mbb_filter` with:

- randomized scalar vs auto equivalence
- randomized scalar vs AVX2 equivalence, allowing AVX2 to fall back to scalar
- dimensions `1,2,4,8,16,32`
- child counts `1,7,8,9,31,64,1000`
- deterministic seed

Extend search-view tests with scalar vs auto/AVX2 adaptive result equivalence.

## Validation

Run:

```bash
cd navigamer_cpp
make test_simd_mbb_filter
make test_search_graph_view test_query_benchmark
make test_all
./navigamer demo --size 200
./navigamer query-benchmark ... --simd-mode auto ...
```

The query benchmark must have zero equality failures and zero false negatives.

