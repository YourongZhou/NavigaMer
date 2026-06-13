# Exact MBB Rectangle Index Design

## Goal

Add an optional exact rectangle-index lookup to NavigaMer's C++ adaptive search
so a parent world can enumerate child worlds whose MBB rows intersect the query
rectangle without scanning every child MBB row.

The rectangle lookup is filter-equivalent to the existing scan. It must not
change construction edges, leaf attachments, center-distance verification,
strict-containment behavior, overlap traversal, or final search results.

## Scope

This change applies only to query-time child-world MBB filtering in
`BioGeometrySearchEngine::search_adaptive()`.

All CLI paths that use adaptive search support the new search configuration:
`demo`, `query`, `run`, `map150`, `benchmark`, `boundary`, and
`layer-radius-experiment`.

The following are explicitly out of scope:

- Changes to `phase2_inter_tier_rebinding()`
- Changes to `attach_leaves()`
- Changes to construction edge or leaf-attachment semantics
- Approximate rectangle search, nearest-neighbor search, or top-k search
- Persisted index serialization
- Applying the rectangle index to greedy, exhaustive, or brute-force search

## Rectangle Index

Add `MBBRectIndex` in:

- `navigamer_cpp/include/mbb_rect_index.hpp`
- `navigamer_cpp/src/mbb_rect_index.cpp`

The class stores exact rectangles in structure-of-arrays form:

```cpp
class MBBRectIndex {
 public:
  struct Rect {
    uint32_t child_id = 0;
    std::vector<int> lo;
    std::vector<int> hi;
  };

  void build(const std::vector<Rect>& rects);
  std::vector<uint32_t> query_intersect(
      const std::vector<int>& q_lo,
      const std::vector<int>& q_hi) const;
  size_t size() const;
  size_t dim() const;
};
```

`child_id` is the child's stable position in the parent's `child_nodes` and
`child_beacon_mbbs` vectors, not a parsed node identifier.

`build()` validates that every rectangle has the same dimension and that each
dimension satisfies `lo <= hi`. Invalid input leaves the index empty.
An empty index is unavailable to search and causes scan fallback.

`query_intersect()` returns a child ID if and only if every dimension overlaps:

```text
child.hi[i] >= q_lo[i] && child.lo[i] <= q_hi[i]
```

For a valid zero-dimensional index, it returns every child ID. It returns an
empty result for an unavailable index or query dimension mismatch. Search
requires nonempty query beacon dimensions before using an index, so
zero-dimensional parents and other unavailable conditions cause scan fallback
rather than an empty survivor set.

The first implementation uses exact SoA filtering. It may choose dimension
order for efficiency, but cannot alter the returned set.

## World Integration And Build

`WorldNode` holds an optional `std::shared_ptr<MBBRectIndex>`. A shared pointer
keeps existing `WorldNode` copy/move behavior simple and does not affect the
current `shared_ptr<WorldNode>` graph ownership.

Add `min_rect_index_fanout` to `BuildRangeConfig`, defaulting to `64`. During
`phase3_collapse_and_compute_mbb()`, after each non-finest primary world's
`child_nodes` and `child_beacon_mbbs` are finalized:

1. Clear any previous rectangle index.
2. Check that fanout is at least the configured threshold.
3. Check that beacons and MBB rows are nonempty and dimensionally aligned.
4. Build one rectangle per child using the existing MBB `min_dist` and
   `max_dist` values.
5. Store the index only if its size and dimension match the source rows.

No MBB values are recomputed or redefined. Rebuilding an index clears old
rectangle indexes along with the other phase-3 metadata.

## Search Configuration And Data Flow

Add:

```cpp
enum class MBBFilterMode {
  Scan,
  RectIndex,
};

struct SearchConfig {
  MBBFilterMode mbb_filter_mode = MBBFilterMode::Scan;
};
```

`BioGeometrySearchEngine` accepts an optional `SearchConfig`. Scan remains the
default for compatibility.

Refactor adaptive child enumeration into one internal helper. It first computes
the existing query-to-beacon distance vector. Scan mode executes the current
per-child MBB pruning logic.

Rect mode constructs:

```text
q_lo[i] = V_Q[i] - tolerance
q_hi[i] = V_Q[i] + tolerance
```

and queries the parent's rectangle index. Returned child positions select the
same `child_nodes` entries that survived the old MBB scan.

Rect mode falls back to the unchanged scan path when:

- The rectangle index is absent or empty
- Parent fanout was below the configured build threshold
- Beacon, MBB-row, index, or query dimensions do not agree
- A returned child position is out of range
- Rectangle querying throws an exception

After survivor generation, adaptive search continues through the existing
`search_layer_adaptive()` logic. Center-distance verification, strict
containment, overlap traversal, visited-node handling, and leaf verification
remain unchanged.

## CLI

Add global flags:

```text
--mbb-filter-mode scan|rect
--min-rect-index-fanout N
```

Defaults:

```text
--mbb-filter-mode scan
--min-rect-index-fanout 64
```

The mode is passed to every `BioGeometrySearchEngine` used by an adaptive CLI
path. The threshold is passed to every builder. Invalid modes and nonpositive
thresholds produce a clear CLI error.

Update the root README, C++ README, and `CLI_REFERENCE.md`.

## Instrumentation

Extend `SearchStats` with:

- `mbb_scan_child_checks`
- `mbb_rect_index_queries`
- `mbb_rect_candidate_children`
- `mbb_rect_fallback_count`
- `mbb_filter_parent_count`
- `mbb_surviving_child_count`
- `center_distance_calls_after_mbb`

`mbb_filter_mode` is available from `SearchConfig` and emitted by benchmark
output. The benchmark TSV also emits the new counters and:

- `avg_mbb_candidates_per_parent`, calculated per query as rectangle candidate
  children or scan survivors divided by `mbb_filter_parent_count`
- `avg_center_distance_calls_per_query`, equal to the per-query
  `center_distance_calls_after_mbb`

Both modes increment `center_distance_calls_after_mbb` immediately before
center-distance calls made for MBB-surviving child worlds. Top-level world
center checks are excluded. Existing stats keep their current meanings and
ordering; new benchmark columns are appended.

Other adaptive CLI paths use the counters internally but do not change their
existing result TSV schemas unless they already emit search-cost statistics.

## Tests

Add a standalone `test_mbb_rect_index` target covering:

- Handwritten 2D and 3D intersection cases
- Random equivalence against naive intersection for dimensions 1, 2, 4, 8,
  and 16
- Empty, zero-dimensional, inconsistent-dimension, and invalid-bound input

Add a standalone `test_mbb_filter_equivalence` target covering:

- Scan and rect adaptive result-set equality on the same built index and
  queries
- Equality of MBB survivor behavior through end-to-end search
- Rect fallback for missing index, dimension mismatch, empty MBBs, and fanout
  below threshold
- Rect-mode recall relative to brute force
- Counter behavior for scan, rect, and fallback paths

The equivalence test sets a low fanout threshold where needed so small fixtures
actually exercise the rectangle path.

## Validation And Benchmark

Run:

```bash
cd navigamer_cpp
make -j
make test_all
./test_mbb_rect_index
./test_mbb_filter_equivalence
./navigamer demo --size 200 --mbb-filter-mode scan
./navigamer demo --size 200 --mbb-filter-mode rect --min-rect-index-fanout 2
```

Run a small benchmark twice against the same reference, reads, hierarchy, and
tolerance, once in each MBB filter mode. Compare result sets independently of
row order and report query time, MBB candidate/check counts, center-distance
calls, result count, and result equality.

Acceptance requires exact result equality, no recall regression, all existing
recall and distance-bound tests passing, and counters that expose whether rect
lookup reduced survivor-enumeration work.
