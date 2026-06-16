# SearchGraphView Continuous Layout Design

## Scope

This design covers PR 6 of the NavigaMer query optimization series. It adds a
read-only, cache-friendly query view over the finalized in-memory index and
lets adaptive search use that view without changing graph construction, edge
semantics, leaf attachment semantics, pruning semantics, exact verification, or
result sets.

PR 6 does not introduce SIMD and does not change bounded edit distance. SIMD
MBB and leaf-beacon filtering remain PR 7 and PR 8.

## SearchGraphView

Add a query-side structure owned by `BioGeometryIndexBuilder`:

```cpp
struct SearchGraphView {
  std::vector<std::shared_ptr<WorldNode>> nodes;
  std::vector<std::shared_ptr<BioSequence>> leaves;

  std::vector<NodeId> child_ids;
  std::vector<uint32_t> child_begin;
  std::vector<uint32_t> child_end;

  std::vector<LeafId> leaf_ids;
  std::vector<uint32_t> leaf_begin;
  std::vector<uint32_t> leaf_end;

  std::vector<int32_t> mbb_lo;
  std::vector<int32_t> mbb_hi;
  std::vector<uint32_t> mbb_begin;
  std::vector<uint32_t> mbb_dim;
  std::vector<uint32_t> beacon_begin;
  std::vector<uint32_t> beacon_end;

  std::vector<int32_t> leaf_beacon_dists;
  std::vector<uint32_t> leaf_beacon_begin;
  std::vector<uint32_t> leaf_beacon_dim;
};
```

`nodes[id]` maps integer node IDs back to existing `WorldNode` objects.
`leaves[id]` maps integer leaf IDs back to existing `BioSequence` objects.
The original graph remains the build/debug representation.

For each node ID:

- children are `child_ids[child_begin[id] : child_end[id]]`
- attached leaves are `leaf_ids[leaf_begin[id] : leaf_end[id]]`
- MBB data for children is stored as SoA:
  `offset + dim * child_count + child_index`
- leaf beacon distances use the same dim-major layout:
  `offset + dim * leaf_count + leaf_index`

The view is built after integer IDs are assigned. Build validation fails if any
node or leaf attachment still has an invalid integer ID.

## Search Config

Add:

```cpp
enum class GraphViewMode {
  Original,
  Flat,
};
```

`SearchConfig::graph_view_mode` defaults to `Flat` for optimized query mode.
`Original` keeps the current pointer-vector path for regression comparisons.

Add CLI:

```text
--graph-view original|flat
```

`query-benchmark` fixes the baseline profile to `graph_view=original` and
uses the CLI-selected mode for optimized profile. The no-FN gate therefore
compares the legacy structure against the flat view on the same queries.

## Adaptive View Path

PR 6 adds view-backed adaptive helpers parallel to the existing helpers. The
view path still calls the same distance, q-gram, MBB, leaf-beacon, and exact
verification logic. It only reads child IDs and leaf IDs from continuous
arrays, then resolves IDs through the view pointer tables when existing
`WorldNode` or `BioSequence` data is required.

The initial implementation may keep local candidate vectors, but candidate
vectors should store `NodeId` instead of `shared_ptr<WorldNode>` on the view
path. This removes repeated pointer-vector traversal from the hot path while
keeping behavior deterministic.

Ordering must match the original child and leaf vector order. Result IDs must
be exactly equal between `Original` and `Flat`.

## Tests

Add `test_search_graph_view` with:

- every node ID is valid and maps to the same original pointer
- every view child list equals `WorldNode::child_nodes` converted to child IDs
- every view leaf list equals `WorldNode::child_leaves` converted to leaf IDs
- every MBB `lo/hi` value equals the original `child_beacon_mbbs`
- every flattened leaf-beacon distance equals the original
- adaptive `graph_view=original` and `graph_view=flat` return identical result
  sets on scan/rect, q-gram off/on, and visited string/epoch combinations

Include the new target in Make, CMake, and `make test_all`.

## Validation

Run:

```bash
cd navigamer_cpp
make test_search_graph_view
make test_epoch_visited test_search_stats test_mbb_filter test_search_qgram test_query_benchmark
make test_all
./navigamer demo --size 200
./navigamer query-benchmark ... --graph-view flat ...
```

The query benchmark must report zero equality failures and zero false
negatives. Counters should remain logically comparable; performance may be
neutral on small datasets because PR 6 is primarily a layout prerequisite for
PR 7 and PR 8.

