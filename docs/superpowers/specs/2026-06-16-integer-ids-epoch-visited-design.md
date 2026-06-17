# Integer IDs and Epoch Visited Design

## Scope

This design covers PR 5 of the NavigaMer query optimization series. It replaces
adaptive-search visited tracking on the optimized path from per-query
`unordered_set<std::string>` lookups to integer node IDs plus a reusable epoch
visited array.

The change must not alter index construction semantics, edge semantics, leaf
attachment semantics, MBB pruning semantics, q-gram pruning semantics, exact
verification, traversal ordering, or final result sets.

## Data Model

Keep existing string identifiers for compatibility and debugging:

- `WorldNode::node_id`
- `BioSequence::id`

Add integer IDs:

- `using NodeId = uint32_t`
- `using LeafId = uint32_t`
- `WorldNode::integer_id`
- `BioSequence::sequence_id`

IDs are assigned after `attach_leaves()` finishes, when the finalized primary
graph and leaf attachments are stable.

World node IDs are assigned by iterating finalized primary layers in layer
order and node-vector order, skipping already-seen node pointers. Leaf/sequence
IDs are assigned to builder `unique_sequences` values in deterministic
string-ID order. Both ID spaces are contiguous from `0`.

The builder exposes:

- `num_world_nodes()`
- `num_sequences()`
- `validate_integer_ids()`

These are read-only query/test helpers. They do not change construction.

## Search Config

Add:

```cpp
enum class VisitedMode {
  StringSet,
  Epoch,
};
```

`SearchConfig::visited_mode` defaults to `Epoch`. Tests can force
`StringSet` to compare against the legacy behavior.

No CLI flag is added in PR 5. The no-FN gate and tests exercise both modes
directly through `SearchConfig`.

## Search Scratch

Add:

```cpp
struct SearchScratch {
  std::vector<uint32_t> visited_epoch;
  uint32_t current_epoch = 0;
  std::vector<std::shared_ptr<WorldNode>> frontier;
  std::vector<std::shared_ptr<WorldNode>> next_frontier;
  std::vector<std::shared_ptr<WorldNode>> mbb_candidates;
  std::vector<std::shared_ptr<WorldNode>> verified_children;

  void begin_query(size_t node_count);
  bool mark_visited(NodeId id);
};
```

`begin_query()` resizes `visited_epoch` only when needed. It increments the
epoch for each query; on overflow it fills the array with zero and restarts at
epoch `1`.

Adaptive epoch mode uses a `thread_local SearchScratch`, so parallel callers do
not share visited state. The string-set mode remains function-local and
preserves the old allocation behavior for regression comparison.

The vectors are included in PR 5 to establish the scratch object and reusable
capacity boundary. This PR only has to move overlap storage into scratch where
it is straightforward; broader frontier/view layout changes belong to PR 6.

## Adaptive Search Behavior

The legacy adaptive visited logic is retained as a separate string-set path.
The epoch path performs the same logical checks at the same decision points:

1. Skip already-visited candidates before center-distance work.
2. Mark the contained node before descending.
3. Mark each overlap node before descending.

Counters keep their current logical meaning:

- `visited_check_count` increments at every visited check.
- `visited_hit_count` increments when a check rejects a node.

The only allowed counter differences are incidental visited implementation
differences if a future test explicitly allows them. Final result IDs must be
exactly equal between modes.

## Tests

Add `test_epoch_visited` with:

- integer world IDs are unique, contiguous, and `< num_world_nodes()`
- sequence IDs are unique, contiguous, and `< num_sequences()`
- `validate_integer_ids()` returns true after build
- `SearchScratch::mark_visited()` returns true then false in the same epoch
- a new epoch allows marking the same ID again
- overflow fallback clears the array and restarts at epoch `1`
- adaptive string-set mode and epoch mode return identical result sets on the
  same index, query set, tolerance, MBB scan/rect settings, and q-gram off/on

Include the new target in Make, CMake, and `make test_all`.

## Validation

Run:

```bash
cd navigamer_cpp
make test_epoch_visited
./test_epoch_visited
make test_search_stats test_mbb_filter test_search_qgram test_recall test_dist
make test_all
./navigamer demo --size 200
```

Run the existing `query-benchmark` mixed synthetic gate with default epoch mode
and with a direct test using string-set mode. Result sets must be identical.

## Deferred

PR 5 does not change q-gram signature cache storage, child edge storage,
leaf-beacon storage, MBB layout, SIMD behavior, distance implementation, or CLI
flags. Those belong to PR 6 through PR 9.
