# Search-Side Q-Gram Safe Prefilter Design

## Goal

Add an optional no-false-negative q-gram prefilter to adaptive search after
child MBB survivor generation and before exact child-world center distance
verification. The prefilter may reduce center distance calls, but it must not
change adaptive result sets, traversal semantics, construction edges, leaf
attachments, or leaf refinement.

## Scope

The first version applies only to child primary worlds returned by
`get_mbb_surviving_children()`. It does not filter coarsest-layer candidates,
leaf candidates, greedy search, exhaustive search, brute-force search,
`phase2_inter_tier_rebinding()`, or `attach_leaves()`.

For each MBB-surviving child with threshold
`tau = child.radius + query_tolerance`, adaptive search may prune the child only
when:

`qgram_l1(query, child.center) > 2 * q * tau`

Every child that is not pruned still receives the existing exact center
distance verification and the existing containment/overlap handling.

## Q-Gram Signature API

Extend `qgram_filter.hpp` and `qgram_filter.cpp` with a reusable signature API:

```cpp
struct QGramSignature {
  int q = 0;
  size_t sequence_length = 0;
  size_t total_qgrams = 0;
  bool safe_for_pruning = false;
  std::vector<QGramEntry> entries;
};

QGramSignature compute_qgram_signature(const std::string& sequence, int q);
size_t qgram_l1_distance(
    const QGramSignature& lhs, const QGramSignature& rhs);
bool qgram_can_prune_edit_distance(
    const QGramSignature& lhs, const QGramSignature& rhs, int tau);
```

`QGramEntry` stores a compact encoded q-gram and its count. Entries are sorted
by encoded q-gram so L1 distance can be computed by a linear merge without
allocating a hash table during search.

The compact representation supports positive q values whose ACGT encoding fits
in a 64-bit integer. A sequence containing any non-ACGT character, an invalid q,
or an unsupported q produces a signature with `safe_for_pruning == false`.
`qgram_can_prune_edit_distance()` returns false for unsafe signatures,
incompatible q values, negative tau, or arithmetic-overflow risk. Strings
shorter than q produce a safe empty signature; the L1 bound remains valid.

The existing string-key q-gram functions and `QGramCountIndex` remain intact
for construction range joins and ambiguous-character support. The new compact
API is additive and is used by search only.

## Signature Ownership And Lifecycle

`BioGeometrySearchEngine` owns an immutable map from `WorldNode::node_id` to
`QGramSignature`. The engine constructor builds this cache only when
`SearchConfig::search_qgram_prefilter` is true and
`SearchConfig::search_qgram_q > 0`.

The constructor visits every node in every finalized primary layer, ensuring
all child worlds reachable by adaptive traversal are covered. Each unique node
ID is built once. The engine does not build signatures for leaves, auxiliary
nodes, or beacons.

This ownership keeps the search q independent from construction `--qgram-q`,
adds no cache memory when the feature is disabled, avoids changing
`WorldNode` copy/move behavior, and provides read-only state during OpenMP
searches.

Missing or unsafe child signatures always cause no-prune fallback. Query
signature computation happens once per `search_adaptive()` call and is passed
through adaptive traversal helpers as immutable state.

## Search Integration

Add to `SearchConfig`:

```cpp
bool search_qgram_prefilter = false;
int search_qgram_q = 5;
```

The feature is effectively disabled when the flag is false or q is not
positive.

After MBB survivor generation, `search_layer_adaptive()` receives a marker that
the candidates are child-world MBB survivors. For each unvisited survivor it:

1. Increments `center_distance_calls_before_qgram`.
2. Checks the query and child-center signatures when the prefilter is enabled.
3. Safely prunes and continues only when the q-gram necessary condition fails.
4. Otherwise increments `center_distance_calls_after_qgram`.
5. Runs bounded exact edit distance with `tau = node.radius + tolerance`.
6. Applies the existing strict containment fast path and overlap traversal
   logic without semantic changes.

The coarsest primary layer is not counted or q-gram filtered because its
candidates were not produced by child MBB filtering.

Replacing the current full center distance call with
`compute_distance_bounded(query, center, tau)` is valid because traversal only
needs the exact value when it is at most tau. A value above tau is rejected
before containment/overlap logic.

## Instrumentation

Add per-query `SearchStats` fields:

- `search_qgram_prefilter_enabled`
- `search_qgram_q`
- `search_qgram_signature_build_count`
- `search_qgram_signature_missing_count`
- `search_qgram_checks`
- `search_qgram_pruned_children`
- `search_qgram_passed_children`
- `center_distance_calls_before_qgram`
- `center_distance_calls_after_qgram`

`search_qgram_signature_build_count` reports the engine cache size for enabled
searches. Missing and unsafe signatures both count as conservative fallback and
increment `search_qgram_signature_missing_count`.

`qgram_prune_ratio` is derived as pruned children divided by q-gram checks.
`result_count` is derived from the final result vector for CLI and benchmark
output. The invariant
`center_distance_calls_after_qgram <= center_distance_calls_before_qgram`
must always hold.

Keep `center_distance_calls_after_mbb` as a compatibility alias for
`center_distance_calls_before_qgram`; document the relationship and emit the
new explicit columns in benchmark TSV output.

## CLI And Documentation

Add global search flags:

- `--search-qgram-prefilter off|on`, default `off`
- `--search-qgram-q N`, default `5`

The parser accepts only `off` and `on` for the prefilter flag. A non-positive
search q disables the prefilter conservatively instead of affecting
construction q validation.

Update root `README.md`, `navigamer_cpp/README.md`, and
`navigamer_cpp/CLI_REFERENCE.md`. Benchmark TSV output gains the q-gram fields,
prune ratio, and result count so scan/rect and on/off runs can be compared
directly.

## Tests

Extend `test_qgram_filter` with compact signature count checks, randomized
edit-distance-bound checks, short-string behavior, incompatible/invalid
signatures, and ambiguous-character conservative fallback.

Add `test_search_qgram_prefilter` covering:

- adaptive q-gram on/off exact result equality;
- scan/rect crossed with q-gram on/off exact result equality;
- ambiguous `N` fallback with no result change;
- missing-signature fallback with no result change;
- strict containment fast-path result equality;
- counters and the before/after center-call invariant;
- at least one fixture where q-gram pruning reduces exact center calls.

Register the new test in Make and CMake and include it in `make test_all`.
Existing recall, distance-bound, construction q-gram, range-join, and MBB tests
remain required.

## Benchmark

Use `data/human/chr1_subset` to prepare 250 bp stride-1 reference windows and a
small deterministic query set. Run benchmark for:

1. scan + q-gram off
2. scan + q-gram on
3. rect + q-gram off
4. rect + q-gram on

Run against the first 2 kb and, if runtime is reasonable, the first 10 kb.
Report average query time, result equality, MBB survivors, q-gram checks,
q-gram pruned children, before/after center calls, prune ratio, and result
count. Timing improvement is not required; counters must explain cases where
MBB filtering leaves too few candidates for q-gram filtering to help.

## Scaling Boundary

The compact engine-owned cache is intended to support E. coli-scale indexes
and partitioned larger references because it stores signatures only for
primary world centers. It does not make a full human-genome stride-1 250 bp
in-memory index practical by itself; that requires separate work on
partitioning or persistence.
