# Construction Q-Gram Safe Filter Design

## Goal

Reduce exact edit-distance calls during indexed `phase2_inter_tier_rebinding()`
and `attach_leaves()` by adding an exact q-gram count filter and a hybrid
pigeonhole/q-gram candidate mode.

This change is construction-side only. It must not change the materialized
hierarchy, leaf attachments, or search results relative to full construction.

## Correctness Contract

Every candidate generator returns a superset of all target items whose
Levenshtein edit distance from the query is at most `tau`.

The implementation relies on two necessary conditions:

1. `abs(len(query) - len(target)) <= tau`
2. If `edit_distance(query, target) <= tau`, then
   `qgram_l1(query, target) <= 2 * q * tau`

The q-gram L1 distance is:

```text
sum_g abs(count_query(g) - count_target(g))
```

No candidate generator may accept an edge or leaf attachment. Builder code
must continue to call `compute_distance_bounded(a, b, tau)` for every returned
candidate, and only an exact result `<= tau` may add an edge or attachment.

The intersection used by hybrid mode is safe because both input candidate sets
are supersets of all true matches.

## Scope

The change includes:

- A standalone string-keyed q-gram count index.
- Candidate modes `auto`, `pigeonhole`, `qgram`, `hybrid`, and `full`.
- Integration into the existing `ExactRangeJoinIndex`.
- Construction statistics for phase 2 and leaf attachment.
- CLI flags, build-system updates, tests, documentation, and a small benchmark
  report.

The change does not modify:

- `BioGeometrySearchEngine::search_adaptive()`.
- Strict containment behavior.
- Phase-2 edge conditions.
- Leaf attachment conditions.
- Search-time MBB rectangle filtering.
- Any persisted index format.

## Q-Gram Count Index

Add:

- `navigamer_cpp/include/qgram_filter.hpp`
- `navigamer_cpp/src/qgram_filter.cpp`

`QGramCountIndex` owns target metadata and an inverted posting index. The first
version uses `std::string` q-gram keys so all input characters, including `N`,
are handled deterministically without encoding-specific ambiguity.

### Public Interface

```cpp
class QGramCountIndex {
 public:
  struct Item {
    size_t item_id = 0;
    std::string sequence;
  };

  struct QueryStats {
    size_t total_items = 0;
    size_t length_filtered_items = 0;
    size_t qgram_candidates = 0;
    size_t full_scan_fallbacks = 0;
    size_t required_shared_nonpositive = 0;
    size_t pruned_by_l1 = 0;
  };

  explicit QGramCountIndex(int q = 5);
  void build(const std::vector<Item>& items);
  std::vector<size_t> query(
      const std::string& query_sequence, int tau,
      QueryStats* stats = nullptr) const;
  int q() const;
  size_t size() const;
};
```

Tests may use public helper functions to compute q-gram counts, total q-grams,
and q-gram L1 distance directly.

### Stored Data

Each target stores:

- External `item_id`.
- Sequence length.
- `total_qgrams = max(0, length - q + 1)`.
- Dense internal index.

The postings map stores:

```text
qgram -> [(internal_index, count_in_target), ...]
```

Dense internal indices allow query-time shared counts to use a vector instead
of an unordered map.

### Query Algorithm

For each query:

1. Reject negative `tau`.
2. Apply the safe length filter to every target.
3. If `q <= 0` or the query/target is shorter than `q`, conservatively include
   the affected length-compatible target.
4. Count query q-grams.
5. For every query q-gram posting, accumulate
   `shared[target] += min(query_count, target_count)`.
6. For each remaining length-compatible target, compute:

```text
required_shared =
    ceil((query_total_qgrams + target_total_qgrams - 2*q*tau) / 2)
```

7. If `required_shared <= 0`, include the target and increment the
   corresponding counter.
8. Otherwise include the target only when
   `shared[target] >= required_shared`.

The returned external IDs are sorted and unique.

## Exact Range Join Integration

Extend `RangeJoinConfig` with:

```cpp
enum class RangeCandidateMode {
  Auto,
  PigeonholeOnly,
  QGramOnly,
  Hybrid,
  FullScan,
};

int min_seed_len = 8;
int max_seed_len = 20;
int qgram_q = 5;
RangeCandidateMode candidate_mode = RangeCandidateMode::Auto;
```

Add parse/name helpers for CLI and summary output.

`ExactRangeJoinIndex::build()` stores the existing items and builds the q-gram
index. Pigeonhole posting lists remain lazy by seed length.

### Mode Semantics

- `full`: return every length-compatible item.
- `pigeonhole`: use pigeonhole when the adaptive seed length is valid;
  otherwise return every length-compatible item and record a full-scan
  fallback.
- `qgram`: use the q-gram filter. Conservative q-gram cases remain q-gram
  queries and return all affected length-compatible items.
- `hybrid`: intersect the pigeonhole and q-gram candidate sets. If pigeonhole
  cannot use a valid seed, its safe candidate set is the length-compatible
  full set, so the intersection reduces to the q-gram result.
- `auto`: use pigeonhole when its adaptive seed length is at least
  `min_seed_len`; otherwise use q-gram.

All modes return sorted, unique IDs.

`RangeJoinQueryResult` records:

- Candidate IDs.
- Actual mode used.
- Block and seed lengths.
- Full-scan fallback count.
- Length-pruned count.
- Q-gram candidate count.
- Q-gram L1-pruned count.
- Required-shared-nonpositive count.

For `hybrid`, `candidate_item_ids.size()` is the final intersection size, while
`qgram_candidate_count` records the size of the q-gram side before
intersection.

## Builder Integration

The builder continues to own exact verification. No edge or leaf condition is
changed.

For indexed phase 2:

```text
candidate_parents = parent_index.query(child.center, max_parent_radius + child.radius)
for parent in candidate_parents:
    tau = parent.radius + child.radius
    skip if length difference > tau
    add edge only if compute_distance_bounded(parent.center, child.center, tau) <= tau
```

For indexed leaf attachment:

```text
candidate_worlds = world_index.query(sequence, max_finest_radius)
for world in candidate_worlds:
    skip if length difference > world.radius
    attach only if compute_distance_bounded(sequence, world.center, world.radius) <= world.radius
```

The second, per-pair length check is required because candidate generation may
use a layer-wide maximum threshold that is larger than an individual node's
threshold. Pairs rejected there do not count as exact distance calls.

## Statistics

Extend builder statistics separately for phase 2 and leaf attachment:

- Total possible pairs.
- Final candidate pairs returned by range join.
- Exact distance calls after per-pair length filtering.
- Accepted edges or attachments.
- Full-scan fallback count.
- Pigeonhole query count.
- Q-gram query count.
- Hybrid query count.
- Q-gram candidate pairs before hybrid intersection.
- Q-gram pairs pruned by L1.
- Length-pruned pairs, including generator-level and per-pair pruning.
- Required-shared-nonpositive count.
- Candidate reduction ratio.
- Exact distance reduction ratio.

The existing statistic names remain where practical. New fields make the
candidate, verification, and accepted-result stages distinct.

Builder summary output prints the configured candidate mode and q value, then
the expanded phase-2 and leaf counters.

## CLI

Add global construction flags:

```text
--range-candidate-mode auto|pigeonhole|qgram|hybrid|full
--qgram-q N
```

Defaults:

```text
--range-candidate-mode auto
--qgram-q 5
```

These flags affect only indexed construction selected by `--link-mode indexed`
and/or `--leaf-attach-mode indexed`. Full construction ignores candidate
generation mode and continues exhaustive exact comparison.

Invalid candidate modes and non-positive q values produce a clear CLI error.

## Tests

### Q-Gram Unit Tests

Add a dedicated q-gram test binary covering:

- Basic multiset counts for `q=2`.
- Randomized verification of the L1 necessary condition for `q=3,4,5`.
- Randomized no-false-negative index queries.
- Short sequences and thresholds where q-gram cannot prune.
- Literal and deterministic handling of `N`.

### Range Join Tests

Extend range-join tests to force every candidate mode:

- For random substitutions and indels, every true match is a candidate.
- Exact verification of candidates equals full-scan matches.
- Hybrid candidates equal the intersection of standalone pigeonhole and q-gram
  candidates for the same query.
- Auto chooses pigeonhole for valid long seeds and q-gram otherwise.
- Returned IDs remain sorted and unique.

### Construction Equivalence Tests

Extend full/indexed construction equivalence tests:

- Compare exact primary edge sets.
- Compare exact finest-world leaf attachment sets.
- Compare adaptive search result sets.
- Run forced q-gram, forced hybrid, and auto indexed construction.
- Include ambiguous bases.
- Assert counters distinguish candidates, exact calls, and accepted pairs.

### Regression Validation

Run:

```bash
cd navigamer_cpp
make -j
make test_all
./navigamer demo --size 200 --range-candidate-mode qgram
./navigamer demo --size 200 --range-candidate-mode hybrid
./navigamer demo --size 200 --range-candidate-mode auto
```

The existing recall, distance-bound, construction-range, and MBB rectangle
tests must remain green.

## Benchmark

Use `data/human/chr1_subset` if available and compare:

- Full construction.
- Indexed pigeonhole.
- Indexed q-gram.
- Indexed hybrid.
- Indexed auto.

Use 250 bp windows, stride 1, tolerance 2, and the current default radii unless
resource limits require a smaller deterministic subset. Record:

- Build wall time.
- Phase-2 possible pairs, candidates, exact calls, and edges.
- Leaf possible pairs, candidates, exact calls, and attachments.
- Fallback and per-mode query counts.
- Q-gram L1 pruning.
- Equality of graph, leaf attachments, and search results on the validation
  dataset.

The benchmark report must state that performance varies with threshold and
sequence distribution; correctness equivalence is the acceptance condition.

## Expected Result

`auto` preserves current pigeonhole behavior when seeds are selective and
replaces weak-seed full-scan fallback with q-gram filtering. `hybrid` may reduce
candidates further by intersecting two safe supersets. In cases where q-gram
cannot prune, counters expose that fact and the implementation remains exact.
