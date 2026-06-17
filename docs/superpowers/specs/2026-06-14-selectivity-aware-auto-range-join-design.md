# Selectivity-Aware Auto Range Join Design

## Goal

Improve `ExactRangeJoinIndex` auto candidate selection so it detects
pigeonhole posting-list explosion from the actual returned candidate set and
invokes q-gram or hybrid filtering when pigeonhole is insufficiently
selective.

This remains a construction-side optimization. It must not change phase-2
edges, leaf attachments, exact verification, or search results.

## Correctness Contract

Pigeonhole and q-gram candidate sets are both safe supersets of all targets
within edit-distance threshold `tau`. Their intersection is also a safe
superset.

The builder remains responsible for final verification:

```text
candidate generation -> per-pair length check -> compute_distance_bounded
```

Only exact bounded edit distance `<= tau` may add a phase-2 edge or leaf
attachment.

## Configuration

Extend `RangeJoinConfig`:

```cpp
size_t auto_pigeonhole_max_candidates = 4096;
double auto_pigeonhole_max_ratio = 0.25;
bool auto_hybrid_on_large_candidates = true;
```

Validation:

- `auto_pigeonhole_max_candidates` may be zero. This allows ratio-only
  selection.
- `auto_pigeonhole_max_ratio` must be finite and in `[0.0, 1.0]`.
- The boolean CLI value accepts only `true` or `false`.

## Auto Selection Algorithm

Auto selection uses the number of length-compatible target items as the ratio
denominator. This prevents length-incompatible targets from making a
pigeonhole result appear more selective than it is.

For each auto query:

1. Compute the adaptive pigeonhole block and seed lengths.
2. If the seed is shorter than `min_seed_len`, skip pigeonhole and run q-gram.
3. Otherwise run pigeonhole candidate generation.
4. Compute:

```text
pigeonhole_candidate_count = pigeonhole_candidates.size()
compatible_count = total target items passing abs(length difference) <= tau
candidate_ratio =
    compatible_count == 0
        ? 0.0
        : pigeonhole_candidate_count / compatible_count
```

5. Accept pigeonhole if either condition is true:

```text
pigeonhole_candidate_count <= auto_pigeonhole_max_candidates
OR candidate_ratio <= auto_pigeonhole_max_ratio
```

6. If neither condition is true, mark pigeonhole as rejected for large
   candidates and run q-gram.
7. If `auto_hybrid_on_large_candidates` is true, return the sorted intersection
   of pigeonhole and q-gram candidates.
8. Otherwise return q-gram candidates directly.

Q-gram conservative behavior for short sequences or nonpositive required
shared counts is considered available and safe. In the extreme case where
q-gram returns every length-compatible target, hybrid reduces to the
pigeonhole result and q-gram-only fallback returns the compatible full set.

When pigeonhole is unavailable because the seed is too short, auto returns
q-gram directly, preserving the current auto behavior.

## Query Result Instrumentation

Extend `RangeJoinQueryResult` with per-query auto decision data:

```cpp
size_t compatible_item_count = 0;
size_t pigeonhole_candidate_count = 0;
double pigeonhole_candidate_ratio = 0.0;
size_t auto_pigeonhole_accepted = 0;
size_t auto_pigeonhole_rejected_large_candidates = 0;
size_t auto_qgram_invoked = 0;
size_t auto_hybrid_invoked = 0;
size_t auto_final_candidate_pairs = 0;
double auto_candidate_ratio_sum = 0.0;
```

The counter fields are zero or one per range-join query except pair counts and
ratio sum:

- `auto_pigeonhole_accepted`: one when auto accepts an available pigeonhole
  result.
- `auto_pigeonhole_rejected_large_candidates`: one when both selectivity
  thresholds fail.
- `auto_qgram_invoked`: one whenever auto runs q-gram, including short-seed
  fallback.
- `auto_hybrid_invoked`: one when auto returns an intersection after rejecting
  a large pigeonhole result.
- `auto_final_candidate_pairs`: final number of candidate IDs returned by auto.
- `auto_candidate_ratio_sum`: the measured pigeonhole candidate ratio when
  pigeonhole ran; zero when pigeonhole was unavailable.

`mode_used` continues to describe the final candidate form:

- Accepted pigeonhole: `PigeonholeOnly`
- Direct q-gram: `QGramOnly`
- Intersection: `Hybrid`

Forced pigeonhole, q-gram, hybrid, and full modes do not increment auto
counters.

## Builder Statistics

Add phase-2 and leaf variants of:

- `auto_pigeonhole_accepted`
- `auto_pigeonhole_rejected_large_candidates`
- `auto_qgram_invoked`
- `auto_hybrid_invoked`
- `auto_final_candidate_pairs`
- `auto_candidate_ratio_sum`
- `auto_candidate_ratio_avg`

The builder accumulates the per-query result fields. Average ratio is:

```text
auto_candidate_ratio_sum /
    (auto_pigeonhole_accepted + auto_pigeonhole_rejected_large_candidates)
```

If no auto query ran pigeonhole, the average is zero.

Existing candidate, exact-call, edge, attachment, q-gram, and length-pruning
counters retain their meanings.

## CLI

Add global construction flags:

```text
--auto-pigeonhole-max-candidates N
--auto-pigeonhole-max-ratio R
--auto-hybrid-on-large-candidates true|false
```

Defaults:

```text
--auto-pigeonhole-max-candidates 4096
--auto-pigeonhole-max-ratio 0.25
--auto-hybrid-on-large-candidates true
```

These flags affect only `--range-candidate-mode auto`.

Old auto behavior is reproducible for benchmarks with:

```text
--range-candidate-mode auto
--auto-pigeonhole-max-candidates <very-large-value>
--auto-pigeonhole-max-ratio 1.0
```

Because acceptance uses OR semantics, either sufficiently large threshold
causes available pigeonhole results to be accepted.

## Tests

### Range Join Selection Tests

Add a deterministic posting-list explosion dataset where:

- Pigeonhole seed length is valid.
- Pigeonhole returns most length-compatible targets.
- Q-gram excludes a meaningful subset.

Assert:

- Default auto rejects the large pigeonhole result.
- Default auto returns hybrid candidates.
- Auto hybrid candidates equal forced hybrid candidates.
- `auto_hybrid_on_large_candidates=false` returns forced q-gram candidates.
- Large thresholds reproduce old auto and return pigeonhole candidates.
- Boundary cases for candidate count and ratio are accepted because comparisons
  are inclusive.
- Every forced and auto mode exact-verifies to the full-distance true match
  set.

### Construction Equivalence Tests

Build full and selectivity-aware auto construction on a deterministic
overlap-heavy input and assert exact equality of:

- Primary edge sets.
- Finest-world leaf attachment sets.
- Adaptive search result sets.

Assert auto rejection, q-gram invocation, hybrid invocation, and final
candidate counters are nonzero where expected.

### Regression Validation

Run:

```bash
cd navigamer_cpp
make -j
make test_all
cmake -S . -B /home/tmp/navigamer-auto-build -DCMAKE_BUILD_TYPE=Release
cmake --build /home/tmp/navigamer-auto-build -j
```

## Benchmark

Use the first 2,000 bp of `data/human/chr1_subset`, with 250 bp windows at
stride 1 and current default radii. Compare:

- Full construction.
- Forced pigeonhole.
- Forced q-gram.
- Forced hybrid.
- Old auto reproduced with permissive thresholds.
- New default auto.

Record:

- Build wall time.
- Phase-2 exact calls and edges.
- Leaf exact calls and attachments.
- Auto decision counters and average candidate ratio.
- Search-result equality.

Expected behavior: new auto detects the overlap-heavy pigeonhole candidate
explosion and approaches forced hybrid candidate/exact-call counts without
changing accepted edges, attachments, or search results.
