# Bounded Myers Distance Design

## Goal

Enable the bounded Myers edit-distance backend by default for query/adaptive search while preserving index construction, pruning semantics, and final result sets. Keep DP available as the explicit reference mode.

## Scope

This PR only affects bounded center-distance calls made during adaptive query search. Index construction keeps using the current bounded DP implementation so edge construction, leaf attachment, and MBB construction remain unchanged.

The existing `compute_distance_bounded(const std::string&, const std::string&, int)` interface remains DP-backed and keeps its current semantics. New call sites that need mode selection use an explicit wrapper.

## Distance Modes

Add:

```cpp
enum class DistanceMode {
  DP,
  Myers,
  Auto,
};
```

`myers` is the adaptive search default. It enables the bounded Myers backend when its preconditions are met and falls back to DP otherwise. `dp` remains the reference mode. `auto` remains conservative and routes to DP.

## Myers Backend

The implementation uses single-word Myers for DNA strings where the shorter input length is at most 64, and multiword Myers up to 256 shorter-input bases. Both inputs must contain only `A`, `C`, `G`, and `T`. The shorter string is used as the bit-vector pattern because edit distance is symmetric.

Fallback to DP is required for:

- negative thresholds, after preserving the current exception behavior;
- shorter input length greater than 256;
- empty input after simple exact handling;
- `N`, lowercase, or any non-ACGT character;
- any unsupported path.

Semantics:

- if true edit distance is `<= tau`, return the true edit distance;
- if true edit distance is `> tau`, return `tau + 1`;
- never return a value below the true edit distance;
- never return `> tau` when true edit distance is `<= tau`.

## Search Integration

`SearchConfig` carries `distance_mode`, defaulting to `DistanceMode::Myers`. Adaptive search uses `compute_distance_bounded_with_mode()` only for child-center checks after MBB/qgram filtering. Full distances, beacon distances, leaf exact verification, exhaustive search, brute-force search, and index construction remain unchanged.

The CLI adds:

```text
--distance-mode dp|myers|auto
```

The query benchmark keeps the baseline profile fixed to `dp` and reports both profile distance modes in JSON.

## Correctness Gates

Add distance-level differential tests:

- random ACGT strings across lengths `1,2,5,20,50,64,65,100,150,250`;
- thresholds `0,1,2,5,10,20,50`;
- known insertion, deletion, and substitution examples;
- strings containing `N` or non-ACGT characters, where Myers mode must match DP fallback behavior.

Add search-level equivalence:

- build one index once;
- run adaptive search with `distance_mode=dp` and `distance_mode=myers`;
- compare result IDs exactly across multiple queries and tolerances.

The final gate is `make test_all`, a `demo` run with `--distance-mode myers`, and a query-benchmark no-FN/equality run with optimized distance mode set to Myers.
