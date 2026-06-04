# Layer / Radius Experiment Summary

## Goal

This experiment is intended to measure how **primary-layer count** and **radius decay schedule** affect **search cost** in the NavigaMer DAG / world-based index, while keeping the **finest-layer radius** fixed.

The key control variable is:

- `r_leaf` must be fixed within each comparison group

This keeps the final search resolution constant, so changes in cost can be attributed to:

- number of primary layers `L`
- radius decay factor `alpha`

rather than to a different finest search radius.

## Experiment Design

### Search configurations

The experiment command sweeps:

- `L ∈ {2, 3, 4, 5}`
- `r_leaf ∈ {4, 8, 12}`
- `alpha ∈ {0.5, 0.7}`

For each combination, the primary-layer radius schedule is generated geometrically:

`radius[layer_idx] = round(r_leaf / alpha^(L - 1 - layer_idx))`

with the final layer forced to:

`radius[L - 1] = r_leaf`

Examples:

- `r_leaf = 8, alpha = 0.5, L = 4` gives `32|16|8|4`
- `r_leaf = 8, alpha = 0.7, L = 4` gives `23|16|11|8`

Auxiliary layers are still generated automatically between adjacent primary layers and are collapsed before query-time navigation.

### Fixed inputs

The experiment reuses the same:

- reference sequence
- query set
- query length
- search tolerance
- random seed

across all `(L, r_leaf, alpha)` combinations in one run.

### Recorded metrics

The current implementation records per-query search-cost statistics only:

- `query_time_ms`
- `world_access_count`
- `node_access_count`
- `edge_access_count`
- `anchor_distance_count`
- `bound_check_count`
- `candidate_count`
- `candidate_verify_count`

Notes on definitions:

- `world_access_count`: number of world / DAG nodes checked during traversal
- `node_access_count`: number of finest-layer stored sequence nodes inspected for the query
- `edge_access_count`: number of parent-to-child world edges examined
- `anchor_distance_count`: number of `d(query, beacon)` computations
- `bound_check_count`: number of pruning checks based on MBB or leaf-beacon bounds
- `candidate_count`: number of candidates reaching exact verification
- `candidate_verify_count`: number of exact leaf verifications

At the moment, `candidate_count` and `candidate_verify_count` are effectively the same stage in the current search path, because the implementation verifies every surviving leaf candidate exactly.

## Important Adjustment: Dense Stride

### Why sparse windows were a problem

With sparse windowing, the finest layer tended to degenerate into:

- one finest node per window

This happened because:

- query/reference windows were too dissimilar from one another
- `r_leaf` values such as `4`, `8`, and `12` were too small to merge many non-overlapping windows

As a result, the experiment would mostly measure:

- traversal overhead

rather than:

- meaningful clustering or pruning at the finest layer

### Fix

The experiment command was extended to support:

- explicit `--stride`

and this should be preferred over `--stride-mode` for the layer/radius study.

Recommended setting for the main study:

- keep `--length 250`
- set `--stride 1`

This creates highly overlapping windows and makes it much more likely that multiple windows map to the same finest-layer node.

## Current Results

### Result 1: sparse / low-overlap windows degenerate quickly

A small smoke run without dense overlap produced:

- `windows = 5`
- effectively one candidate per query path
- `node_access_count = 1`
- `candidate_count = 1`

Example output:

| L | r_leaf | alpha | radius_schedule | world_access | node_access | edge_access | candidate_count |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.5 | `8|4` | 2 | 1 | 1 | 1 |
| 3 | 4 | 0.5 | `16|8|4` | 3 | 1 | 2 | 1 |

Interpretation:

- increasing `L` increased traversal work
- but did not reduce candidate work

This is not a useful regime for comparing the benefit of extra layers.

### Result 2: `stride = 1` restores finest-layer merging

A smoke run with dense overlap:

```bash
./navigamer layer-radius-experiment \
  --ref ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT \
  --length 8 \
  --stride 1 \
  --tolerance 2 \
  --query-edits 2 \
  --queries-per-cell 2 \
  --L-values 2 \
  --r-leaf-values 4 \
  --alpha-values 0.5 \
  --out /tmp/layer_radius_stride1.csv
```

produced:

- `windows = 33`
- `33 -> 4 unique`
- finest-layer compression of `75%`

and per-query rows:

| L | r_leaf | alpha | radius_schedule | world_access | node_access | edge_access | anchor_distance | bound_check | candidate_count |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.5 | `8|4` | 2 | 4 | 1 | 2 | 5 | 3 |

Interpretation:

- dense overlap allows multiple windows to merge into the same finest node
- `node_access_count` and `candidate_count` become non-trivial
- the experiment can now measure a real tradeoff between:
  - extra traversal cost from larger `L`
  - potential reduction in downstream candidate work

## What These Results Mean

### What is already clear

1. The experiment logic is working.
2. The per-query CSV contains the intended cost breakdown.
3. Sparse windows are a poor default for this study.
4. `stride = 1` is a much better setting when the goal is to study the effect of layers and radius schedules on search cost.

### What is not yet established

The current results are smoke-test scale only. They confirm that the experimental setup is valid, but they are not yet enough to support strong claims such as:

- whether increasing `L` consistently reduces candidate work
- whether query time has an optimal `L`
- whether `alpha = 0.5` or `alpha = 0.7` is better at a given `r_leaf`

Those conclusions require a full run on a realistic subset such as:

- `data/human/chr1_subset`

using:

- `--length 250`
- `--stride 1`
- fixed query count and seed

## Recommended Main Run

```bash
cd navigamer_cpp
./navigamer layer-radius-experiment \
  --ref ../data/human/chr1_subset \
  --length 250 \
  --stride 1 \
  --tolerance 2 \
  --query-edits 2 \
  --queries-per-cell 20 \
  --L-values 2,3,4,5 \
  --r-leaf-values 4,8,12 \
  --alpha-values 0.5,0.7 \
  --out /tmp/layer_radius_search_stats.csv
```

This will generate one CSV row per query per parameter combination and is the recommended input for downstream Python aggregation and plotting.

## Summary

The experiment is now implemented and usable.

The main methodological takeaway so far is:

- **use dense overlap (`stride = 1`) for this study**

because otherwise the finest layer collapses into a nearly one-window-one-node regime, which makes the layer/radius comparison much less informative.

The current smoke results validate the implementation and support proceeding to a full sweep on `chr1_subset`.
