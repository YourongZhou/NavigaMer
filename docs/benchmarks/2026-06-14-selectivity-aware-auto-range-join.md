# Selectivity-Aware Auto Range Join Benchmark

## Setup

- Dataset: first 2,000 bp of `data/human/chr1_subset`
- Indexed sequences: 1,751 windows of length 250, stride 1
- Tolerance: 2
- Primary radii: `30,15,5`
- Q-gram length: 5

Old auto used:

```text
--range-candidate-mode auto
--auto-pigeonhole-max-candidates 18446744073709551615
--auto-pigeonhole-max-ratio 1.0
```

New auto used the requested defaults: maximum 4,096 candidates, maximum ratio
0.25, and hybrid enabled. A strict diagnostic used maximum 0 candidates and
maximum ratio 0.10.

## Results

| Mode | Wall s | Phase2 exact calls | Edges | Leaf exact calls | Attachments | Auto rejected / hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 71.20 | 521,085 | 6,269 | 1,376,286 | 3,928 | 0 / 0 |
| pigeonhole | 22.57 | 189,237 | 6,269 | 320,761 | 3,928 | 0 / 0 |
| qgram | 21.06 | 105,551 | 6,269 | 42,196 | 3,928 | 0 / 0 |
| hybrid | 21.68 | 105,551 | 6,269 | 42,196 | 3,928 | 0 / 0 |
| old auto | 22.71 | 177,878 | 6,269 | 320,761 | 3,928 | 0 / 0 |
| new default auto | 22.46 | 177,878 | 6,269 | 320,761 | 3,928 | 0 / 0 |
| strict auto (`0`, `0.10`) | 21.37 | 105,551 | 6,269 | 42,196 | 3,928 | 1,213 / 1,213 phase2; 1,751 / 1,751 leaf |

All modes produced identical sorted search-result keys:

```text
16fe74766d2476a8344f334b49bcfb1ba33aba6db8743c43e26f065cd4040f28
```

## Interpretation

The selectivity-aware implementation works: strict auto matched forced hybrid
exact-call counts and remained exact.

The requested default new-auto did not differ from old-auto on this 2 kb
benchmark. Each per-query candidate set was smaller than the default absolute
threshold of 4,096, so the specified OR acceptance rule accepted pigeonhole
regardless of ratio. This is expected from the configured policy:

```text
accept if candidate_count <= 4096 OR candidate_ratio <= 0.25
```

To make default auto approach hybrid on datasets with fewer than 4,096
length-compatible targets, either the absolute threshold must be lowered or the
acceptance rule must change from OR to AND. This implementation preserves the
requested defaults and OR semantics.
