# Construction Q-Gram Candidate Mode Benchmark

## Setup

- Source: deterministic first 400 bp of `data/human/chr1_subset`
- Indexed sequences: 151 overlapping 250 bp windows at stride 1
- Query: first 250 bp of the same source
- Tolerance: 2
- Primary radii: `30,15,5`
- Q-gram length: 5
- Timing: `/usr/bin/time` wall seconds

The 400 bp prefix keeps full pairwise construction practical in the current
environment. It is intentionally overlap-heavy, which makes pigeonhole posting
lists large and exercises the q-gram fallback use case.

Representative command:

```bash
./navigamer benchmark \
  --ref /home/tmp/navigamer_chr1_prefix_400.fa \
  --reads "$QUERY" \
  --window 250 \
  --stride 1 \
  --tolerance 2 \
  --primary-radii 30,15,5 \
  --range-candidate-mode qgram
```

Full construction used `--link-mode full --leaf-attach-mode full`. The other
indexed runs used `--range-candidate-mode pigeonhole|qgram|hybrid|auto`.

## Results

| Mode | Wall s | Phase2 possible | Phase2 candidates / exact | Phase2 edges | Phase2 fallbacks | Phase2 qgram L1 pruned | Leaf possible | Leaf candidates / exact | Attachments | Leaf fallbacks | Leaf qgram L1 pruned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1.06 | 3,612 | 3,612 / 3,612 | 495 | 0 | 0 | 9,664 | 9,664 / 9,664 | 317 | 0 | 0 |
| pigeonhole | 0.64 | 3,612 | 3,612 / 3,612 | 495 | 37 | 0 | 9,664 | 9,664 / 9,664 | 317 | 0 | 0 |
| qgram | 0.64 | 3,612 | 3,263 / 3,263 | 495 | 0 | 349 | 9,664 | 3,225 / 3,225 | 317 | 0 | 6,439 |
| hybrid | 0.63 | 3,612 | 3,263 / 3,263 | 495 | 37 | 349 | 9,664 | 3,225 / 3,225 | 317 | 0 | 6,439 |
| auto | 0.64 | 3,612 | 3,612 / 3,612 | 495 | 0 | 0 | 9,664 | 9,664 / 9,664 | 317 | 0 | 0 |

Q-gram and hybrid reduced leaf candidate and exact-distance calls by 66.63%.
On this overlap-heavy sample, pigeonhole candidates were the full compatible
set, so hybrid reduced to the q-gram candidate set. Auto selected pigeonhole
whenever its seed length met the configured minimum and therefore did not
reduce leaf candidates on this sample.

## Correctness Check

All modes produced:

- 495 phase-2 edges.
- 317 leaf attachments.
- The same benchmark search hit keys (`query_id`, `hit_id`, `distance`,
  `ref_positions`).

The sorted hit-key files for all five modes had the same SHA-256:

```text
1a55d3b95ac2a2deb529a6fba1fa7716969ec3550972db77d53ecdfe84dca317
```

These measurements demonstrate this dataset only. Candidate reduction and
timing depend on sequence distribution, q, radii, and threshold; exact graph,
attachment, and search-result equivalence remain the required behavior.
