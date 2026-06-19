# Phase1 Exact-Seed Build Optimization Design

## Goal

Reduce NavigaMer build time enough for the full E. coli reference with
250-base windows and stride 1 to complete within 24 hours, without changing
query-time data structures, query semantics, or no-false-negative guarantees.

## Current Bottleneck

Phase1 processes unique sequences online and searches each candidate group for
the best covering world. The current large-fanout path scans q=5 posting lists
to compute a safe q-gram L1 filter. On the measured E. coli prefixes,
`phase1_qgram_touched_candidates` grows approximately quadratically:

- 32 kb: 40.99 million
- 64 kb: 164.26 million
- 128 kb: 660.14 million

Exact edit-distance calls grow approximately linearly, so the bottleneck is the
posting traversal rather than edlib verification. Parallelizing this traversal
alone retains the quadratic work and becomes memory-bandwidth-bound.

## Selected Design

Add a build-only incremental pigeonhole seed index to each large-fanout
`Phase1CoverGroupIndex`.

For a query sequence and cover radius `tau`, partition the query into `tau + 1`
non-overlapping blocks. If a candidate is within edit distance `tau`, at least
one block must occur unchanged as a substring of that candidate. The union of
the corresponding exact-seed posting lists is therefore a recall-safe
candidate superset.

The index will:

1. Build lazily when a candidate group reaches the configured Phase1 fanout.
2. Append new world centers incrementally as Phase1 creates them.
3. Encode A/C/G/T seeds as integer keys and store posting vectors by seed
   length; ambiguous sequences use the existing safe fallback.
4. Use the existing seed-length bounds. If pigeonhole blocks are too short,
   postings are unavailable, or another safety precondition fails, fall back
   to the existing q-gram/scan path.
5. Verify every emitted candidate with bounded exact edit distance.
6. Keep the existing `phase1_better_cover` distance and index tie-breaking.

This changes only an ephemeral construction helper. `BioSequence`,
`WorldNode`, MBBs, leaf links, persisted indexes, and search code remain
unchanged.

## Semantics

For a fixed input order, the optimized builder must produce the same selected
cover at every Phase1 step as the full-scan builder. Candidate generation may
emit extra candidates, but it must never omit a sequence whose edit distance is
within the requested radius. Exact verification and the existing tie-break
then recover the same best cover.

The initial implementation will not change deduplication order, radius
schedules, or online insertion order. Those are separate optimization axes and
would change the constructed graph.

## Parallelism

The online outer Phase1 loop remains serial because each sequence can create a
world that must be visible to the next sequence. CPU parallelism remains
available for sufficiently large exact candidate sets. GPU verification is
deferred until profiling shows a large batchable exact-verification workload;
the current bottleneck is pointer-heavy posting traversal, which is a poor GPU
target.

## Instrumentation

Add Phase1 counters for:

- pigeonhole queries
- seed posting entries visited
- seed-union candidates
- pigeonhole fallbacks
- final candidates sent to exact verification

Add periodic Phase1 progress output for long builds without changing normal
TSV/CSV schemas.

## Validation

Correctness gates:

1. Pigeonhole candidates contain every brute-force match across substitutions,
   insertions, deletions, length differences, and ambiguous-base fallbacks.
2. Phase1 full-scan and optimized builders produce identical primary edges,
   leaf links, cover hit/miss counts, and adaptive query hits.
3. `test_recall` and `test_distance_bound` pass.
4. Persistence round-trip tests pass because the persisted representation is
   unchanged.

Performance gates:

1. Compare old and new Phase1 on E. coli 32, 64, and 128 kb prefixes with the
   same 250-base window, stride, radii, and thread settings.
2. Confirm query benchmark results and index topology are unchanged.
3. Fit conservative scaling from the largest measured prefixes. The projected
   full E. coli build must be below 24 hours.
4. If the first stage does not meet the target, proceed to a separately
   validated locality-incumbent or batched-construction design rather than
   weakening candidate safety.

## Non-Goals

- Changing query algorithms or query-time structures
- Changing the persisted index format
- Relaxing exact edit-distance verification
- Accepting probabilistic seeds, minimizers, or filters that can introduce
  false negatives
- Changing the radius schedule as part of this optimization
