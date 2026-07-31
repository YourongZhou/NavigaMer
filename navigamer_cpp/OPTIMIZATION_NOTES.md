# Array-index optimization notes

This document records why each optimization exists, how it works, and why it
does not introduce false negatives. The measurements below use 4,751
overlapping 250-base windows from a fixed 5,000-base synthetic reference and
512 fixed queries at edit-distance tolerance 2.

## Correctness contract

All candidate-generation filters are conservative. A filter may return extra
candidates, but it may remove a candidate only when a length, pigeonhole, or
q-gram bound proves that the edit-distance threshold cannot be met. Every
remaining candidate is still checked by an exact edit-distance implementation.
Ambiguous DNA symbols and unsupported packed-key lengths fall back to a larger
candidate set or a full scan.

The regression reference is exhaustive or brute-force search. Optimized build
outputs are also compared using node counts, edge counts, candidate counts, and
leaf-attachment counts.

## Implemented optimizations

### Canonical flat-array index

Problem: the former graph used one heap allocation per world plus
`shared_ptr`, nested vectors, string IDs, and duplicated pointer links. This
increased allocator metadata, reference-count traffic, and cache misses.

Method: worlds, child IDs, leaf IDs, beacon IDs, MBB bounds, and leaf-beacon
distances are stored in separate contiguous arrays. A node is an integer ID
whose record contains offsets and counts into those arrays. The search engine
uses this representation directly; it does not reconstruct a pointer graph.

Safety: IDs replace object addresses only. Layer ranges, edge multiplicity,
leaf attachments, radii, MBB values, and exact verification are preserved.

### Packed rolling seed keys

Problem: seed indexes previously allocated a `std::string` for every seed and
re-encoded every overlapping seed in `O(k)` time.

Method: A/C/G/T seeds of at most 32 bases are represented by 2-bit
`uint64_t` keys. After the first seed, the next overlapping seed is produced by
a shift, mask, and one new base, reducing construction from `O(n*k)` to
`O(n)`. Duplicate seed keys for one item are sorted and removed before posting
insertion.

Safety: 2-bit encoding is one-to-one for A/C/G/T at a fixed seed length.
Sequences containing other symbols are recorded as unindexable candidates, and
queries containing other symbols fall back to full scan.

### Packed q-gram postings

Problem: q-gram posting maps used heap-allocated string keys and repeated
substring hashing.

Method: q-grams of at most 32 bases use the same 2-bit `uint64_t` encoding.
Counts are built from rolling codes, sorted, and run-length encoded.

Safety: the q-gram L1 inequality is unchanged. Unsupported q values or
non-ACGT input disable pruning for the affected item/query, which can add work
but cannot remove a true hit.

### Parallel exact leaf attachment

Problem: candidate lookup and exact verification for every finest-layer world
were serial even though each world's leaf set is independent.

Method: OpenMP workers use independent range-query workspaces, statistics, and
per-world result vectors. Results are sorted by sequence ID and merged after
the parallel region, so construction is deterministic.

Safety: workers execute the same conservative candidate lookup, q-gram bound,
and exact Edlib threshold check. Only scheduling changes; no pruning rule
changes.

### Removal of speculative path-reuse distances

Problem: with path reuse enabled, each visited non-leaf world computed an
unbounded exact distance for every child, including children that the current
query would never visit. On the benchmark, this speculative cache population
made search about three times slower.

Method: cache distances only for centers and leaves that normal exact search
actually evaluates. A later near-query reuse attempt that lacks a cached value
already follows the exact fallback path.

Safety: the removed values were not used to decide the current query. Missing
future cache entries disable a triangle-bound shortcut and therefore cause
more exact verification, not false negatives.

## Measured result

On the fixed benchmark and release `-O2` build:

| Measurement | Pointer-era baseline | Current | Change |
| --- | ---: | ---: | ---: |
| Mean adaptive query time | 19.096 ms | about 1.32 ms | about 14.5x faster |
| Build time, 16 threads | about 2.39 s | about 1.07 s | about 2.2x faster |
| Peak build RSS | about 94.9 MiB | about 60.0 MiB | about 37% lower |

The query comparison uses identical query/hit/distance/reference-position
fields. Build comparisons use identical structural counters. Timings should be
read as same-host engineering measurements, not hardware-independent claims.

## Why the 100x memory and 10x build targets are not met

A 100x reduction from 94.9 MiB would require peak RSS below 0.95 MiB. The raw
4,751 by 250-base input alone contains 1,187,750 base bytes (about 1.13 MiB)
before string objects, IDs, occurrences, nodes, edges, allocator state, and
code/runtime pages. The serialized array index is about 2.9 MB, and loading it
for one query uses roughly 8 MiB RSS. Therefore a 100x whole-process RSS
reduction is below the data-size lower bound for this benchmark.

A 10x build target relative to 2.39 seconds is below 0.239 seconds. Phase 1
alone currently takes about 0.40 seconds and is intentionally
insertion-order-dependent: each sequence selects the best current covering
world, and creating or selecting that world changes the candidates available
to later sequences and lower layers. Parallel snapshot construction would
change the tree. Phase 1 performs 58,662 exact threshold-distance calls on this
benchmark; Edlib is substantially faster here than the available DP and Myers
backends.

Further large reductions would require a different contract or a new exact
distance backend, for example a verified SIMD/GPU batch kernel, direct
edit-distance operations over packed sequences, or a build algorithm whose
tree is allowed to differ while retaining recall. Those are algorithm/backend
projects rather than additional array-layout changes, and packed decoding or
accelerator transfer can also violate the requirement that query speed must
not regress.
