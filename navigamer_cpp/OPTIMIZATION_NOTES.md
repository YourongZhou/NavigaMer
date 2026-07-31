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

### Exact base-count lower bounds

Problem: Phase 1 and leaf attachment still called Edlib for candidates whose
base composition alone proved they could not meet the radius.

Method: each sequence stores four A/C/G/T counts. The lower bound is half the
L1 distance between those count vectors, rounded up. One edit can change that
L1 distance by at most two, so a candidate is skipped only when this lower
bound exceeds the radius. Phase 1 also uses the bound to prove that a candidate
cannot improve the current best distance or win its deterministic
lowest-index tie break.

Safety: the bound is a mathematical lower bound on edit distance, not a
heuristic. If either sequence contains a non-ACGT symbol, the bound becomes
zero and disables pruning. The full-scan construction mode remains an
unfiltered exact reference.

### Reuse of Phase 1 hint distances

Problem: Phase 1 first checked a locality hint with exact Edlib, then usually
recomputed the same candidate distance during the best-cover scan.

Method: a successful hint initializes the best-cover result with its already
known exact distance. The scan still considers every candidate that could beat
it or win the deterministic tie break.

Safety: this removes only a duplicate computation. Candidate ordering,
best-distance comparison, and tie behavior are unchanged.

### Compact, build-local q-gram signatures

Problem: leaf attachment retained a generic q-gram signature for every leaf
and every finest-layer world. A generic entry occupies 16 bytes on the tested
64-bit build, even though `q=5` needs only a 10-bit code and the observed
counts fit in 16 bits.

Method: build-time leaf signatures use a 4-byte `{uint16_t code, uint16_t
count}` entry when representable. In world-to-sequence mode, each worker
creates only the current world's signature and releases it after that world is
processed. Generic signatures remain the fallback for larger codes or counts.
The reusable q-gram inverted index also stores 32-bit internal item indexes
and counts, and range-index construction consumes sequence views instead of
making a second copy of every string.

Safety: compact entries contain the same exact code/count pairs. Mixed or
unsupported representations disable the postfilter instead of pruning. Item
views are consumed synchronously while their owning range-index strings are
alive; no view is retained.

### Move range-index inputs

Problem: range-index construction copied the complete vector of
`{item_id, sequence}` records and then copied the sequences again while
building q-gram postings.

Method: production call sites move temporary item vectors into the index.
Q-gram construction reads non-owning views of the index-owned strings and
stores only posting metadata.

Safety: this changes ownership during one-time index construction only. The
stored strings and all query-visible index contents are identical.

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
| Mean adaptive query time | 19.096 ms | about 1.33 ms | about 14.4x faster |
| Build time, 16 threads | about 2.39 s | about 0.85 s | about 2.8x faster |
| Peak build RSS | about 94,500 KB | about 34,900 KB | about 63% lower |

The query comparison uses identical query/hit/distance/reference-position
fields. Build comparisons use identical structural counters. Timings should be
read as same-host engineering measurements, not hardware-independent claims.
An alternating 10-run query A/B comparison between the previous checkpoint and
this one measured 1.3274 versus 1.3302 ms over 5,120 query samples, a 0.2%
difference inside run-to-run noise; the current build changes add no
per-query algorithmic work. At 64 build threads, the same input takes about
0.61 seconds (about 3.9x faster than the pointer-era run) with higher RSS from
worker-local state.

## Why the 100x memory and 10x build targets are not met

A 100x reduction from about 94.5 MB would require peak RSS below 0.95 MB. The raw
4,751 by 250-base input alone contains 1,187,750 base bytes (about 1.13 MiB)
before string objects, IDs, occurrences, nodes, edges, allocator state, and
code/runtime pages. The serialized array index is about 2.9 MB, and loading it
for a query uses multiple megabytes of runtime memory. Therefore a 100x
whole-process RSS reduction is below the data-size lower bound for this
benchmark.

A 10x build target relative to 2.39 seconds is below 0.239 seconds. Phase 1
alone currently takes about 0.25 seconds and is intentionally
insertion-order-dependent: each sequence selects the best current covering
world, and creating or selecting that world changes the candidates available
to later sequences and lower layers. Parallel snapshot construction would
change the tree. Safe lower-bound pruning reduced Phase 1 scan calls from
58,662 to 27,419, while 18,707 already-computed hint distances are now reused.
Including 20,784 hint checks, Phase 1 still needs 48,203 exact
threshold-distance decisions. It already exceeds the whole-build 10x target
before Phase 2, MBB construction, leaf attachment, input storage, and output
materialization.

A specialized DNA Edlib path that removed repeated alphabet discovery and
reused allocation workspaces was also prototyped and rejected: Phase 1 remained
about 0.25 seconds and total build remained in the same 0.82--0.86 second
range. This shows that those allocations are not the limiting cost on this
workload.

Further large reductions would require a different contract or a new exact
distance backend, for example a verified SIMD/GPU batch kernel, direct
edit-distance operations over packed sequences, or a build algorithm whose
tree is allowed to differ while retaining recall. Those are algorithm/backend
projects rather than additional array-layout changes, and packed decoding or
accelerator transfer can also violate the requirement that query speed must
not regress.
