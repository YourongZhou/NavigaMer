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
known exact distance. A failed hint is also remembered so that the scan does
not repeat the same rejected comparison. Once a best cover exists, later
bounded checks use only the largest distance that could improve it, including
the deterministic lowest-index tie break, instead of always using the full
layer radius. The scan still considers every candidate that could beat the
current result.

Safety: these changes remove duplicate work and tighten an exact verifier's
stopping threshold only when a larger distance cannot change the selected
cover. Candidate ordering, best-distance comparison, and tie behavior are
unchanged.

### Cross-layer Phase 1 distance cache

Problem: the same input sequence can be compared with the same world center
again at a lower hierarchy layer or through a later hint. Phase 1 previously
discarded the bounded exact result after each layer.

Method: Phase 1 keeps an epoch-tagged array indexed by center sequence ID for
the current input sequence. An accepted bounded check records the exact
distance and can answer every later threshold. A rejected check at threshold
`t` can answer only later checks whose threshold is at most `t`; a wider
threshold is recomputed. Scans large enough to run in parallel do not access
the mutable cache. The fixed benchmark reuses about 15,000 checks, and the
16,000-base E. coli benchmark reuses about 123,000.

Safety: a bounded exact result at most `t` is the true edit distance. A result
greater than `t` proves rejection only for thresholds at most `t`. Cache
entries are reset logically for every input sequence, so results are never
shared across different queries. Parallel scans retain their former
independent exact checks.

### Prepared DNA Edlib patterns

Problem: the same sequence is compared against many targets in every build
phase. Standard Edlib repeatedly discovers the alphabet, constructs the Myers
`Peq` bit masks, allocates block storage, and encodes the target for every
comparison.

Method: an uppercase A/C/G/T-only prepared API caches the query's `Peq` masks.
The masks are constructed in one pass over the query, target characters are
mapped inside the Myers column loop, and each worker reuses thread-local block
storage. A move-only RAII handle avoids an unnecessary shared-ownership control
allocation. Phase 1 prepares each input sequence once; Phase 2 prepares the
side with fewer consecutive runs in each batch; MBB construction and leaf
attachment prepare their repeated outer-loop sequence once.

Safety: the Myers recurrence and global edit-distance result are unchanged.
Prepared objects are immutable after construction and therefore safe to share
for reads. Invalid input, ambiguous DNA, allocation failure, or any prepared
backend error falls back to the original Edlib implementation. Randomized
tests compare prepared bounded and unbounded results with both dynamic
programming and the original Edlib path.

### Compact seed postings and epoch unions

Problem: range-join seed postings stored 64-bit external IDs and each query
allocated an `unordered_set` to union posting lists. The index also retained a
separate hash table solely to recover item lengths.

Method: postings store the item's position in the contiguous item array:
16-bit indexes for at most 65,535 items and 32-bit indexes otherwise. A
worker-local epoch array deduplicates posting hits without per-query hash-node
allocations. Item IDs and lengths are read from the contiguous item record.
The final candidate vector reserves the already known number of touched items,
avoiding geometric reallocation while applying the length filter.
If an array index cannot be represented, the operation conservatively uses a
full scan.

Safety: internal positions are translated back to the original item ID before
returning candidates. Epoch marking changes only duplicate removal. Capacity
overflow and unindexable DNA expand the candidate set rather than remove an
item.

### Phase 2 base-count lower bound

Problem: range joining uses the maximum parent radius for a layer so that one
index query safely covers every parent. Many resulting parent-child pairs
cannot meet their smaller pair-specific radius, but q-gram signatures cost
more to construct than they save on this workload.

Method: Phase 2 computes four A/C/G/T counts per center and rejects a pair only
when half the count-vector L1 distance, rounded up, exceeds that pair's exact
radius. This skipped 261,886 exact calls on the fixed benchmark. The optional
Phase 2 q-gram postfilter remains available but off by default.

Safety: a single edit changes the base-count L1 distance by at most two, so the
value is a strict edit-distance lower bound. Ambiguous symbols disable the
bound. Every surviving pair is still checked with exact bounded Edlib.

### Compact, build-local q-gram signatures

Problem: leaf attachment retained a generic q-gram signature for every leaf
and every finest-layer world. A generic entry occupies 16 bytes on the tested
64-bit build, even though `q=5` needs only a 10-bit code and the observed
counts need only 6 bits.

Method: build-time leaf signatures pack code and count into one 2-byte entry
when `q <= 7` and both fields fit. In world-to-sequence mode, each worker
creates only the current world's signature and releases it after that world is
processed. Generic signatures remain the fallback for larger codes or counts.
The reusable q-gram inverted index stores 32-bit internal item indexes and
counts, and range-index construction consumes sequence views instead of making
a second copy of every string.

Safety: compact entries contain the same exact code/count pairs. Mixed or
unsupported representations disable the postfilter instead of pruning. Item
views are consumed synchronously while their owning range-index strings are
alive; no view is retained.

### Direct leaf verification by default

Problem: after prepared Edlib made exact checks much cheaper, constructing
leaf q-gram signatures cost more than the exact calls they removed.

Method: indexed leaf attachment now directly verifies range candidates by
default. `--leaf-qgram-postfilter on` is retained for unusually broad or
repetitive candidate regimes. On the fixed workload, disabling it reduced
Phase 4 from roughly 130 ms to roughly 74 ms.

Safety: removing a prefilter can only send more candidates to the exact
verifier. It cannot introduce a false negative, and accepted links remain
identical.

### Move range-index inputs

Problem: range-index construction copied the complete vector of
`{item_id, sequence}` records and then copied the sequences again while
building q-gram postings.

Method: production call sites move temporary item vectors into the index.
Q-gram construction reads non-owning views of the index-owned strings and
stores only posting metadata.

Safety: this changes ownership during one-time index construction only. The
stored strings and all query-visible index contents are identical.

### Lazy Phase 2 q-gram materialization

Problem: every Phase 2 layer eagerly built both seed postings and a q-gram
index, although many child queries are resolved by the pigeonhole index and
never inspect q-grams.

Method: the q-gram index is materialized on the first query that needs it.
Construction is protected by the existing per-index mutex; after construction
the immutable index is shared by worker queries. Seed-posting hash maps reserve
one bucket per indexed item before insertion to avoid repeated small rehashes
on the overlapping-window workload.

Safety: only construction time changes. The deferred index is built from the
same index-owned strings with the same q-gram code and count representation.
An eager/deferred A/B build produced byte-identical 2.9 MB serialized indexes
with SHA-256
`ddd4ea35b53b7f12887ab21b2c7638d097df3a50cebcad4719edf9d35cd3bc0f`.

### Incremental postings for consecutive leaf windows

Problem: leaf-range construction indexed every overlapping reference window
independently. For a 250-base window and a 20-base seed, shifting the window by
one base preserves 230 of its 231 positional seeds, but the former path
regenerated and sorted all 231 seeds again.

Method: the leaf sequence index explicitly enables a shifted-window posting
builder. It uses the fast path only after proving that the previous sequence
without its first base exactly equals the current sequence without its last
base. A ring buffer removes the outgoing seed and appends the incoming seed,
while a small count map preserves per-item seed deduplication for repeats.
Runs shorter than eight shifts retain the sort path because initializing the
rolling state costs more than it saves. Phase 2 does not enable this option:
its sparse world centers made the same optimization slower.

Safety: the strict overlap equality proves that every retained positional seed
is unchanged. Counts ensure that a seed is emitted exactly once per item,
including homopolymers and other repeated seeds. Non-ACGT input, short
sequences, a broken overlap, and short runs use the original construction.
The fixed benchmark's serialized index remained byte-identical, while leaf
index construction fell from about 49--50 ms to about 18 ms. On the
16,000-base E. coli case it fell from about 170 ms to about 68--74 ms.

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
| Mean adaptive query time | 19.096 ms | 1.332 ms | 14.3x faster |
| Build time, 16 threads | about 2.39 s | about 0.388 s | about 6.2x faster |
| Build time, 64 threads | about 2.39 s | about 0.292 s | about 8.2x faster |
| Peak build RSS, 16 threads | about 94,500 KB | about 31,200 KB | about 67% lower |

The query comparison uses identical query/hit/distance/reference-position
fields. Build comparisons use identical structural counters. Timings should be
read as same-host engineering measurements, not hardware-independent claims.
An alternating 10-run query A/B comparison between the pointer-free checkpoint
and the exact-build optimizations measured 1.3311 versus 1.3323 ms over 5,120
query samples, a 0.09% difference inside run-to-run noise; wall time was
identical. The new build-only cache and lazy indexes are destroyed before
search and do not change the persisted query representation. Query semantic
fields were byte-identical to both the previous checkpoint and the fixed
reference. At 64 build threads, peak RSS remains about 34 MB because more
worker-local state is live concurrently.

## Why the 100x memory and 10x build targets are not met

A 100x reduction from about 94.5 MB would require peak RSS below 0.95 MB. The raw
4,751 by 250-base input alone contains 1,187,750 base bytes (about 1.13 MiB)
before string objects, IDs, occurrences, nodes, edges, allocator state, and
code/runtime pages. The serialized array index is about 2.9 MB, and loading it
for a query uses multiple megabytes of runtime memory. Therefore a 100x
whole-process RSS reduction is below the data-size lower bound for this
benchmark.

A 10x build target relative to 2.39 seconds is below 0.239 seconds. Phase 1
currently takes about 0.105 seconds and is intentionally
insertion-order-dependent: each sequence selects the best current covering
world, and creating or selecting that world changes the candidates available
to later sequences and lower layers. Parallel snapshot construction would
change the tree. At 64 threads, Phase 1 and Phase 2 together take about
0.23 seconds, leaving only about 9 ms of the 10x budget for Phase 0, MBB
construction, leaf attachment, ID assignment, and graph materialization;
those remaining exact stages currently take about 55--60 ms. Reaching 10x while
preserving byte-identical topology therefore requires a materially faster
exact-distance backend, not another container substitution.

An allocation-only Edlib workspace prototype was rejected because it did not
improve wall time. The successful prepared path is different: it also caches
the query-dependent `Peq` masks and fuses target encoding into the Myers loop.
Other rejected A/B experiments include non-owning range-item storage (about
0.2 MB saved with added lifetime risk and no speed gain), direct construction
of compact leaf q-grams (no stable gain), enabling the Phase 2 q-gram
postfilter (about 0.625 s instead of 0.473 s), and speculative query path
distances (roughly three times slower). A prepared scalar Myers backend was
more than twice as slow as prepared Edlib. A flat sorted `(seed,item)` array
increased the fixed 64-thread build from about 0.32 seconds to about 0.42
seconds; at 16 threads it used about 34 MB instead of roughly 28--29 MB because
its 8-byte entry duplicated a seed for every posting. A safe 64-bin q-gram
lower bound pruned only about 1% of Phase 2 pairs and slowed the 16,000-base
build. Lower Phase 1 index thresholds, OpenMP schedule/affinity changes, and a
CUDA Phase 2 prototype were also slower.

Further large reductions would require a different contract or a new exact
distance backend, for example a verified SIMD/GPU batch kernel, direct
edit-distance operations over packed sequences, or a build algorithm whose
tree is allowed to differ while retaining recall. Those are algorithm/backend
projects rather than additional array-layout changes, and packed decoding or
accelerator transfer can also violate the requirement that query speed must
not regress.
