# NavigaMer — C++ reference implementation

This directory contains the **C++17 v8** reference indexer and CLI (`navigamer`) used for the paper implementation. The build pipeline follows a **top-down extended hierarchy**, **inter-tier DAG wiring**, **auxiliary-tier collapse with beacon sequences and MBBs**, and **leaf attachment** to the finest primary layer. Construction stores nodes in a `BuildWorldNodeRecord` array and all relationships as integer IDs, without allocating a `WorldNode` pointer graph. Radius/tier metadata is implied by the layer arrays; only primary nodes allocate geometry records, and child-MBB/leaf distances share one mutually exclusive vector. The finalized index uses `WorldNodeRecord` arrays plus flat child, leaf, beacon, MBB, and leaf-beacon arrays. Generic inputs use a `BioSequence` array; reference-backed inputs instead use one shared reference and a block-compressed monotone representative-position map. Adaptive search returns 32-bit `LeafId` values.

## Build

Requires **g++** (or Clang) with **OpenMP**.

```bash
make -j
# or
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Output: `./navigamer` (Makefile) or `build/navigamer` (CMake).

## CLI (summary)

```bash
./navigamer demo   [--size N] [--primary-radii 30,15,5 | --r-sw 5 --r-mw 15 --r-lw 30]
./navigamer build  --ref <fasta|sequence> --reads <fastq|sequence> [--index index.navidx] [same primary-layer flags]
./navigamer build-scale --ref <fasta|sequence> --window 250 --stride 1 --prefix-lengths 50000 --out build_scale.csv [--index index.navidx] [same primary-layer flags]
./navigamer build-sharded --ref <fasta|sequence> --window 250 --stride 1 --shard-windows N --shard-build-jobs N --index index.navshard [same primary-layer flags]
./navigamer query  --reads <fastq|sequence> --query <sequence> [--index index.navidx] [--tolerance 2] [--mode adaptive|greedy|exhaustive]
./navigamer query-index --index <index.navidx|index.navshard> --query <sequence> [--tolerance 2] [--mode adaptive|greedy|exhaustive]
./navigamer query-index-batch --index <index.navidx|index.navshard> --reads <fastq> [--tolerance 2] [--out out.tsv] [--path-trace-out trace.tsv]
./navigamer run    --ref <fasta|sequence> --reads <fastq|sequence> [--tolerance 2] [--out out.tsv]
./navigamer map150 --ref <fasta|sequence> --reads <fastq|sequence> --tolerance <N> --out out.tsv [--locator refpos|seqan]
./navigamer benchmark --ref <fasta> --reads <fastq> [--tolerance 2] [--window 200] [--stride 1] [--out out.tsv]
./navigamer query-benchmark --ref <fasta|sequence> --out detail.tsv --summary-out summary.tsv --json-out summary.json [--window 200] [--query-length 200] [--query-benchmark-ablations 0|1] [--proximal-oracle 0|1] [--proximal-oracle-k 1,2,4]
./navigamer candidate-verify --ref <fasta|sequence> --reads <fastq> --candidates candidates.tsv --out detail.tsv --summary-out summary.tsv [--window 150] [--stride 1] [--tolerance 5] [--truth source|exhaustive]
./navigamer locality-benchmark --index index.navidx --ref <fasta|sequence> --out summary.tsv [--scenarios low-fanout,high-fanout,repeat,batch-locality,oracle,all]
./navigamer query-locality-benchmark --index index.navidx --ref <fasta|sequence> --out summary.tsv [same flags as locality-benchmark]
./navigamer query-locality-report --ref <fasta|sequence> --out-dir report_dir [--index index.navidx] [--scenarios all]
./navigamer boundary --ref <fasta> [--length 250] [--error-rates csv] [--tolerance-rates csv] [--queries-per-cell 200] [--stride-mode sparse|dense] [--seed 42] [--out out.tsv]
./navigamer layer-radius-experiment --ref <fasta> [--length 250] [--tolerance 2] [--query-edits 2] [--queries-per-cell 200] [--stride 1 | --stride-mode sparse|dense] [--seed 42] [--L-values 2,3,4,5] [--r-leaf-values 4,8,12] [--alpha-values 0.5,0.7] [--out out.csv]
```

**Full syntax and defaults:** [`CLI_REFERENCE.md`](CLI_REFERENCE.md).

All adaptive-search commands accept `--mbb-filter-mode scan|rect` and
`--min-rect-index-fanout N` (default `64`). Both modes apply exact
all-dimension filtering over the flat MBB arrays; `rect` retains its fanout
threshold and reporting counters and falls back for small fanout.
They also accept `--visited-mode string|epoch` (default `epoch`),
`--graph-view original|flat` (default `flat`),
`--simd-mode auto|scalar|avx2|avx512` (default `auto`),
`--distance-mode dp|myers|edlib|auto` (default `myers`),
`--build-distance-mode dp|edlib|auto` (default `edlib`),
`--search-prefetch off|on` (default `off`),
`--search-qgram-prefilter off|on` (default `off`), and `--search-qgram-q N`
(default `5`). `string` keeps the legacy per-query string visited set for
regression comparisons; `epoch` uses integer node IDs and a reused epoch array.
`original` remains accepted as a compatibility label, but both values traverse
the canonical array index. SIMD mode applies to flat child-MBB and leaf-beacon
filtering; unsupported modes
conservatively fall back to scalar and keep the same survivor set. The search q is independent from
construction `--qgram-q`. Enabled search-side filtering runs only on
MBB-surviving child-world centers before bounded exact center verification;
unsafe or missing signatures fall back to no pruning. Distance mode affects
only adaptive bounded child-center checks after MBB/q-gram filtering. `myers`
is the default mode and uses the optional Myers backend through 256bp ACGT
shorter-input length, falling back to DP otherwise. `edlib` uses the vendored
Edlib bounded distance backend. `dp` remains the reference mode; `auto` is
conservative and currently uses DP. Build distance mode is separate and affects
only index construction exact/bounded distance calls; its default is `edlib`.
Adaptive profiling additionally accepts `--query-profile 0|1` (default `0`) and
records per-query timing/counter buckets in `SearchStats`, `benchmark`, and
`query-benchmark` output without changing search results.

Adaptive path reuse additionally accepts `--path-reuse 0|1` (default `1`).
When enabled, adaptive search keeps thread-local warm-start caches for exact
parent-local anchor-distance vectors on repeated queries and cached child
shortlists keyed by cheap query-derived fingerprints. This remains an
ordering/cache hint only: it never becomes the sole pruning reason, preserves
exact verification, and records `path_reuse_attempt_count`,
`path_reuse_hit_count`, `anchor_cache_hit_count`, and
`child_shortlist_reuse_hit_count`, plus locality-summary counters such as
	`child_shortlist_cache_hit_count`, `safe_child_candidate_cache_hit_count`, and
	`productive_world_reuse_hit_count`. Near-query reuse appends triangle-bound
	center, leaf, and direct-verify counters plus `center_distance_reduction`,
	`world_access_reduction`, and `p95_speedup` to locality/query-benchmark TSVs.
	Leaf verification uses bounded exact edit distance; when path reuse is enabled,
	it caches leaf distances up to the configured near-query neighbor bound so the
	next nearby query can safely prune leaves by triangle inequality without
	rebuilding the index.
	Batch-oriented commands group queries by the same query-derived fingerprint
while keeping emitted output rows in original query order.

Adaptive router hints additionally accept `--router-hints 0|1`,
`--router-hint-qgram-q N`, `--router-hint-minimizer-k N`, and
`--router-hint-minimizer-w N`. The current implementation builds parent-local
child-center range hints from q-gram/pigeonhole candidate supersets and uses
q-gram plus minimizer scores only to reprioritize post-MBB child traversal.
This remains a `RouterHint` only: it never becomes the sole pruning reason and
always preserves full fallback enumeration and exact verification.

Adaptive local routing additionally accepts `--local-router 0|1`,
`--local-router-max-anchors N`, `--local-router-max-children N`, and
`--local-router-score anchor-envelope`. The current router uses parent-local
beacon-envelope scoring only to reorder post-MBB child traversal; it never
becomes the sole pruning reason and always preserves full fallback enumeration
and exact verification.

Adaptive safe child routing additionally accepts `--safe-child-router 0|1`,
`--safe-child-router-min-fanout N`,
`--safe-child-router-max-candidates N`,
`--safe-child-router-max-ratio R`,
`--safe-child-router-min-seed-len N`,
`--safe-child-router-mode auto|pigeonhole|qgram|mbb|full-fallback`, and
`--safe-child-router-validate 0|1`. This is a `SafeCandidateRouter`: it may
reduce child enumeration only when a parent-local radius-bucketed child-center
range query or parent-local MBB interval query returns a safe candidate
superset. Radius buckets use `tolerance + child.radius`; MBB mode rejects a
child only when the query-to-anchor distance is outside that child's stored
MBB interval by more than `tolerance`. Candidate sets that are too broad or
cannot be proven safe fall back to full enumeration, and every survivor still
goes through bounded center verification and final exact leaf verification.

Adaptive query planning additionally accepts `--query-planner 0|1`,
`--planner-router-min-fanout N`, and
`--planner-safe-child-router-min-fanout N`. The current planner is
conservative: it records a per-query strategy and can skip optional
q-gram/router ordering work on low-fanout indexes, but it never skips MBB
filtering, bounded center verification, or final exact leaf verification.
TSV/JSON outputs include planner strategy counters and `planner_decision_ms`.

`query-benchmark` can enable proximal-anchor oracle diagnostics with
`--proximal-oracle 1` and `--proximal-oracle-k 1,2,4`. The extra output compares
actual anchor sources, traversed frontier anchors, true-path anchors, global
nearest anchors, and deterministic random anchors using exact edit-distance
envelopes. It records diagnostics only and does not change search results.

Adaptive safe best-first ordering additionally accepts `--best-first 0|1`
(default `0`). The current implementation uses conservative parent-local MBB
lower bounds and tighter envelope spans to reprioritize post-MBB child worlds
before bounded center verification. It records queue/bound counters in
`SearchStats` and may prune only when that lower bound itself is a conservative
`SafeBound`.

Query-side optimization safety contract:

- `RouterHint` may affect only ordering, warm-starts, or candidate priority.
- `SafeCandidateRouter` may reduce enumeration only with a safe candidate
  superset and must full-fallback otherwise.
- `SafeBound` may prune only when it is conservative and no-false-negative.
- `ExactVerifier` remains the final authority for returned hits.

Build commands also expose Phase1 helper thresholds for tuning the extended
sketch step: `--phase1-metric-min-fanout N` (default `12`),
`--phase1-qgram-min-fanout N` (default `12`), and
`--phase1-qgram-max-touched N` (default `250000`). These switch parent-local
candidate groups from direct scan to the metric helper and then to the q-gram
helper; every surviving candidate is still bounded-exact verified before a
world link is accepted.

Long builds emit timestamped phase progress to stderr every 600 seconds by
default. `--progress-interval-seconds N` changes the interval; zero disables
periodic heartbeats but retains phase start/finish reports. `build-scale` can
persist a reference-window index with `--index <file>` when exactly one prefix
length is requested. Multiple prefixes with one output index are rejected.
`build-sharded` instead partitions window starts into independently decodable
logical graph payloads, each capped by `--shard-windows`, and packs up to 1,024
payloads into one `.navpack` container. Reference slices overlap only
where needed to materialize boundary windows; every window start belongs to
exactly one part and output coordinates remain relative to the original
contig. File-backed references use sparse byte-offset checkpoints matched to
the shard source span (bounded to 4 KiB--1 MiB), so each small shard is decoded
without repeatedly rescanning up to a full MiB of FASTA text. Valid completed
packs are reused on restart, damaged packs are rebuilt
as one atomic group, and payloads are written directly into that group's
temporary pack without per-shard temporary files. The automatic policy keeps one part
below 8 OpenMP threads, builds two parts at 8--15 threads, and for the default
small parts build up to 20 parts at the recommended at-most-8,192-window size,
or up to 16 parts through 16,384 windows, with at least four threads each;
larger parts stay capped at four concurrent builders. Use
`--shard-build-jobs N` to cap concurrent builders and their
aggregate memory; nested teams remain inside the original OpenMP thread
budget. Parallel builders suppress per-part progress/summary output, avoiding
large contended stderr streams at human-genome shard counts while retaining the
outer completed-index summary.
For a human stride-1 build, `--shard-windows 5000` is the recommended
starting point when construction time and peak memory matter. Packing keeps
the resulting logical-shard count manageable, while smaller bounded parts
limit Phase-2 join growth, packed-offset widths, mapped query working sets,
and build memory. Use `10000` when fewer logical shards are more important
than build cost, and benchmark adjacent sizes on the target reference before
committing to a full build.

Indexed leaf attachment directly verifies range candidates by default. Use
`--leaf-qgram-postfilter on` to apply a safe q-gram L1 necessary condition
before bounded exact verification. This can reduce exact calls for unusually
broad candidate sets, but its signature-building overhead is slower on the
default prepared-DNA distance path. Either setting is no-false-negative:
accepted links are still determined by bounded exact edit distance.

`query-benchmark` fixes the baseline profile to MBB scan, legacy string
visited mode, the `original` compatibility label (which uses the same canonical
array traversal), `dp` distance mode, and search q-gram disabled. It compares
that with the profile selected by `--mbb-filter-mode`,
`--visited-mode`, `--graph-view`, `--simd-mode`, `--distance-mode`,
`--search-qgram-prefilter`, `--search-qgram-q`, `--router-hints`,
`--local-router`, `--best-first`, `--safe-child-router`, `--path-reuse`, and
`--query-planner`.
With
`--query-benchmark-ablations 1`, it also derives one ablation profile per
enabled query-side optimization stage by disabling only that stage within the
optimized stack.
It deterministically generates random-region, ordinary-region,
low-complexity-region, no-hit, single-hit, and multi-hit queries. Step 0 runs
queries serially even though `--threads` is recorded and applied to OpenMP.
One best-effort eviction-buffer cold sample and configured warm samples are
reported per query/profile. The summary TSV and JSON add baseline-relative
speedup/work-ratio columns so optimized and ablation profiles can be compared
without post-processing. With `--proximal-oracle 1`, detail rows add
actual/frontier/true-path/global/random envelope fields for k1/k2/k4,
nearest-anchor distances, and global-oracle gap fields; summary rows add mean
envelopes and fractions where the global oracle is materially better than the
observed actual/frontier anchors. Any repeated-result, cross-profile, or
brute-force no-FN mismatch makes the command return `2`.

`candidate-verify` is the exact verifier used for external seed baselines such
as randstrobe, strobemer, or spaced-seed candidate TSVs. It reads query FASTQ
records, candidate window IDs, and the same reference/window geometry, then
exactly verifies each candidate with bounded edit distance before reporting final
matches. `--truth source` uses `source_pos=` FASTQ annotations as the expected
origin window, while `--truth exhaustive` scans all reference windows for a
small-run no-FN audit. Candidate generation time remains the external tool's
time; `verify_ms` is the exact extension/verification time, and `truth_ms` is
quality-audit time only.

## Module map

| Header / source | Purpose |
| --------------- | ------- |
| `include/structure.hpp`, `src/structure.cpp` | `BioSequence`, `MBB`, legacy `WorldNode` declaration, and default radii |
| `include/tools.hpp`, `src/tools.cpp` | Levenshtein `compute_distance`, helpers |
| `include/qgram_filter.hpp`, `src/qgram_filter.cpp` | Exact q-gram multiset count filter and inverted index |
| `include/range_join.hpp`, `src/range_join.cpp` | Exact full, adaptive-pigeonhole, q-gram, hybrid, and auto candidate generation |
| `include/phase2_distance_verifier.hpp`, `src/phase2_distance_verifier.cpp` | CPU batch exact verifier used by Phase2 rebinding |
| `include/mbb_rect_index.hpp`, `src/mbb_rect_index.cpp` | Exact SoA rectangle lookup for parent-local child MBB filtering |
| `include/index_builder.hpp`, `src/index_builder.cpp` | ID-array construction plus packing into `SequenceStore`, `WorldNodeRecord`, and flat relationship arrays |
| `include/index_persistence.hpp`, `src/index_persistence.cpp` | Array-format v38 binary persistence and manifest signatures |
| `include/sharded_index.hpp`, `src/sharded_index.cpp` | Lossless shard planning, resumable part construction, exact-minimizer shard routing, bundle manifests, and validated loading |
| `include/candidate_verifier.hpp`, `src/candidate_verifier.cpp` | Exact edit-distance verifier and TP/FP/FN accounting for external seed candidate TSVs |
| `include/search_engine.hpp`, `src/search_engine.cpp` | `search_adaptive`, `verify_leaf_candidates`, `search_greedy`, `search_exhaustive`, `search_brute_force` |
| `include/io_utils.hpp`, `src/io_utils.cpp` | FASTA/FASTQ load, TSV output |
| `include/map150.hpp`, `src/map150.cpp` | Fixed-150bp mapper pipeline: stride-1 reference windows, `2t` candidate search, occurrence location, final exact verifier |
| `src/main.cpp` | CLI entry points |
| `include/experiment_utils.hpp`, `src/experiment_utils.cpp` | Radius-schedule helpers for layer/radius search-cost experiments |

**Note:** Generic-read TSV paths emit genomic coordinates from `BioSequence::ref_positions`. `map150 --locator refpos` instead uses the reference-backed sequence store: leaves are `LeafId` values with representative/duplicate reference positions, so no individual 150-mer strings are retained. `--locator seqan` remains an optional build-time backend that fills `BioSequence::bwt_interval` as a suffix-array interval and uses that stored interval as the occurrence lookup handle during mapping.

**Note:** `build` and `query` can persist and reuse an index with
`--index <file>`. The binary file stores a manifest signature derived from input
fingerprints and construction parameters, followed by the sequence store, node
records, layer ranges, child/leaf/beacon IDs, MBB rows, and leaf-beacon rows.
Format v61 bit-packs each node to the minimum whole-byte width supported by
that shard's actual offset and count ranges (9 bytes per node in the 100k-window
reference benchmark, with wider automatic fallbacks). Base-relative child
payloads store a minimum whole-byte forward base delta from `node_id + 1`.
When a parent's child IDs form one exact consecutive range, the base is the
entire representation and child `i` is decoded as `base + i`; otherwise the
smallest exact delta encoding is used. Finalized arrays load directly. Older
files must be rebuilt.
Center sequence IDs use one verified byte-offset cycle when a layer's adjacent
ID deltas repeat exactly. Arithmetic lookup reconstructs the same `LeafId`;
irregular layers retain aligned eight-node blocks with one exact 16/32-bit base
and fixed-width exact deltas.
Paired child-MBB coordinates store one exact beacon-pair distance and rank
each child's quantized pair only among states allowed by the triangle
inequality. Decoding reproduces the same conservative bins exactly, so pruning
and recall semantics are unchanged while infeasible states consume no code
space.
Non-finest beacon payload begins use four-node records: one exact absolute
base plus three exact block-local deltas. The representation retains
constant-time decoding while avoiding an independent offset for every node.
All explicit beacon ID encodings share one contiguous byte payload, which
makes those begins monotone and permits the block compression.
Explicit beacon IDs use exact ZigZag deltas with one shard-wide 1..32-bit
width chosen by evaluating every width. Direct signed bytes and absolute IDs
remain available per node; choosing 16 bits can exactly match the former
8/16/32-bit payload, so the optimizer never makes this array larger.
A v16 `.navshard` bundle stores the common v61 construction manifest once and
points to independently loadable graph-payload byte ranges in `.navpack`
containers. This removes the repeated manifest from every logical shard without
changing its mapped graph arrays. When the bundle
has multiple shards and windows of at least 24 bases, a memory-mapped
`.route` sidecar. The router uses 16-mer minimizers from 24- to 64-base seeds
in `d + 1` disjoint query blocks. Sorted keys use exact 16-entry blocks with
one 32-bit base and minimum-width packed adjacent deltas; the parallel shard
IDs use exactly `ceil(log2(shard_count))` bits per entry. Any target
within edit distance `d` must contain one whole block exactly, so omitting
shards without any of those minimizers is no-FN-safe.
Pack paths and contig names are interned once in the manifest; logical-shard
descriptors contain only fixed-width numeric fields and occupy 40 bytes in
memory on the supported 64-bit build. Unsupported short/ambiguous queries or
an unavailable sidecar fall back to an exact scan of every part in groups of
at most 64 resident shards. Selected ranges are memory-mapped directly rather than loading
their whole pack. `query-index` and `query-index-batch` search selected parts
in parallel and merge identical sequences and their occurrences. Single-query
loading maps only routed parts; batch loading maps the union of all routed
parts, with oversized routed selections split at the same 64-shard cap. Any
fallback query conservatively searches every part without mapping
the whole human index at once. Batch planning and active-engine lookup use
only the sorted IDs in the current bounded group, with no bitmap or reverse
map proportional to the total shard count. FASTQ records and route plans are
streamed in blocks of at most 8,192 queries; unchanged resident shard groups
are reused across block boundaries, so memory is independent of total FASTQ
record count. Each block is executed in online subplans that stop after at
least 65,536 selected shard IDs. A query route is never split, bounding the
route table by that budget plus one complete route; stderr reports
`peak_route_ids`. Large sorted per-minimizer shard ranges are merged directly
into that unique route; the ordinary small-sort path expands at most 4,096 IDs
(16 KiB). Single-shard results bypass cross-shard hash merging. Non-sharded
queries stream one record at a time, and path tracing retains only the previous
query trace needed for overlap statistics. Bundle query
loading validates signatures, counts, mapped pack bounds, layer ranges, shard coordinates, and
the bundle checksum without an O(total nodes) rescan; build/restart reuse still
performs full part validation. Path tracing is not supported for a bundle
because node IDs are shard-local. `query-index` is the
pure load-and-search command for one query, so repeated invocations include
index load time each time. Use `locality-benchmark --index <navidx> --ref
<fasta> --out <tsv>` or the alias `query-locality-benchmark` to load once and
separate persisted-index load time, search-engine initialization time, and
query-only latency. `--scenarios low-fanout,high-fanout,repeat,batch-locality,
oracle,all` selects deterministic query streams for router gating, nearby
window routing, repeat stress, batch locality, and source-oracle diagnostics.
The locality summary reports the actual loaded-index fanout distribution
(`mean_fanout`, `p95_fanout`, `max_fanout`) plus router/path-reuse ratios so a
run can distinguish low-fanout gating from high-fanout router usage.
`--batch-schedules` defaults to source-sorted oracle query ordering for locality
benchmarks and can compare original, random, minimizer, q-gram signature, router
signature, and source-sorted oracle query ordering; the source oracle schedule is
diagnostic only.

The router builder writes each completed shard's sorted 32-bit minimizer list
contiguously to a temporary spool, then memory-maps and k-way merges the lists
while streaming the key column and packed shard IDs. The spool contains exactly
four bytes per minimizer, with no per-shard page padding. List starts are
implicit prefix sums, so build metadata stores only one 32-bit count per shard;
the merge heap reserves exactly one cursor per non-empty shard, uses a 12-byte
cursor below 2^32 entries, and advances by absolute entry index. Query-time
layout is unchanged.

Reference-window deduplication uses a contiguous exact open-addressed table with
one 64-bit slot per bucket and a maximum load of 7/8. The stored 32-bit hash is
only a probe hint; candidate equality is confirmed byte-for-byte against the
shared reference, making correctness independent of collisions. Ambiguous-base
windows are counted before allocation, and the build-only table is released
before occurrence sorting.

Dense q-gram postings preserve the existing 16-bit item/count fast path through
65,536 items. Above that boundary, short sequences with at most 255 q-grams use
one lossless 32-bit `24-bit item index + 8-bit count` word per posting rather
than the previous 8-byte pair. Inputs exceeding either bound automatically use
the wide representation.
Before materializing a deferred q-gram index, the range join tests the maximum
possible q-gram total over the length-compatible range. If its exact L1
threshold is non-positive, every compatible item must survive, so it returns
that same superset without building postings.
Use `--query-fastq-out <path>` to export the deterministic generated queries
with `source_pos=` read-header annotations for matched external baseline
candidate-recovery checks.
`query-locality-report --ref <fasta|sequence> --out-dir <dir>` wraps that
persisted benchmark and writes `summary.tsv`, `summary.json`, and `report.md`;
if `--index` is omitted, it reuses a manifest-compatible
`query_locality.navidx` in the report directory or builds it when missing or
stale. `run`, `benchmark`,
`map150`, and `boundary` still
build in-memory indexes for their current workflows. `boundary` avoids repeated
rebuilds within a parameter sweep by building once per stride mode and reusing
that in-memory index across the full rate grid. The Phase2 distance backend is
not part of the manifest signature because it changes only how exact checks are
executed during construction.

**Note:** The legacy three-primary-layer configuration remains the default CLI path, but the generalized implementation accepts arbitrary primary-layer lists such as `--primary-radii 40,28,18,10`. One auxiliary tier is generated automatically between each adjacent pair of primary layers and collapsed before search.

**Note:** Phase-2 rebinding and leaf attachment default to exact indexed range
joins. Use `--link-mode full` and/or `--leaf-attach-mode full` for the original
full-pairwise construction. `--range-candidate-mode auto` uses adaptive
pigeonhole seeds when they are at least 8 bp. Phase-2 rebinding accepts them by
actual candidate count and early-aborts to the q-gram safe fallback when the
seed union exceeds the configured threshold. Leaf attachment keeps the exact
pigeonhole candidate superset so a rare fallback cannot materialize a q-gram
index over the entire finest tier. The legacy ratio flag is ignored, so normal
pigeonhole queries do not full-scan all length-compatible targets just to
compute a denominator. `--leaf-attach-direction auto` chooses world-to-sequence
leaf attachment when there are fewer finest worlds than unique sequences;
explicit `seq-to-world` and `world-to-seq` are also supported. Forced `qgram`
and `hybrid` modes are still available. All modes exact verify every surviving
candidate before adding a link.

**Note:** Every build prints an aggregate `Build timing` section to stderr.
The timing fields are wall-clock milliseconds collected with
`std::chrono::steady_clock`; high-frequency loops are timed in aggregate to keep
profiling overhead low. In parallel Phase 2 indexed rebinding,
`phase2_rebinding_ms`, `phase2_index_build_ms`, and
`phase2_edge_insert_ms` remain wall-clock timings, while the Phase 2
candidate-query and exact-verify worker fields are accumulated per-thread time.
The `build-scale` command rebuilds in memory for each requested reference prefix
and writes phase timing, substep timing, construction counters, range candidate
mode, and q-gram length to CSV. With one prefix and `--index`, it also writes a
loadable persisted index whose manifest includes the reference fingerprint,
actual prefix length, window length, and stride. Reference windows are
reference-backed and contig-aware: only all-ACGT windows contained in one
contig are indexed. Representative offsets are strictly monotone by `LeafId`
and are encoded in 256-entry blocks. Arithmetic progressions store only one
16-byte base/step record; irregular blocks automatically choose the smallest
exact local bitset or 8/16/32-bit representation. Repeated windows add sorted
8-byte `(LeafId, source offset)` records for all
valid occurrences, including positions between sampled stride starts. Its
`BioSequence` identity and sequence are implicit in the array index and the
fixed window length. Query output resolves all occurrences directly from these
records and reports contig-local coordinates without scanning the reference.
All construction/search kernels consume `std::string_view` values into the
single stored reference instead of owning one object or string per window.
On AVX2 hosts, reference-backed Phase 2 construction verifies four exact
high-tolerance distances per Myers kernel and falls back to scalar Edlib on
unsupported inputs. A periodic edit-distance lower bound exits only when all
four candidates are provably outside the threshold, preserving every edge.
Indexed sequences and reference windows are limited to 255 bases. Therefore
every exact sequence-to-beacon edit distance fits in 8 bits. Format version 61
stores an all-ACGT shared reference in 2-bit form and restores one contiguous
byte view per loaded shard, so edit-distance query kernels retain direct
character access; references containing other IUPAC characters are kept
losslessly as raw bytes. An all-linear representative-position stream uses one
`base + LeafId * stride` mapping, omitting its per-256-position table; any
irregular stream retains the existing exact block encoding. It keeps
periodic center-ID layers as one exact byte-offset cycle after verifying every
reconstructed center; nonperiodic layers retain exact block bases and deltas.
It keeps leaf MBB distances as one base-3 byte per world whenever the
shard-wide exact alphabet has at most three values and every world has at most
five leaves;
otherwise it retains the normal exact bit packing. It keeps
leaf IDs implicit whenever every leaf list is an exact center-relative
consecutive interval, including endpoint clipping; the shard-wide interval
radius reconstructs every ID without a leaf-ID payload. When both this layout
and the dense ternary leaf-MBB layout apply, the clipped interval also implies
the exact leaf count and every other finest-node field is shard-wide, so the
finest layer needs no node-record bytes. It keeps
the leaf-MBB byte array implicit too when every exact distance is verified to
equal twice the consecutive leaf-ID displacement from its center; query then
reconstructs the identical ternary code from the clipped interval. It keeps
non-leaf beacons as one 4-bit shard-local pattern code per world whenever at
most 16 exact three-ID center-relative patterns cover the shard, eliminating
both their payload offsets and repeated beacon IDs. It keeps
the common non-leaf MBB bit width once per shard when the dense beacon layout
is active; an exact one-bit-per-node exception map restores the sole alternate
width when present. It keeps
contiguous child-range bases as a 32-node absolute base plus exact one-byte
deltas when every block fits, reducing the former fixed-width base stream
without changing child IDs. It keeps
long literal inputs only as manifest fingerprints, and omits redundant
child-payload offsets when a shard's child ranges are fully contiguous. It
also interns byte-identical child MBB distance blocks.
When packed leaf-ID and leaf-MBB streams have identical per-node byte starts,
the leaf-MBB offset is derived from the leaf-ID offset after an exact per-leaf
build-time check, preserving the packed MBB bytes and all pruning decisions.
If a shard's leaves all use packed IDs, center-only beacons, and one shared
packed-ID width, those repeated fields live once in the leaf layout instead of
once per leaf record; every other shard uses the general record layout.
For this uniform leaf layout, one exact byte offset per eight leaf nodes
replaces the absolute payload offset in every leaf record; the bounded
in-block sum reconstructs the same offset before leaf decoding and pruning.
Its guaranteed single center beacon is likewise stored once in the leaf
layout, rather than repeating a four-bit count of one in every leaf record.
When a shard's non-leaf children are all exact contiguous ranges, the packed
link kind and fixed width are stored once in the child layout as well.
Center-ID block bases use exact 16-bit values whenever all shard sequence IDs
fit, with an automatic 32-bit fallback for larger indexes. Eight-node blocks
minimize the exact base-plus-packed-delta representation on the target shards.
Beacon-offset blocks are selected from 2/4/8/16/32 entries per shard by exact
payload size; every stored offset remains unchanged.
Child MBB values are
packed for at most 10 deterministic beacons per non-finest parent. Reducing the
beacon cap only removes safe metric constraints and therefore cannot create a
false negative. Coarse layers store child-center distances as
`floor(distance / 12)`; the last transition into the finest world layer uses
`floor(distance / 6)` to preserve more pruning precision while fitting the
observed distance range in three bits. Search decodes the bin midpoint and
widens every corresponding metric bound by the matching maximum reconstruction
error, six or three, so the representation
may retain extra children but cannot prune a true result. Values use the exact
minimum bit width required by each parent's largest quantized value, with each
node starting on a byte boundary for constant-time lookup. Non-finest and
finest nodes use separate shard-local record layouts, so each region stores
its exact MBB offset at the minimum width its payload requires unless the
checked leaf-MBB offset derivation above applies.
A shard may contain up to 512 MiB of
packed child-MBB data. The child-layer radius plus the
layer-specific quantization error reconstructs a conservative interval during
search. Finest-layer leaf distances remain exact, but each node packs them at
the minimum 1..8-bit width required by its largest value. The width shares the
same node field as its exact byte offset, and nodes start on byte boundaries
for constant-time decoding. This avoids both per-distance padding and a
separate heap allocation for every child or leaf row. Finest-layer
construction also reuses the cleared child-ID buffer for leaf IDs. Leaf IDs
use exact ZigZag deltas from the world center and the smallest profitable
per-node 1..16-bit width; its width shares the leaf offset field, so the
encoding needs no side array. Finalized
nodes use one shard-wide compact bit layout and are addressed directly by array
index; the narrower finest-node layout reduces its common record by one byte
without adding query-time offset decoding;
layer offsets imply the layer ID and both radii. A packed 32-bit word stores
the common 24-bit child-or-leaf count, 4-bit beacon count, 2-bit beacon
encoding, and 2-bit link encoding;
rare overflows use an exact side-table record with full 32-bit counts. Beacon
IDs use the narrowest exact per-node
encoding: signed 8-bit or 16-bit deltas from the node center, falling back to a
full 32-bit ID only when necessary. A finest-layer node's sole beacon is its
center and consumes no side-array entry. Only the dense non-finest NodeId
prefix stores explicit beacon offsets in a separate array, eliminating the
otherwise constant zero field from every finest-layer node.
Each parent selects the smallest exact constant-time child-ID representation:
one 32-bit minimum ID followed by fixed-width 1..16-bit offsets, a base plus
8-bit offsets, 16-bit forward deltas, or full 32-bit IDs. The packed width
shares the node's child-offset field instead of requiring a side array. Every
encoding preserves every edge without truncation.
Leaf IDs use signed 8-bit or 16-bit deltas from the node center when the whole
leaf list fits, with exact 32-bit IDs as the fallback.
Repeated reference positions use one pair for a singleton duplicate, while
larger duplicate groups share one group offset and a flat 32-bit position
array.
Finalized node, edge, beacon, MBB, leaf, reference-record, and repeated
occurrence arrays start on 64-byte boundaries in the index and load through
read-only memory mappings, so SIMD scans do not inherit accidental cache-line
misalignment and index loading does not duplicate the arrays into heap memory.
Query access still uses cached raw pointers and sizes.
Because finest nodes use leaf links and all other nodes use child links, those
two mutually exclusive ranges share one offset/count pair. Together with the
packed count word, a finalized world node is therefore 16 bytes instead of 32,
with no range tag and no lossy count limit.
Adaptive, greedy, and exhaustive searches reuse exact edit distances already
computed for the same sequence ID within one query. This fixed open-addressed
cache occupies 16 KiB per thread regardless of index size. It stores only exact
8-bit distances; a bounded result above its threshold and an overfull probe
chain both bypass the cache, preserving the original pruning and recall
semantics.

## Parameter sweeps

For long-sequence boundary studies, `boundary` outputs one aggregated TSV row per `(error_rate, tolerance_rate)` cell with source-recovery and pruning metrics for fixed-length `L=250` windows derived from a reference FASTA such as `chr1_subset`. Broader experiment orchestration and comparative baseline workflows live under the repository-level `methods/` directory.

## Tests

| Target | Command |
| ------ | ------- |
| Recall (adaptive vs brute force, 0 FN under test protocol) | `make test_recall && ./test_recall` |
| Distance bounds (violations report) | `make test_distance_bound && ./test_distance_bound` |
| 150bp mapper recall and verifier checks | `make test_map150 && ./test_map150_recall` |
| Bounded edit distance | `make test_bounded && ./test_bounded_edit_distance` |
| Bounded Myers edit distance | `make test_bounded_myers && ./test_bounded_myers_bin` |
| Exact range join | `make test_range_join && ./test_range_join` |
| Q-gram count filter | `make test_qgram && ./test_qgram_filter` |
| Full/indexed construction equivalence | `make test_build_range && ./test_build_range_equivalence` |
| Build timing statistics | `make test_build_timing_stats && ./test_build_timing_stats` |
| Build-scale CSV smoke | `make test_build_scale_smoke && ./test_build_scale_smoke` |
| Exact MBB rectangle lookup | `make test_mbb_rect && ./test_mbb_rect_index` |
| Scan/rect adaptive equivalence and fallback | `make test_mbb_filter && ./test_mbb_filter_equivalence` |
| Search q-gram on/off and scan/rect equivalence | `make test_search_qgram && ./test_search_qgram_prefilter` |
| Safe child router no-FN / candidate superset / fallback | `make test_safe_child_router && ./test_safe_child_router_no_false_negative` |
| Persisted index round-trip and manifest matching | `make test_index_persistence && ./test_index_persistence_bin` |
| Sharded window/coordinate equivalence, restart/repair, substitution/indel router no-FN, and fallback | `make test_sharded_index` |
| Phase2 CPU verifier behavior | `make test_phase2_distance_verifier && ./test_phase2_distance_verifier_bin` |
| Build heartbeat formatting and timer | `make test_build_progress` |

Search q-gram benchmark results and commands are recorded in
[`SEARCH_QGRAM_PREFILTER_BENCHMARK.md`](SEARCH_QGRAM_PREFILTER_BENCHMARK.md).
