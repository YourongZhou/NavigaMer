# NavigaMer CLI reference

The `navigamer` executable (`src/main.cpp`) implements the commands below. Paths are optional: if `--ref` or `--reads` is **not** an existing file, the argument is treated as a **literal DNA string** (see I/O rules).

## Requirements

- **C++17**, **OpenMP**
- Build: `make` or CMake (see [`README.md`](README.md) in this directory)
- Optional SeqAn3 FM-index locator for `map150`: `make WITH_SEQAN3=1` or CMake with `-DNAVIGAMER_WITH_SEQAN3=ON`

## Global hierarchy flags

Used by all pipelines that build the index:

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--primary-radii` | *(none)* | Comma-separated primary-layer radii from coarsest to finest, e.g. `40,28,18,10` |
| `--r-sw` | `5` | Small-world radius (`R_SW` in `structure.hpp`) |
| `--r-mw` | `15` | Mid-world radius |
| `--r-lw` | `30` | Large-world radius |
| `--link-mode` | `indexed` | Phase-2 rebinding: `full` or exact `indexed` range join |
| `--leaf-attach-mode` | `indexed` | Leaf attachment: `full` or exact `indexed` range join |
| `--leaf-attach-direction` | `auto` | Indexed leaf attachment direction: `auto`, `seq-to-world`, or `world-to-seq`; `auto` uses `world-to-seq` when finest worlds are fewer than unique sequences |
| `--leaf-qgram-postfilter` | `off` | Optional safe q-gram L1 postfilter for indexed leaf attachment candidates before bounded exact verification |
| `--range-min-seed-length` | `8` | Full-scan fallback below this adaptive seed length |
| `--range-max-seed-length` | `20` | Maximum adaptive pigeonhole seed length |
| `--range-candidate-mode` | `auto` | Indexed construction candidates: `auto`, `pigeonhole`, `qgram`, `hybrid`, or `full` |
| `--qgram-q` | `5` | Positive q-gram length used by q-gram and hybrid candidate generation |
| `--auto-pigeonhole-max-candidates` | `4096` | Auto accepts pigeonhole when its candidate count is at most this value |
| `--auto-pigeonhole-max-ratio` | `0.25` | Compatibility no-op; parsed and validated, but auto no longer computes or uses a candidate ratio |
| `--auto-hybrid-on-large-candidates` | `true` | Compatibility flag; normal auto early-aborts oversized seed unions and uses q-gram safe fallback |
| `--build-distance-mode` | `edlib` | Index construction edit-distance backend: default `edlib`, reference `dp`, or `auto` (Myers when supported, otherwise Edlib) |
| `--min-rect-index-fanout` | `64` | Minimum child-world fanout required to build an exact MBB rectangle index |
| `--phase1-metric-min-fanout` | `12` | Minimum Phase1 parent-local candidate fanout before building/querying the metric helper instead of scanning |
| `--phase1-qgram-min-fanout` | `12` | Minimum Phase1 parent-local candidate fanout before using the q-gram helper instead of the metric helper |
| `--phase1-qgram-max-touched` | `250000` | Maximum Phase1 q-gram touched/candidate set size before conservatively falling back |
| `--progress-interval-seconds` | `600` | Timestamped build heartbeat interval on stderr; `0` disables periodic heartbeats but keeps phase-boundary reports |
| `--mbb-filter-mode` | `scan` | Adaptive child-MBB filtering: original `scan` or exact `rect` lookup |
| `--visited-mode` | `epoch` | Adaptive visited tracking: legacy per-query `string` set or integer-ID `epoch` array |
| `--graph-view` | `flat` | Array traversal mode; `original` is accepted as a compatibility alias for `flat` |
| `--simd-mode` | `auto` | Flat child-MBB and leaf-beacon filter backend: `auto`, `scalar`, `avx2`, or `avx512`; unsupported SIMD falls back to scalar |
| `--distance-mode` | `myers` | Adaptive bounded child-center distance backend: default Myers through 256bp ACGT shorter-input length, optional `edlib`, reference `dp`, or `auto` (Myers when supported, otherwise Edlib) |
| `--search-prefetch` | `off` | Best-effort adaptive traversal prefetch hints: `off` or `on`; affects only memory-access hints, not pruning or verification semantics |
| `--search-qgram-prefilter` | `off` | Safe child-world center q-gram prefilter: `off` or `on` |
| `--search-qgram-q` | `5` | Search-only q-gram length; non-positive values disable the prefilter |
| `--query-profile` | `0` | Enable (`1`) or disable (`0`) per-query profiling timers in adaptive search; counters remain available either way |
| `--path-reuse` | `1` | Enable (`1`) or disable (`0`) thread-local warm-start caches and query-derived batch scheduling hints |
| `--router-hints` | `0` | Enable (`1`) or disable (`0`) q-gram/minimizer/pigeonhole router hints before local-router / best-first ordering |
| `--router-hint-qgram-q` | `5` | Router-hint q-gram length used for cached child-center signatures and parent-local range hints |
| `--router-hint-minimizer-k` | `4` | Router-hint minimizer k-mer length |
| `--router-hint-minimizer-w` | `8` | Router-hint minimizer window length in bases; must be at least `k` to produce a usable sketch |
| `--local-router` | `0` | Enable (`1`) or disable (`0`) parent-local child routing hints after safe MBB filtering |
| `--local-router-max-anchors` | `4` | Maximum parent-local beacon dimensions used by the router score |
| `--local-router-max-children` | `64` | Reporting threshold for the router's top-k shortlist counters; `0` means all routed children |
| `--local-router-score` | `anchor-envelope` | Router scoring mode; current implementation supports only `anchor-envelope` |
| `--best-first` | `0` | Enable (`1`) or disable (`0`) safe best-first ordering of post-MBB child worlds |
| `--safe-child-router` | `0` | Enable (`1`) or disable (`0`) parent-local safe child candidate generation before MBB filtering |
| `--safe-child-router-min-fanout` | `64` | Minimum parent child fanout before building/querying the safe child router |
| `--safe-child-router-max-candidates` | `4096` | Maximum candidate children accepted from the router before full enumeration fallback |
| `--safe-child-router-max-ratio` | `0.5` | Maximum candidate/child ratio accepted from the router before full enumeration fallback |
| `--safe-child-router-min-seed-len` | `8` | Minimum seed length for the parent-local range helper |
| `--safe-child-router-mode` | `auto` | Candidate helper mode: `auto`, `pigeonhole`, `qgram`, `mbb`, or `full-fallback` |
| `--safe-child-router-validate` | `0` | Debug validation mode; exact-checks possible children and throws if a routed candidate set misses one |
| `--query-planner` | `0` | Enable (`1`) or disable (`0`) the adaptive query planner that records per-query routing strategy and may safely bypass expensive router stack work on low-fanout indexes |
| `--planner-direct-verify-max-candidates` | `32` | Reserved direct-verify planning threshold; currently reported in profile JSON but direct q-gram verification is not selected |
| `--planner-router-min-fanout` | `64` | Minimum observed primary child fanout before the planner keeps router-hint/local-router/best-first/q-gram ordering work enabled |
| `--planner-safe-child-router-min-fanout` | `64` | Minimum observed primary child fanout before the planner keeps safe-child-router work enabled |
| `--planner-allow-direct-qgram-verify` | `1` | Reserved switch for future direct q-gram verification planning; current implementation remains exact traversal only |
| `--proximal-oracle` | `0` | Enable (`1`) or disable (`0`) query-benchmark proximal-anchor oracle diagnostics; instrumentation only |
| `--proximal-oracle-k` | `1,2,4` | Comma-separated k values recorded in query-benchmark configuration; TSV currently emits k1/k2/k4 envelope columns |
| `--index` | *(none)* | Persisted NavigaMer index or shard-manifest path for `build`, single-prefix `build-scale`, `build-sharded`, `query`, and `query-index` |

If `--primary-radii` is present, it takes precedence and the legacy three-radius flags are ignored. The implementation automatically inserts one auxiliary tier between each adjacent pair of primary layers during build and collapses those auxiliary tiers into beacons + MBB rows before query-time navigation.

Indexed construction is exact. Pigeonhole mode uses
`block_len = floor(L / (tau + 1))` and
`seed_len = min(range_max_seed_length, block_len)`, falling back to the
length-compatible full set when the seed is too short. Q-gram mode safely
prunes only pairs with `qgram_l1(a,b) > 2*q*tau`. Hybrid mode intersects the
pigeonhole and q-gram safe candidate supersets. Auto runs pigeonhole when its
seed is long enough, accepting it when candidate count is at most the
configured maximum. If the seed union grows beyond the maximum, auto stops
collecting pigeonhole candidates immediately and invokes q-gram as a safe
fallback. It does not full-scan all length-compatible targets to compute a
candidate ratio; `--auto-pigeonhole-max-ratio` is retained only so older
command lines still parse. Full candidate mode returns every
length-compatible item.

Indexed leaf attachment directly verifies range candidates by default. With
`--leaf-qgram-postfilter on`, it applies the same no-false-negative q-gram L1
condition before bounded exact verification. This can reduce exact calls for
unusually broad candidate sets, but its signature-building overhead is slower
on the default prepared-DNA distance path. Accepted leaf links are determined
by bounded edit distance in either mode.

Old seed-length-only auto behavior can be reproduced with a permissive
candidate-count threshold such as
`--auto-pigeonhole-max-candidates 18446744073709551615`; the ratio flag no
longer affects acceptance.

Every returned candidate is still verified with bounded exact edit distance;
candidate generation never directly adds an edge or leaf attachment. Builder
summaries distinguish possible pairs, returned candidates, exact calls,
accepted results, length pruning, q-gram L1 pruning, per-mode query counts,
fallback counts, seed candidates before length filtering, seed length-pruned
candidates, pigeonhole early-abort counts, final range candidates, and
reduction ratios.

Phase1 sketch construction uses the same bounded exact verifier after any
helper path. For each parent-local candidate group, fanout below
`--phase1-metric-min-fanout` scans directly; fanout between
`--phase1-metric-min-fanout` and `--phase1-qgram-min-fanout` uses the metric
helper; larger fanout uses the q-gram helper unless it exceeds
`--phase1-qgram-max-touched` and falls back conservatively.

Rectangle filtering is also exact. Rect mode returns a child world if and only
if its existing MBB row intersects the query rectangle in every beacon
dimension. It changes only survivor enumeration; center-distance verification,
containment/overlap traversal, and leaf verification remain unchanged. Missing
or inconsistent indexes, small fanout, dimension mismatches, and query
exceptions fall back to scan.

Search-side q-gram filtering is also exact and no-false-negative. It runs only
after child MBB survivor generation and safely prunes when
`qgram_l1(query, child.center) > 2*q*(child.radius+tolerance)`. Passing children
still receive bounded exact center edit-distance verification. It does not
change coarsest-layer search, strict containment, overlap traversal, leaf
refinement, construction rebinding, or leaf attachment. World-center
signatures are cached per search engine. Non-ACGT centers/queries, unsupported
q values, and missing signatures conservatively fall back to no pruning.

Adaptive profiling (`--query-profile 1`) records timing buckets such as
`query_total_ms`, `anchor_distance_ms`, `mbb_filter_ms`,
`center_distance_ms`, `leaf_collect_ms`, `leaf_verify_ms`, and
`result_dedup_ms`, plus counters such as world access, anchor distance calls,
center distance calls, and raw candidates. `query`, `benchmark`, and
`query-benchmark` keep the same result semantics; profiling changes only
instrumentation output.

Adaptive path reuse (`--path-reuse 1`) keeps thread-local warm-start caches for
exact parent-local anchor-distance vectors on repeated queries and cached child
shortlists keyed by cheap query-derived fingerprints. It records
`path_reuse_attempt_count`, `path_reuse_hit_count`,
`anchor_cache_hit_count`, and `child_shortlist_reuse_hit_count`. These caches
affect only ordering or exact memoization, never become a sole pruning reason,
and batch-oriented commands group queries by the same query-derived fingerprint
while preserving emitted output order.

Adaptive router hints (`--router-hints 1`) run after safe MBB survivor
generation and before local-router / best-first ordering. The current PR3 path
builds parent-local child-center range hints with q-gram/pigeonhole candidate
supersets, then uses q-gram and minimizer similarity only to reprioritize the
survivor list. Query, benchmark, and query-benchmark output add
`router_hint_invoked_count`, `router_qgram_ranked_count`,
`router_minimizer_ranked_count`, `router_pigeonhole_query_count`,
`router_candidate_count`, `router_candidate_hit_count`, and
`router_fallback_count`. These counters are advisory only: all non-predicted
children still remain in the exact fallback order.

Adaptive local routing (`--local-router 1`) ranks MBB-surviving child worlds by
their parent-local beacon-envelope fit before bounded center verification.
`--local-router-max-anchors` limits how many beacon dimensions contribute to the
score, and `--local-router-max-children` controls the reported top-k shortlist
size without suppressing fallback traversal of the remaining children. Query and
benchmark output add local-router invocation, shortlist, and fallback counters.

Adaptive safe child routing (`--safe-child-router 1`) runs before child MBB
filtering for sufficiently high-fanout parents. It builds a parent-local
child-center range index and queries it at `tolerance + max_child_radius`,
which is a safe superset for any child satisfying
`d(query, child.center) <= tolerance + child.radius`. If the candidate set is
too broad, the range helper falls back to full scan, or validation is requested
and detects a miss, traversal falls back to the original full child
enumeration. Returned candidates still go through MBB filtering, bounded center
distance, and exact leaf verification. Counters include
`safe_child_router_invoked_count`, `safe_child_router_fallback_count`,
`safe_child_router_candidate_count`, and
`safe_child_router_pruned_by_not_candidate_count`.

Adaptive query planning (`--query-planner 1`) runs once per adaptive query and
records which routing strategy the query used. The current planner is
conservative: on low-fanout indexes it disables optional q-gram/router ordering
work for that query, and on high-fanout indexes it keeps the selected optimized
stack active. It never bypasses MBB filtering, bounded center verification, or
final exact leaf verification. Query, benchmark, and query-benchmark output
include `planner_invoked_count`, `planner_strategy_baseline_count`,
`planner_strategy_router_count`, `planner_strategy_safe_child_router_count`,
`planner_strategy_path_reuse_count`, `planner_fallback_count`, and
`planner_decision_ms`.

Proximal-anchor oracle diagnostics (`--proximal-oracle 1`) are available for
`query-benchmark`. They record actual anchor-source nodes, traversed frontier
nodes, true-path anchors implied by brute-force hits, global nearest anchors,
and deterministic random anchors, then report exact edit-distance envelopes.
This path only adds diagnostic output; it does not change traversal, pruning, or
result verification.

Adaptive safe best-first ordering (`--best-first 1`) then reprioritizes those
post-MBB child worlds using conservative parent-local MBB lower bounds with a
tighter-envelope tie-break before bounded center verification. It records
`best_first_invoked_count`, `best_first_bound_candidate_count`,
`child_safe_bound_pruned_count`, `frontier_max_size`, and
`frontier_total_pushed`. Any pruning from this path must remain a conservative
`SafeBound`.

Query-side safety contract:

- `RouterHint`: may be incomplete or wrong and must never be the sole pruning reason.
- `SafeCandidateRouter`: may reduce child enumeration only when it returns a
  mathematically safe candidate superset; otherwise it must full-fallback.
- `SafeBound`: may prune only when the bound is conservative and recall-safe.
- `ExactVerifier`: only exact edit distance `<= tolerance` may emit final hits.

## I/O conventions (`io_utils`)

- **`--ref`**: If the value is a **path to a file**, load **FASTA** (`>` header, sequence lines). Otherwise treat the whole argument as one reference sequence (ID `ref`).
- **`--reads`**: If the value is a **file path**, load **FASTQ** (`@` id, sequence, `+`, quality). Otherwise treat it as a **single read** sequence (ID `query_0`).

## Commands

### `demo`

Synthetic reference (~50 kb) and reads (length 20, zero mutation rate). Compares adaptive vs exhaustive vs brute force on a sample of reads.

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--size` | `500` | Number of reads |
| `--primary-radii` or `--r-sw`, `--r-mw`, `--r-lw` | `30,15,5` by default through the legacy flags | Primary-layer radii |

### `build`

Deduplicates, builds the index, and prints layer sizes. If `--index <file>` is
provided, writes an array-format v17 binary index with a manifest signature,
build parameters, input fingerprints, sequence records, layer ranges,
child/leaf/beacon ID arrays, MBB rows, and leaf-beacon distance rows. Older
pointer-graph index formats are rejected and must be rebuilt.
The in-memory construction path also uses node arrays and integer IDs; it does
not construct an intermediate `WorldNode` pointer graph.

**Required:** `--ref`, `--reads`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--index` | *(none)* | Output path for the persisted index |

Every build prints a `Build timing` section to stderr. Timing fields are
aggregate wall-clock milliseconds and include Phase0 deduplication, Phase1
sketch construction, Phase2 rebinding, Phase3 MBB computation, Phase4 leaf
attachment, integer ID assignment, graph-view flattening, and selected
range-join/MBB/leaf substeps. Existing construction counters remain in the
summary.

### `build-scale`

Builds one index per requested reference prefix and writes construction timing
plus construction counters to CSV. With `--index <file>`, exactly one prefix
must be requested and the resulting reference-window index is serialized.
Reference windows use one stored reference sequence plus representative
offsets; they are not materialized as independent `std::string` objects.
FASTA contig boundaries are retained, cross-contig windows are never created,
and windows containing characters outside A/C/G/T are counted as invalid and
skipped. Repeated valid windows use a sparse occurrence array for direct
contig-local output-coordinate lookup; occurrences between stride-selected
window starts are retained as well.

**Required:** `--ref`, `--prefix-lengths`, `--out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--window` | `200` | Reference-window length |
| `--stride` | `1` | Step between window starts |
| `--prefix-lengths` | *(required)* | Comma-separated reference prefix lengths; values larger than the reference use the full reference |
| `--out` | *(required)* | Output CSV path |
| `--index` | *(none)* | Persist the single requested prefix as a loadable binary index |
| `--progress-interval-seconds` | `600` | Periodic progress interval; `0` keeps only phase start/finish reports |

After each prefix, stderr prints the top build phases, top substeps, and
candidate/exact reduction percentages for Phase2 and leaf attachment.
Heartbeat lines include timestamp, phase, completed/total work items,
percentage, elapsed time, observed rate, and ETA. Progress output never changes
the CSV shape or persisted-index signature.

### `build-sharded`

Builds a lossless reference-window bundle for inputs that would exceed one
index's 32-bit node, relationship, or MBB-array limits. Each window start is
assigned to exactly one part. Adjacent parts store only the reference overlap
needed to materialize boundary windows, and every emitted coordinate remains
relative to the original contig.

Each part is an ordinary v17 `.navidx` file. A completed part is reused only
after its input fingerprint, construction signature, reference slice, contig,
and source coordinates validate. Damaged or incompatible parts are rebuilt,
and newly completed parts are installed atomically. The final `.navshard`
manifest is written only after all parts are valid.

**Required:** `--ref`, `--index`, `--shard-windows`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--window` | `200` | Reference-window length |
| `--stride` | `1` | Step between window starts |
| `--shard-windows` | *(required)* | Maximum number of window starts assigned to one shard |
| `--index` | *(required)* | Output `.navshard` manifest; part files are created beside it |
| `--progress-interval-seconds` | `600` | Periodic progress interval inside each shard build |

### `query`

Builds an index from `--reads`, then searches for `--query`. With
`--index <file>`, `query` first compares the requested build manifest with the
stored manifest. If the signatures match, it loads the persisted index and skips
construction. If the file is missing, unreadable, or has different inputs or
construction parameters, it rebuilds from `--reads` and writes the new index to
that path. With `--index` and no `--reads`, `query` loads the index directly.

**Required:** `--query` plus either `--reads` or `--index`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Max edit distance |
| `--mode` | `adaptive` | `adaptive` \| `greedy` \| `exhaustive`; all modes exactly verify returned leaves |
| `--ref` | optional | Placeholder in current flow |
| `--index` | *(none)* | Persisted index to reuse, create, or load directly |

### `query-index`

Loads a persisted index or `.navshard` bundle and searches `--query`. Bundle
parts are searched in parallel and identical hit sequences are merged. This
command never rebuilds and does not accept `--reads`; use
`query --reads ... --index ...` for automatic reuse-or-rebuild behavior.

**Required:** `--index`, `--query`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Max edit distance |
| `--mode` | `adaptive` | `adaptive` \| `greedy` \| `exhaustive` |

### `query-index-batch`

Loads one persisted index or `.navshard` bundle and searches every record in
`--reads` without reloading or rebuilding between queries. Bundle parts are
searched in parallel; identical sequences and all of their original-contig
occurrences are merged before TSV output. This is the warm-load batch entry
point used for source-sorted, duplicated-read, and near-repeat locality
experiments. It currently supports `--mode adaptive` only; every emitted hit is
still produced by final bounded exact edit-distance verification.
Bundle loading checks manifests, mapped ranges, compact layer layout, shard
coordinates, and checksums without touching every persisted node and edge.

**Required:** `--index`, `--reads`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Max edit distance |
| `--mode` | `adaptive` | Currently only `adaptive` is accepted |
| `--out` | *(none)* | Optional per-query hit/stat TSV |
| `--path-trace-out` | *(none)* | Optional locality trace TSV for one `.navidx`; rejected for `.navshard` because node IDs are shard-local |

The main TSV includes query latency, result counts, adaptive work counters,
`search_prefetch_enabled`, and path-class counters such as
`query_path_class`, `path_contained_step_count`, `path_overlap_step_count`, and
`path_uncovered_step_count`. The optional trace TSV records world node IDs
evaluated and leaf sequence IDs exactly verified for each input query so batch
ordering and reuse can be audited without changing query results.

### `locality-benchmark` / `query-locality-benchmark`

Loads one persisted NavigaMer index once, generates clustered query streams from
`--ref`, and reports load time, engine initialization time, and query-only
latency separately. It emits `same_template`, `nearby_windows`, and
`random_windows` datasets for `baseline`, `path_reuse`, and `optimized`
profiles by default. `query-locality-benchmark` is a compatibility alias with
the same flags. Use this instead of repeated `query-index` calls when measuring
batch locality or prefetch/path-reuse effects, because repeated `query-index`
includes index load time on every query. The TSV also reports primary-DAG
fanout distribution and router/path-reuse invocation ratios, so low-fanout
sanity runs can show that router stages were gated while high-fanout runs can
show whether local routing and safe child routing actually fired.

**Required:** `--index`, `--ref`, `--out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--query-count` | `256` | Queries per locality dataset |
| `--query-length` | `250` | Generated query length |
| `--query-edits` | `--tolerance` | Substitution edits per generated query |
| `--tolerance` | `2` | Search edit-distance threshold |
| `--seed` | `42` | Deterministic query generation seed |
| `--scenarios` / `--scenario` | *(unset)* | Comma-separated scenario presets: `low-fanout`, `high-fanout`, `repeat`, `batch-locality`, `oracle`, or `all`; when set, these override `--locality-datasets` |
| `--locality-profiles` | `baseline,path_reuse,optimized` | Comma-separated profiles to run; use `baseline,path_reuse` to isolate path reuse before testing the full optimized stack |
| `--locality-datasets` | `same_template,nearby_windows,random_windows` | Comma-separated query streams; use `same_template,nearby_windows` for larger clustered-query runs |
| `--batch-schedules` | `source-oracle` | Comma-separated internal query schedules: `original`, `random`, `minimizer`, `qgram-signature`, `router-signature`, or `source-oracle`; `source-oracle` is an upper-bound diagnostic only |
| `--query-fastq-out` | *(unset)* | Optional FASTQ export of the generated locality queries, with `source_pos=` in each read header, for matched external baseline runs |

Scenario presets are implemented as deterministic query-stream selections:
`low-fanout` maps to random windows, `high-fanout` maps to nearby windows for
router-opportunity measurement on a high-fanout index, `repeat` cycles through
nearby repeated-source windows, `batch-locality` emits clustered batches, and
`oracle` emits source-position-aware windows intended for comparison with
`source-oracle` scheduling. The benchmark still reports the real fanout
distribution from the loaded index; choosing `high-fanout` does not fabricate a
high-fanout index.

Output columns include load/init/query timing, result and mismatch counts,
mean/p50/p95 query latency, mean world/center/leaf work counters,
`mean_fanout`, `p50_fanout`, `p95_fanout`, `max_fanout`,
`router_invoked_ratio`, `local_router_invoked_ratio`,
`safe_child_router_invoked_ratio`, `path_reuse_hit_ratio`, and average
router/path-reuse counters. Aggregate locality reuse columns include
`anchor_cache_hit_count`, `child_shortlist_cache_hit_count`,
`safe_child_candidate_cache_hit_count`, and
`productive_world_reuse_hit_count`. Near-query reuse columns include
	`near_query_triangle_pruned_count`, `near_query_center_distance_reused_count`,
	`near_query_bound_fallback_count`, `near_query_direct_verify_count`,
	`near_query_leaf_triangle_pruned_count`,
	`near_query_leaf_distance_reused_count`,
	`near_query_leaf_bound_fallback_count`,
	`center_distance_reduction`, `world_access_reduction`, and `p95_speedup`.
`batch_schedule_mode` records the schedule used for that row. The optimized
locality profile enables deterministic path reuse while
leaving router hints, safe child routing, local routing, and best-first ordering
off by default; persisted locality runs already use exact rect MBB filtering, so
the heavier router stages are measured through explicit query and
query-benchmark flags instead of the default locality optimized profile.

### `query-locality-report`

Builds or reuses a persisted index, runs the same locality benchmark matrix,
and writes a small report bundle:

- `summary.tsv`: the locality-benchmark TSV
- `summary.json`: machine-readable rows and gate status
- `report.md`: compact Markdown table for review

If `--index` is omitted, the command reuses a manifest-compatible
`query_locality.navidx` in `--out-dir` or builds reference windows from `--ref`
and saves that file when it is missing or stale before running query-only
measurement. `source-oracle` batch scheduling remains an upper-bound diagnostic
only.

**Required:** `--ref`, `--out-dir`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--index` | *(none)* | Optional persisted index to reuse instead of building one in the report directory |
| `--out-dir` | required | Directory for `summary.tsv`, `summary.json`, `report.md`, and an auto-built `query_locality.navidx` |
| `--reference-subset-length` | `0` | Prefix length used when auto-building the report index; `0` uses the full input |
| `--window`, `--stride`, `--query-count`, `--query-length`, `--query-edits`, `--tolerance`, `--seed`, `--scenarios`, `--locality-profiles`, `--locality-datasets`, `--batch-schedules` | see `locality-benchmark` | Passed through to the locality benchmark |

### `run`

Full pipeline: load ref + reads, build, **adaptive** search for every read, optional TSV.

**Required:** `--ref`, `--reads`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Threshold |
| `--out` | *(none)* | If set, write TSV; otherwise only stderr summary |

Uses OpenMP over reads.

### `map150`

Fixed-length mapper path for 150 bp reads. Builds an in-memory index from all forward reference 150-mers at stride 1, searches each read on both strands with `candidate_tolerance = 2 * --tolerance`, then verifies candidate occurrence neighborhoods exactly and emits only alignments with edit distance `<= --tolerance`.

**Required:** `--ref`, `--reads`, `--out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Final edit-distance threshold; the candidate search uses `2 * tolerance` |
| `--mode` | `adaptive` | Currently only `adaptive` is accepted |
| `--locator` | `refpos` | `refpos` uses stored reference-window positions; `seqan` requires optional SeqAn3 support |
| `--out` | *(required)* | Output TSV path; header is written even when there are no hits |

Safety constraints: every read must be exactly 150 bp, reference and reads must contain only A/C/G/T, and the finest primary radius must be greater than `2 * tolerance`.

### `benchmark`

Slices the reference into windows of length `--window` with stride `--stride`; each window is one indexed sequence with coordinates. Query sequences come from `--reads`. Uses **adaptive** search; TSV includes search statistics.

**Required:** `--ref`, `--reads` (queries)

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--tolerance` | `2` | Edit threshold |
| `--window` | `200` | Window length on the reference |
| `--stride` | `1` | Step between window starts |
| `--out` | *(none)* | Output TSV path |

If a query has no hit, a placeholder row is still emitted with stats.

### `query-benchmark`

Builds one shared in-memory index, deterministically generates six query
classes (`random_region`, `ordinary_region`, `low_complexity_region`,
`no_hit`, `single_hit`, and `multi_hit`), and compares:

- baseline: fixed `scan` MBB filtering, legacy `string` visited mode, the
  `original` compatibility label (same canonical array traversal), scalar MBB
  filtering, `dp` distance mode, search q-gram disabled, and `best-first`
  disabled
- optimized: `--mbb-filter-mode`, `--visited-mode`, `--graph-view`,
  `--simd-mode`, `--distance-mode`, `--search-qgram-prefilter`,
  `--search-qgram-q`, `--router-hints`, `--local-router`,
  `--safe-child-router`, `--path-reuse`, `--best-first`, and
  `--query-planner`
- optional ablations: with `--query-benchmark-ablations 1`, one additional
  profile per enabled query-side optimization stage that disables only that
  stage inside the optimized stack
- exact brute-force result IDs computed before timing

**Required:** `--ref`, `--out`, `--summary-out`, `--json-out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--reference-subset-length` | `0` | Reference prefix length; `0` uses the full input |
| `--window` | `200` | Indexed reference-window length |
| `--stride` | `1` | Indexed window stride |
| `--query-length` | `200` | Generated query length |
| `--tolerance` | `2` | Exact edit-distance threshold |
| `--seed` | `42` | Deterministic generation seed |
| `--threads` | `1` | Recorded/applied OpenMP thread count; Step 0 query execution remains serial |
| `--queries-per-class` | `1` | Queries generated for each class |
| `--warmup-iterations` | `2` | Untimed warmups per query/profile |
| `--measured-iterations` | `10` | Timed warm samples per query/profile |
| `--cold-cache-bytes` | `268435456` | Best-effort eviction buffer touched before each cold sample; `0` disables it |
| `--query-benchmark-ablations` | `0` | Enable (`1`) or disable (`0`) derived ablation profiles such as `ablation_no_search_qgram`, `ablation_no_router_hints`, `ablation_no_safe_child_router`, `ablation_no_path_reuse`, or `ablation_no_query_planner` when those stages are enabled in the optimized profile |
| `--proximal-oracle` | `0` | Enable (`1`) or disable (`0`) proximal-anchor oracle diagnostics in detail/summary TSV and JSON |
| `--proximal-oracle-k` | `1,2,4` | Positive comma-separated k values for proximal-oracle envelope configuration; detail/summary TSV emits k1/k2/k4 |
| `--out` | required | Detailed per-sample TSV |
| `--summary-out` | required | Per-class/profile plus `all` aggregate TSV |
| `--json-out` | required | Configuration, build, memory, aggregate, mismatch, and gate JSON |

The timed region contains only `search_adaptive`. Results must be stable across
repeated executions and exactly match between profiles and brute force. The
command writes all outputs before returning and exits `0` when the gate passes,
`2` on a result/no-FN mismatch, and `1` on configuration or runtime errors.
Current/peak RSS telemetry is best effort. Candidate-set comparison and
per-query allocation counting are explicitly reported as `unavailable`. Summary
rows also include baseline-relative columns such as
`cold_avg_speedup_vs_baseline`, `warm_avg_speedup_vs_baseline`,
`avg_world_access_ratio_vs_baseline`,
`avg_center_distance_ratio_vs_baseline`, and
`avg_raw_candidate_ratio_vs_baseline`.

### `candidate-verify`

Exact verification and TP/FP/FN accounting for external seed candidate TSVs.
This command is intended for fair comparisons with randstrobe, strobemer, and
spaced-seed tools that emit candidate reference-window IDs. It does not generate
seed candidates itself; measure that step separately, then add this command's
`verify_ms` to get the full seed-to-final-match mapping time. `truth_ms` is
quality-audit time and should not be included in mapper runtime.

**Required:** `--ref`, `--reads`, `--candidates`, `--out`, `--summary-out`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--ref` | required | Reference FASTA or literal sequence used to reconstruct candidate windows |
| `--reads` | required | Query FASTQ; `source_pos=` in headers is used by `--truth source` |
| `--candidates` | required | Candidate TSV with `read_id`, `tau`, `raw_candidate_count`, and `candidate_window_ids` columns |
| `--window` | `200` | Candidate/reference window length |
| `--stride` | `1` | Step between reference-window starts; `window_id * stride` gives the reference start |
| `--tolerance` | `2` | Fallback edit-distance threshold when a candidate row is missing; row-level `tau` is preferred |
| `--truth` | `source` | `source` checks the annotated origin window; `exhaustive` scans every reference window for a full small-run no-FN audit |
| `--out` | required | Per-query detail TSV |
| `--summary-out` | required | One-row aggregate TSV |

All reported matches are bounded exact edit-distance verified against the query
sequence. Source truth mode can count a verified non-origin window as FP;
exhaustive truth mode uses the same exact verifier over all windows, so verified
candidate FPs should be zero unless the inputs are inconsistent.

Detail output columns:
`read_id`, `tau`, `raw_candidate_count`, `verified_match_count`,
`truth_match_count`, `tp_count`, `fp_count`, `fn_count`,
`verified_window_ids`, `truth_window_ids`.

Summary output columns:
`query_count`, `raw_candidate_count`, `verified_match_count`,
`truth_match_count`, `tp_count`, `fp_count`, `fn_count`, `verify_ms`,
`truth_ms`.

### `boundary`

Builds one in-memory index from reference windows of fixed length `--length` and sweeps a full `error_rate × tolerance_rate` grid without rebuilding the index for each cell. This command is intended for capability-boundary exploration on long reference slices such as `chr1_subset`.

**Required:** `--ref`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--length` | `250` | Fixed window/query length; current implementation only accepts `250` |
| `--error-rates` | `0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20` | Comma-separated substitution error rates; each rate becomes `round(rate * 250)` edits |
| `--tolerance-rates` | `0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20` | Comma-separated tolerance rates; each rate becomes `round(rate * 250)` edit distance |
| `--queries-per-cell` | `200` | Number of mutated queries evaluated for each `(error_rate, tolerance_rate)` cell |
| `--stride-mode` | `sparse` | `sparse` uses stride `250`; `dense` uses stride `62` |
| `--seed` | `42` | Random seed used for query sampling and mutation |
| `--out` | *(none)* | Output TSV path for the aggregated boundary table |

`boundary` currently uses substitution-only mutations and, for each cell,
additionally samples up to 50 queries for `brute_force` agreement checks. This
experiment command builds and reuses an in-memory index within the run; it does
not use the persisted-index path.

### `layer-radius-experiment`

Builds one index per `(L, r_leaf, alpha)` combination, reuses a fixed query set across the full sweep, and writes one CSV row per query with search-cost instrumentation only.

**Required:** `--ref`

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--length` | `250` | Reference window and query length |
| `--tolerance` | `2` | Edit-distance threshold used during search |
| `--query-edits` | `--tolerance` | Number of substitution edits applied when generating the fixed query set |
| `--queries-per-cell` | `200` | Number of fixed queries reused across every `(L, r_leaf, alpha)` combination |
| `--stride` | *(unset)* | Explicit reference-window stride; if set, overrides `--stride-mode` |
| `--stride-mode` | `sparse` | `sparse` uses stride `length`; `dense` uses stride `length / 4` |
| `--seed` | `42` | Seed for fixed query generation |
| `--L-values` | `2,3,4,5` | Comma-separated primary-layer counts |
| `--r-leaf-values` | `4,8,12` | Comma-separated finest-layer radii |
| `--alpha-values` | `0.5,0.7` | Comma-separated geometric decay factors in `(0,1)` |
| `--out` | `layer_radius_search_stats.csv` | Output CSV path |

Each primary-layer radius schedule is generated geometrically from `(L, r_leaf, alpha)` and written to the CSV as a pipe-delimited string such as `64|32|16|8`.

## TSV columns

**`run`:**  
`query_id`, `hit_id`, `distance`, `ref_positions`, `read_id`, `read_len`, `ref_id`, `strand`, `query_start`, `reference_start`, `aligned_length`, `score`, `edit_distance`, `query_fragment`, `reference_fragment`, `bwt_start`, `bwt_end`

**`benchmark`** adds:  
`dist_calcs`, `leaf_verify_count`, `candidate_count_for_prune`, `beacon_prune_count`,
`mbb_filter_mode`, `mbb_scan_child_checks`, `mbb_rect_index_queries`,
`mbb_rect_candidate_children`, `mbb_rect_fallback_count`,
`mbb_surviving_child_count`, `mbb_scalar_checks`, `mbb_simd_batches`,
`mbb_simd_fallbacks`, `best_first_invoked_count`,
`best_first_reordered_count`, `best_first_bound_candidate_count`,
`child_safe_bound_pruned_count`, `center_distance_calls_after_mbb`,
`frontier_max_size`, `frontier_total_pushed`,
`leaf_beacon_scalar_checks`, `leaf_beacon_simd_batches`,
`leaf_beacon_simd_fallbacks`,
`search_qgram_prefilter_enabled`, `search_qgram_q`,
`search_qgram_signature_build_count`, `search_qgram_signature_missing_count`,
`search_qgram_checks`, `search_qgram_pruned_children`,
`search_qgram_passed_children`, `center_distance_calls_before_qgram`,
`center_distance_calls_after_qgram`, `qgram_prune_ratio`, `result_count`,
`avg_mbb_candidates_per_parent`,
`avg_center_distance_calls_per_query`, `query_time_ms`

**`query-benchmark` detail:**  
`query_id`, `query_class`, `profile`, `profile_rank`, `sample_kind`, `iteration`,
`first_profile`, `latency_ms`, `result_count`, `brute_force_result_count`,
`result_equal`, `no_fn`, `world_access_count`, `node_access_count`,
`edge_access_count`, `mbb_checks`, `mbb_survivors`, `mbb_scalar_checks`,
`mbb_simd_batches`, `mbb_simd_fallbacks`, `best_first_invoked_count`,
`best_first_reordered_count`, `best_first_bound_candidate_count`,
`child_safe_bound_pruned_count`, `planner_strategy_router_count`,
`planner_decision_ms`, `qgram_checks`,
`center_exact_distance_calls`, `leaf_beacon_checks`,
`leaf_beacon_scalar_checks`, `leaf_beacon_simd_batches`,
`leaf_beacon_simd_fallbacks`, `frontier_max_size`, `frontier_total_pushed`,
`leaf_exact_distance_calls`, `visited_checks`, `visited_hits`,
`candidate_count`, `verified_candidate_count`,
`actual_envelope_k1`, `actual_envelope_k2`, `actual_envelope_k4`,
`frontier_oracle_envelope_k1`, `frontier_oracle_envelope_k2`,
`frontier_oracle_envelope_k4`, `true_path_oracle_envelope_k1`,
`true_path_oracle_envelope_k2`, `true_path_oracle_envelope_k4`,
`global_oracle_envelope_k1`, `global_oracle_envelope_k2`,
`global_oracle_envelope_k4`, `random_envelope_k1`, `random_envelope_k2`,
`random_envelope_k4`, `actual_nearest_anchor_dist`,
`frontier_oracle_nearest_anchor_dist`,
`true_path_oracle_nearest_anchor_dist`,
`global_oracle_nearest_anchor_dist`, `random_nearest_anchor_dist`,
`global_oracle_gap_vs_actual_k1`, `global_oracle_gap_vs_actual_k2`,
`global_oracle_gap_vs_actual_k4`, `global_oracle_gap_vs_frontier_k1`,
`global_oracle_gap_vs_frontier_k2`, `global_oracle_gap_vs_frontier_k4`

The summary TSV reports cold/warm average, p50, p95, and p99 latency; query,
sample, result, equality-failure, and false-negative totals; baseline-relative
speedup/work-ratio columns; and average logical counters for each
query-class/profile pair plus `all`. With `--proximal-oracle 1`, it also
reports mean actual/frontier/true-path/global/random envelopes for k1/k2/k4 and
fractions where the global oracle envelope is materially better than actual or
frontier anchors.

`center_distance_calls_after_mbb` is retained as a compatibility alias for
`center_distance_calls_before_qgram`. The actual bounded center edit-distance
calls are reported by `center_distance_calls_after_qgram`.

`candidate_count_for_prune` and `beacon_prune_count` include both hierarchy-level MBB pruning and finest-layer leaf-beacon refinement.

**`map150`:**
`query_id`, `hit_id`, `distance`, `ref_id`, `strand`, `query_start`, `reference_start`, `aligned_length`, `score`, `edit_distance`, `query_fragment`, `reference_fragment`, `bwt_start`, `bwt_end`, `dist_calcs`, `leaf_verify_count`, `candidate_count_for_prune`, `beacon_prune_count`

For `map150 --locator refpos`, `bwt_start` and `bwt_end` are `-1`. With the optional SeqAn locator they represent the half-open suffix-array interval for the matched 150-mer leaf, not genomic coordinates; mapping uses that stored interval as the occurrence lookup handle.

**`boundary`:**  
`length`, `stride_mode`, `num_index_seqs`, `error_rate`, `error_edits`, `tolerance_rate`, `tolerance_edits`, `query_count`, `source_recovery_rate`, `any_hit_rate`, `avg_hit_count`, `avg_dist_calcs`, `avg_leaf_verify_count`, `avg_candidate_count_for_prune`, `avg_beacon_prune_count`, `avg_pruning_rate`, `bf_sample_count`, `bf_source_recovery_rate`, `bf_agreement_rate`, `bf_source_mismatch_count`

**`build-scale`:**
`prefix_len`, `window_count`, `invalid_window_count`, `unique_count`, `world_node_count`,
`finest_world_count`, `total_build_ms`, `phase0_dedup_ms`,
`phase1_sketch_ms`, `phase2_rebinding_ms`, `phase2_index_build_ms`,
`phase2_candidate_query_ms`, `phase2_exact_verify_ms`,
`phase2_distance_batches`,
`phase2_edge_insert_ms`, `phase3_mbb_ms`,
`phase3_collect_beacons_ms`, `phase3_collapse_children_ms`,
`phase3_child_mbb_distance_ms`, `phase3_rect_index_build_ms`,
`phase4_attach_ms`, `leaf_index_build_ms`, `leaf_candidate_query_ms`,
`leaf_exact_verify_ms`, `leaf_tuple_emit_ms`, `leaf_tuple_merge_sort_ms`,
`leaf_populate_ms`, `leaf_beacon_distance_ms`, `assign_ids_ms`,
`graph_view_ms`, `phase2_total_possible_pairs`, `phase2_candidate_pairs`,
`phase2_exact_distance_calls`, `phase2_edges_added`,
`leaf_total_possible_pairs`, `leaf_candidate_pairs`,
`leaf_exact_distance_calls`, `leaf_attachments_added`,
`phase2_full_scan_fallback_count`, `leaf_full_scan_fallback_count`,
`phase2_seed_candidate_pairs_before_length_filter`,
`phase2_seed_length_pruned_candidates`,
`phase2_pigeonhole_early_abort_count`,
`phase2_range_final_candidate_pairs`,
`leaf_seed_candidate_pairs_before_length_filter`,
`leaf_seed_length_pruned_candidates`, `leaf_pigeonhole_early_abort_count`,
`leaf_range_final_candidate_pairs`,
`leaf_attach_direction_used`, `range_candidate_mode`, `qgram_q`,
`phase2_candidate_query_worker_ms`, `phase2_exact_verify_worker_ms`

`phase2_rebinding_ms`, `phase2_index_build_ms`, and
`phase2_edge_insert_ms` are wall-clock timings. The Phase 2 worker fields are
per-thread accumulated query and exact-verification time for parallel indexed
rebinding.

**`layer-radius-experiment`:**  
`dataset`, `query_id`, `query_length`, `L`, `r_leaf`, `alpha`, `radius_schedule`, `query_time_ms`, `world_access_count`, `node_access_count`, `edge_access_count`, `anchor_distance_count`, `bound_check_count`, `candidate_count`, `candidate_verify_count`

## Standalone test binaries

| Binary | Purpose |
| ------ | ------- |
| `test_recall` | Randomized recall check: adaptive vs brute force |
| `test_distance_bound` | Distance-bound checks across search modes |
| `test_hierarchy_config` | Hierarchy-config validation for multi-primary-layer builds |
| `test_search_stats_bin` | Radius-schedule and search-cost instrumentation checks |
| `test_map150_recall` | Fixed-150bp mapper recall, strand, duplicate, and verifier checks |
| `test_bounded_edit_distance` | Banded thresholded distance vs full Levenshtein |
| `test_bounded_myers_bin` | Optional bounded Myers backend vs full Levenshtein and DP fallback |
| `test_qgram_filter` | Q-gram counts, L1 bound, ambiguous bases, and index no-false-negative checks |
| `test_range_join` | Pigeonhole/q-gram/hybrid no-false-negative and verified-pair checks |
| `test_build_range_equivalence` | Full vs q-gram/hybrid/auto construction and search-result equivalence |
| `test_build_timing_stats` | Construction timing field and summary smoke checks |
| `test_build_scale_smoke` | `build-scale` CSV timing smoke check |
| `test_sharded_index_bin` | Monolithic/sharded window and coordinate equivalence, restart/repair, and adaptive no-FN checks |
| `test_build_progress_bin` | Timestamped build progress formatting, forced boundaries, and periodic heartbeat |
| `test_mbb_rect_index` | Exact rectangle intersection and randomized naive-scan equivalence |
| `test_mbb_filter_equivalence` | Adaptive scan/rect result equality, recall, counters, and fallback |
| `test_search_qgram_prefilter` | Search q-gram on/off, scan/rect, ambiguous-base fallback, containment, and center-call reduction checks |
| `test_search_distance_mode_bin` | Adaptive `dp` vs `myers` bounded center-distance equivalence |
| `test_query_benchmark_gate` | Deterministic query classes, dual-profile output, exact result equality, and no-FN gate |
| `test_phase2_distance_verifier_bin` | CPU Phase2 exact verifier batch equivalence |

Build with `make test_recall` / `make test_distance_bound`.
