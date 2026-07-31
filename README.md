# NavigaMer

[![Build](https://img.shields.io/badge/build-passing-success)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)

## Overview

**NavigaMer** (*Multilateration-Based Indexing and Navigation for Error-Tolerant Read Mapping*) is a multi-tiered indexer that formulates read mapping as **geometric localization in a coordinate-free metric space** under **edit distance**. Rather than embedding sequences in a continuous sketch space (with embedding distortion) or relying only on fixed seeds under high mutation rates, NavigaMer uses **beacon-mediated multilateration** and **triangle-inequality pruning** over a hierarchy of metric “worlds.” The **adaptive** search aims for **zero false negatives** (perfect recall relative to the indexed sequence set) within a user-specified edit-distance threshold, while pruning candidates that cannot contain a match.

## Methodology ↔ Code (for better code readability)

| Concept (paper) | Implementation |
| --------------- | -------------- |
| **Extended world hierarchy (sketch)** | `BioGeometryIndexBuilder::phase1_build_extended_sketch()` — `navigamer_cpp/src/index_builder.cpp` |
| **DAG topology & overlap binding** | `BioGeometryIndexBuilder::phase2_inter_tier_rebinding()` — same file |
| **Beacon extraction & tier collapse + MBBs** | `BioGeometryIndexBuilder::phase3_collapse_and_compute_mbb()` — one auxiliary tier between each adjacent pair of primary layers is collapsed into **Metric Bounding Boxes (MBBs)** for consistency checks |
| **Leaf beacon refinement** | `BioGeometryIndexBuilder::attach_leaves()` precomputes finest-layer leaf-to-beacon distances; `BioGeometrySearchEngine::verify_leaf_candidates()` applies the final local beacon sieve before exact verification |
| **Hierarchical multilateration search** | `BioGeometrySearchEngine::search_adaptive()` — `navigamer_cpp/src/search_engine.cpp` (MBB-based pruning plus finest-layer leaf refinement via triangle inequality) |

The finalized index uses `SequenceStore`, `WorldNodeRecord`, and flat ID/data
arrays declared in `navigamer_cpp/include/index_builder.hpp`. Construction also
uses an array of `BuildWorldNodeRecord` values and integer relationships; it
does not allocate a temporary `WorldNode` pointer graph. Edit distance is in
`navigamer_cpp/src/tools.cpp`; FASTA/FASTQ/TSV I/O is in
`navigamer_cpp/src/io_utils.cpp`.

Reference-window builds keep one reference string plus a representative source
offset per unique window. Distance, q-gram, seed-index, and range-join code read
non-owning sequence views, so the index does not allocate one sequence string
per reference window. Persisted format version 4 stores this reference-backed
representation directly.

## Installation

**Requirements:** Linux, **C++17**, **OpenMP**, optional **CMake >= 3.14**.

```bash
cd navigamer_cpp
make -j
# or:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Binary: `navigamer_cpp/navigamer` (or `navigamer_cpp/build/navigamer` with CMake).

**Python** is only needed for notebooks and baseline/reproducibility scripts; there is no separate Python implementation path for the paper algorithm:

```bash
pip install -r reproducibility/requirements.txt
```

## Quick start

```bash
cd navigamer_cpp
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mode adaptive
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mode adaptive --query-profile 1
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mode adaptive --query-profile 1 --path-reuse 1
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --router-hints 1 --router-hint-qgram-q 3 --router-hint-minimizer-k 3 --router-hint-minimizer-w 5
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --mbb-filter-mode rect --min-rect-index-fanout 2
./navigamer query --reads ACGTACGTACGTACGT --query ACGTACGTACGTACGT --tolerance 2 --search-qgram-prefilter on --search-qgram-q 5
./navigamer build --ref ref --reads ACGTACGTACGTACGT --index /tmp/navigamer.navidx
./navigamer query-index --index /tmp/navigamer.navidx --query ACGTACGTACGTACGT --tolerance 2
./navigamer demo --size 200 --range-candidate-mode hybrid --qgram-q 5
./navigamer demo --size 200
./navigamer demo --primary-radii 30,15,5
./navigamer build-scale --ref ../data/human/chr1_subset --window 250 --stride 1 --prefix-lengths 10000,50000 --out /tmp/build_scale.csv
./navigamer build-scale --ref ../data/human/chr1_subset --window 250 --stride 1 --prefix-lengths 50000 --index /tmp/reference.navidx --out /tmp/build_scale.csv
./navigamer query-benchmark --ref ACGTGCACTGATTGCATAGCTACGAAAAAAAAAAAACCCCGGGGCCCCCCCCCGGGCCCC --window 12 --stride 12 --query-length 12 --tolerance 1 --primary-radii 12,6,2 --min-rect-index-fanout 1 --mbb-filter-mode rect --search-qgram-prefilter on --search-qgram-q 3 --path-reuse 1 --router-hints 1 --query-benchmark-ablations 1 --proximal-oracle 1 --proximal-oracle-k 1,2,4 --warmup-iterations 1 --measured-iterations 2 --cold-cache-bytes 0 --out /tmp/query_detail.tsv --summary-out /tmp/query_summary.tsv --json-out /tmp/query_summary.json
```

Fixed-length 150 bp mapper path with final gap-aware verification:

```bash
cd navigamer_cpp
READ=$(printf 'ACGT%.0s' {1..37})AC
REF=TTTT${READ}GGGG
./navigamer map150 --ref "$REF" --reads "$READ" --tolerance 1 --out /tmp/navigamer_map150.tsv
```

Optional boundary sweep using the bundled small reference:

```bash
cd navigamer_cpp
./navigamer boundary --ref ../data/human/chr1_subset --length 250 --stride-mode sparse --queries-per-cell 1 --error-rates 0 --tolerance-rates 0 --out /tmp/navigamer_boundary.tsv
```

Layer/radius sweep for search-cost instrumentation:

```bash
cd navigamer_cpp
./navigamer layer-radius-experiment \
  --ref ../data/human/chr1_subset \
  --length 250 \
  --stride 1 \
  --tolerance 2 \
  --query-edits 2 \
  --queries-per-cell 50 \
  --L-values 2,3,4,5 \
  --r-leaf-values 4,8,12 \
  --alpha-values 0.5,0.7 \
  --out /tmp/layer_radius_search_stats.csv
```

Full CLI reference: [`navigamer_cpp/CLI_REFERENCE.md`](navigamer_cpp/CLI_REFERENCE.md). C++ layout and tests: [`navigamer_cpp/README.md`](navigamer_cpp/README.md).

## E. coli Comparison Workflow

The experiment-side unified comparison entrypoint lives at
`experiments/ecoli_1p1m/candidate_tool`.

```bash
cd experiments/ecoli_1p1m
make candidate_tool -j
./candidate_tool compare \
  --ref /path/to/reference.fa \
  --reads /path/to/reads.fq \
  --tau 2 \
  --window 150 \
  --stride 1 \
  --out-dir /tmp/ecoli_compare \
  [--rebuild] \
  [--tensor-top-k 64] \
  [--oracle on|off] \
  [--navigamer-bin ../../navigamer_cpp/navigamer] \
  [--navigamer-index ../../navigamer_cpp/.tmp_experiments/ecoli_1p1m.navidx]
```

`compare` materializes or reuses the baseline candidate indexes, runs the
baseline methods plus NavigaMer on the same reads, computes brute-force oracle
neighbors under edit distance when `--oracle on`, and writes `per_read.tsv` and
`summary.tsv` into `--out-dir`. With `--navigamer-index`, the NavigaMer bridge
loads that persisted `.navidx` once and calls `navigamer query-index-batch`;
without it, the bridge falls back to `navigamer benchmark` on reference windows.

The C++ `build` and `query` commands support explicit persisted indexes with
`--index <file>`. The v3 index file stores a manifest with input fingerprints
and construction parameters plus the canonical sequence, node, child, leaf,
beacon, MBB, and leaf-beacon arrays. Older pointer-graph index files must be
rebuilt. `query --index <file> --query <seq>` and
`query-index --index <file> --query <seq>` load the index directly. When
`query` is given both `--reads` and `--index`, it compares the requested build
manifest with the stored manifest and reuses the file only on an exact
signature match; otherwise it rebuilds and overwrites the index. Experiment
commands such as `run`, `benchmark`, `map150`, and `boundary` still build
in-memory indexes for their current workflows. The `boundary` command reuses a
single in-memory index across the full `error_rate × tolerance_rate` grid
inside one run.

`build-scale` can also persist its reference-window index with `--index <file>`
when exactly one prefix length is requested. Multiple prefixes with one index
path are rejected instead of overwriting the file. Builds emit timestamped
phase progress to stderr every 600 seconds by default; use
`--progress-interval-seconds N` to change the interval or `0` to disable
periodic heartbeats while retaining phase-boundary reports.

Indexed construction supports exact `auto`, `pigeonhole`, `qgram`, `hybrid`,
and `full` candidate modes. Q-gram filtering uses the necessary condition
`qgram_l1(a,b) <= 2*q*tau`; hybrid intersects the pigeonhole and q-gram safe
candidate supersets. Every surviving pair is still verified by bounded exact
edit distance before an edge or leaf attachment is added.

Phase1 sketch construction has separate helper thresholds for experiments:
`--phase1-metric-min-fanout` switches parent-local candidate groups from direct
scan to the metric helper, `--phase1-qgram-min-fanout` switches larger groups to
the q-gram helper, and `--phase1-qgram-max-touched` controls conservative
fallback when q-gram candidate expansion is too broad. These knobs do not skip
bounded exact verification.

Indexed leaf attachment directly verifies range candidates by default. The
optional `--leaf-qgram-postfilter on` applies a safe q-gram L1 condition before
bounded exact verification. It can reduce exact calls for unusually broad
candidate sets, but its signature-building overhead is slower on the default
prepared-DNA distance path. Either setting is no-false-negative: accepted links
are still determined by bounded exact edit distance.

Auto construction accepts pigeonhole candidates when their count is at most
`4096`; if the seed union grows beyond that threshold, it early-aborts the
pigeonhole collection and invokes the q-gram safe fallback. The legacy
`--auto-pigeonhole-max-ratio` flag is parsed for command compatibility but no
longer drives auto selection, so normal pigeonhole queries do not full-scan all
length-compatible targets just to compute a ratio.

Index construction now reports aggregate build timing to stderr. The timing
breakdown covers Phase0 deduplication, Phase1 sketch construction, Phase2
rebinding, Phase3 MBB computation, Phase4 leaf attachment, ID assignment,
graph-view flattening, and selected range-join, MBB, and leaf-attachment
substeps. Phase2 rebinding, index build, and edge insert timings are wall-clock
milliseconds; Phase2 candidate-query and exact-verify worker fields are
accumulated per-thread time. The `build-scale` command writes the same timing
breakdown and construction counters to CSV for multiple reference prefix
lengths.

Adaptive search supports `--mbb-filter-mode scan|rect` (default `scan`). The
`rect` mode uses an exact in-memory rectangle index over the existing per-child
MBB rows and falls back to the original scan whenever the index is unavailable
or inconsistent. `--min-rect-index-fanout` controls the build threshold and
defaults to `64`.

Flat adaptive traversal supports `--simd-mode auto|scalar|avx2|avx512`
(default `auto`) for child MBB rectangle filtering and leaf-beacon filtering.
Unsupported SIMD paths fall back to the scalar filter and preserve the same
survivor set.

Adaptive bounded child-center distance supports `--distance-mode dp|myers|edlib|auto`
(default `myers`). Myers supports ACGT inputs through 256bp shorter input
length and falls back to DP for unsupported inputs. `edlib` uses the vendored
Edlib bounded distance backend. `dp` remains the reference mode; `auto`
currently remains DP. Index construction separately supports
`--build-distance-mode dp|edlib|auto` and defaults to `edlib`.

Adaptive query profiling supports `--query-profile 0|1` (default `0`). When
enabled, `search_adaptive()` records per-query timing buckets such as
`query_total_ms`, `anchor_distance_ms`, `mbb_filter_ms`,
`center_distance_ms`, `leaf_collect_ms`, `leaf_verify_ms`, and
`result_dedup_ms`, along with counters including world access, anchor distance
calls, center distance calls, and raw candidate counts. The `benchmark` and
`query-benchmark` commands surface these values in TSV output and print query
time/world-access summaries to stderr.

Adaptive path reuse supports `--path-reuse 0|1` (default `1`). When enabled,
adaptive search keeps thread-local warm-start caches for exact per-parent
anchor-distance vectors on repeated queries and cached child shortlists keyed by
query-derived fingerprints. These caches affect ordering or exact memoization
only, never become a pruning reason, and record counters such as
`path_reuse_attempt_count`, `path_reuse_hit_count`,
`anchor_cache_hit_count`, `child_shortlist_reuse_hit_count`,
`safe_child_candidate_cache_hit_count`, and
`productive_world_reuse_hit_count`. Batch commands also group queries by
`source_pos=` FASTQ annotations when present, otherwise by a cheap query-derived
fingerprint, while preserving output order so related queries are more likely to
hit the same thread-local warm cache.

Adaptive local routing accepts `--local-router 0|1`,
`--local-router-max-anchors N`, `--local-router-max-children N`, and
`--local-router-score anchor-envelope`. This remains a `RouterHint` only: it
ranks child worlds after safe MBB filtering using parent-local beacon
envelopes, records invocation/shortlist/fallback counters in query and
benchmark output, and still preserves full fallback traversal plus exact final
verification.

Adaptive router hints accept `--router-hints 0|1`,
`--router-hint-qgram-q N`, `--router-hint-minimizer-k N`, and
`--router-hint-minimizer-w N`. When enabled, adaptive search builds parent-local
child-center range hints with q-gram/pigeonhole candidate supersets and then
uses q-gram plus minimizer signals only to prioritize post-MBB child worlds
before local-router and best-first stages. Query and benchmark output add
router-hint invocation, ranked-child, predicted-child, and fallback counters;
all non-predicted children still remain in the exact fallback order.

Adaptive safe child routing accepts `--safe-child-router 0|1`,
`--safe-child-router-min-fanout N`, `--safe-child-router-max-candidates N`,
`--safe-child-router-max-ratio R`,
`--safe-child-router-min-seed-len N`,
`--safe-child-router-mode auto|pigeonhole|qgram|mbb|full-fallback`, and
`--safe-child-router-validate 0|1`. Unlike `RouterHint`, it can reduce child
enumeration, but only by generating a safe parent-local candidate superset.
Range-helper modes query child centers by child-radius bucket at
`tolerance + child.radius`; MBB mode rejects a child only when the local
query-to-anchor distance is outside that child's MBB interval by more than
`tolerance`. Overly broad or unsafe cases fall back to full enumeration. All
candidates still pass bounded center distance and final exact leaf
verification.

Adaptive query planning accepts `--query-planner 0|1`,
`--planner-router-min-fanout N`, and
`--planner-safe-child-router-min-fanout N`. The current planner is
conservative: it records a per-query strategy and can skip optional
q-gram/router ordering work on low-fanout indexes, but it never skips MBB
filtering, bounded center verification, or final exact leaf verification.
TSV/JSON outputs include planner strategy counters and `planner_decision_ms`.

`query-benchmark` can additionally enable proximal-anchor oracle diagnostics
with `--proximal-oracle 1` and `--proximal-oracle-k 1,2,4`. These diagnostics
compare the observed anchor/frontier choices with true-path, global nearest,
and deterministic random anchor envelopes. They are instrumentation only:
search traversal, pruning, and final exact verification are unchanged.

Adaptive safe best-first ordering accepts `--best-first 0|1` (default `0`).
When enabled, adaptive search reorders post-MBB child-world survivors using
conservative parent-local MBB lower bounds and tighter envelope spans before
bounded center verification. This path records queue/bound counters such as
`best_first_invoked_count`, `best_first_bound_candidate_count`,
`child_safe_bound_pruned_count`, `frontier_max_size`, and
`frontier_total_pushed`, and may prune only through those conservative
`SafeBound` lower bounds.

Query-side optimization work follows this safety contract:

- `RouterHint`: may be incomplete or wrong; affects only ordering, warm-start,
  or candidate priority.
- `SafeCandidateRouter`: may reduce enumeration only with a safe candidate
  superset and must full-fallback otherwise.
- `SafeBound`: may prune only when impossibility is conservatively proven.
- `ExactVerifier`: only exact edit distance `<= tolerance` may produce a hit.

Adaptive child-world traversal also supports the optional
`--search-prefetch on`, which issues best-effort lookahead prefetch hints for
array child/MBB/leaf data without changing pruning or verification semantics.
It is experimental and intended for locality A/B measurements. It also supports
`query-index-batch --path-trace-out <tsv>` for locality diagnostics: the trace
TSV records per-input-read world node IDs evaluated, leaf sequence IDs exactly
verified, and adjacent-input-read path-overlap/Jaccard fields. The main hit TSV
also reports adaptive path diagnostics via `query_path_class`,
`path_contained_step_count`, `path_overlap_step_count`, and
`path_uncovered_step_count`.

Adaptive search also supports
the optional
`--search-qgram-prefilter on` with independent `--search-qgram-q` (default
`5`). After MBB filtering, it safely rejects a child center only when
`qgram_l1(query, center) > 2*q*(child.radius+tolerance)`. Every passing child
still receives bounded exact edit-distance verification. The default is
`off`; unsupported q values or non-ACGT sequences conservatively fall back to
no pruning.

`query-benchmark` is a deterministic correctness and latency gate for adaptive
search. It compares a fixed baseline (`scan`, scalar MBB filtering, search
q-gram off, `dp` distance mode), the optimized profile selected by
adaptive-search flags, and exact brute-force IDs across six query classes. With
`--query-benchmark-ablations 1`, it also adds one ablation profile for each
enabled PR0-PR4 query-side stage (`search_qgram`, `router_hints`,
`local_router`, `best_first`, `safe_child_router`, `path_reuse`,
`query_planner`) by disabling that stage alone while
keeping the rest of the optimized stack intact. The summary TSV and JSON now
include baseline-relative speedup and work-ratio columns so each ablation can
be compared directly against the fixed baseline. With `--proximal-oracle 1`,
detail rows also include actual/frontier/true-path/global/random envelope
columns for k1/k2/k4, nearest-anchor distances, and oracle gap fields; summary
rows add mean envelopes plus fractions where the global oracle is materially
better than observed actual/frontier anchors. The command writes detailed TSV,
aggregate TSV, and JSON output and returns exit status `2` on any result
mismatch or false negative.

For persisted-index query-only measurements, `locality-benchmark` loads the
index once and emits same-template, nearby-window, random, repeat,
batch-locality, and oracle-oriented query streams. `query-locality-benchmark`
is an alias with the same flags. `--scenarios low-fanout,high-fanout,repeat,
batch-locality,oracle,all` selects benchmark presets while the summary reports
the actual loaded-index fanout distribution plus router/local-router/safe-child
router invocation ratios, path-reuse hit ratio, and aggregate reuse counters
including `anchor_cache_hit_count`, `child_shortlist_cache_hit_count`,
`safe_child_candidate_cache_hit_count`, and
`productive_world_reuse_hit_count`. `--batch-schedules` defaults to
source-oracle for locality benchmarks and can compare internal query orders such
as original, random, minimizer, qgram-signature, router-signature, and
source-oracle; source-oracle is reported only as an upper-bound diagnostic.
`--query-fastq-out` exports the generated query stream with `source_pos=` read
header annotations so external candidate baselines can be run on the same
reads. `candidate-verify` consumes those FASTQ records plus external seed
candidate TSVs, exactly verifies every candidate by edit distance, and reports
TP/FP/FN. Its `verify_ms` is the extension/verification part to add to external
candidate-generation time; `truth_ms` is evaluation-only audit work.
Leaf-stage near-query reuse is query-only: it caches bounded exact leaf
distances from the previous nearby query and uses triangle lower bounds to skip
provably impossible leaf verifications without changing or rebuilding the
persisted index.

`query-locality-report --ref <fasta|sequence> --out-dir <dir>` wraps the
persisted locality benchmark and writes `summary.tsv`, `summary.json`, and
`report.md`. If `--index` is omitted, it reuses a manifest-compatible
`query_locality.navidx` in the report directory or builds it when missing or
stale. The same locality profiles,
scenario presets, datasets, and batch schedules are supported.

The default CLI path still uses the legacy three primary layers (`LW/MW/SW`) via `--r-lw`, `--r-mw`, and `--r-sw`, but the C++ implementation now also supports any number of primary layers `K >= 2` through `--primary-radii coarse,...,fine`. One auxiliary tier is inserted automatically between each adjacent pair of primary layers during index construction and collapsed away before query-time navigation.

## Tests

```bash
cd navigamer_cpp
make test_recall test_distance_bound
./test_recall
./test_distance_bound
make test_build_timing_stats test_build_scale_smoke
./test_build_timing_stats
./test_build_scale_smoke
```

## Repository layout

| Path | Role |
| ---- | ---- |
| `navigamer_cpp/` | C++ v8 reference implementation and `navigamer` CLI |
| `data/human/chr1_subset` | Small reference sequence used by README examples |
| `methods/` | Comparative baselines, experiment notebooks, and plotting/evaluation workflows |
| `reproducibility/` | Optional Python dependencies (`requirements.txt`) for notebooks and baseline workflows |
