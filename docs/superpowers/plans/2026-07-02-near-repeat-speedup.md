# Near-Repeat Query Speedup Plan

## Goal

Make NavigaMer faster than strobemer/randstrobe and spaced-seed baselines on the E. coli 1.1M source-sorted near-repeat workload while preserving exact edit-distance verification and no false negatives relative to the baseline adaptive/exhaustive NavigaMer query path.

The target workload is not the already-winning duplicated-read case. This plan targets adjacent or source-sorted similar reads where the final answer cannot simply be copied, but traversal evidence from neighboring queries can be reused safely.

## Current Evidence

- Duplicated real-read locality already wins on E. coli 1.1M:
  - NavigaMer optimized: about 0.50-1.46 s/query depending on duplicate factor/order.
  - NavigaMer baseline: about 13.8 s/query.
  - No FN / mismatch count: 0.
- Source-sorted near-repeat workloads currently show correctness but little speedup:
  - `source_sorted_stride1`: about 2619.6 ms baseline vs 2580.1 ms optimized.
  - `source_sorted_mutated_tau5`: about 2957.3 ms baseline vs 2930.6 ms optimized.
  - No FN / mismatch count: 0 for tau5.
- Current reuse is mostly ordering/cache warmth. It does not yet safely eliminate enough center-distance or world traversal work.

## Core Idea

Use neighboring query similarity to derive correctness-preserving lower bounds by triangle inequality.

For a previous query `q0`, current query `q1`, and a world/center sequence `c`:

```text
d(q1, c) >= max(0, d(q0, c) - d(q0, q1))
```
If the stored previous distance `d(q0, c)` is known and:

```text
max(0, d(q0, c) - d(q0, q1)) > tolerance + world_radius
```

then that world/child can be safely skipped for `q1` without computing `d(q1, c)`.

This is the important distinction from the current path reuse:

- Existing reuse mostly reorders candidate traversal.
- V2 should reuse previous distance evidence to safely prune work.
- Exact verification still runs for every reported match.
- Any unproven complement falls back to normal traversal.

## Implementation Tasks

### 1. Add Failing Safety/Performance Tests First

Edit:

- `navigamer_cpp/src/test_path_reuse_no_false_negative.cpp`
- `navigamer_cpp/src/test_query_benchmark_gate.cpp`

Add a near-repeat reuse regression that builds a high-fanout synthetic index, generates adjacent mutated queries, and compares:

- baseline adaptive query result IDs
- optimized near-repeat reuse query result IDs
- brute-force/exhaustive result IDs where feasible

Required assertions:

```cpp
assert(sorted_baseline_ids == sorted_optimized_ids);
assert(stats.false_negative_count == 0);
assert(stats.near_query_reuse_hit_count > 0);
assert(stats.near_query_triangle_pruned_count > 0);
assert(optimized_center_distance_count < baseline_center_distance_count);
```

For benchmark-gate coverage, add a compact `source_sorted_mutated_tau5` or `source_sorted_stride1` run and assert the output TSV contains the new counters:

- `near_query_triangle_pruned_count`
- `near_query_center_distance_reused_count`
- `near_query_bound_fallback_count`
- `near_query_direct_verify_count`

Validation command:

```bash
cd navigamer_cpp
make test_path_reuse_no_false_negative test_query_benchmark_gate -j
./test_path_reuse_no_false_negative
./test_query_benchmark_gate
```

### 2. Add Near-Query Reuse State

Edit:

- `navigamer_cpp/include/search_engine.hpp`
- `navigamer_cpp/src/search_engine.cpp`

Add a small, bounded cache owned by the query batch/search context, not global mutable state.

Proposed structures:

```cpp
struct NearQueryReuseEntry {
    std::string query;
    std::vector<std::pair<int, int>> world_center_distances;
    std::vector<int> productive_world_ids;
    std::vector<int> verified_match_ids;
    int tolerance = 0;
};

struct NearQueryReuseConfig {
    bool enabled = false;
    int max_neighbor_edit_distance = 8;
    int max_entries = 2;
    double min_qgram_jaccard = 0.35;
};
```

Store only distances that were already computed during normal traversal. Do not introduce approximate-only match reporting.

Add `SearchStats` counters:

```cpp
long long near_query_reuse_attempt_count = 0;
long long near_query_reuse_hit_count = 0;
long long near_query_triangle_pruned_count = 0;
long long near_query_center_distance_reused_count = 0;
long long near_query_bound_fallback_count = 0;
long long near_query_direct_verify_count = 0;
```

### 3. Compute Neighbor Similarity Lazily

Edit:

- `navigamer_cpp/src/search_engine.cpp`
- optionally `navigamer_cpp/include/query_router.hpp` if existing qgram helpers should be reused

For each query in a sorted/local batch:

1. Compare only with the previous one or two queries.
2. Build qgram signatures lazily.
3. Compute exact edit distance between neighboring queries only if cheap qgram similarity passes.
4. Enable near-query bound reuse only when:

```text
neighbor_edit_distance <= min(tolerance, max_neighbor_edit_distance)
```

or a configured looser gate is explicitly enabled for diagnostics.

This prevents low-locality random batches from paying reuse overhead.

### 4. Implement Safe Triangle-Bound Pruning

Edit:

- `navigamer_cpp/src/search_engine.cpp`

Inside adaptive traversal, before computing a center distance for a world/child:

1. Look up the previous known distance to the same world/center.
2. Compute:

```cpp
int lower_bound = std::max(0, previous_center_distance - neighbor_query_distance);
```

3. If:

```cpp
lower_bound > tolerance + world.radius
```

skip this world/child safely and increment:

```cpp
near_query_triangle_pruned_count++;
near_query_center_distance_reused_count++;
```

4. Otherwise fall back to normal exact center-distance computation and increment:

```cpp
near_query_bound_fallback_count++;
```

No result can be dropped unless the lower-bound inequality proves it cannot match.

### 5. Add Direct Verify-First Warm Start

Edit:

- `navigamer_cpp/src/search_engine.cpp`

Before traversal, exact-verify previously productive leaves and previous verified matches against the current query.

Rules:

- Verification must use the same exact edit-distance threshold as normal query.
- Verified matches can be emitted immediately.
- This does not replace traversal unless the planner later proves the remaining space is safely pruned.
- Deduplicate emitted sequence IDs before returning.

Expected counters:

```cpp
near_query_direct_verify_count++;
productive_world_reuse_hit_count++;
```

### 6. Reuse Child Shortlists Only As Safe Hints

Edit:

- `navigamer_cpp/src/search_engine.cpp`

Use previous child shortlists and safe-child candidate sets as ordering hints first.

Then apply triangle-bound pruning to the complement:

- If complement children are all safely pruned, do not enumerate them.
- If any complement child lacks a previous distance or bound proof, fall back to baseline enumeration for that child.

This keeps no-FN intact while allowing real child enumeration reduction when neighboring queries are close.

### 7. Teach The Planner When To Use Near-Query Reuse

Edit:

- `navigamer_cpp/src/search_engine.cpp`
- `navigamer_cpp/include/search_engine.hpp`

Planner policy:

```text
low fanout                         -> baseline adaptive
random/nonlocal batch              -> no near-query reuse
near query + high fanout           -> triangle-bound reuse
near query + previous productive   -> direct verify-first
safe child router selective        -> safe child router
safe child router high fallback    -> disable router for that parent/query
```

Add a diagnostic counter for planner decisions if not already represented:

- `planner_near_reuse_enabled_count`
- `planner_near_reuse_disabled_count`

### 8. Extend Locality Benchmark Output

Edit:

- `navigamer_cpp/include/query_benchmark.hpp`
- `navigamer_cpp/src/query_benchmark.cpp`

Add TSV columns:

- `near_query_triangle_pruned_count`
- `near_query_center_distance_reused_count`
- `near_query_bound_fallback_count`
- `near_query_direct_verify_count`
- `center_distance_reduction`
- `world_access_reduction`
- `p95_speedup`

Keep existing column order stable where possible by appending new columns.

### 9. Run Correctness Gates

Required before any performance claim:

```bash
cd navigamer_cpp
make -j
make test_recall test_distance_bound test_path_reuse_no_false_negative test_query_benchmark_gate -j
./test_recall
./test_distance_bound
./test_path_reuse_no_false_negative
./test_query_benchmark_gate
./navigamer demo --size 200
```

Pass condition:

- all tests pass
- no optimized result set differs from baseline/adaptive/exhaustive references
- false-negative and mismatch columns remain 0 in benchmark TSVs used for claims

### 10. Run E. coli 1.1M Near-Repeat Benchmarks

Prefer the existing copied index:

```bash
NAVIDX=/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/onepoint1m_builds/20260702_w150_s1_coarser_s2w_omp4/navigamer_ecoli_1p1m_w150_s1_coarser_s2w_omp4.navidx
REF=/home/luting/projects/AnchorMapping/NavigaMer/data/ecoli/ecoli_1p1m.fa

cd navigamer_cpp
OMP_NUM_THREADS=4 ./navigamer locality-benchmark \
  --index "$NAVIDX" \
  --ref "$REF" \
  --out /tmp/navigamer_1p1m_near_repeat_v3_q8.tsv \
  --query-count 8 \
  --query-length 150 \
  --query-edits 5 \
  --tolerance 5 \
  --scenarios source-sorted-stride1,source-sorted-mutated-tau5 \
  --locality-profiles baseline,optimized \
  --batch-schedules source-oracle,random

OMP_NUM_THREADS=4 ./navigamer locality-benchmark \
  --index "$NAVIDX" \
  --ref "$REF" \
  --out /tmp/navigamer_1p1m_near_repeat_v3_tau8_q8.tsv \
  --query-count 8 \
  --query-length 150 \
  --query-edits 8 \
  --tolerance 8 \
  --scenarios source-sorted-mutated-tau8 \
  --locality-profiles baseline,optimized \
  --batch-schedules source-oracle
```

Minimum report fields:

- mean query ms
- p95 query ms
- no-FN / mismatch count
- center-distance reduction
- world-access reduction
- near-query triangle-pruned count
- direct verify count
- speedup versus NavigaMer baseline

### 11. Compare Against Existing Candidate Baselines

Use existing E. coli baseline indexes if present:

```bash
BASE=/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/full_compare_20260629_w150_tau2/run/compare_tau2
TOOL=/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/full_compare_20260629_w150_tau2/bin/candidate_tool

find "$BASE/indexes" -maxdepth 1 -type f | sort
test -x "$TOOL"
```

For the final comparison table, include at least:

- NavigaMer baseline adaptive
- NavigaMer optimized near-repeat V3
- randstrobe / strobemer candidate baseline
- spaced seed candidate baseline

The table must separate:

- candidate generation time if baselines only produce candidates
- exact verification time if available
- total time if both are available
- recall/no-FN status under the same edit threshold

Do not claim a total-speed win over a method unless the comparison includes equivalent exact verification or clearly labels candidate-only timing.

### 12. Success Criteria

Primary success:

- E. coli 1.1M `source_sorted_mutated_tau5` or `source_sorted_stride1`
- optimized NavigaMer no-FN/mismatch = 0
- optimized NavigaMer p95 query time faster than randstrobe/strobemer and spaced-seed equivalent exact-verified timing

Secondary acceptable success:

- If external baselines remain candidate-only, optimized NavigaMer must beat their candidate-only time only when clearly labeled as a scoped candidate-generation comparison.
- If not faster yet, the run must show a concrete bottleneck:
  - triangle prune too low
  - previous-distance coverage too low
  - fanout too low
  - exact verification dominates
  - router candidate ratio too high

## Automation Plan

After the current V2 autorun finishes, start a V3 development autorun with this prompt:

```text
Implement the near-repeat triangle-bound reuse plan in docs/superpowers/plans/2026-07-02-near-repeat-speedup.md. Preserve exact verification and no-FN. Add tests first, then implement safe triangle-bound pruning and direct verify-first reuse, then run C++ correctness tests and E. coli 1.1M source-sorted near-repeat benchmarks. Produce a table comparing NavigaMer baseline, NavigaMer optimized, randstrobe/strobemer, and spaced-seed timings. Do not claim a win unless exact verification/no-FN conditions are met.
```

Suggested wait wrapper:

```bash
V2_PID=3556101
while kill -0 "$V2_PID" 2>/dev/null; do
  sleep 60
done

setsid nohup codex exec \
  --model gpt-5.5 \
  -c model_reasoning_effort=\"high\" \
  "$(cat <<'PROMPT'
Implement the near-repeat triangle-bound reuse plan in docs/superpowers/plans/2026-07-02-near-repeat-speedup.md. Preserve exact verification and no-FN. Add tests first, then implement safe triangle-bound pruning and direct verify-first reuse, then run C++ correctness tests and E. coli 1.1M source-sorted near-repeat benchmarks. Produce a table comparing NavigaMer baseline, NavigaMer optimized, randstrobe/strobemer, and spaced-seed timings. Do not claim a win unless exact verification/no-FN conditions are met.
PROMPT
)" \
  > .codex_logs/near_repeat_speedup_v3_launcher_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
