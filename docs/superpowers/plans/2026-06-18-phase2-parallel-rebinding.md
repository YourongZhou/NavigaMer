# Phase 2 Parallel Rebinding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Phase 2 rebinding wall time without changing the edge set, leaf attachment semantics, search semantics, or no-FN guarantees.

**Architecture:** Make `ExactRangeJoinIndex` query operations read-only and thread-safe by moving mutable query state into per-thread workspaces and prebuilding lazy seed postings before parallel use. Then split Phase 2 into parallel compute and deterministic serial commit: threads query candidate parents and exact-verify edges into tuple buffers, then the main thread sorts tuples and populates `parent->child_nodes`.

**Tech Stack:** C++17, OpenMP, existing `ExactRangeJoinIndex`, `QGramCountIndex`, bounded edit distance helpers, existing Makefile test targets.

---

## Files

- Modify: `navigamer_cpp/include/range_join.hpp`
  - Add a per-query workspace type.
  - Add const/thread-safe query overloads.
  - Add a seed-posting preparation method for serial prebuild before parallel queries.
- Modify: `navigamer_cpp/src/range_join.cpp`
  - Remove shared mutable query workspace from hot query path.
  - Make q-gram query use caller-provided workspace.
  - Make pigeonhole query read prebuilt postings without mutating the index during parallel execution.
- Modify: `navigamer_cpp/include/index_builder.hpp`
  - Add Phase 2 parallel config and stats fields if needed.
- Modify: `navigamer_cpp/src/index_builder.cpp`
  - Parallelize indexed Phase 2 rebinding with local stats and edge tuple buffers.
  - Keep full-link mode serial unless separately requested.
  - Preserve deterministic `child_nodes` ordering.
- Modify: `navigamer_cpp/src/main.cpp`, `README.md`, `navigamer_cpp/README.md`, `navigamer_cpp/CLI_REFERENCE.md`
  - Only if a build-thread CLI flag or new stats fields are exposed.
- Add or modify tests in `navigamer_cpp/src/`
  - Thread-safe range join query equivalence.
  - Phase 2 parallel rebinding equivalence.
  - Existing build range and timing tests.

## Task 1: Add Thread-Safe Range Join Query API

**Files:**
- Modify: `navigamer_cpp/include/range_join.hpp`
- Modify: `navigamer_cpp/src/range_join.cpp`
- Test: `navigamer_cpp/src/test_range_join.cpp`

- [ ] **Step 1: Write a failing parallel-query equivalence test**

Add a test function in `navigamer_cpp/src/test_range_join.cpp` that builds one `ExactRangeJoinIndex`, runs the same query set serially and in OpenMP parallel, and asserts identical candidate IDs.

```cpp
void test_parallel_range_join_queries_are_deterministic() {
  std::vector<navigamer::RangeJoinItem> items;
  const std::vector<std::string> bases = {
      "ACGTACGTACGTACGTACGTACGTACGTACGT",
      "ACGTACGTACGTACGTACGTACGTACGTTCGT",
      "TTTTACGTACGTACGTACGTACGTACGTACGA",
      "GGGGACGTACGTACGTACGTACGTACGTACCC",
  };
  for (size_t i = 0; i < bases.size(); ++i) items.push_back({i, bases[i]});

  navigamer::RangeJoinConfig config;
  config.candidate_mode = navigamer::RangeCandidateMode::Auto;
  config.min_seed_len = 4;
  config.max_seed_len = 12;
  config.qgram_q = 4;

  navigamer::ExactRangeJoinIndex index(config);
  index.build(items);
  index.prepare_seed_lengths({8, 10, 12});

  std::vector<std::string> queries = bases;
  queries.push_back("ACGTACGTACGTACGTACGTACGTACGTAAAA");

  std::vector<std::vector<size_t>> serial;
  for (const auto& query : queries) {
    navigamer::RangeJoinQueryWorkspace workspace;
    serial.push_back(index.query(query, 3, &workspace).candidate_item_ids);
  }

  std::vector<std::vector<size_t>> parallel(queries.size());
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < queries.size(); ++i) {
    navigamer::RangeJoinQueryWorkspace workspace;
    parallel[i] = index.query(queries[i], 3, &workspace).candidate_item_ids;
  }

  assert(serial == parallel);
}
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
cd navigamer_cpp && make test_range_join && ./test_range_join
```

Expected before implementation: compile failure because `RangeJoinQueryWorkspace`, `prepare_seed_lengths`, or the new query overload does not exist.

- [ ] **Step 3: Add the public API**

In `navigamer_cpp/include/range_join.hpp`, add:

```cpp
struct RangeJoinQueryWorkspace {
  QGramQueryWorkspace qgram;
};

class ExactRangeJoinIndex {
 public:
  void prepare_seed_lengths(const std::vector<int>& seed_lengths);
  RangeJoinQueryResult query(
      const std::string& query_sequence, int tau,
      RangeJoinQueryWorkspace* workspace) const;
```

Keep the old `query(const std::string&, int)` overload and have it create a local workspace.

- [ ] **Step 4: Make query methods const and workspace-driven**

Change private methods in `range_join.hpp` and `range_join.cpp` so query work does not mutate shared state:

```cpp
const PostingLists* find_postings_for_seed_len(int seed_len) const;
RangeJoinQueryResult pigeonhole_query(
    const std::string& query_sequence, int tau, int block_len, int seed_len,
    size_t early_abort_candidate_limit) const;
RangeJoinQueryResult qgram_query(
    const std::string& query_sequence, int tau,
    RangeJoinQueryWorkspace* workspace) const;
```

If seed postings are missing in a const query, return a safe `full_scan(query_sequence, tau, true)` rather than constructing postings lazily.

- [ ] **Step 5: Prebuild seed postings serially**

Implement:

```cpp
void ExactRangeJoinIndex::prepare_seed_lengths(
    const std::vector<int>& seed_lengths) {
  for (int seed_len : seed_lengths) {
    if (seed_len >= config_.min_seed_len) {
      (void)postings_for_seed_len(seed_len);
    }
  }
}
```

This method is called before entering OpenMP regions.

- [ ] **Step 6: Verify range-join tests**

Run:

```bash
cd navigamer_cpp && make test_range_join test_qgram && ./test_range_join && ./test_qgram_filter
```

Expected: all tests pass; candidate sets are unchanged.

## Task 2: Parallelize Indexed Phase 2 Compute

**Files:**
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/include/index_builder.hpp` only if new counters/config are added.
- Test: `navigamer_cpp/src/test_build_range_equivalence.cpp`

- [ ] **Step 1: Add a failing Phase 2 parallel equivalence test**

Extend `test_build_range_equivalence.cpp` with a case that builds the same synthetic dataset twice under indexed construction, once with `omp_set_num_threads(1)` and once with `omp_set_num_threads(4)`, then compares the final graph edge IDs.

```cpp
void test_indexed_parallel_phase2_matches_single_thread() {
  auto sequences = make_mutated_window_dataset();

  navigamer::BuildRangeConfig config;
  config.link_mode = navigamer::BuildRangeMode::Indexed;
  config.leaf_attach_mode = navigamer::BuildRangeMode::Indexed;
  config.leaf_attach_direction = navigamer::LeafAttachDirection::WorldToSeq;
  config.range_join.candidate_mode = navigamer::RangeCandidateMode::Auto;

  omp_set_num_threads(1);
  auto single = build_and_collect_edges(sequences, config);

  omp_set_num_threads(4);
  auto parallel = build_and_collect_edges(sequences, config);

  assert(single == parallel);
}
```

- [ ] **Step 2: Run the failing/guard test**

Run:

```bash
cd navigamer_cpp && make test_build_range && ./test_build_range_equivalence
```

Expected before implementation: test may pass if no parallel path exists, but it acts as a regression guard after the code change.

- [ ] **Step 3: Introduce local edge and stats buffers**

Inside indexed `BioGeometryIndexBuilder::phase2_inter_tier_rebinding()`, add local structs:

```cpp
struct Phase2EdgeTuple {
  size_t parent_idx = 0;
  size_t child_idx = 0;
};

struct Phase2LocalStats {
  BioGeometryIndexBuilder::Statistics stats;
  double candidate_query_worker_ms = 0.0;
  double exact_verify_worker_ms = 0.0;
};
```

Only store final verified edges, not all candidate pairs.

- [ ] **Step 4: Precompute seed lengths before the parallel loop**

For the current parent/child layer pair, compute the query tau values used by children and prepare their seed lengths:

```cpp
std::vector<int> seed_lengths;
for (const auto& child : children) {
  if (!child->center_ptr) continue;
  const int query_tau = max_parent_radius + child->radius;
  const int block_count = query_tau + 1;
  const int block_len = static_cast<int>(
      child->center_ptr->seq.size() / static_cast<size_t>(block_count));
  seed_lengths.push_back(std::min(range_config_.range_join.max_seed_len, block_len));
}
std::sort(seed_lengths.begin(), seed_lengths.end());
seed_lengths.erase(std::unique(seed_lengths.begin(), seed_lengths.end()),
                   seed_lengths.end());
parent_index.prepare_seed_lengths(seed_lengths);
```

- [ ] **Step 5: Parallelize child query and exact verification**

Replace the serial indexed `for (auto& child : children)` loop with:

```cpp
const int thread_count = std::max(1, omp_get_max_threads());
std::vector<std::vector<Phase2EdgeTuple>> thread_edges(thread_count);
std::vector<Phase2LocalStats> thread_stats(thread_count);

#pragma omp parallel
{
  const int tid = omp_get_thread_num();
  RangeJoinQueryWorkspace workspace;
  auto& local_edges = thread_edges[static_cast<size_t>(tid)];
  auto& local = thread_stats[static_cast<size_t>(tid)];

  #pragma omp for schedule(dynamic, 8)
  for (size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
    auto& child = children[child_idx];
    if (!child->center_ptr) continue;

    const int query_tau = max_parent_radius + child->radius;
    const auto query_start = std::chrono::steady_clock::now();
    RangeJoinQueryResult candidates =
        parent_index.query(child->center_ptr->seq, query_tau, &workspace);
    const auto query_end = std::chrono::steady_clock::now();
    local.candidate_query_worker_ms +=
        std::chrono::duration<double, std::milli>(query_end - query_start).count();

    accumulate_phase2_candidate_stats(local.stats, candidates);

    const auto verify_start = std::chrono::steady_clock::now();
    for (size_t parent_idx : candidates.candidate_item_ids) {
      auto& parent = parents[parent_idx];
      const int tau = parent->radius + child->radius;
      if (std::llabs(static_cast<long long>(parent->center_ptr->seq.size()) -
                     static_cast<long long>(child->center_ptr->seq.size())) > tau) {
        local.stats.phase2_length_pruned_pairs++;
        continue;
      }
      local.stats.phase2_exact_distance_calls++;
      const int dist = build_distance_bounded(
          parent->center_ptr->seq, child->center_ptr->seq, tau,
          range_config_.distance_mode);
      if (dist <= tau) {
        local_edges.push_back({parent_idx, child_idx});
        local.stats.phase2_edges_added++;
      }
    }
    const auto verify_end = std::chrono::steady_clock::now();
    local.exact_verify_worker_ms +=
        std::chrono::duration<double, std::milli>(verify_end - verify_start).count();
  }
}
```

Do not push into `parent->child_nodes` inside the parallel region.

- [ ] **Step 6: Merge stats and commit edges deterministically**

After the parallel region:

```cpp
std::vector<Phase2EdgeTuple> edges;
for (auto& local_edges : thread_edges) {
  edges.insert(edges.end(), local_edges.begin(), local_edges.end());
}
std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
  return std::tie(a.parent_idx, a.child_idx) < std::tie(b.parent_idx, b.child_idx);
});
edges.erase(std::unique(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
  return a.parent_idx == b.parent_idx && a.child_idx == b.child_idx;
}), edges.end());

{
  ScopedTimer timer(&stats_.phase2_edge_insert_ms);
  for (const auto& edge : edges) {
    parents[edge.parent_idx]->child_nodes.push_back(children[edge.child_idx]);
  }
}
```

Merge every Phase 2 counter from `thread_stats` into `stats_`. Keep `phase2_rebinding_ms` as the authoritative wall time; if per-thread query/verify timings are accumulated, document them as worker-ms or add explicit worker-ms fields.

- [ ] **Step 7: Verify deterministic equivalence**

Run:

```bash
cd navigamer_cpp && make test_build_range && ./test_build_range_equivalence
```

Expected: full/indexed edge sets and single-thread/parallel indexed edge sets match exactly.

## Task 3: Preserve and Clarify Phase 2 Timing

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `README.md`, `navigamer_cpp/README.md`, `navigamer_cpp/CLI_REFERENCE.md`
- Test: `navigamer_cpp/src/test_build_timing_stats.cpp`
- Test: `navigamer_cpp/src/test_build_scale_smoke.cpp`

- [ ] **Step 1: Decide timing semantics**

Keep these fields as wall-clock fields:

```cpp
phase2_rebinding_ms
phase2_index_build_ms
phase2_edge_insert_ms
```

Add explicit worker-time fields if query/verify are measured inside threads:

```cpp
double phase2_candidate_query_worker_ms = 0.0;
double phase2_exact_verify_worker_ms = 0.0;
```

- [ ] **Step 2: Update timing tests**

In `test_build_timing_stats.cpp`, assert the new worker fields are nonnegative and that total phase wall time remains nonnegative.

- [ ] **Step 3: Update build-scale CSV**

If worker fields are added, append:

```text
phase2_candidate_query_worker_ms
phase2_exact_verify_worker_ms
```

Do not rename existing columns.

- [ ] **Step 4: Verify timing output**

Run:

```bash
cd navigamer_cpp && make test_build_timing_stats_check test_build_scale_smoke
./test_build_timing_stats
./test_build_scale_smoke
```

Expected: timing tests pass and CSV columns remain backward-compatible.

## Task 4: Optional Safe Candidate Reduction After Parallelism

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Test: `navigamer_cpp/src/test_build_range_equivalence.cpp`

- [ ] **Step 1: Add an optional q-gram post-filter guard**

After candidate generation and before exact verification, allow a safe q-gram L1 lower-bound post-filter for candidate pairs:

```cpp
if (qgram_can_prune_edit_distance(child_signature, parent_signature, tau)) {
  local.stats.phase2_qgram_pruned_by_l1++;
  continue;
}
```

This can only prune when the q-gram lower bound proves `edit_distance > tau`; otherwise the pair still goes to exact bounded verify.

- [ ] **Step 2: Keep it off until benchmarked**

Gate the post-filter behind a config defaulting to disabled or reuse only existing safe q-gram config after tests show no regressions.

- [ ] **Step 3: Verify no-FN equivalence**

Run:

```bash
cd navigamer_cpp && make test_build_range test_dist test_all
./test_build_range_equivalence
```

Expected: edge sets match full construction exactly.

## Task 5: Benchmark and Report Scaling

**Files:**
- No source edits unless a benchmark flag is added.

- [ ] **Step 1: Build release binary**

Run:

```bash
cd navigamer_cpp && make -j
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
cd navigamer_cpp
make test_range_join test_qgram test_build_range test_build_timing_stats_check test_build_scale_smoke
./test_range_join
./test_qgram_filter
./test_build_range_equivalence
./test_build_timing_stats
./test_build_scale_smoke
```

- [ ] **Step 3: Run recall/distance guards**

Run:

```bash
cd navigamer_cpp
make test_recall test_distance_bound test_dist test_all
./test_recall
./test_distance_bound
```

- [ ] **Step 4: Run build-scale with explicit OpenMP threads**

Run with representative thread counts:

```bash
cd navigamer_cpp
OMP_NUM_THREADS=1 ./navigamer build-scale \
  --ref ../data/ecoli/fasta/ecoli.fa \
  --window 250 \
  --stride 1 \
  --prefix-lengths 10000,50000,100000 \
  --primary-radii 30,15,5 \
  --range-candidate-mode auto \
  --leaf-attach-direction auto \
  --out build_scale_phase2_threads1.csv

OMP_NUM_THREADS=16 ./navigamer build-scale \
  --ref ../data/ecoli/fasta/ecoli.fa \
  --window 250 \
  --stride 1 \
  --prefix-lengths 10000,50000,100000 \
  --primary-radii 30,15,5 \
  --range-candidate-mode auto \
  --leaf-attach-direction auto \
  --out build_scale_phase2_threads16.csv
```

- [ ] **Step 5: Report speedup and bottleneck**

Report:

```text
prefix_len,total_build_ms,phase2_rebinding_ms,phase2_exact_distance_calls,phase2_candidate_pairs,leaf_attach_direction_used
```

Expected first win: wall-time reduction in Phase 2 with unchanged candidate pairs, exact calls, and edges added. If exact calls remain superlinear, the next optimization is safe candidate reduction, not more threading.

## Acceptance Criteria

- Indexed construction edge sets match full construction on existing equivalence tests.
- Single-thread and multi-thread indexed Phase 2 produce identical `child_nodes` ordering.
- No shared mutable query workspace is used by parallel Phase 2.
- No thread writes directly to `parent->child_nodes`.
- `phase2_rebinding_ms` remains wall-clock time.
- Build-scale shows lower Phase 2 wall time with multiple threads.
- No ANN, approximate top-k, HNSW, LSH, containment-only graph, sparse graph, or search semantic changes are introduced.

## Self-Review

- Spec coverage: covers Phase 2 thread safety, parallel compute, deterministic commit, timing, tests, and scaling benchmark.
- Placeholder scan: no `TBD` or open implementation holes remain; optional post-filter is explicitly gated and not required for first acceptance.
- Type consistency: `RangeJoinQueryWorkspace`, `prepare_seed_lengths`, `Phase2EdgeTuple`, and worker timing names are consistent across tasks.
