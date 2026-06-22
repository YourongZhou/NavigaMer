# Phase1 Pigeonhole Build Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase1's dense q=5 posting traversal with a recall-safe incremental exact-seed candidate path so full E. coli construction projects below 24 hours without changing query topology or semantics.

**Architecture:** Add a focused build-only `IncrementalPigeonholeIndex` that lazily maintains exact substring postings by seed length. `Phase1CoverGroupIndex` queries it before the existing q-gram/metric fallback and still performs bounded exact verification plus the existing best-cover tie-break. No persisted or query-time structure changes.

**Tech Stack:** C++17, OpenMP, edlib, existing Make/CMake test targets.

---

### Task 1: Incremental Pigeonhole Candidate Index

**Files:**
- Create: `navigamer_cpp/include/phase1_seed_index.hpp`
- Create: `navigamer_cpp/src/phase1_seed_index.cpp`
- Create: `navigamer_cpp/src/test_phase1_seed_index.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing candidate-superset test**

Create randomized fixed-length and mixed-length candidates. Append them incrementally, generate substitution/indel queries, and assert every brute-force match is present whenever `result.safe` is true. Add explicit ambiguous-base and too-short-seed cases that assert `safe == false`.

```cpp
navigamer::IncrementalPigeonholeIndex index({8, 20});
for (size_t i = 0; i < candidates.size(); ++i) {
  index.append(i, candidates[i]);
}
auto result = index.query(query, tau);
if (result.safe) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (navigamer::compute_distance(query, candidates[i]) <= tau) {
      assert(std::binary_search(result.candidate_indices.begin(),
                                result.candidate_indices.end(), i));
    }
  }
}
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cd navigamer_cpp && make test_phase1_seed_index`

Expected: compilation fails because `phase1_seed_index.hpp` and `IncrementalPigeonholeIndex` do not exist.

- [ ] **Step 3: Implement the minimal incremental index**

Expose this API:

```cpp
struct Phase1SeedIndexConfig {
  int min_seed_len = 8;
  int max_seed_len = 20;
};

struct Phase1SeedQueryResult {
  bool safe = false;
  int block_len = 0;
  int seed_len = 0;
  size_t posting_entries_visited = 0;
  std::vector<size_t> candidate_indices;
};

class IncrementalPigeonholeIndex {
 public:
  explicit IncrementalPigeonholeIndex(Phase1SeedIndexConfig config = {});
  void append(size_t item_id, const std::string& sequence);
  Phase1SeedQueryResult query(const std::string& sequence, int tau);
  size_t size() const;
};
```

Implementation requirements:

- Lazily build each requested seed length over already appended items.
- Append later items to every materialized seed-length index.
- Encode A/C/G/T seeds in `uint64_t`; reject unsupported lengths or ambiguous bases safely.
- Store each seed at most once per candidate.
- Include unindexable but length-compatible candidates instead of dropping them.
- Deduplicate with an epoch array and sort final candidate IDs.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `cd navigamer_cpp && make test_phase1_seed_index && ./test_phase1_seed_index`

Expected: `phase1 seed index tests passed`.

- [ ] **Step 5: Commit the helper**

```bash
git add navigamer_cpp/include/phase1_seed_index.hpp \
        navigamer_cpp/src/phase1_seed_index.cpp \
        navigamer_cpp/src/test_phase1_seed_index.cpp \
        navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "feat: add incremental phase1 seed index"
```

### Task 2: Integrate Exact Seeds Into Phase1

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/test_phase1_hybrid.cpp`

- [ ] **Step 1: Extend the equivalence test first**

Keep the existing scan-vs-hybrid graph, leaf-link, and adaptive-hit assertions. Add assertions proving the new path ran and reduced the candidate set:

```cpp
assert(hybrid_stats.phase1_pigeonhole_queries > 0);
assert(hybrid_stats.phase1_seed_posting_entries_visited > 0);
assert(hybrid_stats.phase1_pigeonhole_candidates > 0);
assert(hybrid_stats.phase1_candidate_pairs <
       hybrid_stats.phase1_total_possible_pairs);
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cd navigamer_cpp && make test_phase1_hybrid`

Expected: compilation fails because the new statistics fields do not exist.

- [ ] **Step 3: Add Phase1 source and counters**

Add `Pigeonhole` to the internal `Phase1CoverSource` and add these public statistics fields:

```cpp
size_t phase1_pigeonhole_queries = 0;
size_t phase1_seed_posting_entries_visited = 0;
size_t phase1_pigeonhole_candidates = 0;
size_t phase1_pigeonhole_fallbacks = 0;
```

- [ ] **Step 4: Query pigeonhole before q-gram**

After the candidate group reaches `phase1_qgram_min_fanout`, synchronize its incremental seed index and query with the layer radius. If `safe`, return the exact-seed superset. Otherwise increment the fallback counter and continue through the existing q-gram/metric path.

Exact verification remains:

```cpp
scan = find_best_phase1_cover_by_indices(
    *candidates, candidate_query.candidate_indices, sequence,
    radius, range_config_.distance_mode);
```

- [ ] **Step 5: Run Phase1 and range-equivalence tests**

Run:

```bash
cd navigamer_cpp
make test_phase1_hybrid test_build_range
./test_phase1_hybrid_bin
./test_build_range_equivalence
```

Expected: both pass and scan/hybrid topology remains identical.

- [ ] **Step 6: Commit the integration**

```bash
git add navigamer_cpp/include/index_builder.hpp \
        navigamer_cpp/src/index_builder.cpp \
        navigamer_cpp/src/test_phase1_hybrid.cpp
git commit -m "perf: use exact seeds for phase1 candidates"
```

### Task 3: Long-Build Visibility and Documentation

**Files:**
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`
- Modify: `navigamer_cpp/src/test_build_timing_stats.cpp`

- [ ] **Step 1: Add failing statistics assertions**

Extend the timing-stat test to assert counter consistency:

```cpp
assert(stats.phase1_pigeonhole_candidates <=
       stats.phase1_total_possible_pairs);
assert(stats.phase1_pigeonhole_queries == 0 ||
       stats.phase1_seed_posting_entries_visited > 0);
```

- [ ] **Step 2: Run and verify RED**

Run: `cd navigamer_cpp && make test_build_timing_stats`

Expected: failure until statistics are printed and populated consistently.

- [ ] **Step 3: Add progress and summary output**

For builds with at least 100,000 unique sequences, print progress every 100,000 processed sequences with elapsed seconds, percentage, created nodes, seed queries, and posting entries visited. Extend the existing Phase1 summary line with the new counters.

- [ ] **Step 4: Update all CLI documentation**

Document that hybrid Phase1 uses exact pigeonhole seeds first, falls back safely, and does not alter query structures or no-FN behavior.

- [ ] **Step 5: Run timing and smoke tests**

Run:

```bash
cd navigamer_cpp
make test_build_timing_stats test_build_scale_smoke
./test_build_timing_stats
./test_build_scale_smoke
```

Expected: both pass and a 128 kb build emits at least one progress line.

### Task 4: Correctness and Query Regression Gates

**Files:**
- No production changes expected

- [ ] **Step 1: Build all affected targets**

Run: `cd navigamer_cpp && make -j`

- [ ] **Step 2: Run construction correctness tests**

Run:

```bash
./test_phase1_seed_index
./test_phase1_hybrid_bin
./test_range_join
./test_build_range_equivalence
./test_index_persistence_bin
```

- [ ] **Step 3: Run no-FN guards**

Run:

```bash
./test_recall
./test_distance_bound
./test_map150_recall
```

Expected: all summaries report zero failures.

- [ ] **Step 4: Run query topology and speed checks**

Build scan and optimized indexes from the same deterministic input and assert identical primary edges and leaf links through `test_phase1_hybrid`. Run the existing `query-benchmark` baseline on the same reference/query set; optimized query median must not regress beyond measurement noise because the graph is identical.

### Task 5: E. coli Scaling Gate

**Files:**
- Generated outputs only under `.tmp_experiments/`; do not commit

- [ ] **Step 1: Run fixed-window prefix benchmarks**

Run 32, 64, and 128 kb prefixes with `window=250`, `stride=1`, default radii, and fixed OpenMP settings. Capture Phase1 time, posting entries, candidate counts, total build time, peak RSS, and query benchmark latency.

- [ ] **Step 2: Compare against recorded baseline**

Baseline Phase1 times are approximately 9.1 s, 28.6 s, and 101.9 s. Confirm seed posting work grows materially slower than the old q-gram touched counts of 40.99 M, 164.26 M, and 660.14 M.

- [ ] **Step 3: Extend the largest prefix if projection is uncertain**

Run 256 or 512 kb until the conservative upper projection is stable. Do not claim the target from a small-prefix linear extrapolation.

- [ ] **Step 4: Evaluate the 24-hour gate**

Project from the two largest measurements and include a cache/memory safety margin. If the upper estimate is below 24 hours, launch the full E. coli build. Otherwise return to design for a locality-incumbent or batch-parallel Phase1 stage without weakening recall.

