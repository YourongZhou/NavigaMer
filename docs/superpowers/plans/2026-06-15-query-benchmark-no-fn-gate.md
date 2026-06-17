# Query Benchmark and No-FN Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic dual-profile adaptive-query benchmark that reports cold/warm latency and search counters while failing on any baseline/optimized/brute-force result mismatch.

**Architecture:** Add logical hot-path counters to `SearchStats`, then implement benchmark configuration, deterministic query generation, result comparison, aggregation, and fixed-schema TSV/JSON output in a focused `query_benchmark` module. Keep the existing `benchmark` command unchanged; `main.cpp` only parses the new `query-benchmark` flags and invokes the module.

**Tech Stack:** C++17, OpenMP, `std::chrono::steady_clock`, existing NavigaMer index/search/I/O APIs, Make, CMake.

---

### Task 1: Add Logical Leaf-Beacon and Visited Counters

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Modify: `navigamer_cpp/src/test_search_stats.cpp`

- [ ] **Step 1: Write failing counter assertions**

Extend `test_search_stats.cpp` after the existing adaptive-search assertions:

```cpp
assert(stats.leaf_beacon_check_count > 0);
assert(stats.mbb_check_count == stats.edge_access_count);
assert(stats.visited_check_count > 0);
assert(stats.visited_hit_count <= stats.visited_check_count);
assert(stats.leaf_exact_distance_call_count == stats.leaf_verify_count);
assert(stats.center_exact_distance_call_count ==
       stats.world_access_count);
```

Also assert that the q-gram-enabled search preserves these logical
relationships.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd navigamer_cpp
make test_search_stats
```

Expected: compilation fails because the five logical counter fields do not
exist.

- [ ] **Step 3: Add logical counters without changing search decisions**

Add these fields to `SearchStats`:

```cpp
size_t leaf_beacon_check_count = 0;
size_t mbb_check_count = 0;
size_t leaf_exact_distance_call_count = 0;
size_t center_exact_distance_call_count = 0;
size_t visited_check_count = 0;
size_t visited_hit_count = 0;
```

Increment `leaf_beacon_check_count` once for every leaf row actually checked by
`leaf_beacon_prunable_row`. Increment `leaf_exact_distance_call_count` beside
the existing leaf `compute_distance`. Increment
`center_exact_distance_call_count` beside every adaptive center
`compute_distance` or `compute_distance_bounded`.

Increment `mbb_check_count` once per child in scan mode and by
`children.size()` for each successful rect-index query. Fallback paths count
only through the scan helper, preserving `mbb_check_count == edge_access_count`.

Replace each adaptive visited test with an equivalent counted form:

```cpp
stats.visited_check_count++;
if (visited_nodes.count(node->node_id)) {
  stats.visited_hit_count++;
  continue;
}
```

Do not change insertion points, containment handling, traversal order, or the
visited container.

- [ ] **Step 4: Verify GREEN and adaptive correctness guards**

Run:

```bash
cd navigamer_cpp
make test_search_stats test_recall test_dist test_mbb_filter test_search_qgram
```

Expected: all five targets pass.

- [ ] **Step 5: Commit the logical counters**

```bash
git add navigamer_cpp/include/search_engine.hpp \
        navigamer_cpp/src/search_engine.cpp \
        navigamer_cpp/src/test_search_stats.cpp
git commit -m "add logical adaptive search counters"
```

### Task 2: Add Benchmark Data Model, Result Comparison, and Aggregation

**Files:**
- Create: `navigamer_cpp/include/query_benchmark.hpp`
- Create: `navigamer_cpp/src/query_benchmark.cpp`
- Create: `navigamer_cpp/src/test_query_benchmark_gate.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write failing tests for result comparison and percentiles**

Create `test_query_benchmark_gate.cpp` with focused tests for:

```cpp
using navigamer::compare_result_ids;
using navigamer::nearest_rank_percentile;

assert(nearest_rank_percentile({1.0, 2.0, 3.0, 4.0}, 0.50) == 2.0);
assert(nearest_rank_percentile({1.0, 2.0, 3.0, 4.0}, 0.95) == 4.0);

auto equal = compare_result_ids({"a", "b"}, {"b", "a"}, {"a", "b"});
assert(equal.baseline_equals_optimized);
assert(equal.baseline_no_fn);
assert(equal.optimized_no_fn);

auto mismatch = compare_result_ids({"a"}, {"b"}, {"a", "b"});
assert(!mismatch.baseline_equals_optimized);
assert((mismatch.baseline_only == std::vector<std::string>{"a"}));
assert((mismatch.optimized_only == std::vector<std::string>{"b"}));
assert((mismatch.brute_force_missing_from_baseline ==
        std::vector<std::string>{"b"}));
assert((mismatch.brute_force_missing_from_optimized ==
        std::vector<std::string>{"a"}));
```

Register `test_query_benchmark_gate` in Make and CMake before implementation.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
cd navigamer_cpp
make test_query_benchmark
```

Expected: compilation fails because `query_benchmark.hpp` and its API are
absent.

- [ ] **Step 3: Define the benchmark API and fixed records**

Define these public types in `query_benchmark.hpp`:

```cpp
enum class QueryClass {
  RandomRegion,
  OrdinaryRegion,
  LowComplexityRegion,
  NoHit,
  SingleHit,
  MultiHit,
};

struct QueryBenchmarkConfig {
  std::string ref_input;
  size_t reference_subset_length = 0;
  int window_length = 200;
  int stride = 1;
  int query_length = 200;
  int tolerance = 2;
  unsigned seed = 42;
  int threads = 1;
  size_t queries_per_class = 1;
  size_t warmup_iterations = 2;
  size_t measured_iterations = 10;
  size_t cold_cache_bytes = 256ULL * 1024ULL * 1024ULL;
  std::string detail_tsv_path;
  std::string summary_tsv_path;
  std::string json_path;
};

struct GeneratedBenchmarkQuery {
  QueryClass query_class;
  BioSequence query;
  std::vector<std::string> brute_force_ids;
};

struct ResultComparison {
  bool baseline_equals_optimized = false;
  bool baseline_no_fn = false;
  bool optimized_no_fn = false;
  std::vector<std::string> baseline_only;
  std::vector<std::string> optimized_only;
  std::vector<std::string> brute_force_missing_from_baseline;
  std::vector<std::string> brute_force_missing_from_optimized;
};

struct QueryBenchmarkRunResult {
  bool gate_passed = false;
  size_t mismatch_count = 0;
  std::vector<std::vector<std::string>> detail_rows;
  std::vector<std::vector<std::string>> summary_rows;
  std::string json_summary;
};

const char* query_class_name(QueryClass value);
double nearest_rank_percentile(std::vector<double> values, double quantile);
ResultComparison compare_result_ids(std::vector<std::string> baseline,
                                    std::vector<std::string> optimized,
                                    std::vector<std::string> brute_force);
```

Use sorted/unique vectors and `std::set_difference` in
`compare_result_ids`. Throw `std::invalid_argument` for an empty percentile
sample or a quantile outside `[0, 1]`.

- [ ] **Step 4: Implement internal records and aggregation**

In `query_benchmark.cpp`, add internal `ExecutionRecord` and
`AggregateRecord` types. `ExecutionRecord` must hold profile, query class,
sample kind (`cold` or `warm`), iteration, latency, result/comparison status,
and a full copy of `SearchStats`.

Implement aggregation grouped by `(query_class, profile)` plus `all` rows.
Use nearest-rank percentiles and calculate averages as:

```cpp
average = values.empty()
              ? 0.0
              : std::accumulate(values.begin(), values.end(), 0.0) /
                    static_cast<double>(values.size());
```

- [ ] **Step 5: Verify GREEN**

Run:

```bash
cd navigamer_cpp
make test_query_benchmark
./test_query_benchmark_gate
```

Expected: result comparison and percentile tests pass.

- [ ] **Step 6: Commit the benchmark core**

```bash
git add navigamer_cpp/include/query_benchmark.hpp \
        navigamer_cpp/src/query_benchmark.cpp \
        navigamer_cpp/src/test_query_benchmark_gate.cpp \
        navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "add query benchmark core model"
```

### Task 3: Generate All Six Deterministic Query Classes

**Files:**
- Modify: `navigamer_cpp/include/query_benchmark.hpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`

- [ ] **Step 1: Add failing deterministic class-generation tests**

Build a purpose-made vector of indexed sequences containing random-looking,
unique, repeated, and low-complexity strings. Add tests that call:

```cpp
auto first = generate_benchmark_queries(
    index_sequences, unique_sequences, 12, 1, 1234, 1);
auto second = generate_benchmark_queries(
    index_sequences, unique_sequences, 12, 1, 1234, 1);

assert(first.size() == 6);
assert(first.size() == second.size());
for (size_t i = 0; i < first.size(); ++i) {
  assert(first[i].query_class == second[i].query_class);
  assert(first[i].query.seq == second[i].query.seq);
  assert(first[i].brute_force_ids == second[i].brute_force_ids);
}
```

Assert the no-hit class has zero brute-force IDs, single-hit has exactly one,
multi-hit has at least two, and every class name occurs once.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd navigamer_cpp
make test_query_benchmark
```

Expected: compilation fails because `generate_benchmark_queries` is absent.

- [ ] **Step 3: Implement deterministic candidate generation**

Add:

```cpp
std::vector<GeneratedBenchmarkQuery> generate_benchmark_queries(
    const std::vector<std::shared_ptr<BioSequence>>& index_sequences,
    const std::vector<std::shared_ptr<BioSequence>>& unique_sequences,
    int query_length,
    int tolerance,
    unsigned seed,
    size_t queries_per_class);
```

Use exact edit distance against `unique_sequences` to classify candidates; do
not include classification time in benchmark measurements.

Use deterministic policies:

- `random_region`: shuffled valid indexed windows.
- `ordinary_region`: highest Shannon-entropy windows that do not qualify as
  multi-hit.
- `low_complexity_region`: lowest Shannon-entropy windows.
- `no_hit`: seeded random A/C/G/T sequences retried until zero hits.
- `single_hit`: sampled or substitution-mutated windows retried until one hit.
- `multi_hit`: sampled or substitution-mutated windows retried until at least
  two hits.

Use `4096` attempts per requested query and throw:

```cpp
throw std::runtime_error(
    "unable to generate query class " + std::string(query_class_name(kind)) +
    " after 4096 deterministic attempts");
```

Reject `query_length <= 0`, empty sequence sets, and index sequences shorter
than `query_length`.

- [ ] **Step 4: Verify GREEN and determinism**

Run:

```bash
cd navigamer_cpp
make test_query_benchmark
./test_query_benchmark_gate
```

Expected: all six classes are generated and repeated calls are identical.

- [ ] **Step 5: Commit deterministic query generation**

```bash
git add navigamer_cpp/include/query_benchmark.hpp \
        navigamer_cpp/src/query_benchmark.cpp \
        navigamer_cpp/src/test_query_benchmark_gate.cpp
git commit -m "add deterministic query benchmark classes"
```

### Task 4: Implement Dual-Profile Measurement and No-FN Gate

**Files:**
- Modify: `navigamer_cpp/include/query_benchmark.hpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`

- [ ] **Step 1: Add failing end-to-end runner tests**

Create a mixed synthetic literal reference that can produce all six classes.
Run the benchmark with `queries_per_class=1`, `warmup_iterations=1`,
`measured_iterations=2`, `cold_cache_bytes=0`, baseline scan/off, and optimized
rect/on. Assert:

```cpp
assert(result.gate_passed);
assert(result.mismatch_count == 0);
assert(result.detail_rows.size() == 6 * 2 * (1 + 2));
assert(!result.summary_rows.empty());
assert(result.json_summary.find("\"gate_passed\":true") != std::string::npos);
assert(result.json_summary.find("\"candidate_set_comparison\":\"unavailable\"")
       != std::string::npos);
```

Add a unit test that feeds deliberately different result ID vectors through
the gate helper and asserts the overall gate fails.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd navigamer_cpp
make test_query_benchmark
```

Expected: compilation fails because `run_query_benchmark` is absent.

- [ ] **Step 3: Implement validated configuration and shared-index setup**

Add:

```cpp
QueryBenchmarkRunResult run_query_benchmark(
    const QueryBenchmarkConfig& config,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& build_config,
    const SearchConfig& optimized_search_config);
```

Validate positive window/query length, stride, threads, queries per class, and
measured iterations; require all three output paths. Load and truncate the
reference prefix, build reference windows, construct one builder, and create:

```cpp
SearchConfig baseline_config;
baseline_config.mbb_filter_mode = MBBFilterMode::Scan;
baseline_config.search_qgram_prefilter = false;

BioGeometrySearchEngine baseline(builder, baseline_config);
BioGeometrySearchEngine optimized(builder, optimized_search_config);
```

Record build duration and builder statistics. Set `omp_set_num_threads` from
the config, but execute the query loop serially.

- [ ] **Step 4: Implement process memory snapshots**

Add a Linux best-effort helper that reads current RSS from `/proc/self/status`
and peak RSS from `getrusage(RUSAGE_SELF, ...)`. Capture snapshots before
build, after build, and after benchmark execution. Represent unavailable
values explicitly and never fail the no-FN gate because memory telemetry is
unavailable. Record per-query allocation counting as `unavailable`.

- [ ] **Step 5: Implement cold/warm measurement and equality checks**

Allocate one reusable `std::vector<uint8_t>` eviction buffer. Before each cold
sample, touch one byte per 64-byte cache line and fold bytes into a volatile
checksum outside the timed region.

For each query, alternate baseline/optimized first-run order by query index.
For each profile, run exactly one cold sample, then untimed warmups, then
`measured_iterations` warm samples. Time only `search_adaptive` with
`steady_clock`.

Normalize result IDs after every execution and require repeated executions for
the same query/profile to match. Compare baseline, optimized, and stored
brute-force IDs. Accumulate all mismatches but return `gate_passed=false` when
any mismatch or false negative occurs.

- [ ] **Step 6: Implement fixed-schema TSV and JSON output**

Detailed TSV columns must include:

```text
query_id,query_class,profile,sample_kind,iteration,first_profile,
latency_ms,result_count,brute_force_result_count,result_equal,no_fn,
world_access_count,node_access_count,edge_access_count,mbb_checks,
mbb_survivors,qgram_checks,center_exact_distance_calls,leaf_beacon_checks,
leaf_exact_distance_calls,visited_checks,visited_hits,candidate_count,
verified_candidate_count
```

Summary TSV must include one row per class/profile plus `all`, with cold and
warm `avg/p50/p95/p99`, query/sample/result totals, failure counts, and average
logical counters.

Build JSON using `std::ostringstream` and a local JSON-string escape helper.
Include schema version, full configuration, build duration/statistics,
profile settings, generation counts, aggregate rows, mismatch diagnostics,
memory snapshots, `candidate_set_comparison:"unavailable"`,
`allocation_counting:"unavailable"`, and gate status.

Write detailed and summary TSV using `write_tsv`; write JSON using
`std::ofstream`. Write outputs before returning a failed gate result.

- [ ] **Step 7: Verify GREEN and existing equivalence tests**

Run:

```bash
cd navigamer_cpp
make test_query_benchmark test_mbb_filter test_search_qgram test_recall test_dist
./test_query_benchmark_gate
```

Expected: all tests pass and the runner reports exact
baseline/optimized/brute-force equality.

- [ ] **Step 8: Commit the runner and gate**

```bash
git add navigamer_cpp/include/query_benchmark.hpp \
        navigamer_cpp/src/query_benchmark.cpp \
        navigamer_cpp/src/test_query_benchmark_gate.cpp
git commit -m "add dual profile query benchmark gate"
```

### Task 5: Expose the `query-benchmark` CLI and Document It

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] **Step 1: Add a failing CLI smoke invocation**

Build the existing CLI, then run:

```bash
cd navigamer_cpp
./navigamer query-benchmark --ref ACGTACGT
```

Expected: command is not recognized or falls through without the required new
validation.

- [ ] **Step 2: Parse and dispatch the new command**

Include `query_benchmark.hpp`. Add these CLI variables and defaults:

```cpp
size_t reference_subset_length = 0;
int query_length = 200;
int threads = 1;
size_t queries_per_class = 1;
size_t warmup_iterations = 2;
size_t measured_iterations = 10;
size_t cold_cache_bytes = 256ULL * 1024ULL * 1024ULL;
std::string summary_out;
std::string json_out;
```

Parse:

```text
--reference-subset-length
--query-length
--threads
--queries-per-class
--warmup-iterations
--measured-iterations
--cold-cache-bytes
--summary-out
--json-out
```

For `query-benchmark`, require `--ref`, `--out`, `--summary-out`, and
`--json-out`. Populate `QueryBenchmarkConfig`, call `run_query_benchmark`, and
return `0` only when `gate_passed` is true; otherwise print the mismatch count
and return `2`.

- [ ] **Step 3: Build and run a deterministic CLI smoke benchmark**

Run:

```bash
cd navigamer_cpp
make -j
./navigamer query-benchmark \
  --ref ../data/human/chr1_subset \
  --reference-subset-length 2000 \
  --window 150 \
  --stride 1 \
  --query-length 150 \
  --tolerance 2 \
  --seed 42 \
  --threads 1 \
  --mbb-filter-mode rect \
  --search-qgram-prefilter on \
  --warmup-iterations 1 \
  --measured-iterations 2 \
  --queries-per-class 1 \
  --cold-cache-bytes 0 \
  --out /tmp/navigamer_query_benchmark.tsv \
  --summary-out /tmp/navigamer_query_benchmark_summary.tsv \
  --json-out /tmp/navigamer_query_benchmark_summary.json
```

Expected: exit `0`, gate passes, all three files exist, and the summary
contains baseline and optimized rows. If this real subset cannot produce one
class, use the mixed synthetic reference from the test for the smoke command
and record that limitation in the benchmark report.

- [ ] **Step 4: Document command semantics and output**

Update both README files and `CLI_REFERENCE.md` with:

- baseline fixed as scan/q-gram-off
- optimized profile flags
- fixed reproducibility inputs
- all six query classes
- best-effort eviction-buffer cold measurement
- serial Step 0 execution despite recorded thread count
- exact result/no-FN gate and exit status
- detailed TSV, summary TSV, and JSON paths/columns
- candidate-set comparison marked unavailable

- [ ] **Step 5: Commit CLI and documentation**

```bash
git add navigamer_cpp/src/main.cpp README.md navigamer_cpp/README.md \
        navigamer_cpp/CLI_REFERENCE.md
git commit -m "expose query benchmark no-fn gate"
```

### Task 6: Full Validation and Before/After Benchmark Report

**Files:**
- Create: `navigamer_cpp/QUERY_BENCHMARK_BASELINE.md`

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
cd navigamer_cpp
make test_all
```

Expected: every existing test plus `test_query_benchmark_gate` passes.

- [ ] **Step 2: Run the required correctness guards explicitly**

Run:

```bash
cd navigamer_cpp
./test_recall
./test_distance_bound
./test_mbb_filter_equivalence
./test_search_qgram_prefilter
./test_query_benchmark_gate
```

Expected: all pass with no result mismatch or false negative.

- [ ] **Step 3: Run fixed benchmark datasets**

Run `query-benchmark` with thread count `1`, a fixed seed, and the same
hierarchy/build settings for:

```text
synthetic random DNA
E. coli subset when available in the repository
chr1_subset first 2 kb
chr1_subset first 10 kb when runtime is practical
synthetic repetitive/low-complexity reference
```

For each dataset, compare baseline scan/q-gram-off with optimized rect/q-gram-on
and preserve generated outputs under `.tmp_experiments/` only.

- [ ] **Step 4: Write the benchmark report**

Create `QUERY_BENCHMARK_BASELINE.md` containing:

- exact commands and dataset availability
- build time
- cold and warm avg/p50/p95/p99 tables
- result equality and no-FN status
- world/node/edge access
- MBB checks/survivors
- q-gram checks
- center exact calls
- leaf beacon/exact calls
- visited checks/hits
- candidate/verified counts
- current/peak RSS before build, after build, and after benchmark
- explanation of speedup or lack of speedup
- explicit note that per-query allocation measurement is unavailable

- [ ] **Step 5: Run final verification**

Run:

```bash
cd navigamer_cpp
make -j
./navigamer demo --size 200
make test_all
cd ..
git diff --check
git status --short
```

Expected: CLI build, smoke run, and all tests pass; no whitespace errors;
generated binaries and `.tmp_experiments/` remain untracked and are not added.

- [ ] **Step 6: Commit the benchmark report**

```bash
git add navigamer_cpp/QUERY_BENCHMARK_BASELINE.md
git commit -m "document query benchmark baseline"
```
