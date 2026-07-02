# Locality Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible benchmark that separates NavigaMer persisted-index load time, search-engine initialization time, and query-only latency for clustered/similar query streams.

**Architecture:** Extend the existing C++ query benchmark path instead of keeping ad hoc scripts. Add a persisted-index locality benchmark entrypoint that loads an index once, generates same-template, nearby-window, and random-window query streams, runs baseline/path-reuse/optimized profiles, verifies no false negatives against baseline on small gates, and writes TSV/JSON summaries with load/init/query timing.

**Tech Stack:** C++17, existing `navigamer_cpp` `index_persistence`, `search_engine`, `query_benchmark` helpers, Makefile tests, TSV/JSON outputs.

---

### Task 1: Define Locality Benchmark Data Model And Tests

**Files:**
- Modify: `navigamer_cpp/include/query_benchmark.hpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`

- [ ] **Step 1: Write the failing test**

Add assertions to `navigamer_cpp/src/test_query_benchmark_gate.cpp` that call new helper APIs:

```cpp
auto clustered = navigamer::generate_locality_benchmark_queries(
    "ACGTACGTACGTACGTACGTACGTACGTACGT", 8, 12, 1, 7);
assert(clustered.same_template.size() == 8);
assert(clustered.nearby_windows.size() == 8);
assert(clustered.random_windows.size() == 8);
assert(clustered.same_template[0].source_pos ==
       clustered.same_template[7].source_pos);
assert(clustered.nearby_windows[0].source_pos + 7 ==
       clustered.nearby_windows[7].source_pos);
assert(clustered.random_windows[0].query.seq.size() == 12);
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd navigamer_cpp && make test_query_benchmark
```

Expected: compile failure because `generate_locality_benchmark_queries` and the result struct do not exist.

- [ ] **Step 3: Write minimal implementation**

Add to `query_benchmark.hpp`:

```cpp
struct LocalityBenchmarkQuery {
  BioSequence query;
  size_t source_pos = 0;
};

struct LocalityBenchmarkQuerySets {
  std::vector<LocalityBenchmarkQuery> same_template;
  std::vector<LocalityBenchmarkQuery> nearby_windows;
  std::vector<LocalityBenchmarkQuery> random_windows;
};

LocalityBenchmarkQuerySets generate_locality_benchmark_queries(
    const std::string& reference,
    size_t query_count,
    int query_length,
    int edits,
    unsigned seed);
```

Implement in `query_benchmark.cpp` using deterministic substitution mutations, a clean A/C/G/T window finder, same-template repeated mutations from one source window, nearby windows from consecutive offsets, and random windows sampled across the reference.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd navigamer_cpp && make test_query_benchmark
```

Expected: `query benchmark gate tests passed`.

### Task 2: Add Persisted Locality Benchmark Runner

**Files:**
- Modify: `navigamer_cpp/include/query_benchmark.hpp`
- Modify: `navigamer_cpp/src/query_benchmark.cpp`
- Modify: `navigamer_cpp/src/test_query_benchmark_gate.cpp`

- [ ] **Step 1: Write the failing test**

Extend `test_query_benchmark_gate.cpp` to build a tiny persisted index in `/tmp`, run the new persisted benchmark API, and assert output shape:

```cpp
navigamer::LocalityBenchmarkConfig locality;
locality.index_path = "/tmp/navigamer_locality_test.navidx";
locality.ref_input = "ACGTGCACTGATTGCATAGCTACGACGTGCACTGATTGCATAGCTACG";
locality.query_count = 4;
locality.query_length = 12;
locality.tolerance = 1;
locality.edits = 1;
locality.out_tsv_path = "/tmp/navigamer_locality_test.tsv";
auto locality_result = navigamer::run_persisted_locality_benchmark(locality);
assert(locality_result.gate_passed);
assert(locality_result.load_ms >= 0.0);
assert(locality_result.rows.size() >= 6);
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd navigamer_cpp && make test_query_benchmark
```

Expected: compile failure because `LocalityBenchmarkConfig` and `run_persisted_locality_benchmark` do not exist.

- [ ] **Step 3: Write minimal implementation**

Add `LocalityBenchmarkConfig`, `LocalityBenchmarkRow`, and `LocalityBenchmarkRunResult` to the header. Implement runner to:

- call `load_index()` once and record `load_ms`;
- construct one `BioGeometrySearchEngine` per profile and record `engine_init_ms`;
- run profiles `baseline`, `path_reuse`, and `optimized` across query sets `same_template`, `nearby_windows`, and `random_windows`;
- write TSV columns `dataset`, `profile`, `query_count`, `load_ms`, `engine_init_ms`, `query_wall_ms`, `mean_query_ms`, `p50_query_ms`, `p95_query_ms`, `fn_count`, `mismatch_count`, `mean_world_access`, `mean_center_distance`, `mean_leaf_verify`, `mean_path_reuse_hits`, `mean_anchor_cache_hits`, `mean_child_shortlist_hits`;
- compare `path_reuse` and `optimized` IDs to `baseline` IDs on small tests and set `gate_passed`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd navigamer_cpp && make test_query_benchmark
```

Expected: query benchmark gate tests pass and `/tmp/navigamer_locality_test.tsv` contains the required columns.

### Task 3: Expose CLI And Documentation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`
- Modify: `navigamer_cpp/README.md`

- [ ] **Step 1: Write the failing test**

Add to `test_query_benchmark_gate.cpp` a check that the result TSV header contains `load_ms`, `engine_init_ms`, and `query_wall_ms`. This fails until the runner writes those columns.

- [ ] **Step 2: Implement CLI**

Add command:

```bash
./navigamer locality-benchmark \
  --index <persisted.navidx> \
  --ref <fasta> \
  --out <summary.tsv> \
  --query-count 1024 \
  --query-length 250 \
  --query-edits 5 \
  --tolerance 5
```

- [ ] **Step 3: Update docs**

Document that this benchmark is for batch/query-only measurement and that single `query-index` includes persisted-index load time.

- [ ] **Step 4: Validate CLI smoke**

Run:

```bash
cd navigamer_cpp && ./navigamer locality-benchmark --index /tmp/navigamer_locality_test.navidx --ref ACGTGCACTGATTGCATAGCTACGACGTGCACTGATTGCATAGCTACG --out /tmp/navigamer_locality_cli.tsv --query-count 4 --query-length 12 --query-edits 1 --tolerance 1
```

Expected: TSV exists with same datasets/profiles as the API test.

### Task 4: Run Full E. coli Measurement

**Files:**
- Create: `.tmp_experiments/optimization_check_v1/full_ecoli_locality_summary.tsv`
- Create: `.tmp_experiments/optimization_check_v1/full_ecoli_locality_run.log`

- [ ] **Step 1: Run persisted-index benchmark**

Run:

```bash
cd navigamer_cpp
/usr/bin/time -v ./navigamer locality-benchmark \
  --index /home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/full_builds/navigamer_full_20260629_andyzhou_w250_s1.navidx \
  --ref /home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/fasta/ecoli.fa \
  --out ../.tmp_experiments/optimization_check_v1/full_ecoli_locality_summary.tsv \
  --query-count 256 \
  --query-length 250 \
  --query-edits 5 \
  --tolerance 5 \
  > ../.tmp_experiments/optimization_check_v1/full_ecoli_locality_run.log 2>&1
```

- [ ] **Step 2: Interpret results**

Compare:

- `load_ms` vs `query_wall_ms`;
- `same_template` and `nearby_windows` vs `random_windows`;
- `path_reuse` vs `baseline`;
- path reuse counters (`mean_path_reuse_hits`, `mean_anchor_cache_hits`, `mean_child_shortlist_hits`) against latency.

- [ ] **Step 3: Decide next optimization target**

If `same_template/nearby_windows` improves with path reuse, scale to 1024/4096 queries. If not, use profile columns to identify whether time is dominated by leaf verification, center distances, or MBB traversal before changing search code.
