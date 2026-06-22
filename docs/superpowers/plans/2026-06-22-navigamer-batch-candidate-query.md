# NavigaMer Batch Candidate Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a recall-safe candidate-only adaptive search API and a one-load/many-query persisted-index CLI path with shared bounded Edlib verification.

**Architecture:** Refactor adaptive leaf handling around an explicit collector mode so hierarchy traversal is shared by verified and candidate-only searches. Add a focused batch-query module that loads one `.navidx`, processes FASTQ reads in bounded parallel batches, verifies candidates with Edlib, and writes deterministic TSV summaries.

**Tech Stack:** C++17, OpenMP, existing NavigaMer search/index persistence APIs, vendored Edlib, Make/CMake.

---

### Task 1: Candidate-Only Search Contract

**Files:**
- Modify: `navigamer_cpp/include/search_engine.hpp`
- Modify: `navigamer_cpp/src/search_engine.cpp`
- Create: `navigamer_cpp/src/test_candidate_search.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing candidate-search test**

Create a deterministic index containing duplicate sequence occurrences. For each query and `scan/rect` configuration, assert that candidate-only results contain every adaptive hit, are unique by `sequence_id`, and perform no leaf exact verification:

```cpp
auto [candidates, candidate_stats] = engine.search_adaptive_candidates(query, 2);
auto [hits, hit_stats] = engine.search_adaptive(query, 2);
assert(is_subset(ids(hits), ids(candidates)));
assert(unique_sequence_ids(candidates));
assert(candidate_stats.leaf_exact_distance_call_count == 0);
assert(candidate_stats.candidate_verify_count == 0);
for (const auto& candidate : candidates) {
  if (navigamer::compute_distance_bounded_edlib(query.seq, candidate->seq, 2) <= 2)
    verified.insert(candidate->id);
}
assert(verified == ids(hits));
```

- [ ] **Step 2: Add and run the failing test target**

Add `test_candidate_search_bin`, `test_candidate_search`, dependency variables, `test_all`, `clean`, and dependency includes following existing Makefile patterns. Add the equivalent CMake executable with all NavigaMer library sources.

Run: `cd navigamer_cpp && make test_candidate_search`

Expected: compilation fails because `search_adaptive_candidates` is absent.

- [ ] **Step 3: Add the public API and internal leaf mode**

Declare:

```cpp
std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
search_adaptive_candidates(const BioSequence& query_seq, int tolerance);
```

Add a private collector passed through adaptive traversal:

```cpp
enum class LeafOutputMode { VerifyAccepted, CollectCandidates };
struct AdaptiveLeafCollector {
  LeafOutputMode mode;
  std::unordered_map<LeafId, std::shared_ptr<BioSequence>> sequences;
};
```

Replace adaptive traversal's string-keyed `unique_results` parameter with
`AdaptiveLeafCollector&`. After the leaf-beacon sieve, insert by `sequence_id`.
Only `VerifyAccepted` calls exact distance and retains accepted sequences.
Keep greedy and exhaustive paths on their existing verified helper or route
them explicitly through `VerifyAccepted`; do not change their results.

- [ ] **Step 4: Implement one shared adaptive entry point**

Add:

```cpp
std::pair<std::vector<std::shared_ptr<BioSequence>>, SearchStats>
search_adaptive_impl(const BioSequence& query, int tolerance,
                     LeafOutputMode mode);
```

Have `search_adaptive()` call `VerifyAccepted` and
`search_adaptive_candidates()` call `CollectCandidates`. Sort returned vectors
by `sequence_id` before returning so batch output is deterministic. Set
`stats.result_count` to the unique returned sequence count.

- [ ] **Step 5: Run focused and safety tests**

Run:

```bash
cd navigamer_cpp
make test_candidate_search test_recall test_distance_bound test_search_qgram
```

Expected: all targets print their existing pass messages; candidate test prints
`candidate search tests passed`.

- [ ] **Step 6: Commit the candidate API**

```bash
git add navigamer_cpp/include/search_engine.hpp navigamer_cpp/src/search_engine.cpp \
  navigamer_cpp/src/test_candidate_search.cpp navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "feat: expose adaptive leaf candidates"
```

### Task 2: Batch Query Module

**Files:**
- Create: `navigamer_cpp/include/index_batch_query.hpp`
- Create: `navigamer_cpp/src/index_batch_query.cpp`
- Create: `navigamer_cpp/src/test_index_batch_query.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing module test**

Build and persist a tiny stride-1 reference-window index, write two FASTQ
queries, invoke `run_index_batch_query`, and assert detail and summary schemas:

```cpp
navigamer::IndexBatchQueryConfig config;
config.index_path = index_path;
config.reads_path = reads_path;
config.tolerance = 1;
config.detail_path = detail_path;
config.summary_path = summary_path;
config.threads = 2;
config.batch_size = 2;
const auto result = navigamer::run_index_batch_query(config);
assert(result.num_reads == 2);
assert(result.false_negative_count == 0);
assert(first_line(detail_path) ==
  "read_id\ttau\tcandidate_window_id\tcandidate_start\tcandidate_sequence_id\tverified_edit_distance\taccepted_by_exact_verification");
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

Run: `cd navigamer_cpp && make test_index_batch_query`

Expected: compilation fails because `index_batch_query.hpp` is absent.

- [ ] **Step 3: Define batch configuration and result types**

Use these stable names:

```cpp
struct IndexBatchQueryConfig {
  std::string index_path, reads_path, detail_path, summary_path, readset;
  int tolerance = 2;
  int threads = 1;
  size_t batch_size = 256;
  SearchConfig search;
};

struct IndexBatchQuerySummary {
  size_t num_reads = 0;
  size_t raw_window_candidates = 0;
  size_t verified_sequence_candidates = 0;
  size_t accepted_window_candidates = 0;
  size_t false_negative_count = 0;
};
```

- [ ] **Step 4: Implement stable occurrence expansion**

Parse the persisted manifest's existing
`reference-windows:v1;prefix=P;window=W;stride=S` `reads_input` value into a
checked `ReferenceWindowLayout`. Reject a non-window index, malformed fields,
negative starts, starts not divisible by `S`, positions outside `P`, duplicate
positions, missing sequence IDs, and spans different from `W`. For this
single-contig experiment, assign `candidate_window_id = start / S`. Sort each
candidate's `ref_positions` by `(ref_id,start,end,strand)` so output never
depends on hash-map iteration.

- [ ] **Step 5: Implement bounded parallel batches and timing**

Load the index and FASTQ once. Process at most `batch_size` reads at a time with
OpenMP. Each read stores only its current detail rows and timing record; write
the completed batch in input order. For every unique sequence candidate call:

```cpp
const int distance = compute_distance_bounded_edlib(
    read.seq, candidate->seq, config.tolerance);
const bool accepted = distance <= config.tolerance;
```

Record retrieval, verification, and total milliseconds separately. Compute
mean, median, p95, and p99 with a sorted-vector percentile helper using nearest
rank. Write `NA` for oracle-only fields.

- [ ] **Step 6: Run module and persistence regressions**

Run:

```bash
cd navigamer_cpp
make test_index_batch_query test_index_persistence
```

Expected: both tests pass and the tiny index is loaded once according to the
captured stderr log.

- [ ] **Step 7: Commit the batch module**

```bash
git add navigamer_cpp/include/index_batch_query.hpp navigamer_cpp/src/index_batch_query.cpp \
  navigamer_cpp/src/test_index_batch_query.cpp navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "feat: add persisted index batch query module"
```

### Task 3: CLI Command and Documentation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`
- Modify: `navigamer_cpp/src/test_index_batch_query.cpp`

- [ ] **Step 1: Add a failing CLI integration assertion**

Extend the batch test to run:

```bash
./navigamer query-index-batch --index tiny.navidx --reads tiny.fq \
  --tolerance 1 --threads 2 --batch-size 2 --mbb-filter-mode rect \
  --search-qgram-prefilter on --search-qgram-q 3 \
  --out detail.tsv --summary-out summary.tsv
```

Assert exit code zero, exact headers, two read IDs, and no index file mtime
change.

- [ ] **Step 2: Run the integration test before dispatch exists**

Run: `cd navigamer_cpp && make navigamer test_index_batch_query && ./test_index_batch_query_bin`

Expected: FAIL because `query-index-batch` is unknown.

- [ ] **Step 3: Parse and dispatch the command**

Add `--batch-size` with default 256 to the existing argument loop. Require
`--index`, `--reads`, `--out`, and `--summary-out`. Dispatch only to
`run_index_batch_query`; never call build or manifest-matching rebuild paths.
Pass the already parsed adaptive search flags and `--threads`.

- [ ] **Step 4: Document exact syntax and semantics**

Add this example to both READMEs and the full option/column definitions to
`CLI_REFERENCE.md`:

```bash
./navigamer query-index-batch --index reference.navidx --reads queries.fq \
  --tolerance 2 --threads 8 --batch-size 256 \
  --mbb-filter-mode rect --search-qgram-prefilter on --search-qgram-q 5 \
  --out candidates.tsv --summary-out summary.tsv
```

State that the index is loaded once, the command never rebuilds it, raw leaf
candidates are verified with bounded Edlib, and timings are separated.

- [ ] **Step 5: Run complete NavigaMer validation**

Run:

```bash
cd navigamer_cpp
make -j
make test_candidate_search test_index_batch_query test_recall test_distance_bound test_index_persistence
./test_candidate_search_bin
./test_index_batch_query_bin
./test_recall
./test_distance_bound
./test_index_persistence_bin
```

Expected: all commands exit zero. Do not invoke the 1.1 Mb index.

- [ ] **Step 6: Commit CLI and docs**

```bash
git add navigamer_cpp/src/main.cpp README.md navigamer_cpp/README.md \
  navigamer_cpp/CLI_REFERENCE.md navigamer_cpp/src/test_index_batch_query.cpp
git commit -m "feat: add query-index-batch CLI"
```
