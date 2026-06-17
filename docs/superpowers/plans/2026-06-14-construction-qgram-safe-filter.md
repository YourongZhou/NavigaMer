# Construction Q-Gram Safe Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact q-gram and hybrid candidate generation to construction range joins without changing phase-2 edges, leaf attachments, or search results.

**Architecture:** A standalone `QGramCountIndex` computes safe q-gram candidate supersets. `ExactRangeJoinIndex` orchestrates full, pigeonhole, q-gram, hybrid, and auto strategies, while `BioGeometryIndexBuilder` remains solely responsible for exact bounded verification and result materialization.

**Tech Stack:** C++17, STL containers, OpenMP, GNU Make, CMake, existing NavigaMer edit-distance and builder APIs.

---

### Task 1: Add Q-Gram Count and L1 Primitives

**Files:**
- Create: `navigamer_cpp/include/qgram_filter.hpp`
- Create: `navigamer_cpp/src/qgram_filter.cpp`
- Create: `navigamer_cpp/src/test_qgram_filter.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write failing basic count and L1 tests**

Add tests that assert:

```cpp
auto counts = navigamer::compute_qgram_counts("ACGTAC", 2);
assert(counts.at("AC") == 2);
assert(counts.at("CG") == 1);
assert(counts.at("GT") == 1);
assert(counts.at("TA") == 1);
assert(navigamer::qgram_total("ACGTAC", 2) == 5);
assert(navigamer::compute_qgram_l1("ACGT", "ACGT", 2) == 0);
```

- [ ] **Step 2: Register and run the test to verify RED**

Run:

```bash
cd navigamer_cpp
make test_qgram
```

Expected: compilation fails because `qgram_filter.hpp` and its functions do not
exist.

- [ ] **Step 3: Implement minimal q-gram primitives**

Define string-keyed counts:

```cpp
using QGramCounts = std::unordered_map<std::string, int>;

QGramCounts compute_qgram_counts(const std::string& sequence, int q);
size_t qgram_total(const std::string& sequence, int q);
size_t compute_qgram_l1(
    const std::string& lhs, const std::string& rhs, int q);
```

For `q <= 0`, throw `std::invalid_argument`. For sequences shorter than `q`,
return zero total and an empty count map.

- [ ] **Step 4: Run the q-gram test to verify GREEN**

Run:

```bash
cd navigamer_cpp
make test_qgram
```

Expected: basic count and L1 tests pass.

- [ ] **Step 5: Add randomized L1-bound and ambiguous-base tests**

For lengths `20, 50, 100, 250`, thresholds `0,1,2,5,10,20`, and q values
`3,4,5`, generate substitution/indel mutations and assert:

```cpp
if (compute_distance(a, b) <= tau) {
  assert(compute_qgram_l1(a, b, q) <=
         static_cast<size_t>(2 * q * tau));
}
```

Include strings containing `N` and assert deterministic counts and the same
necessary condition.

- [ ] **Step 6: Run the expanded q-gram test**

Run:

```bash
cd navigamer_cpp
make test_qgram
```

Expected: all q-gram primitive tests pass.

- [ ] **Step 7: Commit**

```bash
git add navigamer_cpp/include/qgram_filter.hpp navigamer_cpp/src/qgram_filter.cpp \
  navigamer_cpp/src/test_qgram_filter.cpp navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "add qgram count primitives"
```

### Task 2: Implement QGramCountIndex

**Files:**
- Modify: `navigamer_cpp/include/qgram_filter.hpp`
- Modify: `navigamer_cpp/src/qgram_filter.cpp`
- Modify: `navigamer_cpp/src/test_qgram_filter.cpp`

- [ ] **Step 1: Write failing no-false-negative index tests**

Build an index over random strings and assert every full-distance match is in
the sorted, unique candidate list. Cover:

```cpp
QGramCountIndex index(5);
index.build(items);
auto candidates = index.query(query, tau, &stats);
```

Include mixed lengths, `N`, sequences shorter than q, and thresholds where
`required_shared <= 0`.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
cd navigamer_cpp
make test_qgram
```

Expected: compilation fails because `QGramCountIndex` is missing.

- [ ] **Step 3: Implement dense q-gram postings**

Store target metadata by dense internal index and postings shaped as:

```cpp
std::unordered_map<std::string, std::vector<Posting>> postings_;
```

where each posting records internal index and target multiplicity.

- [ ] **Step 4: Implement safe query logic**

Apply length filtering, accumulate multiset shared counts, conservatively
include short sequences, and compare shared counts against:

```cpp
const long long numerator =
    query_total + target_total - 2LL * q_ * tau;
const long long required_shared = (numerator + 1) / 2;
```

Only use this ceiling expression when `numerator > 0`; otherwise include the
target and increment `required_shared_nonpositive`.

- [ ] **Step 5: Run test to verify GREEN**

Run:

```bash
cd navigamer_cpp
make test_qgram
```

Expected: all q-gram index no-false-negative tests pass.

- [ ] **Step 6: Commit**

```bash
git add navigamer_cpp/include/qgram_filter.hpp navigamer_cpp/src/qgram_filter.cpp \
  navigamer_cpp/src/test_qgram_filter.cpp
git commit -m "add exact qgram count index"
```

### Task 3: Add Range Join Candidate Modes and Hybrid

**Files:**
- Modify: `navigamer_cpp/include/range_join.hpp`
- Modify: `navigamer_cpp/src/range_join.cpp`
- Modify: `navigamer_cpp/src/test_range_join.cpp`

- [ ] **Step 1: Write failing forced-mode and hybrid tests**

Extend `test_range_join.cpp` to construct configs for all modes and assert:

```cpp
verified_matches(qgram_result) == full_matches;
verified_matches(hybrid_result) == full_matches;
hybrid_candidates ==
    set_intersection(pigeonhole_candidates, qgram_candidates);
```

Also assert auto selects pigeonhole for a valid long seed and q-gram when the
seed is shorter than `min_seed_len`.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
cd navigamer_cpp
make test_range_join_check
```

Expected: compilation fails because candidate modes and q-gram integration are
missing.

- [ ] **Step 3: Extend range join config and result types**

Add:

```cpp
enum class RangeCandidateMode {
  Auto,
  PigeonholeOnly,
  QGramOnly,
  Hybrid,
  FullScan,
};
```

Add parse/name helpers, `qgram_q`, configured mode, actual mode, and query
statistics to `RangeJoinQueryResult`.

- [ ] **Step 4: Implement strategy orchestration**

Refactor pigeonhole generation into a helper returning a safe candidate result.
Implement:

- Full length-compatible scan.
- Forced pigeonhole with explicit fallback.
- Forced q-gram.
- Hybrid sorted intersection.
- Auto pigeonhole-or-q-gram selection.

Keep every returned ID sorted and unique.

- [ ] **Step 5: Run range join and q-gram tests**

Run:

```bash
cd navigamer_cpp
make test_qgram test_range_join_check
```

Expected: both test binaries pass.

- [ ] **Step 6: Commit**

```bash
git add navigamer_cpp/include/range_join.hpp navigamer_cpp/src/range_join.cpp \
  navigamer_cpp/src/test_range_join.cpp
git commit -m "add qgram and hybrid range join modes"
```

### Task 4: Integrate Builder Statistics and Per-Pair Length Filtering

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/test_build_range_equivalence.cpp`

- [ ] **Step 1: Write failing construction equivalence tests**

Build the same input using full, forced q-gram, forced hybrid, and auto configs.
Assert exact equality of:

```cpp
primary_edges(full_builder) == primary_edges(candidate_builder);
leaf_links(full_builder) == leaf_links(candidate_builder);
hit_sequences(full_builder, query, tau) ==
    hit_sequences(candidate_builder, query, tau);
```

Add input containing `N` and assertions that mode counters are populated.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
cd navigamer_cpp
make test_build_range
```

Expected: compilation fails because the new statistics are missing.

- [ ] **Step 3: Extend builder statistics**

Add phase-2 and leaf fields for:

- Pigeonhole, q-gram, and hybrid query counts.
- Q-gram candidate pairs.
- Q-gram L1-pruned pairs.
- Length-pruned pairs.
- Required-shared-nonpositive counts.
- Leaf exact-distance reduction ratio.

- [ ] **Step 4: Accumulate range query statistics**

In both indexed construction paths, accumulate query result counters before
verification. Before each bounded verification, apply the individual pair's
safe length filter. Increment exact-call counters only immediately before
`compute_distance_bounded`.

- [ ] **Step 5: Expand builder summary output**

Print configured candidate mode, q value, per-mode queries, q-gram counters,
length pruning, and both candidate/exact reduction ratios.

- [ ] **Step 6: Run construction equivalence tests**

Run:

```bash
cd navigamer_cpp
make test_build_range
```

Expected: full, q-gram, hybrid, and auto graph/leaf/search sets are identical.

- [ ] **Step 7: Commit**

```bash
git add navigamer_cpp/include/index_builder.hpp navigamer_cpp/src/index_builder.cpp \
  navigamer_cpp/src/test_build_range_equivalence.cpp
git commit -m "integrate qgram construction range joins"
```

### Task 5: Add CLI Flags and Documentation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] **Step 1: Write failing CLI smoke expectations**

Run:

```bash
cd navigamer_cpp
make -j
./navigamer demo --size 20 --range-candidate-mode qgram --qgram-q 5
```

Expected before implementation: the flags are ignored or absent from summary,
which fails the intended CLI behavior.

- [ ] **Step 2: Add CLI parsing and validation**

Parse:

```text
--range-candidate-mode auto|pigeonhole|qgram|hybrid|full
--qgram-q N
```

Set `BuildRangeConfig::range_join`, reject invalid modes and non-positive q, and
include the flags in usage output.

- [ ] **Step 3: Update documentation**

Document exact mode semantics, hybrid intersection safety, q-gram necessary
condition, defaults, counters, and continued bounded exact verification in all
three requested documentation files.

- [ ] **Step 4: Run CLI smoke tests**

Run:

```bash
cd navigamer_cpp
make -j
./navigamer demo --size 20 --range-candidate-mode qgram --qgram-q 5
./navigamer demo --size 20 --range-candidate-mode hybrid --qgram-q 5
./navigamer demo --size 20 --range-candidate-mode auto --qgram-q 5
```

Expected: all commands complete and summaries identify the configured mode.

- [ ] **Step 5: Commit**

```bash
git add navigamer_cpp/src/main.cpp README.md navigamer_cpp/README.md \
  navigamer_cpp/CLI_REFERENCE.md
git commit -m "document construction qgram candidate modes"
```

### Task 6: Complete Build-System Coverage and Full Verification

**Files:**
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Ensure q-gram source links into every dependent target**

Add `src/qgram_filter.cpp` to CLI/library source lists and all CMake targets that
compile `range_join.cpp`. Add `test_qgram_filter` to Make and CMake test target
lists, including clean and optional SeqAn paths.

- [ ] **Step 2: Run full Make validation**

Run:

```bash
cd navigamer_cpp
make -j
make test_all
```

Expected: all existing and new tests pass.

- [ ] **Step 3: Run required focused regression binaries**

Run:

```bash
cd navigamer_cpp
./test_recall
./test_distance_bound
./test_qgram_filter
./test_range_join
./test_build_range_equivalence
./test_mbb_rect_index
./test_mbb_filter_equivalence
```

Expected: every binary exits successfully.

- [ ] **Step 4: Verify CMake build**

Run:

```bash
cd navigamer_cpp
cmake -S . -B /home/tmp/navigamer-qgram-build -DCMAKE_BUILD_TYPE=Release
cmake --build /home/tmp/navigamer-qgram-build -j
```

Expected: all configured targets compile.

- [ ] **Step 5: Commit**

```bash
git add navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "complete qgram build and test coverage"
```

### Task 7: Run Candidate-Mode Benchmark and Record Summary

**Files:**
- Create: `docs/benchmarks/2026-06-14-construction-qgram-candidate-modes.md`

- [ ] **Step 1: Select a deterministic manageable dataset**

Use `data/human/chr1_subset` with 250 bp stride-1 windows if it completes within
the available environment. If not, document and use a deterministic prefix
large enough to exercise construction pruning.

- [ ] **Step 2: Run full and indexed candidate modes**

Measure wall time and capture builder summaries for:

```bash
./navigamer benchmark ... --link-mode full --leaf-attach-mode full
./navigamer benchmark ... --range-candidate-mode pigeonhole
./navigamer benchmark ... --range-candidate-mode qgram
./navigamer benchmark ... --range-candidate-mode hybrid
./navigamer benchmark ... --range-candidate-mode auto
```

- [ ] **Step 3: Record benchmark and equality evidence**

Write a table containing build time, phase-2/leaf possible pairs, candidates,
exact calls, accepted results, fallback counts, q-gram query counts, and q-gram
L1 pruning. State the dataset subset and exact command lines.

- [ ] **Step 4: Run final diff and status checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated pre-existing generated files remain
uncommitted.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/2026-06-14-construction-qgram-candidate-modes.md
git commit -m "benchmark construction qgram candidate modes"
```
