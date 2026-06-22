# Candidate Baseline Indexes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone C++ tool with persisted contiguous, spaced-seed, safe q-gram, pigeonhole, randstrobe, and faithful TensorSketch/HNSW candidate indexes.

**Architecture:** Keep reference/window encoding, occurrence postings, persistence, and evaluation interfaces separate. Use versioned binary payloads for compatibility checks and emit JSON manifests as trace metadata. TensorSketch is a conditional adapter over the existing `ts::Tensor` and HNSW headers, never a substitute implementation.

**Tech Stack:** C++17, OpenMP, NavigaMer vendored Edlib, `ts::Tensor<int>`, hnswlib, Make, deterministic binary serialization.

---

### Task 1: Experiment Tool Skeleton and Reference Model

**Files:**
- Create: `experiments/ecoli_1p1m/Makefile`
- Create: `experiments/ecoli_1p1m/src/reference_windows.hpp`
- Create: `experiments/ecoli_1p1m/src/reference_windows.cpp`
- Create: `experiments/ecoli_1p1m/src/candidate_tool.cpp`
- Create: `experiments/ecoli_1p1m/src/test_reference_windows.cpp`

- [ ] **Step 1: Write failing reference-window tests**

Test multiline FASTA parsing, uppercase normalization, rejection of multiple
contigs for this experiment, stride-1 numbering, and occurrence-to-window
interval expansion:

```cpp
ReferenceWindows ref = ReferenceWindows::from_fasta(path, 4, 1);
assert(ref.sequence() == "ACGTACGT");
assert(ref.size() == 5);
assert(ref.window(2) == "GTAC");
assert(ref.window_id_for_start(2) == 2);
assert(ref.covering_window_ids(3, 2) == std::vector<uint32_t>({0,1,2,3}));
```

- [ ] **Step 2: Add Make targets and verify failure**

Build `candidate_tool` and `test_reference_windows` with includes from
`navigamer_cpp/include` and Edlib sources. Run
`make -C experiments/ecoli_1p1m test_reference_windows` and expect missing-type
compilation errors.

- [ ] **Step 3: Implement the immutable reference model**

Expose:

```cpp
class ReferenceWindows {
 public:
  static ReferenceWindows from_fasta(const std::string&, uint32_t, uint32_t);
  uint32_t size() const;
  std::string_view window(uint32_t id) const;
  uint32_t start(uint32_t id) const;
  std::vector<uint32_t> covering_window_ids(uint32_t occurrence_start,
                                             uint32_t span) const;
};
```

Use checked 64-bit arithmetic before narrowing IDs. A seed occurrence expands
to starts satisfying `window_start <= occurrence_start` and
`window_start + window_length >= occurrence_start + span`.

- [ ] **Step 4: Add minimal CLI help and build metadata command**

Support `candidate_tool --help` and `candidate_tool inspect-reference --ref ...
--window 150 --stride 1`, printing contig ID, length, and window count as TSV.
Unknown flags and commands return exit code 1.

- [ ] **Step 5: Run and commit**

Run `make -C experiments/ecoli_1p1m test_reference_windows` and expect
`reference window tests passed`.

```bash
git add experiments/ecoli_1p1m/Makefile experiments/ecoli_1p1m/src
git commit -m "feat: scaffold candidate baseline tool"
```

### Task 2: Hashing, Manifest, and Atomic Persistence

**Files:**
- Create: `experiments/ecoli_1p1m/src/sha256.hpp`
- Create: `experiments/ecoli_1p1m/src/sha256.cpp`
- Create: `experiments/ecoli_1p1m/src/index_persistence.hpp`
- Create: `experiments/ecoli_1p1m/src/index_persistence.cpp`
- Create: `experiments/ecoli_1p1m/src/test_index_persistence.cpp`
- Modify: `experiments/ecoli_1p1m/Makefile`

- [ ] **Step 1: Write failing digest and persistence tests**

Assert SHA-256 known vectors, binary round trip, checksum rejection, and
manifest mismatch rejection:

```cpp
assert(sha256_hex("abc") ==
  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
write_index_atomic(path, manifest, payload);
assert(read_index(path, manifest).payload == payload);
flip_last_byte(path);
assert_throws([&] { read_index(path, manifest); }, "checksum");
```

- [ ] **Step 2: Run and observe missing persistence symbols**

Run: `make -C experiments/ecoli_1p1m test_index_persistence`

Expected: compilation fails.

- [ ] **Step 3: Define the binary and manifest contract**

Use fixed magic `ECOLIBL1`, format version 1, little-endian fixed-width fields,
length-prefixed strings/vectors, payload length, and SHA-256 payload digest.
Define `IndexManifest` with method, parameter string, reference path/hash/length,
window/stride/count, build command/seconds, bytes, timestamp, and git commit.

- [ ] **Step 4: Implement atomic writes and JSON sidecar**

Write `index.bin.tmp.<pid>`, flush/close, read it back with the expected
manifest, then rename. Emit `manifest.json` with escaped strings and structured
`parameters` pairs. Compatibility is checked from the typed binary header; JSON
is trace output, not reparsed with string matching.

- [ ] **Step 5: Run and commit**

Run `make -C experiments/ecoli_1p1m test_index_persistence` and expect
`candidate persistence tests passed`.

```bash
git add experiments/ecoli_1p1m/src/sha256.* \
  experiments/ecoli_1p1m/src/index_persistence.* \
  experiments/ecoli_1p1m/src/test_index_persistence.cpp experiments/ecoli_1p1m/Makefile
git commit -m "feat: persist candidate index manifests"
```

### Task 3: Generic Occurrence Index and Contiguous K-mers

**Files:**
- Create: `experiments/ecoli_1p1m/src/occurrence_index.hpp`
- Create: `experiments/ecoli_1p1m/src/occurrence_index.cpp`
- Create: `experiments/ecoli_1p1m/src/candidate_indexes.hpp`
- Create: `experiments/ecoli_1p1m/src/contiguous_index.cpp`
- Create: `experiments/ecoli_1p1m/src/test_seed_indexes.cpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_tool.cpp`
- Modify: `experiments/ecoli_1p1m/Makefile`

- [ ] **Step 1: Write naive-equivalence and round-trip tests**

For random 80 bp references and 20 bp queries, compare `ContiguousIndex::query`
with a naive per-window shared-k-mer implementation for k 5 and 7. Save, load,
and compare candidate IDs again.

- [ ] **Step 2: Run the failing test**

Run: `make -C experiments/ecoli_1p1m test_seed_indexes`

Expected: missing `ContiguousIndex` errors.

- [ ] **Step 3: Implement sorted postings**

Represent postings as a sorted vector of `(uint64_t key, uint32_t position)` and
a directory of `(key, begin, end)`. Build keys with rolling 2-bit A/C/G/T
encoding; reset on ambiguous bases. Serialize vectors directly through the
typed persistence writer.

- [ ] **Step 4: Implement candidate union by window intervals**

For each distinct query k-mer, look up reference positions and mark every valid
covering window in a reusable epoch vector. Return marked IDs in ascending
order. This must exactly match the naive window-level rule.

- [ ] **Step 5: Add CLI build/query commands**

Support:

```bash
candidate_tool build --method contig --k 15 --ref ref.fa --window 150 --stride 1 --out-dir index
candidate_tool query --index index/index.bin --reads reads.fq --tau 2 --out per_read.tsv
```

The query TSV columns are `read_id,tau,raw_candidate_count,candidate_window_ids`.

- [ ] **Step 6: Run and commit**

Run `make -C experiments/ecoli_1p1m test_seed_indexes` and expect pass.

```bash
git add experiments/ecoli_1p1m/src experiments/ecoli_1p1m/Makefile
git commit -m "feat: add persisted contiguous seed index"
```

### Task 4: Spaced Seeds and Randstrobes

**Files:**
- Create: `experiments/ecoli_1p1m/src/spaced_seed_index.cpp`
- Create: `experiments/ecoli_1p1m/src/randstrobe_index.cpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_indexes.hpp`
- Modify: `experiments/ecoli_1p1m/src/test_seed_indexes.cpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_tool.cpp`

- [ ] **Step 1: Add failing deterministic-equivalence tests**

Assert four distinct masks per weight, exact mask weights, save/load equality,
and candidate equality against naive mask extraction. For randstrobes, assert
the same seed produces identical composite keys and a changed seed changes at
least one key on a nontrivial sequence.

- [ ] **Step 2: Run and verify failure**

Run: `make -C experiments/ecoli_1p1m test_seed_indexes`

Expected: missing spaced/randstrobe classes.

- [ ] **Step 3: Implement spaced-mask families**

Expose `make_spaced_masks(weight)` returning four explicit bit vectors with
spans 24, 26, 29, and 32 where possible. Validate `popcount == weight` at
construction. Encode selected bases into a 64-bit key and prefix the mask ID
into the high bits before posting lookup.

- [ ] **Step 4: Implement deterministic order-2 randstrobes**

Encode 15-base strobes in 30 bits. For every first strobe, choose the second
start in `[first+w_min, first+w_max]` minimizing
`splitmix64(first_code ^ rotl(second_code, 17) ^ seed)`, breaking ties by the
smaller position. Pack both codes into 60 bits and index the full span.

- [ ] **Step 5: Persist parameters and add CLI variants**

Store exact masks or randstrobe order, strobe length, window bounds, hash name,
and seed in both typed metadata and JSON. Add `--method spaced --weight W` and
`--method randstrobe --strobe-len 15 --w-min 20 --w-max 50 --seed N`.

- [ ] **Step 6: Run and commit**

Run `make -C experiments/ecoli_1p1m test_seed_indexes` and expect pass.

```bash
git add experiments/ecoli_1p1m/src
git commit -m "feat: add spaced seed and randstrobe indexes"
```

### Task 5: Safe Sliding Q-gram Index

**Files:**
- Create: `experiments/ecoli_1p1m/src/qgram_safe_index.cpp`
- Create: `experiments/ecoli_1p1m/src/test_qgram_safe_index.cpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_indexes.hpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_tool.cpp`
- Modify: `experiments/ecoli_1p1m/Makefile`

- [ ] **Step 1: Write failing full-signature equivalence tests**

For q 3 and 4, variable query lengths, ambiguous symbols, and tau 0 through 3,
compare every returned window with independently computed full q-gram L1.
Also mutate source windows by substitutions/insertions/deletions and assert all
neighbors within tau are retained.

- [ ] **Step 2: Run and observe failure**

Run: `make -C experiments/ecoli_1p1m test_qgram_safe_index`

Expected: missing `QGramSafeIndex`.

- [ ] **Step 3: Persist q-gram code stream**

Store one code or an invalid sentinel for each reference q-gram start, q, and
the first window's `4^(q)` count vector. Reject unsupported q values that cannot
fit the configured dense count vector.

- [ ] **Step 4: Implement constant-update L1 scan**

Build the query counts once. Initialize the first target signature and L1.
When advancing one base, update L1 before and after decrementing the outgoing
code and incrementing the incoming code. Return a window when
`l1 <= 2*q*tau`. If the query or target contains unsupported symbols, compute
that window with the independent conservative full-signature path.

- [ ] **Step 5: Run and commit**

Run `make -C experiments/ecoli_1p1m test_qgram_safe_index` and expect pass.

```bash
git add experiments/ecoli_1p1m/src experiments/ecoli_1p1m/Makefile
git commit -m "feat: add safe sliding qgram index"
```

### Task 6: Conservative Pigeonhole Index

**Files:**
- Create: `experiments/ecoli_1p1m/src/pigeonhole_index.cpp`
- Create: `experiments/ecoli_1p1m/src/test_pigeonhole_index.cpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_indexes.hpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_tool.cpp`
- Modify: `experiments/ecoli_1p1m/Makefile`

- [ ] **Step 1: Write failing exhaustive no-FN tests**

Enumerate all one- and two-edit substitutions, insertions, and deletions of
small source windows. Compare candidate sets with brute-force Edlib and assert
every true neighbor is present for each matching tau.

- [ ] **Step 2: Run and observe failure**

Run: `make -C experiments/ecoli_1p1m test_pigeonhole_index`

Expected: missing `PigeonholeIndex`.

- [ ] **Step 3: Build the tau-specific minimum-block postings**

Set `minimum_block_length = floor((window_length - max_supported_tau) /
(tau + 1))`. Index every reference substring of that length. Persist tau,
nominal read length, supported query-length interval, and block length.

- [ ] **Step 4: Query conservatively around indels**

Split the actual query into `tau+1` blocks. Use each block's first minimum-length
seed to find occurrences, verify the full block at that occurrence, then derive
window starts from `occurrence - query_block_offset + delta` for every
`delta in [-tau,+tau]`. Clip only after generating the full range and return a
sorted union.

- [ ] **Step 5: Run and commit**

Run `make -C experiments/ecoli_1p1m test_pigeonhole_index` and expect pass.

```bash
git add experiments/ecoli_1p1m/src experiments/ecoli_1p1m/Makefile
git commit -m "feat: add conservative pigeonhole index"
```

### Task 7: Faithful TensorSketch/HNSW Persistence

**Files:**
- Create: `experiments/ecoli_1p1m/src/tensor_index.hpp`
- Create: `experiments/ecoli_1p1m/src/tensor_index.cpp`
- Create: `experiments/ecoli_1p1m/src/test_tensor_index.cpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_tool.cpp`
- Modify: `experiments/ecoli_1p1m/Makefile`

- [ ] **Step 1: Write the conditional save-load test**

With `TENSOR_SKETCH_ROOT` present, build dimensions 16 and 32 on a tiny
reference, save HNSW and exact vectors, reload, and assert identical topK labels
and distances within float tolerance. Assert exact L2 results are sorted by
`(distance,window_id)`.

- [ ] **Step 2: Add dependency detection and verify test failure**

The Makefile must locate `metagraph/src/sketch/tensor.hpp` and an hnswlib include
directory, define `NAVIGAMER_HAVE_TENSOR_SKETCH`, and provide a clear strict-mode
error. Run `make -C experiments/ecoli_1p1m test_tensor_index` and expect missing
adapter symbols.

- [ ] **Step 3: Implement Tensor conversion and exact vectors**

Map A/C/G/T to 0/1/2/3 and use:

```cpp
ts::Tensor<int> tensor(4, dimension, 5, seed);
std::vector<double> raw = tensor.compute(encoded_window);
std::vector<float> sketch(raw.begin(), raw.end());
```

Persist row-major float vectors when exact mode is requested.

- [ ] **Step 4: Build, save, and load HNSW once**

Use stable `window_id` labels, configured `M` and `efConstruction`, and
`saveIndex(hnsw.bin)`. Load through the hnswlib file constructor, set `efSearch`
at query time, and request top 10,000 once, capped by index size.

- [ ] **Step 5: Record faithful algorithm metadata**

Manifest parameters must say `algorithm=ts::Tensor`, `subsequence_length=5`,
dimension, seed, metric L2, HNSW parameters, dependency source path, and
dependency git commit when available. Never call this a q-gram CountSketch.

- [ ] **Step 6: Run and commit**

Run `make -C experiments/ecoli_1p1m test_tensor_index` and expect pass.

```bash
git add experiments/ecoli_1p1m/src experiments/ecoli_1p1m/Makefile
git commit -m "feat: persist TensorSketch HNSW indexes"
```

### Task 8: Common Verification and Baseline Build Matrix

**Files:**
- Create: `experiments/ecoli_1p1m/src/evaluation.hpp`
- Create: `experiments/ecoli_1p1m/src/evaluation.cpp`
- Create: `experiments/ecoli_1p1m/src/test_evaluation.cpp`
- Modify: `experiments/ecoli_1p1m/src/candidate_tool.cpp`
- Modify: `experiments/ecoli_1p1m/Makefile`

- [ ] **Step 1: Write failing verifier and summary tests**

Assert every raw candidate is verified by
`compute_distance_bounded_edlib`, accepted IDs match brute force on a small
case, percentile nearest-rank behavior is fixed, and `NA` oracle fields remain
literal `NA`.

- [ ] **Step 2: Run and observe failure**

Run: `make -C experiments/ecoli_1p1m test_evaluation`

Expected: missing evaluator symbols.

- [ ] **Step 3: Implement the common evaluator**

Define `PerReadResult` with raw/verified/accepted counts, candidate IDs,
retrieval/verification/total milliseconds, source recovery, and optional oracle
metrics. Verify candidates only through bounded Edlib. Aggregate sorted samples
into mean, median, p95, and p99.

- [ ] **Step 4: Add matrix build and reuse behavior**

Add `candidate_tool build-matrix` to create all required directories and
parameter variants. Matching manifests write `reused=true`; mismatches fail
unless `--rebuild` is supplied. Write one `build_summary.tsv` per index and a
combined summary.

- [ ] **Step 5: Run all candidate tests**

Run:

```bash
make -C experiments/ecoli_1p1m test_reference_windows test_index_persistence \
  test_seed_indexes test_qgram_safe_index test_pigeonhole_index \
  test_tensor_index test_evaluation
```

Expected: all tests pass.

- [ ] **Step 6: Commit the evaluator**

```bash
git add experiments/ecoli_1p1m/src experiments/ecoli_1p1m/Makefile
git commit -m "feat: add shared candidate evaluator"
```

