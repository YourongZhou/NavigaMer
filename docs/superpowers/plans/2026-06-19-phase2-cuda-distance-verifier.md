# Phase2 CUDA Distance Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional CUDA backend for Phase2 bounded edit-distance verification while preserving CPU behavior, deterministic topology, persisted index compatibility, and recall safety.

**Architecture:** Extract Phase2 exact verification behind a focused batch verifier API. The CPU implementation wraps the existing bounded build distance path; the CUDA implementation is compiled only with `NAVIGAMER_WITH_CUDA=1` and falls back to CPU in `auto` mode when unavailable. Phase2 candidate generation, q-gram filtering, sorting, edge insertion, and query-time structures stay unchanged.

**Tech Stack:** C++17, OpenMP, edlib, optional CUDA 12.x via `nvcc`, existing Makefile and CMake builds.

---

### Task 1: Backend Type and CPU Batch Verifier

**Files:**
- Create: `navigamer_cpp/include/phase2_distance_verifier.hpp`
- Create: `navigamer_cpp/src/phase2_distance_verifier.cpp`
- Create: `navigamer_cpp/src/test_phase2_distance_verifier.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing parser and CPU verifier test**

Create `src/test_phase2_distance_verifier.cpp` with tests that parse
`auto`, `cpu`, and `cuda`, reject an invalid backend, and verify that a CPU
batch returns the same accepted pair IDs as direct bounded CPU calls.

```cpp
std::vector<std::string> parents = {"ACGTACGT", "AAAAACCC"};
std::vector<std::string> children = {"ACGTTCGT", "TTTTTTTT", "AAAAACCA"};
std::vector<navigamer::Phase2DistancePair> pairs = {
    {0, 0, 1}, {0, 1, 2}, {1, 2, 1}};
auto verifier = navigamer::make_phase2_distance_verifier(
    navigamer::Phase2DistanceBackend::Cpu,
    navigamer::BuildDistanceMode::Edlib);
auto result = verifier->verify(parents, children, pairs);
assert(result.accepted_pair_indices == std::vector<size_t>({0, 2}));
```

- [ ] **Step 2: Run the test and verify RED**

Run: `cd navigamer_cpp && make test_phase2_distance_verifier`

Expected: compilation fails because the header and target do not exist.

- [ ] **Step 3: Implement the CPU verifier API**

Expose:

```cpp
enum class Phase2DistanceBackend { Auto, Cpu, Cuda };
const char* phase2_distance_backend_name(Phase2DistanceBackend backend);
Phase2DistanceBackend parse_phase2_distance_backend(const std::string& value);

struct Phase2DistancePair {
  size_t parent_idx;
  size_t child_idx;
  int tau;
};

struct Phase2DistanceBatchResult {
  std::vector<size_t> accepted_pair_indices;
  Phase2DistanceBackend backend_used = Phase2DistanceBackend::Cpu;
  size_t cpu_fallback_count = 0;
};

class Phase2DistanceVerifier {
 public:
  virtual ~Phase2DistanceVerifier() = default;
  virtual Phase2DistanceBatchResult verify(
      const std::vector<std::string>& parent_sequences,
      const std::vector<std::string>& child_sequences,
      const std::vector<Phase2DistancePair>& pairs) = 0;
};
```

The CPU implementation loops over the batch, calls the existing bounded build
distance function through a shared helper, and records accepted pair indices in
input order.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
cd navigamer_cpp
make test_phase2_distance_verifier
./test_phase2_distance_verifier
```

Expected: parser and CPU verifier tests pass.

### Task 2: Phase2 Integration With CPU Semantics

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/test_build_range_equivalence.cpp`
- Modify: `navigamer_cpp/src/test_build_timing_stats.cpp`

- [ ] **Step 1: Write failing integration assertions**

Add config fields for requested/used backend and assert that forcing `cpu`
produces the same topology as the existing indexed construction.

```cpp
config.phase2_distance_backend = navigamer::Phase2DistanceBackend::Cpu;
builder.build(sequences);
assert(builder.get_statistics().phase2_distance_backend_used ==
       navigamer::Phase2DistanceBackend::Cpu);
```

- [ ] **Step 2: Run and verify RED**

Run: `cd navigamer_cpp && make test_build_range`

Expected: compilation fails because the config/stat fields do not exist.

- [ ] **Step 3: Wire Phase2 through the verifier**

For each parent layer, keep the existing per-parent candidate query. Collect
candidate pairs into a per-thread batch, run the verifier, and translate
accepted batch indices back to `(parent_idx, child_idx)` edges. Preserve the
existing q-gram postfilter and length pruning before adding a pair to the
batch.

- [ ] **Step 4: Run integration tests**

Run:

```bash
cd navigamer_cpp
make test_phase2_distance_verifier test_build_range test_build_timing_stats
./test_phase2_distance_verifier
./test_build_range_equivalence
./test_build_timing_stats
```

Expected: all pass, and CPU topology is unchanged.

### Task 3: Optional CUDA Build and CUDA Verifier

**Files:**
- Create: `navigamer_cpp/src/phase2_distance_verifier_cuda.cu`
- Modify: `navigamer_cpp/include/phase2_distance_verifier.hpp`
- Modify: `navigamer_cpp/src/phase2_distance_verifier.cpp`
- Modify: `navigamer_cpp/src/test_phase2_distance_verifier.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Add a CUDA availability/equivalence test**

Extend the verifier test so `auto` always works, `cuda` either reports
unavailable in CPU-only builds or matches CPU when compiled with CUDA support.
Use deterministic random sequences of length 1 to 250 and `tau` values from 0
to 250.

- [ ] **Step 2: Run CPU-only RED/GREEN guard**

Run: `cd navigamer_cpp && make test_phase2_distance_verifier`

Expected: CPU-only build passes and reports CUDA unavailable without requiring
CUDA libraries.

- [ ] **Step 3: Add optional CUDA compilation**

In Makefile, add `NAVIGAMER_WITH_CUDA ?= 0`, `NVCC ?= nvcc`, compile `.cu`
objects only when enabled, and define `-DNAVIGAMER_WITH_CUDA`.

In CMake, add `option(NAVIGAMER_WITH_CUDA ...)`; when enabled, call
`enable_language(CUDA)`, add the `.cu` source to relevant targets, and define
`NAVIGAMER_WITH_CUDA`.

- [ ] **Step 4: Implement the CUDA batch verifier**

Implement a correctness-first kernel that computes bounded edit distance for
each submitted pair and writes one accepted flag per pair. The host wrapper
packs sequences and pair descriptors, launches the kernel in chunks, copies
flags back, and returns accepted pair indices in input order. In `auto`, any
CUDA initialization or allocation failure returns to CPU.

- [ ] **Step 5: Run CUDA verifier tests**

Run:

```bash
cd navigamer_cpp
make clean
make NAVIGAMER_WITH_CUDA=1 test_phase2_distance_verifier
./test_phase2_distance_verifier
```

Expected: CUDA and CPU accepted-pair sets match exactly.

### Task 4: CLI, Persistence Signature, and Documentation

**Files:**
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/include/index_persistence.hpp`
- Modify: `navigamer_cpp/src/index_persistence.cpp`
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`
- Modify: `navigamer_cpp/src/test_index_persistence.cpp`

- [ ] **Step 1: Add failing CLI/persistence tests**

Assert the new CLI option parses, appears in summary output, and does not
invalidate a persisted index built with the same semantic construction
parameters.

- [ ] **Step 2: Run and verify RED**

Run: `cd navigamer_cpp && make test_index_persistence`

Expected: failure until the CLI/config plumbing exists.

- [ ] **Step 3: Add the CLI flag**

Add `--phase2-distance-backend auto|cpu|cuda` to build/run/benchmark paths that
construct an index. Store the requested and used backend in statistics output.
Do not add this field to the semantic persistence signature; record it only as
diagnostic metadata if metadata output already has a non-semantic section.

- [ ] **Step 4: Update documentation**

Document default behavior, CPU-only fallback, CUDA build command, and the fact
that query semantics and persisted index compatibility are unchanged.

### Task 5: Correctness and Performance Gates

**Files:**
- No additional production files expected.

- [ ] **Step 1: Run focused correctness gates**

Run:

```bash
cd navigamer_cpp
make NAVIGAMER_WITH_CUDA=1 test_phase2_distance_verifier test_build_range test_recall test_dist test_map150
./test_phase2_distance_verifier
./test_build_range_equivalence
./test_recall
./test_distance_bound
./test_map150_recall
```

Expected: all pass with identical CPU/GPU topology where CUDA is enabled.

- [ ] **Step 2: Run a prefix benchmark**

Run a 128 kb or larger fixed-window E. coli prefix with CPU and CUDA backends,
same radii, same threads, and CPU0 excluded if the old long-running process is
still active. Compare Phase2 per-layer logs, exact verifier time, total time,
and accepted-edge counts.

- [ ] **Step 3: Report full E. coli estimate**

Use the largest successful CUDA prefix to estimate the full E. coli runtime.
Call out layers still dominated by candidate generation and identify the next
optimization target if the estimate remains above 24 hours.
