# Build-Scale Persistence And Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `build-scale --index` persist a single reference-window index and add low-overhead, timestamped build progress heartbeats every 600 seconds by default.

**Architecture:** Add a focused `BuildProgressReporter` that owns the timer thread, atomic counters, formatting, and ETA calculation. `BioGeometryIndexBuilder` passes a non-owning reporter pointer into phase methods and publishes progress only in coarse batches. Build-scale reuses the existing version-2 serializer with a deterministic synthetic reads descriptor containing prefix/window/stride, so no index format change is required.

**Tech Stack:** C++17, OpenMP, `std::thread`, atomics and condition variables, existing NavigaMer binary persistence, Make and CMake tests.

---

## File Structure

- Create `navigamer_cpp/include/build_progress.hpp`: reporter interface and progress snapshot model.
- Create `navigamer_cpp/src/build_progress.cpp`: heartbeat thread, timestamp/rate/ETA formatting, and forced boundary reports.
- Create `navigamer_cpp/src/test_build_progress.cpp`: deterministic formatting/default tests plus a short periodic heartbeat test.
- Modify `navigamer_cpp/include/index_builder.hpp`: execution-only progress interval and private phase method parameters.
- Modify `navigamer_cpp/src/index_builder.cpp`: phase lifecycle and coarse progress publication.
- Modify `navigamer_cpp/include/index_persistence.hpp` and `src/index_persistence.cpp`: reference-window manifest helper.
- Modify `navigamer_cpp/src/main.cpp`: CLI parsing, build-scale validation, and save call.
- Modify `navigamer_cpp/src/test_build_scale_smoke.cpp`: real CLI persistence/load/rejection regression tests.
- Modify `navigamer_cpp/src/test_index_persistence.cpp`: manifest parameter and execution-only option tests.
- Modify `navigamer_cpp/Makefile` and `CMakeLists.txt`: compile/link reporter and its test.
- Modify `README.md`, `navigamer_cpp/README.md`, and `navigamer_cpp/CLI_REFERENCE.md`: CLI behavior and progress documentation.

### Task 1: Build Progress Reporter

**Files:**
- Create: `navigamer_cpp/src/test_build_progress.cpp`
- Create: `navigamer_cpp/include/build_progress.hpp`
- Create: `navigamer_cpp/src/build_progress.cpp`
- Modify: `navigamer_cpp/Makefile`
- Modify: `navigamer_cpp/CMakeLists.txt`

- [ ] **Step 1: Write the failing reporter test**

Create a test that constructs `BuildProgressReporter(0, output)`, begins a
known phase, publishes 50 of 100 items, forces a report, and finishes. Assert
the text contains `phase=phase1_sketch`, `completed=50`, `total=100`,
`percent=50.0`, `elapsed_s=`, `rate_per_s=`, `eta_s=`, and a timestamp. Also
construct a reporter with a one-second interval, wait up to 1.5 seconds, and
assert a periodic line appears.

```cpp
std::ostringstream output;
navigamer::BuildProgressReporter reporter(0, output);
reporter.begin_phase("phase1_sketch", 100);
reporter.set_completed(50);
reporter.report_now("heartbeat");
reporter.finish_phase();
const std::string text = output.str();
assert(text.find("phase=phase1_sketch") != std::string::npos);
assert(text.find("completed=50") != std::string::npos);
assert(text.find("total=100") != std::string::npos);
assert(text.find("percent=50.0") != std::string::npos);
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd navigamer_cpp
make test_build_progress
```

Expected: compilation fails because `build_progress.hpp` and
`BuildProgressReporter` do not exist.

- [ ] **Step 3: Implement the reporter**

Define a non-copyable reporter with this public surface:

```cpp
class BuildProgressReporter {
 public:
  explicit BuildProgressReporter(int interval_seconds,
                                 std::ostream& output = std::cerr);
  ~BuildProgressReporter();
  void begin_phase(const std::string& phase, uint64_t total);
  void set_completed(uint64_t completed);
  void advance(uint64_t delta);
  void report_now(const char* event);
  void finish_phase();
};
```

Use atomics for completed/total, a mutex for phase/start/output formatting, and
`condition_variable::wait_for` so destruction stops immediately rather than
waiting 600 seconds. Format each report as one assembled string and perform one
stream insertion. Interval zero does not create a timer thread but explicit
boundary reports still work.

- [ ] **Step 4: Wire the target and verify GREEN**

Add `src/build_progress.cpp` to application/library source lists and add
`test_build_progress` to Make/CMake. Run:

```bash
cd navigamer_cpp
make test_build_progress && ./test_build_progress
```

Expected: `build progress tests passed`.

- [ ] **Step 5: Commit the reporter unit**

```bash
git add navigamer_cpp/include/build_progress.hpp \
        navigamer_cpp/src/build_progress.cpp \
        navigamer_cpp/src/test_build_progress.cpp \
        navigamer_cpp/Makefile navigamer_cpp/CMakeLists.txt
git commit -m "feat: add low-overhead build progress reporter"
```

### Task 2: Builder Progress Integration

**Files:**
- Modify: `navigamer_cpp/include/index_builder.hpp`
- Modify: `navigamer_cpp/src/index_builder.cpp`
- Modify: `navigamer_cpp/src/test_build_timing_stats.cpp`
- Modify: `navigamer_cpp/src/test_index_persistence.cpp`
- Modify: `navigamer_cpp/src/main.cpp`

- [ ] **Step 1: Write failing configuration and CLI tests**

Assert `BuildRangeConfig{}.progress_interval_seconds == 600`. Assert changing
the interval from 600 to 0 does not change `make_index_manifest(...).signature`.
Extend the build-scale smoke command with
`--progress-interval-seconds 0` and assert stderr still contains forced
`event=start` and `event=finish` phase reports. Add a negative CLI command and
assert `--progress-interval-seconds -1` returns nonzero.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd navigamer_cpp
make test_build_timing_stats_check test_index_persistence test_build_scale_smoke_check
```

Expected: compilation or assertion failure because the configuration field,
CLI flag, and reporter integration are absent.

- [ ] **Step 3: Add the execution-only configuration and parser**

Add:

```cpp
int progress_interval_seconds = 600;
```

to `BuildRangeConfig`. Parse `--progress-interval-seconds`, reject negative
values, and include the flag in usage text. Do not add the field to
`manifest_signature_payload`.

- [ ] **Step 4: Instrument phase lifecycle and work units**

Construct one reporter in `BioGeometryIndexBuilder::build`. Pass it to private
phase methods. Use these totals and coarse updates:

- Phase1: total unique sequences; publish every 1024 completed sequences.
- Phase2 indexed: total children across adjacent expanded-layer pairs; each
  OpenMP worker publishes every 256 completed children.
- Phase2 full: total parents across adjacent pairs; publish every completed
  parent.
- Phase3: total current-layer nodes processed during collapse/MBB; each worker
  publishes every 64 nodes.
- Phase4: total finest worlds for world-to-sequence/full mode, or total unique
  sequences for sequence-to-world mode; publish every 256 outer-loop items.

Call `begin_phase` after each existing phase banner and `finish_phase` before
moving to the next phase. Keep existing useful layer summaries.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
cd navigamer_cpp
make test_build_progress test_build_timing_stats_check \
     test_index_persistence test_build_scale_smoke_check
```

Expected: all four targets pass; stderr contains boundary progress lines even
when periodic reporting is disabled.

- [ ] **Step 6: Commit builder progress integration**

```bash
git add navigamer_cpp/include/index_builder.hpp \
        navigamer_cpp/src/index_builder.cpp \
        navigamer_cpp/src/test_build_timing_stats.cpp \
        navigamer_cpp/src/test_index_persistence.cpp \
        navigamer_cpp/src/main.cpp
git commit -m "feat: report long-running build progress"
```

### Task 3: Build-Scale Persistence

**Files:**
- Modify: `navigamer_cpp/include/index_persistence.hpp`
- Modify: `navigamer_cpp/src/index_persistence.cpp`
- Modify: `navigamer_cpp/src/main.cpp`
- Modify: `navigamer_cpp/src/test_index_persistence.cpp`
- Modify: `navigamer_cpp/src/test_build_scale_smoke.cpp`

- [ ] **Step 1: Write failing manifest tests**

Add tests for a wished-for helper:

```cpp
auto manifest = navigamer::make_reference_window_index_manifest(
    "ACGTACGTACGT", 12, 6, 1, hierarchy, build_config);
assert(manifest.reads_input ==
       "reference-windows:v1;prefix=12;window=6;stride=1");
assert(manifest.signature != navigamer::make_reference_window_index_manifest(
       "ACGTACGTACGT", 12, 7, 1, hierarchy, build_config).signature);
```

Also assert prefix and stride changes alter the signature.

- [ ] **Step 2: Write failing CLI persistence tests**

In `test_build_scale_smoke.cpp`, run a single-prefix command with
`--index /tmp/navigamer_build_scale_smoke.navidx`; assert the file exists and
is non-empty. Run:

```bash
./navigamer query-index \
  --index /tmp/navigamer_build_scale_smoke.navidx \
  --query ACGTACGTACGT --tolerance 0
```

and assert exit zero. Then run two prefixes with one index path, assert nonzero,
assert stderr contains `--index requires exactly one prefix length`, and assert
the index file was not created.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
cd navigamer_cpp
make test_index_persistence test_build_scale_smoke_check
```

Expected: helper compilation failure or CLI assertion failure because
build-scale still ignores `--index`.

- [ ] **Step 4: Implement the reference-window manifest helper**

Implement:

```cpp
IndexBuildManifest make_reference_window_index_manifest(
    const std::string& ref_input,
    size_t actual_prefix_length,
    int window_size,
    int stride,
    const HierarchyConfig& hierarchy,
    const BuildRangeConfig& range_config);
```

Create the descriptor exactly as tested and delegate to
`make_index_manifest(ref_input, descriptor, hierarchy, range_config)`.

- [ ] **Step 5: Persist the single build-scale index**

Pass `index_path` into `run_build_scale`. Before opening/building, reject an
index path with any prefix count other than one. After `builder.build(windows)`:

```cpp
if (!index_path.empty()) {
  auto manifest = make_reference_window_index_manifest(
      ref_input, actual_prefix, window_size, stride, config, range_config);
  save_index(index_path, builder, manifest);
  const auto stored = read_index_manifest(index_path);
  std::cerr << "Index saved: " << index_path
            << " signature=" << stored.signature
            << " sequences=" << stored.sequence_count << "\n";
}
```

Only write the successful CSV row after this block.

- [ ] **Step 6: Run focused tests and verify GREEN**

```bash
cd navigamer_cpp
make test_index_persistence test_build_scale_smoke_check
```

Expected: manifest, save/load/query, rejection, and existing timing-only smoke
tests all pass.

- [ ] **Step 7: Commit build-scale persistence**

```bash
git add navigamer_cpp/include/index_persistence.hpp \
        navigamer_cpp/src/index_persistence.cpp \
        navigamer_cpp/src/main.cpp \
        navigamer_cpp/src/test_index_persistence.cpp \
        navigamer_cpp/src/test_build_scale_smoke.cpp
git commit -m "feat: persist single-prefix build-scale indexes"
```

### Task 4: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `navigamer_cpp/README.md`
- Modify: `navigamer_cpp/CLI_REFERENCE.md`

- [ ] **Step 1: Update documentation**

Document the persistent full-reference command:

```bash
./navigamer build-scale \
  --ref ../data/ecoli/fasta/ecoli.fa \
  --window 250 --stride 1 \
  --prefix-lengths 4641652 \
  --index /path/to/ecoli_w250_s1.navidx \
  --out /path/to/ecoli_w250_s1.csv
```

State the single-prefix restriction, default 600-second heartbeat,
`--progress-interval-seconds 0` behavior, and that progress goes to stderr.

- [ ] **Step 2: Run formatting and CPU build verification**

```bash
git diff --check
cd navigamer_cpp
make -j
make test_build_progress test_build_scale_smoke_check test_index_persistence \
     test_build_range test_build_timing_stats_check test test_dist
```

Expected: build succeeds; recall reports 11 passed/0 failed; distance-bound
reports 14 passed/0 failed; all focused tests pass.

- [ ] **Step 3: Run a direct persistence smoke command**

```bash
cd navigamer_cpp
rm -f /tmp/navigamer_reference_windows.navidx
./navigamer build-scale \
  --ref ACGTACGTACGTACGTACGTACGTACGTACGT \
  --window 12 --stride 4 --prefix-lengths 32 \
  --primary-radii 12,6,2 \
  --progress-interval-seconds 0 \
  --index /tmp/navigamer_reference_windows.navidx \
  --out /tmp/navigamer_reference_windows.csv
./navigamer query-index \
  --index /tmp/navigamer_reference_windows.navidx \
  --query ACGTACGTACGT --tolerance 0
```

Expected: `Index saved:` appears, index file is non-empty, and query-index
loads it successfully.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md navigamer_cpp/README.md navigamer_cpp/CLI_REFERENCE.md
git commit -m "docs: document persistent build-scale indexes"
```
