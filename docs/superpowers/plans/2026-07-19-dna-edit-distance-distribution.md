# DNA Edit-Distance Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible OpenMP/WFA2-lib program that measures the exact global Levenshtein-distance PMF for random fixed-length DNA pairs and plots the formal one-million-pair result.

**Architecture:** A standalone experiment CMake project fetches pinned WFA2-lib v2.3.6 and builds a small reusable core library plus CLI. Pair-indexed counter-based random generation feeds thread-private reusable WFA aligners and histograms; reporting and plotting consume only the merged histogram.

**Tech Stack:** C++17, OpenMP, WFA2-lib v2.3.6, CMake/CTest, Python 3, matplotlib.

---

## File map

- Create `experiments/dna_edit_distribution/CMakeLists.txt`: dependency pinning, build targets, and CTest registration.
- Create `experiments/dna_edit_distribution/include/dna_edit_distribution.hpp`: public core, statistics, and reporting API.
- Create `experiments/dna_edit_distribution/src/core.cpp`: deterministic DNA generation, DP reference, WFA wrapper, and OpenMP histogram computation.
- Create `experiments/dna_edit_distribution/src/report.cpp`: summary calculation and CSV/JSON publication.
- Create `experiments/dna_edit_distribution/src/main.cpp`: strict CLI parsing and run orchestration.
- Create `experiments/dna_edit_distribution/tests/test_core.cpp`: direct WFA, DP, statistics, and thread-independence tests.
- Create `experiments/dna_edit_distribution/tests/check_cli.py`: CLI output and plotting integration checks.
- Create `scripts/plot_distribution.py`: discrete PMF plotting.
- Create `experiments/dna_edit_distribution/README.md`: complete build, test, run, and plotting commands.

### Task 1: Exact distance and deterministic parallel core

**Files:**
- Create: `experiments/dna_edit_distribution/CMakeLists.txt`
- Create: `experiments/dna_edit_distribution/tests/test_core.cpp`
- Create: `experiments/dna_edit_distribution/include/dna_edit_distribution.hpp`
- Create: `experiments/dna_edit_distribution/src/core.cpp`

- [ ] **Step 1: Add the CMake test target and failing core tests**

The initial tests include the not-yet-created public header and assert the required behavior:

```cpp
CHECK(exact_distance("ACGT", "ACGT") == 0);
CHECK(exact_distance("ACGT", "AGGT") == 1);
CHECK(exact_distance("ACGT", "ACGGT") == 1);
CHECK(exact_distance("ACGT", "ACT") == 1);
CHECK(levenshtein_dp(a, b) == exact_distance(a, b));
CHECK(compute_histogram(31, 200, 7, 1).counts ==
      compute_histogram(31, 200, 7, 4).counts);
```

CMake must pin `GIT_TAG 0db345a8fe862fd7873d3354c499da385583a65a`, link `wfa2::wfa2_static` and `OpenMP::OpenMP_CXX`, and register `dna_edit_distribution_tests` with CTest.

- [ ] **Step 2: Configure/build and verify RED**

Run:

```bash
conda run -n cpp_env_317 cmake -S experiments/dna_edit_distribution -B build -DCMAKE_BUILD_TYPE=Release
conda run -n cpp_env_317 cmake --build build --target dna_edit_distribution_tests -j
```

Expected: compilation fails because `dna_edit_distribution.hpp` and its functions do not exist yet.

- [ ] **Step 3: Implement the minimal tested core API**

Expose these exact interfaces:

```cpp
struct HistogramResult {
  std::vector<std::uint64_t> counts;
  int actual_threads;
};

void generate_pair(std::size_t length, std::uint64_t seed,
                   std::uint64_t pair_index,
                   std::string& first, std::string& second);
int levenshtein_dp(std::string_view first, std::string_view second);

class ExactWfaAligner {
 public:
  ExactWfaAligner();
  ~ExactWfaAligner();
  ExactWfaAligner(const ExactWfaAligner&) = delete;
  ExactWfaAligner& operator=(const ExactWfaAligner&) = delete;
  int distance(std::string_view first, std::string_view second);
 private:
  wavefront_aligner_t* aligner_;
};

HistogramResult compute_histogram(std::size_t length, std::uint64_t pairs,
                                  std::uint64_t seed, int threads);
```

`ExactWfaAligner` uses `distance_metric = edit`, `alignment_scope = compute_score`, and `wf_heuristic_none`. `compute_histogram` creates one `ExactWfaAligner` inside each OpenMP thread, reuses it for that thread's static loop, accumulates a thread-local `length + 1` vector, and throws after the region if WFA reports failure or a result outside `[0, length]`.

- [ ] **Step 4: Build and verify GREEN**

Run:

```bash
conda run -n cpp_env_317 cmake --build build --target dna_edit_distribution_tests -j
conda run -n cpp_env_317 ctest --test-dir build -R '^dna_edit_distribution_tests$' --output-on-failure
```

Expected: one test target passes with zero failures.

- [ ] **Step 5: Commit the core**

```bash
git add experiments/dna_edit_distribution/CMakeLists.txt \
  experiments/dna_edit_distribution/include/dna_edit_distribution.hpp \
  experiments/dna_edit_distribution/src/core.cpp \
  experiments/dna_edit_distribution/tests/test_core.cpp
git commit -m "feat: add exact random DNA distance core"
```

### Task 2: Statistics, CLI, and machine-readable output

**Files:**
- Modify: `experiments/dna_edit_distribution/include/dna_edit_distribution.hpp`
- Modify: `experiments/dna_edit_distribution/CMakeLists.txt`
- Modify: `experiments/dna_edit_distribution/tests/test_core.cpp`
- Create: `experiments/dna_edit_distribution/tests/check_cli.py`
- Create: `experiments/dna_edit_distribution/src/report.cpp`
- Create: `experiments/dna_edit_distribution/src/main.cpp`

- [ ] **Step 1: Add failing summary and CLI integration tests**

Core tests use a fixed histogram and assert population statistics and nearest-rank quantiles:

```cpp
const std::vector<std::uint64_t> counts{0, 1, 3, 1};
const Summary summary = summarize(counts);
CHECK(summary.mean == 2.0);
CHECK(summary.min == 1);
CHECK(summary.median == 2);
CHECK(summary.max == 3);
CHECK(summary.mode == 2);
CHECK(summary.q05 == 1);
CHECK(summary.q95 == 3);
```

`check_cli.py` runs the binary for 64 pairs, parses all three outputs, checks that histogram counts total 64, all distances are in `[0, 12]`, required summary keys exist, JSON parses, and a second one-thread run has identical counts to the four-thread run.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n cpp_env_317 cmake --build build -j
conda run -n cpp_env_317 ctest --test-dir build -R 'dna_edit_distribution_(tests|cli)' --output-on-failure
```

Expected: build/test fails because `Summary`, `summarize`, the CLI, and report writers are absent.

- [ ] **Step 3: Implement summary/reporting and the strict CLI**

Add:

```cpp
struct Summary {
  double mean;
  double standard_deviation;
  int min;
  int median;
  int max;
  int mode;
  int q05;
  int q95;
};

Summary summarize(const std::vector<std::uint64_t>& counts);
void write_results(const std::filesystem::path& output_dir,
                   const RunParameters& parameters,
                   const HistogramResult& histogram,
                   const Summary& summary,
                   const RunTiming& timing);
```

The CLI defaults are `length=150`, `pairs=1000000`, `seed=20260719`, OpenMP maximum threads, and `output_dir=results`. It rejects malformed or unknown arguments, measures only pair generation/alignment/histogram work for `elapsed_seconds`, validates the total count before output, and writes the specified CSV/JSON schemas with WFA version `2.3.6` and pinned commit metadata.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
conda run -n cpp_env_317 cmake --build build -j
conda run -n cpp_env_317 ctest --test-dir build -R 'dna_edit_distribution_(tests|cli)' --output-on-failure
```

Expected: core and CLI tests pass.

- [ ] **Step 5: Commit reporting and CLI**

```bash
git add experiments/dna_edit_distribution
git commit -m "feat: add DNA distance distribution reports"
```

### Task 3: PMF plotting

**Files:**
- Modify: `experiments/dna_edit_distribution/CMakeLists.txt`
- Modify: `experiments/dna_edit_distribution/tests/check_cli.py`
- Create: `scripts/plot_distribution.py`

- [ ] **Step 1: Extend the integration test to require plot files**

The test invokes:

```python
subprocess.run([
    sys.executable, plot_script,
    "--histogram", str(output_dir / "histogram.csv"),
    "--output-dir", str(output_dir),
], check=True, env={**os.environ, "MPLBACKEND": "Agg"})
assert (output_dir / "edit_distance_distribution.png").stat().st_size > 0
assert (output_dir / "edit_distance_distribution.pdf").stat().st_size > 0
```

- [ ] **Step 2: Run the CLI integration test and verify RED**

Run:

```bash
conda run -n cpp_env_317 ctest --test-dir build -R '^dna_edit_distribution_cli$' --output-on-failure
```

Expected: failure because `scripts/plot_distribution.py` does not exist.

- [ ] **Step 3: Implement the discrete PMF plotter**

The script uses `csv.DictReader`, validates count/probability columns, derives mean, length, and total pairs from the histogram, and draws integer PMF bars plus:

```python
ax.axvline(mean, color="tab:red", linestyle="--", linewidth=1.5,
           label=f"Mean = {mean:.2f}")
ax.set_xlabel("Edit distance")
ax.set_ylabel("Probability")
ax.set_title(f"Random DNA edit-distance distribution (L={length}, pairs={pairs:,})")
figure.savefig(output_dir / "edit_distance_distribution.png", dpi=300,
               bbox_inches="tight")
figure.savefig(output_dir / "edit_distance_distribution.pdf",
               bbox_inches="tight")
```

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
conda run -n cpp_env_317 ctest --test-dir build -R '^dna_edit_distribution_cli$' --output-on-failure
```

Expected: CLI/plot integration passes and both plots are nonempty.

```bash
git add scripts/plot_distribution.py \
  experiments/dna_edit_distribution/CMakeLists.txt \
  experiments/dna_edit_distribution/tests/check_cli.py
git commit -m "feat: plot DNA edit distance PMF"
```

### Task 4: Documentation and full validation

**Files:**
- Create: `experiments/dna_edit_distribution/README.md`

- [ ] **Step 1: Document complete commands and schemas**

The README includes environment, configure, build, CTest, 10,000-pair smoke,
1,000,000-pair formal run, plotting, output schemas, deterministic RNG/thread
behavior, exact WFA attributes, and these commands:

```bash
conda activate cpp_env_317
cmake -S experiments/dna_edit_distribution -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/dna_edit_distribution --length 150 --pairs 10000 --seed 20260719 \
  --threads 16 --output-dir results/smoke_10000
./build/dna_edit_distribution --length 150 --pairs 1000000 --seed 20260719 \
  --threads 16 --output-dir results
python scripts/plot_distribution.py --histogram results/histogram.csv \
  --output-dir results
```

- [ ] **Step 2: Run full fresh verification**

Run:

```bash
conda run -n cpp_env_317 cmake -S experiments/dna_edit_distribution -B build -DCMAKE_BUILD_TYPE=Release
conda run -n cpp_env_317 cmake --build build -j
conda run -n cpp_env_317 ctest --test-dir build --output-on-failure
```

Expected: configure/build exit zero and all experiment tests pass.

- [ ] **Step 3: Commit documentation**

```bash
git add experiments/dna_edit_distribution/README.md
git commit -m "docs: document DNA distance experiment"
```

### Task 5: Smoke run, formal experiment, plot, and audit

**Files:**
- Generate, do not commit: `results/smoke_10000/*`
- Generate, do not commit: `results/histogram.csv`
- Generate, do not commit: `results/summary.csv`
- Generate, do not commit: `results/run_metadata.json`
- Generate, do not commit: `results/edit_distance_distribution.png`
- Generate, do not commit: `results/edit_distance_distribution.pdf`

- [ ] **Step 1: Run and audit the 10,000-pair smoke experiment**

Run the documented smoke command using `min(16, available CPUs)` threads, then parse the three outputs with Python. Expected: counts sum to 10,000; distances are in `[0,150]`; probabilities sum to one within `1e-12`; metadata parameters match the CLI.

- [ ] **Step 2: Run the formal 1,000,000-pair experiment**

Run the documented formal command with the same thread count. Expected: exit zero and three output files in `results/`.

- [ ] **Step 3: Plot and audit formal outputs**

Run the documented plot command, verify both image files are nonempty, parse `summary.csv`, independently recompute mean, standard deviation, quantiles, and total count from `histogram.csv`, and compare them to the summary.

- [ ] **Step 4: Inspect the final diff and generated artifact paths**

Run:

```bash
git diff --check HEAD~4..HEAD
git status --short
```

Expected: source changes are limited to the design/plan, experiment directory,
and plotting script; generated results remain untracked or ignored and are not
committed.
