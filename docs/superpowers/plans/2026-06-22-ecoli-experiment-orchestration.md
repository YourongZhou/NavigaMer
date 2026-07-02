# E. coli Experiment Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver resumable scripts for reference preparation, read/oracle generation, candidate evaluation, external mapping, final tables, and a fast end-to-end smoke test.

**Architecture:** Thin shell entrypoints share validated environment/path helpers and call focused C++ or Python tools. All outputs carry hashes, commands, git commits, and explicit skip/failure states; candidate and mapper results remain separate through final collection.

**Tech Stack:** Bash, Python 3 standard library, C++ candidate tool, NavigaMer CLI, optional BWA-MEM2/minimap2/strobealign.

---

### Task 1: Common Configuration and Reference Preparation

**Files:**
- Create: `experiments/ecoli_1p1m/common.sh`
- Create: `experiments/ecoli_1p1m/run_00_prepare_ref.sh`
- Create: `experiments/ecoli_1p1m/scripts/prepare_reference.py`
- Create: `experiments/ecoli_1p1m/scripts/test_prepare_reference.py`

- [ ] **Step 1: Write failing reference preparation tests**

Using `tempfile`, assert multiline FASTA prefixing preserves the first header,
writes exactly 1,100 bases in test mode, refuses a too-short source, and reuses
a matching output without changing mtime.

- [ ] **Step 2: Run and observe missing script**

Run: `python3 -m unittest experiments.ecoli_1p1m.scripts.test_prepare_reference -v`

Expected: import failure.

- [ ] **Step 3: Implement shared shell defaults**

`common.sh` resolves `REPO_ROOT`, exports the six required defaults, defines
`log`, `warn`, `require_file`, `file_bytes`, `git_commit`, and `resolve_nav_index`.
The resolver distinguishes an explicitly exported `NAV_INDEX` from its default
and never invokes a build command.

- [ ] **Step 4: Implement atomic prefix creation and metadata**

Parse FASTA structurally in Python, uppercase sequence lines, preserve the
first header, wrap output at 80 columns, and write a JSON metadata file with
source/prefix SHA-256, lengths, command, time, and git commit. Use `os.replace`.

- [ ] **Step 5: Run tests and commit**

Run the unittest command and a shell syntax check:
`bash -n experiments/ecoli_1p1m/common.sh experiments/ecoli_1p1m/run_00_prepare_ref.sh`.

```bash
git add experiments/ecoli_1p1m/common.sh experiments/ecoli_1p1m/run_00_prepare_ref.sh \
  experiments/ecoli_1p1m/scripts
git commit -m "feat: prepare ecoli experiment reference"
```

### Task 2: Deterministic Read and Oracle Dataset Generation

**Files:**
- Create: `experiments/ecoli_1p1m/run_03_generate_reads.sh`
- Create: `experiments/ecoli_1p1m/scripts/generate_reads.py`
- Create: `experiments/ecoli_1p1m/scripts/test_generate_reads.py`
- Modify: `experiments/ecoli_1p1m/src/candidate_tool.cpp`
- Modify: `experiments/ecoli_1p1m/src/evaluation.cpp`

- [ ] **Step 1: Write failing generator tests**

Assert byte-identical output for the same seed, exact file names/counts,
150-base exact/sub reads, valid variable mixed-read qualities, exact operation
counts, source bounds, and hard-window ranking reproducibility.

- [ ] **Step 2: Run and observe failure**

Run: `python3 -m unittest experiments.ecoli_1p1m.scripts.test_generate_reads -v`

Expected: import failure.

- [ ] **Step 3: Implement read generation**

Use `random.Random(seed)`. Record each substitution/insertion/deletion in an
edit-script JSON field. Generate the requested 10,000/5,000 main sets and
separate 1,000-read oracle sets per prefix. Write FASTQ plus the exact required
ground-truth columns, seed, and edit script.

- [ ] **Step 4: Add exact oracle command**

Add `candidate_tool oracle --ref prefix.fa --reads oracle.fq --taus 1,2,3,5
--out-dir ...`. For every read/window call bounded Edlib and write sorted comma-
separated window IDs with the required header. Parallelize reads, then emit in
input order.

- [ ] **Step 5: Run tests and commit**

Run generator tests plus `make -C experiments/ecoli_1p1m test_evaluation`.

```bash
git add experiments/ecoli_1p1m/run_03_generate_reads.sh \
  experiments/ecoli_1p1m/scripts/generate_reads.py \
  experiments/ecoli_1p1m/scripts/test_generate_reads.py \
  experiments/ecoli_1p1m/src/candidate_tool.cpp experiments/ecoli_1p1m/src/evaluation.cpp
git commit -m "feat: generate ecoli reads and exact oracles"
```

### Task 3: External Mapper Index Builds

**Files:**
- Create: `experiments/ecoli_1p1m/run_01_build_external_mapper_indexes.sh`
- Create: `experiments/ecoli_1p1m/scripts/test_external_index_script.py`

- [ ] **Step 1: Write failing fake-tool tests**

Place fake `bwa-mem2`, `minimap2`, and `strobealign` executables in a temporary
PATH. Assert exact requested commands, minimap2's three variants, `.sti`
discovery/copy, missing-tool exit 127 rows, and TSV columns/order.

- [ ] **Step 2: Run and observe missing script failure**

Run: `python3 -m unittest experiments.ecoli_1p1m.scripts.test_external_index_script -v`

- [ ] **Step 3: Implement timed nonfatal builds**

Create required directories, use nanosecond timestamps where available, capture
each exit code without disabling the remaining matrix, compute recursive index
bytes, shell-escape the recorded command, and atomically write
`external_mapper_index_build.tsv`.

- [ ] **Step 4: Run tests and commit**

Run the unittest and `bash -n` on the script.

```bash
git add experiments/ecoli_1p1m/run_01_build_external_mapper_indexes.sh \
  experiments/ecoli_1p1m/scripts/test_external_index_script.py
git commit -m "feat: build external mapper indexes"
```

### Task 4: Candidate Build and Evaluation Drivers

**Files:**
- Create: `experiments/ecoli_1p1m/run_02_build_candidate_baseline_indexes.sh`
- Create: `experiments/ecoli_1p1m/run_04_evaluate_candidate_retrieval.sh`
- Create: `experiments/ecoli_1p1m/scripts/merge_candidate_results.py`
- Create: `experiments/ecoli_1p1m/scripts/test_candidate_drivers.py`

- [ ] **Step 1: Write failing fake-tool driver tests**

Use fake `candidate_tool` and `navigamer` commands to assert the exact method
matrix, tau-specific pigeonhole selection, four Tensor dimensions/five topK
values, randstrobe inclusion, main-index fallback, no NavigaMer build command,
and nonzero propagation for safe-method false negatives.

- [ ] **Step 2: Run and observe failure**

Run: `python3 -m unittest experiments.ecoli_1p1m.scripts.test_candidate_drivers -v`

- [ ] **Step 3: Implement the build matrix driver**

Call `candidate_tool build-matrix` with reference/window/stride/hash metadata.
Translate `REBUILD_BASELINES=1` to `--rebuild`. Tensor dependency absence writes
a skipped summary unless strict mode is selected. Never pass the NavigaMer path
to a writable option.

- [ ] **Step 4: Implement main and oracle evaluation loops**

For each readset/tau, invoke baseline evaluation and `query-index-batch` with
the same threads and search profile. Run dedicated prefix indexes against
oracle reads. Merge schemas into `candidate_retrieval_per_read.tsv` and
`candidate_retrieval_summary.tsv`; retain retrieval/verification/total times,
source recovery, `NA` or oracle metrics, and index bytes.

- [ ] **Step 5: Run tests and commit**

Run driver unittests and `bash -n` on both scripts.

```bash
git add experiments/ecoli_1p1m/run_02_build_candidate_baseline_indexes.sh \
  experiments/ecoli_1p1m/run_04_evaluate_candidate_retrieval.sh \
  experiments/ecoli_1p1m/scripts/merge_candidate_results.py \
  experiments/ecoli_1p1m/scripts/test_candidate_drivers.py
git commit -m "feat: orchestrate candidate retrieval evaluation"
```

### Task 5: External Mapper Runs and Truth Scoring

**Files:**
- Create: `experiments/ecoli_1p1m/run_05_run_external_mappers.sh`
- Create: `experiments/ecoli_1p1m/scripts/summarize_sam.py`
- Create: `experiments/ecoli_1p1m/scripts/test_summarize_sam.py`

- [ ] **Step 1: Write failing SAM scoring tests**

Cover unmapped, primary, secondary, supplementary, reverse-strand, exact,
within-5, outside-5, and repeated equivalent mappings. Assert mapped and
primary counts count each read once.

- [ ] **Step 2: Run and observe failure**

Run: `python3 -m unittest experiments.ecoli_1p1m.scripts.test_summarize_sam -v`

- [ ] **Step 3: Implement standard-library SAM parsing**

Parse flags and 1-based POS, convert to zero-based start, join ground truth by
read ID, and count a correct primary start when `abs(mapped_start-source_start)
<= 5`. Emit one mapper/readset summary row.

- [ ] **Step 4: Implement mapper command matrix and timing**

Run the specified BWA-MEM2, minimap2 `-ax sr`, and strobealign `--use-index`
commands for every FASTQ. Missing tool or index produces a skipped row and does
not abort other methods. Preserve SAM paths and wall seconds.

- [ ] **Step 5: Run tests and commit**

Run SAM unittests and `bash -n`.

```bash
git add experiments/ecoli_1p1m/run_05_run_external_mappers.sh \
  experiments/ecoli_1p1m/scripts/summarize_sam.py \
  experiments/ecoli_1p1m/scripts/test_summarize_sam.py
git commit -m "feat: evaluate external mapper runs"
```

### Task 6: Result Collection and Paper Tables

**Files:**
- Create: `experiments/ecoli_1p1m/run_06_collect_results.sh`
- Create: `experiments/ecoli_1p1m/scripts/collect_results.py`
- Create: `experiments/ecoli_1p1m/scripts/test_collect_results.py`

- [ ] **Step 1: Write failing schema and category tests**

Provide fixture summaries and assert all six final files, stable columns,
candidate/full-mapper category separation, retained method parameters, and a
fatal error for a missing required column.

- [ ] **Step 2: Run and observe failure**

Run: `python3 -m unittest experiments.ecoli_1p1m.scripts.test_collect_results -v`

- [ ] **Step 3: Implement strict TSV collection**

Use `csv.DictReader(delimiter="\t")`, explicit required-column sets, stable
sort keys, and `csv.DictWriter`. Paper correctness tables contain recall/FN or
mapper correctness; speed tables contain category-appropriate timing and
throughput without ranking across categories.

- [ ] **Step 4: Run tests and commit**

Run collection unittests and `bash -n`.

```bash
git add experiments/ecoli_1p1m/run_06_collect_results.sh \
  experiments/ecoli_1p1m/scripts/collect_results.py \
  experiments/ecoli_1p1m/scripts/test_collect_results.py
git commit -m "feat: collect paper comparison tables"
```

### Task 7: Smoke Test and Experiment Documentation

**Files:**
- Create: `experiments/ecoli_1p1m/run_smoke_test.sh`
- Create: `experiments/ecoli_1p1m/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write the smoke workflow before running it**

Generate a deterministic 5 kb FASTA and 100 reads under a temporary directory,
build all candidate indexes and a small NavigaMer index, run tau 1/2/3/5 oracle
evaluation, require zero FN for NavigaMer/q-gram/pigeonhole, and report but do
not fail heuristic FN. TensorSketch is strict unless
`ALLOW_MISSING_TENSOR=1`.

- [ ] **Step 2: Ignore only generated experiment products**

Keep `experiments/ecoli_1p1m/**` tracked while continuing to ignore `results/`,
compiled candidate binaries, object/dependency files, Python caches, and smoke
temporary output. Do not unignore the existing third-party `methods/` tree.

- [ ] **Step 3: Document setup, categories, persistence, and run order**

Document the seven requested scripts, existing main-index fallback, no-rebuild
guarantee, TensorSketch dependency, randstrobe versus strobealign distinction,
no-FN and heuristic methods, output schemas, resume/rebuild controls, and exact
commands for smoke and full runs.

- [ ] **Step 4: Run the complete smoke and regression gate**

Run:

```bash
bash experiments/ecoli_1p1m/run_smoke_test.sh
cd navigamer_cpp
make test_recall test_distance_bound test_index_persistence test_candidate_search test_index_batch_query
./test_recall
./test_distance_bound
./test_index_persistence_bin
./test_candidate_search_bin
./test_index_batch_query_bin
```

Expected: all commands exit zero; smoke prints zero FN for all safe methods.
Confirm `stat` shows no change to
`navigamer_cpp/.tmp_experiments/ecoli_1p1m.navidx`.

- [ ] **Step 5: Commit documentation and smoke test**

```bash
git add experiments/ecoli_1p1m/run_smoke_test.sh experiments/ecoli_1p1m/README.md .gitignore
git commit -m "docs: add ecoli comparison workflow"
```

