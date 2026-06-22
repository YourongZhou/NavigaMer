# E. coli 1.1 Mb Comparison Workflow Design

## Purpose

Build a reproducible experiment workflow under `experiments/ecoli_1p1m/` for
comparing NavigaMer with candidate-retrieval baselines and external full
mappers on the first 1,100,000 bp of the E. coli K-12 MG1655 reference.

The workflow must reuse the existing persisted NavigaMer index, keep candidate
retrieval separate from end-to-end mapping, use one exact bounded edit-distance
verifier for all candidate methods, and emit traceable paper-ready tables.

The existing main index is currently located at
`navigamer_cpp/.tmp_experiments/ecoli_1p1m.navidx`. Evaluation must never
silently rebuild or overwrite it.

## Scope

The implementation includes:

- reference preparation;
- persisted indexes for all candidate-retrieval baselines;
- persisted indexes and run wrappers for external mappers;
- deterministic exact, substitution, mixed-edit, and repetitive read sets;
- exact brute-force oracle runs on 50 kb and 100 kb prefixes;
- batch querying of one loaded NavigaMer `.navidx`;
- candidate retrieval, exact verification, mapper evaluation, and result
  collection;
- smoke and focused regression tests; and
- experiment documentation.

The implementation does not instrument internal candidates from BWA-MEM2,
minimap2, or strobealign. It does not run the full 1.1 Mb experiment as part of
normal repository validation.

## Method Categories

Candidate-retrieval methods are compared through a shared interface:

```text
build(reference_windows)
query(read, tolerance) -> unique candidate window IDs
```

Every returned window is verified with the same bounded Edlib implementation.
This category contains:

- NavigaMer adaptive retrieval;
- contiguous k-mer indexes for k = 15, 19, and 23;
- spaced-seed indexes for weights 15, 18, and 21 with four masks each;
- safe q-gram indexes for q = 5 and 6;
- pigeonhole indexes for tau = 2, 3, and 5;
- deterministic order-2 randstrobes; and
- TensorSketch indexes with dimensions 64, 128, 256, and 512.

The no-false-negative expectation applies to NavigaMer, safe q-gram, and
pigeonhole retrieval on the oracle protocol. Contiguous k-mers, spaced seeds,
randstrobes, and TensorSketch are heuristic and report rather than fail on
false negatives.

Full mapper methods are evaluated only end to end:

- BWA-MEM2;
- minimap2; and
- strobealign.

Candidate retrieval and full mapping appear in separate result tables. Final
combined tables retain a `category` field so their timings cannot be mistaken
for directly comparable measurements.

## Repository Layout

The workflow adds:

```text
experiments/ecoli_1p1m/
  README.md
  common.sh
  run_00_prepare_ref.sh
  run_01_build_external_mapper_indexes.sh
  run_02_build_candidate_baseline_indexes.sh
  run_03_generate_reads.sh
  run_04_evaluate_candidate_retrieval.sh
  run_05_run_external_mappers.sh
  run_06_collect_results.sh
  run_smoke_test.sh
  Makefile
  src/
    candidate_tool.cpp
    candidate_indexes.hpp
    candidate_indexes.cpp
    index_persistence.hpp
    index_persistence.cpp
    evaluation.hpp
    evaluation.cpp
  scripts/
    generate_reads.py
    summarize_sam.py
    collect_results.py
```

Small additional source files may be split from these modules when needed to
keep index implementations and tests focused. The public responsibilities and
data formats described here remain unchanged.

The C++ implementation also adds a NavigaMer candidate-only search API and the
`query-index-batch` CLI command. Both README files and `CLI_REFERENCE.md` are
updated with the new command.

## Configuration

Each run script exposes these environment-overridable defaults near its top
and sources `common.sh` for validation and logging:

```bash
REF_FASTA=${REF_FASTA:-data/ecoli/ecoli_1p1m.fa}
NAV_INDEX=${NAV_INDEX:-results/ecoli_1p1m/indexes/navigamer/ecoli_1p1m_w150_s1_r30_15_5_auto.navidx}
OUT_DIR=${OUT_DIR:-results/ecoli_1p1m}
THREADS=${THREADS:-8}
READ_LEN=${READ_LEN:-150}
STRIDE=${STRIDE:-1}
```

Scripts resolve paths relative to the repository root and work from any
current directory. Additional controls include `RANDOM_SEED`,
`REBUILD_BASELINES`, `ORACLE_PREFIXES`, and `TENSOR_SKETCH_ROOT`.

NavigaMer index lookup uses this order:

1. an explicitly exported `NAV_INDEX`;
2. the standard result path above; and
3. the existing compatibility path
   `navigamer_cpp/.tmp_experiments/ecoli_1p1m.navidx`.

If no compatible index is found, candidate evaluation fails with a diagnostic
and an explicit build command. It does not initiate a main-index build.

## Reference Preparation

`run_00_prepare_ref.sh` reuses `REF_FASTA` when it is a valid 1,100,000 bp
single-reference FASTA. Otherwise it locates a configurable larger source,
defaulting to `data/ecoli/fasta/ecoli.fa`, preserves its first FASTA header,
and writes exactly the first 1,100,000 sequence bases to `REF_FASTA`.

The script records the source path, source and prefix hashes, sequence ID,
length, timestamp, command, and git commit. It writes through a temporary file
and atomically renames the completed FASTA.

## NavigaMer Candidate-Only Search

The existing `search_adaptive()` must retain its behavior. A new API performs
the same hierarchy traversal, MBB filtering, q-gram prefiltering, and finest
leaf-beacon filtering, but stops before final leaf edit-distance verification.
It returns unique `BioSequence` candidates plus normal search statistics.

Candidates reached through multiple finest worlds are deduplicated by stable
`sequence_id`. Each candidate sequence is then expanded through its persisted
`ref_positions` into stable reference-window IDs. For a single contig and
stride `s`, a valid position at `start` maps to the window ordinal implied by
the ordered reference-window construction; the emitted row always includes
both `window_id` and `candidate_start`.

`query-index-batch` performs these stages:

1. load and validate the `.navidx` once;
2. load all FASTQ queries;
3. retrieve NavigaMer leaf candidates;
4. verify unique candidate sequences with bounded Edlib;
5. expand accepted and rejected sequences to window occurrences; and
6. stream detail and aggregate output.

The command separately records retrieval, verification, and total time. It
also records internal leaf verifier-call counts where useful, but raw candidate
comparisons use unique reference windows.

Candidate-only regression tests compare candidate-only plus shared verification
against the existing `search_adaptive()` result set across deterministic query
classes and search configurations.

## Candidate Indexes

### Contiguous K-mers

The index stores reference k-mer occurrence postings using rolling 2-bit DNA
keys. It does not duplicate each occurrence into every covering window.
During query, each shared k-mer occurrence expands to the exact interval of
reference windows containing that occurrence. The union is identical to a
window-level inverted index for the rule "share at least one k-mer."

This method is heuristic and may have false negatives.

### Spaced Seeds

Each weight has four deterministic built-in masks with span 24 through 32.
Masks distribute selected positions across the span and are distinct within a
family. The exact mask strings and mask-generation version are persisted in the
manifest.

Reference masked-seed occurrence postings expand to covering windows during
query. A hit under any family mask produces a candidate. This method is
heuristic.

### Safe Q-grams

The persisted payload contains an encoded reference q-gram stream and the
state required to initialize the first reference-window signature. Query uses
sliding count updates so each next window changes the current L1 value in
constant time after query-signature construction.

A window is returned exactly when:

```text
qgram_l1(query, window) <= 2 * q * tau
```

The implementation handles unequal query and window lengths without assuming
equal q-gram mass. It falls back conservatively for non-ACGT symbols. Tests
compare every decision with independently constructed full signatures.

### Pigeonhole Seeds

An index is built for each supported tau. A query is divided deterministically
into `tau + 1` contiguous blocks. An exact block occurrence is translated back
to possible window starts using the block's query offset and a conservative
`[-tau, +tau]` displacement range. Boundary uncertainty only broadens the
candidate set.

This is a no-false-negative method under the tested edit-distance model. Any
oracle false negative fails evaluation.

### Randstrobes

The independent candidate baseline uses deterministic order-2 randstrobes with
defaults:

```text
strobe_len = 15
w_min = 20
w_max = 50
```

For each first strobe, the second strobe is selected from the downstream range
by a seeded combined-hash minimum. Composite occurrences are indexed and
expanded to windows covering the full strobemer span. The hash algorithm,
seed, order, lengths, and window bounds are persisted.

This baseline is not an extraction of strobealign's internal candidates and is
reported as heuristic.

### TensorSketch

TensorSketch uses the existing `ts::Tensor<int>` implementation under
`methods/tensor-sketch-alignment`. Its parameter is accurately named
subsequence length `t = 5`; it is not described as a contiguous q-gram count
sketch.

Each dimension builds one HNSW L2 index. The stable reference-window ID is the
HNSW label. Build artifacts include the HNSW index and, when exact oracle-mode
search is enabled, persisted sketch vectors. A query requests top 10,000 once;
prefixes of that ordered result provide topK values 100, 500, 1,000, 5,000,
and 10,000.

The main 1.1 Mb experiment uses HNSW. The 50 kb and 100 kb oracle experiments
also support exact L2 scanning. Reports distinguish embedding/topK loss from
additional approximate-nearest-neighbor loss.

TensorSketch is built once per dimension and reused across all read sets,
tolerances, and topK values.

## Persistence Contract

Every candidate index directory contains:

```text
index.bin
manifest.json
build_summary.tsv
```

TensorSketch additionally stores `hnsw.bin` and optional
`exact_vectors.bin`. Payloads use a versioned binary header, explicit
endianness, checked lengths, and a checksum.

Required manifest metadata includes the task-requested fields plus:

- format and tool versions;
- reference hash algorithm and digest;
- reference length and ordered contig IDs;
- window-ID rule;
- payload checksum;
- spaced-seed masks;
- randstrobe hash and geometry parameters;
- TensorSketch `t`, dimension, and seed; and
- HNSW `M`, `efConstruction`, and query-time `efSearch`.

Index reuse follows these rules:

- a complete matching manifest reuses the index and records `reused=true`;
- a missing index is written to temporary files, validated, and atomically
  renamed;
- an incompatible index fails by default; and
- `REBUILD_BASELINES=1` permits rebuilding candidate baselines only.

No baseline control can overwrite the main NavigaMer index.

## TensorSketch Dependency

The experiment Makefile detects `ts::Tensor` and HNSW headers through
`TENSOR_SKETCH_ROOT`, with the existing local checkout as its default. The
tracked experiment source contains the adapter and persistence implementation,
not a renamed substitute algorithm.

If the dependency is absent, normal baseline workflows mark TensorSketch as
skipped and continue. A strict TensorSketch build or smoke invocation fails
with setup instructions unless the caller explicitly allows the skip. Generated
binaries remain untracked.

## Read Generation

`run_03_generate_reads.sh` creates the requested FASTQ files using a fixed,
recorded random seed.

- Exact and substitution-only reads remain exactly 150 bp.
- Mixed reads start from 150 bp source windows and receive exactly the named
  number of Levenshtein edit operations. Reads containing insertions or
  deletions may therefore range from 145 through 155 bp.
- A mixed read set collectively contains substitutions, insertions, and
  deletions; an individual read need not contain all operation types.
- Hard reads are selected deterministically by high 5-mer multiplicity and low
  5-mer diversity before applying two or five edits.

Ground truth records the requested fields plus the random seed and explicit
edit script. FASTQ qualities use a deterministic constant high-quality score and
match each mutated sequence length.

## Oracle

The default oracle prefixes are 50,000 and 100,000 bp. Each prefix receives a
dedicated set of 1,000 deterministic reads. For tau 1, 2, 3, and 5, the oracle
uses bounded Edlib against every valid reference window and writes the complete
true-neighbor window-ID set.

Candidate methods, including NavigaMer, are evaluated against indexes built
from the same oracle prefix and window numbering. These small NavigaMer indexes
are test artifacts and are never confused with the main 1.1 Mb index. A
250,000 bp oracle runs only when explicitly included in `ORACLE_PREFIXES`.

## Candidate Evaluation

Candidate evaluation runs the same read sets and tolerances for all applicable
methods. Tau-specific pigeonhole indexes run only at their matching tau.

Per query, evaluation records:

- unique raw candidate windows;
- verified candidate windows;
- accepted windows;
- retrieval, verification, and total milliseconds;
- source-window recovery;
- oracle recall and false negatives when available; and
- raw and accepted candidate blowup when available.

Aggregate output includes the requested means and percentiles, total wall time,
throughput, thread count, index bytes, command, git commit, and notes. Oracle
fields are `NA` outside oracle runs; source recovery is never labeled recall.

Candidate details are streamed. NavigaMer writes the requested candidate rows.
Other weak seed methods write per-read counts by default rather than potentially
enormous all-candidate tables.

If NavigaMer, safe q-gram, or pigeonhole has an oracle false negative, the tool
prints a bounded number of concrete examples and exits nonzero. Heuristic
methods report false negatives without failing.

## External Mapper Indexes and Runs

The external index script creates the required output directories, builds all
requested minimap2 variants, and times each command. Missing executables produce
an `exit_code=127` summary row and a warning without aborting other methods.

Strobealign's generated `.sti` files are discovered after successful index
creation, copied into the configured index directory, and recorded by exact
path. Existing compatible external indexes may be reused with that fact noted.

Mapper runs use the same FASTQ files and configured thread count. SAM parsing
counts mapped reads, primary mappings, and starts within plus or minus 5 bp of
ground truth. Secondary and supplementary records do not count as primary.
Mapper timings are not mixed with candidate retrieval timings.

## Result Collection

`run_06_collect_results.sh` validates input schemas and collects index builds,
candidate retrieval, oracle correctness, and mapper summaries into:

```text
results/ecoli_1p1m/final/
  index_build_summary.tsv
  candidate_retrieval_summary.tsv
  mapper_summary.tsv
  paper_table_main.tsv
  paper_table_correctness.tsv
  paper_table_speed.tsv
```

Rows retain category, method parameters, reference and read-set hashes, command,
git commit, timestamp, index path, and thread count. Collection fails on schema
errors instead of silently dropping fields.

## Logging and Failure Behavior

Scripts log timestamped phase starts, commands, reuse decisions, durations,
warnings, and output paths. Secrets are not expected in commands. Output files
are written through temporary paths and renamed only after successful close and
validation.

Reference or index hash mismatches, malformed persisted payloads, invalid
FASTQ, and oracle safety failures are fatal for their relevant workflow.
Missing optional external mappers and TensorSketch dependencies are recorded as
skips unless the invoked command explicitly requests strict availability.

## Validation

Focused C++ tests cover:

- baseline candidates against naive implementations on small references;
- q-gram sliding L1 against full signatures;
- conservative pigeonhole behavior under substitutions and indels;
- randstrobe determinism;
- persistence round trips and rejection of corrupt or incompatible payloads;
- TensorSketch/HNSW save-load query equivalence when available; and
- NavigaMer candidate-only plus shared verification against existing adaptive
  results.

`run_smoke_test.sh` uses a deterministic 5 kb reference and 100 reads. It builds
all available candidate indexes, runs the oracle and evaluation, requires zero
false negatives from NavigaMer, safe q-gram, and pigeonhole, and reports
heuristic false negatives. TensorSketch is strict by default for the repository
environment expected to contain its checkout; an explicit skip control exists
for environments without that optional dependency.

Because the NavigaMer search engine changes, repository validation also runs:

```bash
cd navigamer_cpp
make -j
make test_recall test_distance_bound test_index_persistence
./test_recall
./test_distance_bound
./test_index_persistence_bin
```

A narrow CLI batch smoke test checks TSV shape and one-load/many-query behavior.
No validation command rebuilds the existing 1.1 Mb NavigaMer index.

## Documentation and Handoff

The experiment README explains required and optional dependencies, method
categories, build-once/query-many persistence, run order, output locations,
environment overrides, resume behavior, and interpretation of timing and recall
columns.

The handoff states which changes affect the C++ implementation, which are
experiment orchestration or reporting, all validation commands run, and any
remaining gaps around external tool availability or unexecuted full-scale
experiments.
