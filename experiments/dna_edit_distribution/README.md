# Random DNA edit-distance distribution

This standalone experiment streams independent random fixed-length DNA pairs,
computes their exact global Levenshtein distances with WFA2-lib, and writes the
empirical probability mass function and summary statistics. It does not modify
or call the NavigaMer index/search implementation.

## Method

- Each sequence position is sampled uniformly from `A`, `C`, `G`, and `T`.
- A pair-indexed counter-based SplitMix64 generator makes a run reproducible and
  makes its histogram independent of OpenMP thread count and scheduling.
- Each OpenMP worker owns one WFA aligner and reuses it for all pairs assigned to
  that worker.
- WFA2-lib is pinned to release 2.3.6, commit
  `0db345a8fe862fd7873d3354c499da385583a65a`.
- WFA uses global end-to-end alignment, the `edit` distance metric,
  `compute_score`, and `wf_heuristic_none`. Match cost is zero and substitution,
  insertion, and deletion costs are one.
- Sequence pairs are generated and aligned one at a time. They are never stored
  as a collection in memory or on disk.

## Build and test

The established project environment provides CMake, Python, and matplotlib.
WFA2-lib is downloaded by CMake at configure time, so the first configure needs
network access.

```bash
cd /home/luting/projects/AnchorMapping/NavigaMer
conda activate cpp_env_317
cmake -S experiments/dna_edit_distribution -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The tests check identical, substitution, insertion, and deletion distances;
cross-validate deterministic random pairs against a dynamic-programming
Levenshtein implementation; compare one-thread and four-thread histograms; and
exercise the CSV, JSON, PNG, and PDF outputs.

## Run

All options have defaults, so `./build/dna_edit_distribution` runs the formal
length-150, one-million-pair experiment. Explicit commands for the smoke and
formal runs are:

```bash
./build/dna_edit_distribution \
  --length 150 \
  --pairs 10000 \
  --seed 20260719 \
  --threads 16 \
  --output-dir results/smoke_10000

./build/dna_edit_distribution \
  --length 150 \
  --pairs 1000000 \
  --seed 20260719 \
  --threads 16 \
  --output-dir results
```

Options:

```text
--length N       sequence length (default: 150)
--pairs N        number of sequence pairs (default: 1000000)
--seed N         random seed (default: 20260719)
--threads N      OpenMP threads (default: OpenMP maximum)
--output-dir DIR output directory (default: results)
```

## Plot

The plotting script draws the integer-valued PMF directly; it does not use KDE.

```bash
python scripts/plot_distribution.py \
  --histogram results/histogram.csv \
  --output-dir results
```

It writes:

- `results/edit_distance_distribution.png` at 300 dpi
- `results/edit_distance_distribution.pdf` as a vector PDF

## Output files

`histogram.csv` contains all bins from zero through the sequence length:

```text
edit_distance,count,probability,cumulative_probability
```

`summary.csv` uses `metric,value` rows and includes `length`, `num_pairs`,
`seed`, `mean`, population `standard_deviation`, `min`, nearest-rank `median`,
`max`, `mode`, `q05`, `q95`, `elapsed_seconds`, and `pairs_per_second`.

`run_metadata.json` records the run parameters, requested and actual thread
counts, exact WFA configuration, pinned WFA2-lib release and commit, compiler,
compile time, UTC start/end times, elapsed seconds, and throughput.

Before writing output, the executable verifies that histogram counts sum
exactly to `--pairs`. For equal-length sequences of length `L`, every accepted
distance must lie in `[0,L]`; any WFA error or out-of-range result aborts the
run with a nonzero exit status.
