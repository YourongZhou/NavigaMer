# Random DNA Edit-Distance Distribution Design

## Goal

Build a reproducible C++17/OpenMP experiment that streams independent random
fixed-length DNA sequence pairs through WFA2-lib, records their exact global
Levenshtein edit-distance distribution, writes machine-readable statistics and
run metadata, and produces a discrete PMF plot.

## Scope and location

The experiment is independent of the NavigaMer production CLI and will live in
`experiments/dna_edit_distribution/`. It will have its own CMake entry point so
that this command produces the requested binary path without changing the
existing `navigamer_cpp` build:

```bash
cmake -S experiments/dna_edit_distribution -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The plotting entry point will be `scripts/plot_distribution.py`. The experiment
will not change NavigaMer indexing, search, or CLI behavior.

## Dependency strategy

CMake will build the exact WFA2-lib v2.3.6 source identified by commit
`0db345a8fe862fd7873d3354c499da385583a65a`. Pinning the commit makes the backend
reproducible and provides the version string recorded in metadata. OpenMP is a
required CMake dependency. The established `cpp_env_317` environment supplies
CMake and Python tooling, while the current system GCC and OpenMP runtime build
and run the C++ program.

## Random sequence generation

Pair `i` is a pure function of the user seed and the zero-based pair index. A
counter-based SplitMix64 stream will generate enough 64-bit words for the two
sequences. Consecutive two-bit fields map to `A`, `C`, `G`, and `T`; because the
alphabet size divides the 64-bit state space, this mapping introduces no modulo
bias.

This design gives deterministic output for a fixed `(length, pairs, seed)` and
makes results independent of OpenMP thread count, scheduling, and iteration
partitioning. Only the two current sequence buffers are retained by a worker.

## Exact alignment and parallel data flow

Each OpenMP thread creates one WFA aligner before processing its loop chunk and
deletes it after the loop. The same aligner is reused for every pair handled by
that thread. Its attributes are:

- `distance_metric = edit`
- `alignment_scope = compute_score`
- `heuristic.strategy = wf_heuristic_none`
- exact global end-to-end alignment with unit substitution, insertion, and
  deletion costs

Each thread owns a histogram with bins `0..length`. After the parallel region,
the main thread merges those histograms. No pair or sequence collection is
written to memory or disk. Any WFA failure or distance outside `[0, length]`
terminates the run without publishing partial result files.

## CLI and outputs

The executable accepts:

- `--length`, default `150`, strictly positive
- `--pairs`, default `1000000`, strictly positive
- `--seed`, default `20260719`
- `--threads`, default OpenMP maximum, strictly positive
- `--output-dir`, default `results`

Unknown flags, missing values, invalid integers, and output-directory failures
produce a diagnostic and a nonzero exit status.

`histogram.csv` contains every distance from zero through `length`, including
zero-count bins, with columns `edit_distance,count,probability,cumulative_probability`.
Probabilities use `count / num_pairs`, and the final cumulative probability is
one up to floating-point formatting.

`summary.csv` uses `metric,value` rows and contains at least `length`,
`num_pairs`, `seed`, `mean`, `standard_deviation`, `min`, `median`, `max`,
`mode`, `q05`, `q95`, `elapsed_seconds`, and `pairs_per_second`. Standard
deviation is the population standard deviation. Quantiles are empirical
nearest-rank values: the smallest integer distance whose cumulative count is at
least the requested fraction of all pairs. If multiple bins tie for the mode,
the smallest distance is reported.

`run_metadata.json` records all CLI parameters, requested and actual thread
counts, WFA2-lib version and pinned commit, compiler identification, compile
timestamp, run start/end timestamps, elapsed seconds, and pairs per second.
CSV and JSON are written only after validating that the histogram count equals
`--pairs`.

## Plotting

`scripts/plot_distribution.py` accepts the histogram path and output directory,
reads `histogram.csv`, and draws the discrete PMF directly without KDE. The
x-axis is `Edit distance`, the y-axis is `Probability`, and a dashed vertical
line marks the mean computed from the histogram. The title includes sequence
length and pair count. It writes `edit_distance_distribution.png` at 300 dpi
and a vector `edit_distance_distribution.pdf`.

## Correctness and verification

CTest will exercise real WFA2 calls and cover:

1. identical sequences have distance zero;
2. one substitution, insertion, or deletion has distance one;
3. a reference dynamic-programming Levenshtein implementation agrees with WFA2
   on deterministic random pairs, including unequal lengths used by the indel
   checks;
4. the counter-based generator produces identical histograms with one and
   multiple threads;
5. CLI output counts sum exactly to `--pairs`, distances remain in `[0, length]`,
   and required output fields exist;
6. the Python plotting script generates nonempty PNG and PDF files from a small
   histogram fixture.

After unit and integration tests, a 10,000-pair run at length 150 will write to
`results/smoke_10000/`. Only if that run passes all invariants will the default
1,000,000-pair run write to `results/`, followed by plotting from the formal
run's histogram.
