# Phase2 CUDA Distance Verifier Design

## Goal

Add an optional GPU backend for Phase2 exact distance verification so large
builds can offload the billions of independent bounded edit-distance checks
without changing query semantics, persisted index contents, or recall safety.

## Current Bottleneck

After Phase1 and Phase3 optimizations, large-prefix builds are dominated by
Phase2 inter-tier rebinding. On the 256 kb E. coli prefix with 250-base
windows, several layers perform hundreds of millions to more than one billion
exact bounded edit-distance checks. The CPU path uses the existing build
distance function and is exact, but the workload is embarrassingly parallel.

One layer is different: `L5->6` spends most of its time generating and q-gram
filtering candidates, then verifies only a small number of exact distances.
GPU exact verification will not remove that candidate-generation hotspot by
itself, so the GPU design must be optional and measured layer by layer.

## Selected Design

Introduce a build-only Phase2 distance verifier abstraction with three backend
choices:

- `cpu`: always use the existing bounded CPU verifier.
- `cuda`: require CUDA initialization and fail if unavailable or self-checks
  fail.
- `auto`: try CUDA and fall back to CPU if CUDA is not available.

The verifier receives a batch of `(parent_idx, child_idx, tau)` pairs and
returns the subset whose center sequences are within `tau`. Phase2 keeps its
current candidate generation, statistics accounting, edge sorting, and edge
insertion. The GPU backend only replaces the inner exact-verification loop.

The CUDA implementation will be compiled only when requested by the build
system. Non-CUDA builds keep the current behavior and pass the same tests.

## CUDA Algorithm

Use a custom batched Myers-style verifier specialized for the current build
workload:

- DNA windows are short, currently up to 250 bases.
- The result needed by Phase2 is only `distance <= tau`; no traceback or CIGAR
  is required.
- Each candidate pair is independent.
- Unsupported sequence shapes or initialization failures fall back to CPU in
  `auto` mode.

The initial CUDA implementation may use a straightforward exact dynamic
programming kernel for correctness if that is faster to land safely, but the
public backend contract remains a bounded edit-distance verifier. The verifier
must compare its results against the existing CPU implementation before it is
used in correctness-sensitive builds.

## Semantics

Accepted Phase2 edges must be identical between CPU and CUDA for the same
input sequences, radii, and candidate mode. The backend is not a semantic build
parameter: it may be recorded in logs/statistics for diagnostics, but it must
not affect persisted index compatibility. Query reads the already-built graph
and does not call this backend.

CUDA must not introduce approximate distances, probabilistic filters, changed
tie-breaking, or reordered child lists in the final graph. Phase2 may sort and
deduplicate GPU results before insertion to preserve deterministic topology.

## CLI and Build

Add a build/query construction option:

- `--phase2-distance-backend auto|cpu|cuda`

The default is `auto` only when the executable was compiled with CUDA support;
otherwise it behaves like `cpu`. Documentation should make clear that `cuda`
requires a CUDA-enabled build and that `auto` falls back safely.

Makefile support:

- Default build remains CPU-only.
- `make NAVIGAMER_WITH_CUDA=1` compiles CUDA sources with `nvcc` and defines
  `NAVIGAMER_WITH_CUDA`.

CMake support:

- Add `NAVIGAMER_WITH_CUDA` option.
- Enable the CUDA language and link the CUDA runtime only when requested.

## Statistics

Add Phase2 verifier diagnostics:

- requested backend
- backend actually used
- number of batches
- number of pairs submitted to CUDA
- accepted pairs returned by CUDA
- CUDA worker milliseconds
- CPU fallback count

Existing `phase2_exact_distance_calls`, `phase2_exact_verify_ms`, and edge
counts remain authoritative for algorithmic summaries.

## Validation

Correctness gates:

1. Backend parser accepts `auto`, `cpu`, and `cuda`, and rejects invalid names.
2. CPU backend batch verification matches direct bounded CPU calls.
3. CUDA backend, when available, matches CPU across substitutions, insertions,
   deletions, unequal lengths, ambiguous bases, and `tau` values from 0 to the
   window length.
4. Full Phase2 CPU and CUDA builds produce identical topology on small and
   medium deterministic data sets.
5. `test_recall`, `test_distance_bound`, and `test_map150_recall` pass.

Performance gates:

1. Benchmark CPU vs CUDA on at least one prefix where Phase2 exact calls exceed
   100 million.
2. Report layer-level speedups and identify candidate-generation-dominated
   layers where GPU exact verification is not expected to help.
3. Recompute the full E. coli build estimate from the largest successful
   prefix rather than extrapolating from CPU-only timings.

## Non-Goals

- Changing Phase1 candidate generation.
- Changing query-time search or persisted index contents.
- Replacing edlib everywhere.
- Adding GPU support to leaf attachment in this first step.
- Adopting a large external GPU alignment SDK before the custom backend has
  been tested against the current workload.
