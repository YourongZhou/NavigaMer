# Edlib Distance Backend Design

## Goal

Add Edlib as an optional exact/bounded edit-distance backend for NavigaMer while preserving result-set equivalence and keeping current defaults unchanged.

## Scope

This change adds:

- Query/adaptive bounded child-center backend: `--distance-mode edlib`.
- Build-time distance backend: `--build-distance-mode dp|edlib|auto`, defaulting to `dp`.
- Tests comparing DP, current Myers, and Edlib bounded semantics.
- Search and construction equivalence tests proving the new backend does not change final results or graph construction.

This change does not add `lv89` to production paths. `lh3/lv89` is archived and its README says the implementation is usually slower than Edlib/WFA2. Because the user's query workload has small tolerance, lv89 remains a useful experimental benchmark candidate after Edlib provides a stable external-library baseline.

## Semantics

The existing bounded distance contract remains:

- If true edit distance is `<= tau`, return the true distance.
- If true edit distance is `> tau`, return a value greater than `tau`.
- Never return a value below the true distance.

Edlib will be called in global/NW mode with task `DISTANCE` and `k=tau`. Edlib returns `-1` when distance is greater than `k`; the wrapper maps that to `tau + 1`.

## Architecture

Vendored Edlib source lives under `navigamer_cpp/third_party/edlib/` and is compiled into the CLI and tests by both Makefile and CMake builds.

`DistanceMode` gains `Edlib` for query/adaptive bounded child-center checks. `SearchConfig::distance_mode` remains default `Myers`.

A separate `BuildDistanceMode` is introduced for construction. It is not reused from query `DistanceMode` so users cannot accidentally alter index construction by changing query tuning flags. The default remains `DP`; `auto` remains conservative and maps to `DP` until benchmark evidence justifies otherwise.

## Acceptance

- `make test_bounded_myers` passes with Edlib included in the same differential tests.
- Search `dp`, `myers`, and `edlib` modes return exactly identical result sets.
- Build `dp` and `edlib` modes produce equivalent node IDs, child edges, leaves, and search results on the regression test dataset.
- `make test_all` passes.
- Query benchmark reports `distance_mode=edlib` when selected and gate passes.
