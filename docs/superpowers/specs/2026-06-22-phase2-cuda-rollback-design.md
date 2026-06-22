# Phase 2 CUDA Rollback Design

## Goal

Remove the experimental CUDA Phase 2 distance verifier while preserving all
CPU construction optimizations, index persistence, query behavior, and build
progress reporting.

## Scope

- Delete the CUDA translation unit and CUDA-specific tests and documentation.
- Remove CUDA options from Make, CMake, the CLI, and user documentation.
- Remove the public Phase 2 backend selector and CUDA-only statistics.
- Keep the CPU Phase 2 verifier abstraction, batched verification, and OpenMP
  parallel rebinding.
- Keep persisted index format version 2 unchanged because execution backend
  selection is not part of the serialized search structure.

## Compatibility

`--phase2-distance-backend` will no longer be accepted or documented. Normal
CPU builds and existing persisted indexes remain supported. Query semantics,
TSV hit output, and recall guarantees are unchanged.

## Validation

- Confirm CUDA symbols, flags, sources, and documentation are absent.
- Run a clean CPU build.
- Run Phase 2 verifier, build timing, build range, persistence, and build-scale
  tests.
- Run recall and distance-bound regression suites.
- Build the relevant targets with CMake.

