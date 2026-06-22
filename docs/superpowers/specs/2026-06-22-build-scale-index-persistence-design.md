# Build-Scale Index Persistence Design

## Problem

`build-scale` constructs the exact reference-window index needed for full-genome
experiments, but it currently writes only timing CSV output. The global parser
accepts `--index`, then the `build-scale` dispatch path silently ignores it. A
long build can therefore finish successfully without preserving the index.

## CLI Behavior

- `build-scale --index <path>` persists the constructed index when exactly one
  prefix length is requested.
- `build-scale` without `--index` retains its existing timing-only behavior.
- Supplying `--index` with multiple prefix lengths is an error. One output path
  cannot safely represent multiple independently built indexes, and silently
  overwriting it is forbidden.
- The command reports `Index saved: <path>` with the stored manifest summary
  after serialization succeeds.
- Index serialization completes before the successful CSV data row is written,
  so a reported successful row cannot conceal a failed requested save.

## Progress Reporting

Long-running builds emit a low-overhead progress heartbeat to stderr. The
default interval is 600 seconds and can be changed with
`--progress-interval-seconds <N>`; zero disables periodic heartbeats while
retaining phase-boundary messages.

Each heartbeat includes a timestamp, current phase, completed and total coarse
work items, percentage, phase elapsed time, observed throughput, and an ETA when
the total work is known. Phase transitions and build completion force a report
regardless of the interval.

Workers do not print from hot loops. They accumulate progress locally and
publish it to atomic counters only at coarse batches. A single reporter polls
those counters on the configured interval, keeping synchronization and I/O out
of candidate and distance-verification inner loops. Phase 3 keeps its existing
substage-boundary output because it is short relative to the other phases.

## Manifest Semantics

The persisted format remains version 2. The existing reference fingerprint
captures the FASTA or literal reference content. The existing reads-input field
is populated with a deterministic reference-window descriptor containing:

- actual prefix length after clamping to the reference length;
- window length;
- stride.

The descriptor participates in the existing manifest signature through its
fingerprint. Changing reference content, prefix length, window length, stride,
hierarchy radii, or construction parameters therefore changes the signature.
The execution-only CPU/CUDA Phase2 backend remains excluded, as it does for
other persisted indexes.

## Implementation

`run_build_scale` receives `index_path`. After building the sole requested
prefix, it creates the reference-window manifest and calls the existing
`save_index` implementation. No serialization format or graph representation
changes are required.

CLI validation rejects a non-empty index path when `prefix_lengths.size() != 1`.
The progress interval is carried in the build configuration as an execution-only
setting and is excluded from persisted-index signatures. Documentation is
updated to describe persistence, the single-prefix rule, and heartbeat control.

## Tests

The build-scale smoke test exercises the real CLI and verifies:

1. a single-prefix build with `--index` creates a non-empty index;
2. `query-index` can load that file and execute a query;
3. the stored manifest records a deterministic reference-window descriptor;
4. multiple prefixes with `--index` return nonzero and do not create an index;
5. existing multi-prefix timing-only behavior remains unchanged;
6. the default interval is 600 seconds, zero disables periodic output, and a
   short test interval emits timestamped phase/progress fields;
7. changing the progress interval does not change the manifest signature.

The persisted-index round-trip, recall, and distance-bound tests remain the
broader correctness gates.
