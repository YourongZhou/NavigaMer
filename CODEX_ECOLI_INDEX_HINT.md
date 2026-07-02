# E. coli Persisted Index Hint

Generated at 2026-07-01 after manual filesystem verification.

The E. coli persisted NavigaMer indexes are present under NFS, not under the
repository worktree or `/tmp`.

Use this index first for window-150 / stride-1 matched query benchmarks:

```text
/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/full_compare_20260629_w150_tau2/run/ecoli_full_w150_s1.navidx
```

Observed metadata:

- size: `2919162287 bytes`
- strings header includes:
  `reference-windows:v1;prefix=4641652;window=150;stride=1`
- build CSV:
  `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/full_compare_20260629_w150_tau2/run/ecoli_full_w150_s1.csv`
- reference used in index metadata:
  `/home/andyzhou/nfs/luting_data/AnchorBasedMapping/ecoli/fasta/ecoli.fa`
- equivalent local path observed:
  `/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/fasta/ecoli.fa`

Smoke verification with the current repository binary:

```text
cd navigamer_cpp &&
timeout 60s ./navigamer query-index \
  --index /home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/full_compare_20260629_w150_tau2/run/ecoli_full_w150_s1.navidx \
  --query AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGCATA \
  --tolerance 5 \
  --mode adaptive \
  --query-profile 1
```

Result:

```text
Loaded index: .../ecoli_full_w150_s1.navidx signature=e2cdc0f6618cfd2c sequences=4594199 world_nodes=3150918
Adaptive hits: 3
EXIT_STATUS=0
```

There is also a full window-250 / stride-1 index:

```text
/home/luting/nfs/luting_data/AnchorBasedMapping/ecoli/full_builds/navigamer_full_20260629_andyzhou_w250_s1.navidx
```

Observed metadata:

- size: `3232049773 bytes`
- build log reports successful save with `sequences=4601706`,
  `world_nodes=3157283`, `edges=18201191`, `leaf_links=10096147`
- build wall time in the old log: `13:58:20`

Do not rebuild 1.1M/full E. coli unless this NFS index is incompatible with
the exact benchmark settings. For the current strobemer/spaced-seed and V2
real-locality goals, prefer the `w150_s1` index above.
