# Exact disk

Question: does exact disk spill, restore, and physically release a 1 GiB exact-runtime payload while retaining integrity and output equality?

- Configuration: [`config/claims.json`](config/claims.json)
- Raw evidence: seven files under [`raw/exact-disk/`](raw/exact-disk/)
- Summary: [`summary.json`](summary.json)
- Figure: [`figures/exact-disk.pdf`](figures/exact-disk.pdf) ([PNG](figures/exact-disk.png))
- Method and limitations: [`../../docs/experiments/exact-disk/README.md`](../../docs/experiments/exact-disk/README.md)

The payload is intentionally omitted. Its SHA-256 and runtime bundle/chunk checksums remain correctness evidence. The validator checks phase coverage, no fallback, size/hash/chunk/manifest-commit consistency, material footprint, completed host-cache release and demotion, actual restore reads, run identity/config metadata, and equal output. No new measurement was run during this refactor, and the canonical GPU rerun is incomplete.
