# Result artifact policy

`results/` contains exactly four current evidence families. It is not a chronological run
dump: live and failed experiments belong under ignored `results/tmp/` until their protocol,
raw evidence, interpretation, and semantic validator are reviewed together.

No new measurements were generated during the default-branch refactor, and a canonical GPU
rerun is not complete. The immutable `v0.1.8` tag remains available as the published v0.1
snapshot; the default branch intentionally does not reproduce its old monolithic directory
layout.

## Current families

| Path | Scope | Summary and figure |
|---|---|---|
| [`lifecycle-latency/`](lifecycle-latency/README.md) | Five retained sleep/wake observations for each supported model/system cell | [`summary.json`](lifecycle-latency/summary.json) · [PNG](lifecycle-latency/figures/lifecycle-latency.png) · [PDF](lifecycle-latency/figures/lifecycle-latency.pdf) |
| [`request-driven-switch/`](request-driven-switch/README.md) | Retained 20-request alternating traces for Proposed and llama-swap | [`summary.json`](request-driven-switch/summary.json) · [PNG](request-driven-switch/figures/request-timeline.png) · [PDF](request-driven-switch/figures/request-timeline.pdf) |
| [`backup-reuse-reclaim/`](backup-reuse-reclaim/README.md) | Repeated exact CPU-backup reuse plus one host-pressure reclaim observation | [`summary.json`](backup-reuse-reclaim/summary.json) · [PNG](backup-reuse-reclaim/figures/backup-reuse.png) · [PDF](backup-reuse-reclaim/figures/backup-reuse.pdf) |
| [`exact-disk/`](exact-disk/README.md) | One retained exact-runtime-byte spill, CPU release, and disk restore observation | [`summary.json`](exact-disk/summary.json) · [PNG](exact-disk/figures/exact-disk.png) · [PDF](exact-disk/figures/exact-disk.pdf) |

Each family contains:

- `raw/`: minimal retained producer evidence;
- `summary.json` (and family-specific tabular output): deterministic aggregates;
- `figures/`: deterministic PDF and PNG views;
- `metadata.json`: family schema, provenance disclosure, and exact expected file set;
- `README.md`: a short pointer back to the full experiment protocol.

## Integrity model

Git object identity protects files tracked in the repository. The default branch therefore
does not add a second set of repository-internal digest manifests. `metadata.json` declares
the exact family file set, and semantic validators require that declaration to equal the
files present.

Cryptographic digests remain mandatory where Git does not identify the measured object:

- lifecycle metadata binds downloadable external executables by URL, byte size, and
  SHA-256;
- exact-disk retains the omitted runtime payload's SHA-256 and the runtime manifest's
  per-chunk checksums, offsets, and sizes.

Those runtime/external digests are evidence and must not be removed merely because the
repository no longer keeps whole-tree digest lists.

## Promotion requirements

Before replacing or adding retained measurement evidence, a contribution must provide:

1. a documented question, metric boundary, success predicate, method, and controls;
2. frozen workload/model identity and raw request- or phase-level evidence;
3. benchmark commit/dirty state and runtime-bound engine/controller commits, imported path,
   behavior-affecting configuration or digest, executable/image digest, and environment;
4. deterministic functional output or another explicit correctness post-condition;
5. application accounting plus OS/GPU-visible evidence for physical resource claims;
6. structured failed/blocked attempts excluded from successful aggregate denominators;
7. a deterministic builder and a semantic validator with focused tests;
8. two clean rebuilds from a fresh checkout.

The retained v0.1 E2E evidence does **not** satisfy the modern runtime-binding requirement:
its producer did not bind engine/controller commits, imported path, or configuration hash.
It is explicitly a historical local observation. Deterministic rebuilding verifies the
retained interpretation; it does not repair that provenance gap or create new data.

## Rebuild and validate

From the repository root:

```bash
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results

scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
scripts/tracked-ignore.sh
```

`build_all` reconstructs summaries, figures, and metadata from retained raw inputs and also
rejects stray top-level result families by removing them from its generated view.
`validate_all` enforces exact family shape and semantic contracts: matrix cardinality,
sample counts, finite positive metrics, frozen request identity/order, strict request
success, raw-to-summary equality, backup reuse/reclaim settlement, exact-disk chunk layout,
physical footprint, and output equality.

## Local output and corrections

Use `results/tmp/<experiment>/<run-id>/` for live or exploratory output. Do not commit model
weights, runtime payloads, credentials, broad logs, caches, or unrelated failed runs. Do not
silently edit retained raw evidence to make a validator pass. Correct a protocol or result
through a reviewable new measurement and replace the affected current family atomically;
preserve the immutable published tag.
