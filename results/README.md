# Result artifact policy

`results/` contains exactly five current evidence families. It is not a chronological run
dump: live and failed experiments belong under ignored `results/tmp/` until their protocol,
raw evidence, interpretation, and semantic validator are reviewed together.

## Current families

| Path | Scope | Summary and figure |
|---|---|---|
| [`lifecycle-latency/`](lifecycle-latency/README.md) | Five retained sleep/wake observations for each supported model/system cell | [`summary.json`](lifecycle-latency/summary.json) · [PNG](lifecycle-latency/figures/lifecycle-latency.png) · [PDF](lifecycle-latency/figures/lifecycle-latency.pdf) |
| [`vllm-profiling/`](vllm-profiling/README.md) | Five post-warm-up activation profiles for cold load, stock vLLM L1/L2, and Proposed CPU/exact-disk backup | [`summary.json`](vllm-profiling/summary.json) · [PNG](vllm-profiling/figures/vllm-profiling.png) · [PDF](vllm-profiling/figures/vllm-profiling.pdf) |
| [`request-driven-switch/`](request-driven-switch/README.md) | Retained 20-request alternating JSONL traces and runtime manifests for Proposed and llama-swap | [`summary.json`](request-driven-switch/summary.json) · [PNG](request-driven-switch/figures/request-timeline.png) · [PDF](request-driven-switch/figures/request-timeline.pdf) |
| [`backup-reuse-reclaim/`](backup-reuse-reclaim/README.md) | Repeated exact CPU-backup reuse plus one host-pressure reclaim observation | [`summary.json`](backup-reuse-reclaim/summary.json) · [PNG](backup-reuse-reclaim/figures/backup-reuse.png) · [PDF](backup-reuse-reclaim/figures/backup-reuse.pdf) |
| [`exact-disk/`](exact-disk/README.md) | One retained exact-runtime-byte spill, CPU release, and disk restore observation | [`summary.json`](exact-disk/summary.json) · [PNG](exact-disk/figures/exact-disk.png) · [PDF](exact-disk/figures/exact-disk.pdf) |

Each family contains:

- `config/`: the frozen family claim/workload contract used by validation;
- `raw/`: minimal retained producer evidence;
- `summary.json` (and family-specific tabular output): deterministic aggregates;
- `figures/`: deterministic PDF and PNG views;
- `metadata.json`: family schema, provenance disclosure, and exact expected file set;
- `README.md`: a short pointer back to the full experiment protocol.

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

`build_all` reconstructs summaries, figures, and metadata from retained raw inputs.
`validate_all` rejects any stray top-level result family.
`validate_all` enforces exact family shape and semantic contracts: matrix cardinality,
sample counts, finite positive metrics, frozen request identity/order, strict request
success, raw-to-summary equality, profiling phase accounting, backup reuse/reclaim
settlement, exact-disk chunk layout,
physical footprint, and output equality.

## Local output and corrections

Use `results/tmp/<experiment>/<run-id>/` for live or exploratory output. Do not commit model
weights, runtime payloads, credentials, broad logs, caches, or unrelated failed runs. Do not
silently edit retained raw evidence to make a validator pass. Use `scripts/promote.sh` to
stage a complete candidate, review it, and atomically replace one validated family. Promotion
keeps the previous family under the ignored candidate root; preserve the immutable tag.
