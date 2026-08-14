# Documentation

This index describes the current default-branch workflow.

## Experiment protocols

| Experiment | Question | Result |
|---|---|---|
| [`experiments/lifecycle-latency/`](experiments/lifecycle-latency/README.md) | How long are explicit lifecycle sleep and wake boundaries? | [`results/lifecycle-latency/`](../results/lifecycle-latency/README.md) |
| [`experiments/vllm-profiling/`](experiments/vllm-profiling/README.md) | Which phases dominate vLLM L1/L2 and vllm-switch backup activation? | [`results/vllm-profiling/`](../results/vllm-profiling/README.md) |
| [`experiments/request-driven-switch/`](experiments/request-driven-switch/README.md) | What completion latency does a frozen alternating request trace observe? | [`results/request-driven-switch/`](../results/request-driven-switch/README.md) |
| [`experiments/backup-reuse-reclaim/`](experiments/backup-reuse-reclaim/README.md) | Are exact CPU backups reused, and can host pressure reclaim them physically? | [`results/backup-reuse-reclaim/`](../results/backup-reuse-reclaim/README.md) |
| [`experiments/exact-disk/`](experiments/exact-disk/README.md) | Can exact runtime bytes restore from disk after CPU-backup release? | [`results/exact-disk/`](../results/exact-disk/README.md) |

Each protocol includes the question, metric and boundaries, method, retained result, threats
to validity, limitations, figure links, deterministic rebuild, semantic validation, and a
separately labeled live-run command.

## Reading and reproducing the retained values

Each experiment protocol is an ordered runbook: prerequisites, live collection, failed-run
handling, dry-run promotion, atomic result replacement, rebuild, validation, and diff review.
The current raw evidence records actual runtime repositories/import paths and configurations
or external executable hashes. The benchmark checkout itself was dirty during collection;
its commit, porcelain status, and working-tree fingerprint are retained rather than hidden.

A fresh checkout reproduces summaries, figures, and validator outcomes deterministically.
Reproducing the GPU measurements additionally requires the model files, pinned external
runtimes, compatible hardware/software, and the conditions documented by each family.

The repository does not maintain digest lists over tracked Git files. Lifecycle metadata
retains exact digests for external executables, request manifests retain service config and
binary hashes, and exact-disk retains payload/per-chunk runtime checksums because those bytes
are outside ordinary Git-file integrity semantics.

## Common CPU commands

```bash
uv sync --frozen --group dev
scripts/docs.sh
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
scripts/build_all.sh
scripts/validate_all.sh
git diff --exit-code -- results
scripts/tracked-ignore.sh
```

See [`../README.md`](../README.md) for setup and live entry points,
[`../results/README.md`](../results/README.md) for result policy,
[`../scripts/README.md`](../scripts/README.md) for wrappers, and
[`../CONTRIBUTING.md`](../CONTRIBUTING.md) before changing a protocol or artifact.
