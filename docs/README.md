# Documentation

This index describes the current default-branch workflow. The immutable `v0.1.8` tag is the
reference for its published snapshot; old branch layouts and commands are not current API.
No new measurements were generated while reorganizing the retained evidence, and no
canonical GPU rerun is complete.

## Experiment protocols

| Experiment | Question | Result |
|---|---|---|
| [`experiments/lifecycle-latency/`](experiments/lifecycle-latency/README.md) | How long are explicit lifecycle sleep and wake boundaries? | [`results/lifecycle-latency/`](../results/lifecycle-latency/README.md) |
| [`experiments/vllm-profiling/`](experiments/vllm-profiling/README.md) | Which phases dominate vLLM L1/L2 and Proposed backup activation? | [`results/vllm-profiling/`](../results/vllm-profiling/README.md) |
| [`experiments/request-driven-switch/`](experiments/request-driven-switch/README.md) | What completion latency does a frozen alternating request trace observe? | [`results/request-driven-switch/`](../results/request-driven-switch/README.md) |
| [`experiments/backup-reuse-reclaim/`](experiments/backup-reuse-reclaim/README.md) | Are exact CPU backups reused, and can host pressure reclaim them physically? | [`results/backup-reuse-reclaim/`](../results/backup-reuse-reclaim/README.md) |
| [`experiments/exact-disk/`](experiments/exact-disk/README.md) | Can exact runtime bytes restore from disk after CPU-backup release? | [`results/exact-disk/`](../results/exact-disk/README.md) |

Each protocol includes the question, metric and boundaries, method, retained result, threats
to validity, limitations, figure links, deterministic rebuild, semantic validation, and a
separately labeled live-run command.

## Supporting system notes

These notes explain external-system behavior and remain useful background, but the
experiment documents above define the current result contracts:

- [`systems/llama-swap.md`](systems/llama-swap.md): automatic request routing versus
  separately instrumented process lifecycle boundaries.
- [`systems/swapservellm.md`](systems/swapservellm.md): swap-out/swap-in operating
  assumptions and external executable identity.
- [`systems/serverlessllm.md`](systems/serverlessllm.md): unresolved scale-to-zero contract;
  it has no current numeric result family.
- [`baselines/baseline1-vllm-cold-reload.md`](baselines/baseline1-vllm-cold-reload.md): cold
  process-reload boundary and storage-cache caveat.
- [`baselines/baseline2-vllm-sleep-mode.md`](baselines/baseline2-vllm-sleep-mode.md): vLLM
  L1/L2 semantics. Both baseline commands feed the vLLM profiling workflow, not the
  cross-system lifecycle retained schema.

## Reading the retained values

The current summaries and figures are deterministic transformations of tracked raw evidence,
not new measurements. The historical E2E producer did not runtime-bind controller or engine
commits, dirty states, executable/import paths, configuration hash, or model revision. Those
numbers are a historical local observation, not an exact fresh-checkout runtime reproduction.
A fresh checkout can reproduce the summaries, figures, and validator outcomes, but not the
original GPU execution from the retained metadata alone.

The repository does not maintain digest lists over tracked Git files. Lifecycle metadata
retains exact digests for external executables, and exact-disk retains payload/per-chunk
runtime checksums because those bytes are outside ordinary Git-file integrity semantics.

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
