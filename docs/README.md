# Documentation index

This directory is organized by purpose.

## Baseline reproduction guides

- `baselines/baseline1-vllm-cold-reload.md`
- `baselines/baseline2-vllm-sleep-mode.md`
- `baselines/baseline3-engine-checkpoint-hotswap.md`

These are the entry points for rerunning baseline1-3.

## External system setup notes

- `systems/serverlessllm.md`
- `systems/swapservellm.md`

These notes record the exact local runtime assumptions that were discovered while getting baseline3 to run.

## Reports

- `reports/baseline3-qwen2p5-0p5b.md`
- `reports/vllm-qwen2p5-0p5b-clean-hbm.md`
- `reports/vllm-qwen2p5-0p5b-clean-hbm-memory.md`

## Archive

- `archive/migration.md`: historical note about moving this harness out of an earlier prototype location.
- `plans/`: implementation plans kept for auditability; they may contain old paths from before the repository cleanup.

## Result data policy

Only curated future-baseline data should be tracked under `../results/baselines/`. Transient experiments should go to `results/tmp/` or a new ignored runtime directory.
