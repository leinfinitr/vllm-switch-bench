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
- `reports/figures/baseline3-qwen2p5-0p5b-comparison.png`
- `reports/vllm-qwen2p5-0p5b.md`

## Archive

- `plans/`: implementation plans kept for auditability; they may contain old paths from before the repository cleanup.

## Result data policy

Keep the latest curated per-system runs and the latest merged baseline comparison under `../results/baselines/`. Older raw result directories should be pruned after their findings are reflected in `docs/reports/`. Transient experiments should go to `results/tmp/` or another ignored runtime directory.
