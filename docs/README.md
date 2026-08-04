# Documentation

Current user-facing documentation is English and portable. Historical reports are retained separately and may contain original language, paths, schemas, and claims.

## Current runbooks and policy

| Document | Status | Purpose |
|---|---|---|
| [`release-artifact.md`](release-artifact.md) | Current | v0.1 artifact layout, builder, manifests, and final-rerun gate |
| [`baselines/baseline1-vllm-cold-reload.md`](baselines/baseline1-vllm-cold-reload.md) | Current | vLLM cold-reload boundary and command |
| [`baselines/baseline2-vllm-sleep-mode.md`](baselines/baseline2-vllm-sleep-mode.md) | Current | vLLM L1/L2 lifecycle boundaries |
| [`baselines/baseline3-engine-checkpoint-hotswap.md`](baselines/baseline3-engine-checkpoint-hotswap.md) | Historical harness | Legacy Baseline3 aggregation; not the v0.1 release protocol |
| [`systems/llama-swap.md`](systems/llama-swap.md) | Current | llama-swap automatic switching and measurement semantics |
| [`systems/swapservellm.md`](systems/swapservellm.md) | Current | SwapServeLLM operational lifecycle requirements |
| [`systems/serverlessllm.md`](systems/serverlessllm.md) | Blocked | ServerlessLLM automatic scale-to-zero gate and known blockers |

## Historical reports

[`archive/reports/`](archive/reports/) contains dated or schema-bound reports retained for audit. They are **historical**, not current CLI documentation, and are not translated or path-normalized when doing so would obscure their provenance. The current release artifact summary is [`../results/release-v0.1/README.md`](../results/release-v0.1/README.md).

## Plans

Implementation and process plans are not retained on the default release branch. Issue and pull-request history owns that narrative. Durable semantics belong in the current runbooks above; immutable experiment provenance belongs in result bundles.
