# 文档说明

这里保留会影响复现实验和理解结论的文档。

## 复现说明

- `baselines/baseline1-vllm-cold-reload.md`：vLLM 冷启动。
- `baselines/baseline2-vllm-sleep-mode.md`：vLLM Sleep Mode。
- `baselines/baseline3-engine-checkpoint-hotswap.md`：ServerlessLLM / SwapServeLLM 系统级切换对比。
- `systems/serverlessllm.md`：ServerlessLLM 本机运行要点。
- `systems/swapservellm.md`：SwapServeLLM 本机运行要点。

外部系统文档必须明确区分“服务能启动/推理”和“满足可绘图 lifecycle
post-condition”。如果当前 adapter 或运行时有已知口径问题，文档应把它标为
blocker，而不是继续展示可能产生误导数字的命令。

## 报告

- `reports/baseline3-qwen2p5-*.md`：Baseline3 对比结论。
- `reports/vllm-pin-compare.md`：vLLM sleep_l1 / sleep_l2 pin/no-pin breakdown，包含 L2 reload weights 拆分说明。
- `reports/phase1-two-model-pool.md`：历史两模型 repeated `sleep_l1` 结果；其中 metadata/eviction 字段是 legacy schema，不是当前 release protocol。
- `reports/cumem-copy-microbench.md`：PCIe copy、CuMemAllocator synthetic copy 和 safetensors 粒度 microbench。
- `reports/request-driven-multi-model-switch.md`：当前 request-driven controller 与 pinned backup 证据。
- `reports/cross-system-single-gpu-evaluation.md`、`reports/model-switch-and-routing-evaluation-2026-07-21.md`：历史跨系统阶段报告，结论边界以各文档自带 provenance 为准。

## 当前阶段 artifact

- `../results/osdi_20260723/`：当前 lifecycle/E2E 图、摘要、raw evidence 和校验和。
- `../results/request_switch/latest/`：request-driven switching curated artifact。

旧 `results/baselines/`、`results/cross_system/` 和 `results/model_switch_eval/`
按历史 schema 原样保留，不应为了适配新脚本而重写。

`plans/` 是历史实施计划归档，可能保留当时的路径和步骤，不作为当前复现入口。
