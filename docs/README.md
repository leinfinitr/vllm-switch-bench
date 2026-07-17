# 文档说明

这里保留会影响复现实验和理解结论的文档。

## 复现说明

- `baselines/baseline1-vllm-cold-reload.md`：vLLM 冷启动。
- `baselines/baseline2-vllm-sleep-mode.md`：vLLM Sleep Mode。
- `baselines/baseline3-engine-checkpoint-hotswap.md`：ServerlessLLM / SwapServeLLM 系统级切换对比。
- `systems/serverlessllm.md`：ServerlessLLM 本机运行要点。
- `systems/swapservellm.md`：SwapServeLLM 本机运行要点。

## 报告

- `reports/baseline3-qwen2p5-*.md`：Baseline3 对比结论。
- `reports/vllm-pin-compare.md`：vLLM sleep_l1 / sleep_l2 pin/no-pin breakdown，包含 L2 reload weights 拆分说明。
- `reports/phase1-two-model-pool.md`：历史两模型 repeated `sleep_l1` 结果；其中 metadata/eviction 字段是 legacy schema，不是当前 release protocol。
- `reports/cumem-copy-microbench.md`：PCIe copy、CuMemAllocator synthetic copy 和 safetensors 粒度 microbench。

`plans/` 是历史实施计划归档，可能保留当时的路径和步骤，不作为当前复现入口。
