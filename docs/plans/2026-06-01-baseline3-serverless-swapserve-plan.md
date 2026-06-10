# 归档：Baseline3 ServerlessLLM / SwapServeLLM 接入计划

这是 2026-06-01 的实施计划归档。计划的目标是把 ServerlessLLM 和 SwapServeLLM 接入 Baseline3，对比 vLLM 内部 sleep mode 之外的系统级切换方案。

## 已沉淀到当前仓库的内容

- Baseline3 入口：`scripts/run_baseline3.sh`
- 聚合 driver：`src/bench_baseline3.py`
- ServerlessLLM adapter：`src/bench_serverless_llm.py`
- SwapServeLLM adapter：`src/bench_swapserve_llm.py`
- 当前复现说明：`docs/baselines/baseline3-engine-checkpoint-hotswap.md`
- 外部系统说明：`docs/systems/serverlessllm.md`、`docs/systems/swapservellm.md`

## 当前结论

ServerlessLLM 和 SwapServeLLM 可作为系统级相关工作对比，但它们独立于 vLLM 内部实现。后续如果目标是给 vLLM 提交 PR，主线应优先放在 vLLM Sleep Mode 内部路径和 allocator 行为上。

## 归档说明

原计划中的逐步实现细节、旧命令和旧路径已经被当前文档替代；本文只保留背景和最终沉淀位置。
