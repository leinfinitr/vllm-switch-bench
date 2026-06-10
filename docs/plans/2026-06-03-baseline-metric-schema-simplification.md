# 归档：Baseline 指标 schema 简化计划

这是 2026-06-03 的实施计划归档。计划目标是统一 vLLM、ServerlessLLM、SwapServeLLM 的输出字段，让 Baseline3 报告能稳定比较 startup、evict、restore、推理延迟和显存占用。

## 已沉淀到当前仓库的内容

- 统一 CSV/JSON 写出逻辑：`src/benchlib/schema.py`
- Baseline3 报告工具：`src/tool/analyze_baseline3.py`
- Baseline3 绘图工具：`src/tool/plot_baseline3.py`
- 当前报告：`docs/reports/baseline3-qwen2p5-0p5b.md`、`docs/reports/baseline3-qwen2p5-1p5b.md`、`docs/reports/baseline3-qwen2p5-3b.md`

## 当前字段语义

- `startup_latency_s`：服务进入可用状态的启动耗时；外部常驻系统可为空。
- `evict_latency_s`：释放或切出模型的耗时。
- `restore_latency_s`：恢复或切入模型的耗时。
- `restore_latency_estimated`：该 restore 是否由请求差值估算。
- `ttft_*`、`latency_*`：切换前后请求行为。
- `memory_*_mib`：ready / evict 阶段采样到的资源占用。

## 归档说明

原计划中的旧路径和旧命令已被当前 `src/README.md`、`docs/baselines/*.md` 和 `docs/reports/*.md` 替代。
