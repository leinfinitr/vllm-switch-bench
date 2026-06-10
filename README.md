# LLM Switch Bench

本仓库用于在本机复现和比较 LLM 服务生命周期相关基线，包括 vLLM 冷启动、vLLM Sleep Mode、ServerlessLLM 和 SwapServeLLM。实现仓库保持独立，本仓库只保存 benchmark harness、运行脚本、必要说明和 curated 结果。

## 主要入口

- `scripts/run_baseline3.sh`：按 `configs/baseline3.local.yaml` 运行 Baseline3 聚合对比。
- `scripts/run_profiling.sh`：运行 vLLM `sleep_l1` / `sleep_l2` 的 pin/no-pin profiling 对比。
- `src/bench_vllm_lifecycle.py`：vLLM cold reload、sleep/wake 的底层 benchmark。
- `src/bench_vllm_pin_compare.py`：多模型 pin/no-pin profiling 矩阵。
- `src/tool/`：只读取已有结果的报告、绘图、合并工具。

## 基线含义

- Baseline1：vLLM cold reload，衡量重新启动服务并加载模型的成本。
- Baseline2：vLLM Sleep Mode，当前仓库测的是单模型 sleep/wake 近似，不包含多模型调度器。
- Baseline3：ServerlessLLM、SwapServeLLM 等外部系统级切换方案，与 vLLM 内部 sleep mode 做横向对比。

## 常用命令

```bash
cd /home/ljl/research-systems/llm-switch-bench
uv venv --python 3.12 .venv
uv pip install pytest psutil requests pandas matplotlib
```

运行测试：

```bash
uv run pytest tests -q
```

运行 Baseline3：

```bash
scripts/run_baseline3.sh
```

运行 vLLM pin/no-pin profiling：

```bash
METHOD=sleep_l1 OUT_DIR=results/profiling/sleep_l1_pin_compare scripts/run_profiling.sh
METHOD=sleep_l2 OUT_DIR=results/profiling/sleep_l2_pin_compare scripts/run_profiling.sh
```

## 当前结论入口

- Baseline3 总结：`docs/reports/baseline3-qwen2p5-0p5b.md`、`docs/reports/baseline3-qwen2p5-1p5b.md`、`docs/reports/baseline3-qwen2p5-3b.md`
- vLLM pin/no-pin profiling：`docs/reports/vllm-pin-compare.md`
