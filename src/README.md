# `src/` 目录说明

这个目录放置 benchmark 执行入口、共享库和结果后处理工具。原则上：

- `src/` 根目录下的脚本会启动 benchmark、服务进程或完整实验矩阵。
- `src/tool/` 下的脚本只处理已有结果，不参与 benchmark 运行过程。
- `src/benchlib/` 是 benchmark 入口共享的内部库，不建议直接命令行执行。

## Benchmark 执行入口

### `bench_vllm_lifecycle.py`

vLLM lifecycle benchmark 主入口，用于测试 cold reload、sleep/wake 等方法。

常用方式：

```bash
.venv/bin/python src/bench_vllm_lifecycle.py \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --workdir /home/ljl/research-systems/llm-switch-bench \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 1 \
  --out-dir results/profiling/manual_vllm_lifecycle
```

### `bench_baseline3.py`

Baseline3 聚合入口。它读取 config，按需调用或导入 vLLM、ServerlessLLM、SwapServeLLM 的结果，输出统一的 `summary.json`、`summary.csv` 和 `metadata.json`。

推荐通过脚本运行：

```bash
scripts/run_baseline3.sh
```

也可以直接运行：

```bash
.venv/bin/python src/bench_baseline3.py \
  --config configs/baseline3.local.yaml \
  --out-dir results/baselines/baseline3/qwen2p5_0p5b
```

模型路径、prompt、repeat 和外部系统地址由 config 指定；脚本本身不绑定具体模型。

### `bench_serverless_llm.py`

ServerlessLLM adapter benchmark 入口。通常由 `bench_baseline3.py` 调用；单独调试 ServerlessLLM 时也可以直接执行。

### `bench_swapserve_llm.py`

SwapServeLLM adapter benchmark 入口。通常由 `bench_baseline3.py` 调用；单独调试 SwapServeLLM 时也可以直接执行。

### `run_sleep_l1_pin_compare.py`

sleep_l1 pinned/non-pinned CPU backup profiling 对照实验入口。默认覆盖 Qwen2.5 0.5B、1.5B、3B，并为 3B 使用 `gpu_memory_utilization=0.85`。

推荐通过脚本运行：

```bash
scripts/run_sleep_l1_profiling.sh
```

快速 dry-run：

```bash
DRY_RUN=1 MODELS=qwen2p5_0p5b PIN_MODES=true REPEATS=1 scripts/run_sleep_l1_profiling.sh
```

完整默认输出目录：

```text
results/profiling/sleep_l1_pin_compare/
```

## 共享库

### `benchlib/`

`benchlib/` 提供 benchmark 入口共享的配置、HTTP、资源采样和结果 schema 工具：

- `config.py`: config 加载与 repo metadata 采集。
- `http.py`: HTTP polling/request helper。
- `resources.py`: GPU/CPU/process 资源采样。
- `sampling.py`: 后台采样循环。
- `schema.py`: summary CSV 写出字段定义。

这些模块由 benchmark 脚本 import，不作为独立 CLI 使用。

## 结果后处理工具

`src/tool/` 下的脚本不启动 benchmark，只读取已经生成的结果目录。

### `tool/analyze_results.py`

生成 vLLM lifecycle 单次结果目录的 Markdown 报告。

```bash
.venv/bin/python src/tool/analyze_results.py \
  results/baselines/vllm/qwen2p5_0p5b/<timestamp> \
  --out docs/reports/vllm-qwen2p5-0p5b.md
```

### `tool/analyze_baseline3.py`

生成 Baseline3 聚合结果的 Markdown 报告。

```bash
.venv/bin/python src/tool/analyze_baseline3.py \
  results/baselines/baseline3/qwen2p5_0p5b/<timestamp> \
  --out docs/reports/baseline3-qwen2p5-0p5b.md
```

### `tool/plot_baseline3.py`

从 Baseline3 `summary.csv` 生成对比图。

```bash
.venv/bin/python src/tool/plot_baseline3.py \
  results/baselines/baseline3/qwen2p5_0p5b/<timestamp> \
  --out docs/reports/figures/baseline3-qwen2p5-0p5b-comparison.png
```

### `tool/merge_results.py`

合并同一 harness 生成的多个结果目录，适合补跑部分 method/prompt 后整理为一个结果目录。

```bash
.venv/bin/python src/tool/merge_results.py \
  results/baselines/vllm/qwen2p5_0p5b/<merged_timestamp> \
  results/baselines/vllm/qwen2p5_0p5b/<run_a> \
  results/baselines/vllm/qwen2p5_0p5b/<run_b>
```

### `tool/summarize_memory.py`

从 lifecycle event logs 提取阶段级显存/内存快照。

```bash
.venv/bin/python src/tool/summarize_memory.py \
  results/baselines/vllm/qwen2p5_0p5b/<timestamp> \
  --out-csv docs/reports/memory-summary.csv \
  --out-md docs/reports/memory-summary.md
```

## 推荐工作流

1. 用 `scripts/run_baseline3.sh` 或 `scripts/run_sleep_l1_profiling.sh` 复现实验。
2. 在 `results/` 中保留最新 curated 输出。
3. 用 `src/tool/` 下的工具生成 Markdown、CSV 或图。
4. 将最终报告放到 `docs/reports/`，将原始实验输出保留在 `results/`。
