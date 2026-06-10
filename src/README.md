# `src/` 目录说明

这个目录放置 benchmark 执行入口、共享库和结果后处理工具。原则上：

- `src/` 根目录下的脚本会启动 benchmark、服务进程或完整实验矩阵。
- `src/tool/` 下的脚本只处理已有结果，不参与 benchmark 运行过程。
- `src/benchlib/` 是 benchmark 入口共享的内部库，不建议直接命令行执行。

## 当前测试环境速记

当前仓库的 benchmark 环境是“两层来源”：

- Python 解释器来自本仓库的 uv 虚拟环境：`/home/ljl/research-systems/llm-switch-bench/.venv/bin/python`。
- vLLM Python 包来自用户本地源码 checkout 的 editable 安装：`/home/ljl/research-systems/vllm`。

因此，vLLM lifecycle 测试不是直接调用系统 Python，也不是使用一份普通的 PyPI/wheel `vllm`。实际流程是：

1. 用本仓库 `.venv/bin/python` 启动 benchmark harness。
2. `bench_vllm_lifecycle.py` 再用 `--python .venv/bin/python` 启动 vLLM OpenAI server。
3. 这个解释器执行 `python -m vllm.entrypoints.openai.api_server` 时，`import vllm` 解析到 `/home/ljl/research-systems/vllm/vllm`。

可用下面命令快速确认当前环境：

```bash
.venv/bin/python -c "import sys, vllm; print(sys.executable); print(vllm.__file__); print(getattr(vllm, '__version__', 'unknown'))"
```

当前检查结果应类似：

```text
/home/ljl/research-systems/llm-switch-bench/.venv/bin/python
/home/ljl/research-systems/vllm/vllm/__init__.py
```

本地 `.venv` 中的安装记录也能证明这一点：

- `.venv/lib/python3.12/site-packages/vllm-0.1.dev16944+gb3269454b.dist-info/direct_url.json` 记录 `file:///home/ljl/research-systems/vllm` 且 `editable=true`。
- `.venv/lib/python3.12/site-packages/__editable___vllm_0_1_dev16944_gb3269454b_finder.py` 将 `vllm` 映射到 `/home/ljl/research-systems/vllm/vllm`。
- 已保留的 vLLM curated run metadata 记录了 `--python .venv/bin/python`，例如 `results/baselines/vllm/qwen2p5_0p5b/20260603_150331/metadata.json`。

一句话结论：**测试进程使用本仓库 uv `.venv` 的解释器，但 vLLM 代码来自用户自己在 `/home/ljl/research-systems/vllm` 编译/安装的本地源码。**

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

### `bench_vllm_pin_compare.py`

vLLM sleep mode pinned/non-pinned CPU backup profiling 对照实验入口。支持 `sleep_l1` 和 `sleep_l2`，默认覆盖 Qwen2.5 0.5B、1.5B、3B，并为 3B 使用 `gpu_memory_utilization=0.85`。

推荐通过脚本运行：

```bash
scripts/run_profiling.sh
```

快速 dry-run：

```bash
DRY_RUN=1 METHOD=sleep_l1 MODELS=qwen2p5_0p5b PIN_MODES=true REPEATS=1 scripts/run_profiling.sh
```

测试 `sleep_l2` 并输出到独立目录：

```bash
METHOD=sleep_l2 OUT_DIR=results/profiling/sleep_l2_pin_compare scripts/run_profiling.sh
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

1. 用 `scripts/run_baseline3.sh` 或 `scripts/run_profiling.sh` 复现实验。
2. 在 `results/` 中保留最新 curated 输出。
3. 用 `src/tool/` 下的工具生成 Markdown、CSV 或图。
4. 将最终报告放到 `docs/reports/`，将原始实验输出保留在 `results/`。
