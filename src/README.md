# `src/` 目录说明

这个目录放置 benchmark 执行入口、共享库和结果后处理工具。原则上：

- `src/` 根目录下的脚本会启动 benchmark、服务进程或完整实验矩阵。
- `src/microbench/` 下的脚本启动 CUDA/vLLM allocator microbench，用于解释 sleep/wake copy 与 allocation 成本。
- `src/tool/` 下的脚本只处理已有结果，不参与 benchmark 运行过程。
- `src/benchlib/` 是 benchmark 入口共享的内部库，不建议直接命令行执行。

## Python 与 vLLM 来源

benchmark 应在仓库自己的 uv 环境中执行；vLLM 可以来自 wheel，也可以来自待测源码的 editable install。不要假设固定的 checkout 路径。运行前记录实际解释器、模块路径和版本：

```bash
.venv/bin/python -c "import sys, vllm; print(sys.executable); print(vllm.__file__); print(getattr(vllm, '__version__', 'unknown'))"
```

summary metadata 会记录 bench repo 和所加载 vLLM module 的 git revision/dirty state；论文实验应保存该 metadata，并确保 treatment/control 使用同一解释器、模型和依赖环境。

## Benchmark 执行入口

### `bench_vllm_lifecycle.py`

vLLM lifecycle benchmark 主入口，用于测试 cold reload、sleep/wake 等方法。`sleep_l2` 的 restore 会拆成 `wake_weights`、`reload_weights`、`wake_kv_cache` 三段；当 vLLM 写出 sleep profile JSONL 时，脚本会同步生成 `sleep_profile_summary.csv`。

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

vLLM sleep mode pinned/non-pinned CPU backup profiling 对照实验入口。支持 `sleep_l1` 和 `sleep_l2`；所有模型通过 `--model NAME=PATH[,GPU_UTILIZATION]` 或 `MODEL_SPECS` 显式选择。

推荐通过脚本运行：

```bash
scripts/run_profiling.sh
```

快速 dry-run：

```bash
MODEL_SPECS='small=/models/small,0.45' DRY_RUN=1 \
  METHOD=sleep_l1 PIN_MODES=true REPEATS=1 scripts/run_profiling.sh
```

测试 `sleep_l2` 并输出到独立目录：

```bash
METHOD=sleep_l2 OUT_DIR=results/profiling/sleep_l2_pin_compare scripts/run_profiling.sh
```

完整默认输出目录：

```text
results/profiling/sleep_l1_pin_compare/
```

### `bench_vllm_repeated_sleep_l1.py`

离线 vLLM `LLM` API profiling 入口。模型通过 `--models NAME=PATH` 显式选择，脚本按给定顺序多轮执行 `wake_up()`、推理和 `sleep(level=1)`。它将 sleep/wake allocator event、coordinator 累计 counter 的 step delta、worker RSS 与 host `MemAvailable` 展平到 `repeated_sleep_l1_steps.csv`。

```bash
.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --models small=/models/small large=/models/large \
  --out-dir results/profiling/repeated_sleep_l1 \
  --iterations 5
```

pressure/no-pressure 实验分别使用 `--expect-release` / `--no-expect-release`；no-pressure control 可增加 `--expect-reuse`，要求后续 sleep 同时观测到 positive reuse 和 zero D2H。若验证物理回收，同时设置观测窗口和 `--min-worker-rss-reclaim-bytes`。每轮确定性推理的 token IDs/text 必须与同模型首次结果一致；pressure 模式还会按当前 run 的 client ID 前缀读取 controller final stats，强制检查 host-cache flush error 为零、`released >= requested` 且 `pending == 0`（allocator 可按整块合法 over-release）。summary 记录模型、全部参数、bench/vLLM git metadata、GPU、host memory、run-local coordinator stats 和 assertion failures。

历史结果与旧 schema 见 `docs/reports/phase1-two-model-pool.md`；当前 CLI 以 `--help` 为准。

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

### `tool/plot_vllm_pin_compare.py`

从 `results/profiling/sleep_l1_pin_compare` 和 `results/profiling/sleep_l2_pin_compare` 读取 breakdown 汇总，为每个模型生成 pin/no-pin 对比图。新版图里会在可用时把 `sleep_l2` 的 `reload_weights` 单独画出。

```bash
.venv/bin/python src/tool/plot_vllm_pin_compare.py
```

### `tool/plot_phase1_two_model_pool.py`

读取 `bench_vllm_repeated_sleep_l1.py` 生成的 steps CSV，画出两模型交替 repeated sleep 的 sleep breakdown 和推理延迟。建议显式传入最新 curated run 的 CSV。

```bash
.venv/bin/python src/tool/plot_phase1_two_model_pool.py \
  --csv results/profiling/phase1_two_model_pool/20260702_165801/phase1_two_model_repeated_sleep_steps.csv \
  --out results/profiling/phase1_two_model_pool/20260702_165801/phase1_two_model_sleep_breakdown.pdf
```

## Microbench 入口

### `microbench/microbench_pcie_copy.py`

测量 pinned/pageable host memory 下的 CPU<->GPU copy 带宽，可同时覆盖 torch `copy_` 和 vLLM `CudaRTLibrary.cudaMemcpy` 路径。

```bash
.venv/bin/python src/microbench/microbench_pcie_copy.py \
  --include-vllm-cudart \
  --repeats 7 \
  --csv results/profiling/microbench/microbench_pcie_copy_<timestamp>.csv
```

### `microbench/microbench_cumem_sleep_copy.py`

在 vLLM `CuMemAllocator` memory pool 下创建 synthetic CUDA tensors，直接调用 `allocator.sleep()` / `allocator.wake_up()`，用于隔离 backup allocation、D2H/H2D copy 和 create/map 成本。

```bash
.venv/bin/python src/microbench/microbench_cumem_sleep_copy.py \
  --out-dir results/profiling/microbench/cumem_copy_microbench_<timestamp>/1GiB_41 \
  --repeats 1 \
  --cases 1GiB:41
```

### `microbench/microbench_cumem_safetensor_sizes.py`

按模型 safetensors 中每个 tensor 的 byte size 建立 synthetic allocations，更接近真实权重粒度。

```bash
.venv/bin/python src/microbench/microbench_cumem_safetensor_sizes.py \
  --out-dir results/profiling/microbench/cumem_safetensor_sizes_microbench_<timestamp>/Qwen2.5-1.5B-Instruct \
  --repeats 1 \
  /home/ljl/models/hf/Qwen2.5-1.5B-Instruct
```

最新 microbench 结论见 `docs/reports/cumem-copy-microbench.md`。

## 推荐工作流

1. 用 `scripts/run_baseline3.sh` 或 `scripts/run_profiling.sh` 复现实验。
2. 在 `results/` 中保留最新 curated 输出。
3. 用 `src/tool/` 下的工具生成 Markdown、CSV 或图。
4. 将最终报告放到 `docs/reports/`，将原始实验输出保留在 `results/`。
