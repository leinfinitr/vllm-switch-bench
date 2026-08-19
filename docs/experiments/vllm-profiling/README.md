# vLLM profiling

## 研究问题与指标

该实验分析原生 vLLM L1/L2 与 vllm-switch CPU/exact-disk 的 sleep/wake 阶段，并显式区分：

- 同一进程内第一次与稳态 L1/CPU/exact-disk 循环；
- vLLM L2 的 cold page cache 与 warm page cache；
- 连续 wake 总延迟与内部 active phase。

Sleep 延迟从调用进程终止或 sleep 开始，到进程退出或 sleep 返回。Wake 延迟从进程启动或第一个 restore 调用开始，到所有 restore 阶段返回；推理和 L2 cache treatment 不计入 wake。L2 的 wake 总量使用覆盖 `wake(weights) -> reload_weights -> wake(kv_cache)` 的连续外层计时，而不是三段 HTTP/RPC active time 的简单相加。

每个机制运行三个独立 process blocks，每个 block 内执行三个 sleep/wake cycles。跨 block 报告 median 和 min/max：

- first：每个 block 的 cycle 0；
- steady：每个 block 的两个后续 cycle 的算术平均；
- L2 cold：每个 block 一个经验证的 cold cycle；
- L2 warm：每个 block 两个 warm cycles 的算术平均。

每个 stacked bar 使用最接近该操作跨 block 中位数的真实样本。各 phase 必须闭合到连续 wall time。Exact-disk 的 read/hash/H2D 并发执行，因此使用 pipeline wall time而不是把 worker 时间相加。

## L2 page-cache 处理

三个 L2 blocks 轮换 cold cycle 位置：

```text
block 0: cold, warm, warm
block 1: warm, cold, warm
block 2: warm, warm, cold
```

Cold treatment 只针对本地 `*.safetensors` 文件调用 `POSIX_FADV_DONTNEED`，不修改系统级 cache，也不使用 `/proc/sys/vm/drop_caches`。Treatment 与验证发生在 L2 sleep 完成后、wake timer 开始前。

Cold 样本必须同时满足：

- wake 前 `mincore` resident ratio 不超过 5%；
- timed wake 期间 process-tree physical read bytes 至少为 checkpoint bytes 的 90%。

Warm 样本必须同时满足：

- wake 前 resident ratio 至少 90%；
- physical read bytes 不超过 checkpoint bytes 的 10%。

不满足条件的 cycle 标记为 invalid，不进入 retained 结果。

## 固定范围

未限定的 `vLLM` 指原生 upstream baseline；fork-specific 机制使用 `vllm-switch`。固定 workload 为 Qwen2.5-0.5B-Instruct、float16、max model length 1024、eager execution、0.80 GPU memory utilization 和一张 RTX 3080。原生 vLLM 与 vllm-switch 使用相同 upstream base；每个实际 engine commit、Python、Torch、CUDA 和 imported module path分别记录在 source provenance 中。

## Retained result

新结果由当前协议的本地 GPU rerun 生成。发布目录保留：

- [JSON summary](../../../results/vllm-profiling/summary.json)
- [Compact per-block samples](../../../results/vllm-profiling/raw/profile-samples.json)
- [PNG figure](../../../results/vllm-profiling/figures/vllm-profiling.png)
- [PDF figure](../../../results/vllm-profiling/figures/vllm-profiling.pdf)

完整 server logs、sleep profile JSONL、per-file residency 和调试产物保留在 ignored `results/tmp/`，不进入 Git。Compact rows保留图表所需的总量、阶段、process/cycle 分类，以及 L2 cache residency/physical-read 证据。

## 重新测量

从仓库根目录执行，GPU 必须空闲。以下变量使用本机有效路径：

```bash
uv sync --frozen --group dev

BENCH_ROOT=$PWD
RUN_ROOT="$BENCH_ROOT/results/tmp/vllm-profiling/run-001"
MODEL=/path/to/Qwen2.5-0.5B-Instruct
VLLM_REPO=/path/to/native-vllm-profiling
VLLM_PYTHON=/path/to/native-vllm-python
VLLM_SWITCH_REPO=/path/to/vllm-switch
VLLM_SWITCH_PYTHON="$VLLM_SWITCH_REPO/.venv/bin/python"
```

Cold process reference 使用三个独立进程：

```bash
scripts/vllm-profiling.sh \
  --model "$MODEL" \
  --served-model-name qwen-0.5b \
  --python "$VLLM_PYTHON" \
  --workdir "$VLLM_REPO" \
  --methods cold_reload \
  --prompts short_short \
  --repeats 3 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --enforce-eager \
  --out-dir "$RUN_ROOT/cold"
```

原生 vLLM L1/L2 使用三个 process blocks × 三个 cycles：

```bash
scripts/vllm-profiling.sh \
  --model "$MODEL" \
  --served-model-name qwen-0.5b \
  --python "$VLLM_PYTHON" \
  --workdir "$VLLM_REPO" \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 3 \
  --cycles-per-process 3 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --enforce-eager \
  --idle-s 0 \
  --out-dir "$RUN_ROOT/vllm"
```

vllm-switch CPU/exact-disk 使用完全相同的 block/cycle harness：

```bash
scripts/vllm-profiling.sh \
  --model "$MODEL" \
  --served-model-name qwen-0.5b \
  --python "$VLLM_SWITCH_PYTHON" \
  --workdir "$VLLM_SWITCH_REPO" \
  --methods cpu_backup exact_disk \
  --prompts short_short \
  --repeats 3 \
  --cycles-per-process 3 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype float16 \
  --enforce-eager \
  --idle-s 0 \
  --out-dir "$RUN_ROOT/vllm-switch"
```

每个 block 必须有 `block-summary.json`、`ok=true`、三个 cycle、输出相等，且 L2 cache evidence 全部通过。

## 更新 results/

```bash
COLD_SUMMARY="$RUN_ROOT/cold/<timestamp>/summary.json"
VLLM_BLOCKS="$RUN_ROOT/vllm/<timestamp>"
SWITCH_BLOCKS="$RUN_ROOT/vllm-switch/<timestamp>"

scripts/promote.sh vllm-profiling \
  --candidate-root "$RUN_ROOT/candidate-dry" \
  --collected-at YYYY-MM-DD \
  --cold-summary "$COLD_SUMMARY" \
  --vllm-blocks "$VLLM_BLOCKS" \
  --switch-blocks "$SWITCH_BLOCKS"
```

Review candidate 后使用新的 candidate root 加 `--apply`，然后执行：

```bash
scripts/build_all.sh vllm-profiling
uv run python -m vllm_switch_bench.validation.vllm_profiling.validate
scripts/build_all.sh vllm-profiling
uv run python -m vllm_switch_bench.validation.vllm_profiling.validate
git diff --exit-code -- results
```

## 威胁与限制

该实验只覆盖一个模型、一个 host/GPU，并只有三个独立 process blocks。Allocator、driver、filesystem 和其他共享服务器活动仍会影响结果。`POSIX_FADV_DONTNEED` 是 advisory，因此 cold/warm 分类依赖实际 `mincore` 和 physical-read 证据，而不是依赖系统调用成功本身。结果不建立 throughput、capacity、tail latency 或一般性系统优越性。
