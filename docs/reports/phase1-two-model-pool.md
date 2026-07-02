# vLLM repeated sleep_l1 CPU backup pool 验证

本文记录最近两模型 repeated `sleep_l1` profiling 的 curated 结果，用于验证 pinned CPU backup pool 是否能跨 sleep 周期复用。相关提交包括 `c94a302`、`d54ee17` 和 `fa89239`。

## 入口与结果

复现入口：

```bash
.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --out-dir results/profiling/phase1_two_model_pool \
  --iterations 5
```

最新 curated run：`results/profiling/phase1_two_model_pool/20260702_165801/`

主要文件：

- `phase1_two_model_repeated_sleep_summary.json`：完整步骤和 sleep profile events。
- `phase1_two_model_repeated_sleep_steps.csv`：每个模型、每次 sleep 的扁平化指标。
- `phase1_two_model_sleep_breakdown.png` / `.pdf`：sleep breakdown 与 inference latency 可视化。

绘图命令建议显式传入最新 CSV：

```bash
.venv/bin/python src/tool/plot_phase1_two_model_pool.py \
  --csv results/profiling/phase1_two_model_pool/20260702_165801/phase1_two_model_repeated_sleep_steps.csv \
  --out results/profiling/phase1_two_model_pool/20260702_165801/phase1_two_model_sleep_breakdown.pdf
```

## 实验设置

- 模型：`qwen2p5_0p5b` 与 `qwen2p5_1p5b`。
- 迭代：5 轮，每轮按 0.5B、1.5B 顺序执行。
- 流程：首次为 load，后续为 `wake_up()`，然后推理并 `sleep(level=1)`。
- prompt：`short_short`。
- `gpu_memory_utilization=0.55`，`max_model_len=1024`，`dtype=float16`。

## 结果摘要

| model | first load | first sleep | first misses | reuse sleep mean | wake mean | infer mean | reuse count |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2p5_0p5b` | 8.553s | 0.422s | 41 | 0.050s | 0.104s | 0.088s | 41 |
| `qwen2p5_1p5b` | 9.866s | 0.991s | 76 | 0.102s | 0.318s | 0.119s | 76 |

首次 `sleep_l1` 会为 weights 分配 pinned CPU backup：0.5B 记录 41 次 miss，1.5B 记录 76 次 miss。后续同模型重复 sleep 时 `cpu_backup_pool_reuse_count` 分别稳定为 41 和 76，`copy_d2h_s` 与 `cpu_backup_alloc_s` 在 steps CSV 中降为 0，sleep 主要剩下 unmap/release 成本。

## 解释

这个结果支持 pin/no-pin 报告里的优化方向：不应简单把 backup 改成 pageable memory，因为 copy 会变慢；更有价值的是保留 pinned copy，同时复用已分配的 CPU backup buffer。两模型交替运行时，pool 仍能按模型权重 allocation 粒度复用，说明该方向可以覆盖多模型切换场景。

当前脚本只把 sleep allocator event 展平成 steps CSV，wake 侧仍主要看 `wake_latency_s` 和 summary JSON 中的原始 profile events。若要继续做论文图，下一步可以把 wake allocator event 也扁平化到 steps CSV，直接展示 H2D copy 与 create/map 的复用后稳定成本。
