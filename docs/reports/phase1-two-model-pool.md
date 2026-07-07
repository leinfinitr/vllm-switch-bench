# vLLM repeated sleep_l1 CPU backup pool 验证

本文记录两模型 repeated `sleep_l1` profiling 的 curated 结果，用于验证 pinned CPU backup pool 是否能跨 sleep 周期复用，以及 CPU backup metadata coordinator / daemon eviction 对复用路径的影响。相关提交包括 `c94a302`、`d54ee17`、`fa89239`、`f07f6af` 和 `4a0d344`。

## 入口与结果

本地 pool baseline 复现入口：

```bash
.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --out-dir results/profiling/phase1_two_model_pool \
  --iterations 5
```

metadata coordinator / daemon eviction 复现入口：

```bash
.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --out-dir results/profiling/phase1_metadata_coordinator \
  --iterations 3 \
  --coordinator-url http://127.0.0.1:19090

.venv/bin/python src/bench_vllm_repeated_sleep_l1.py \
  --out-dir results/profiling/phase2_daemon_eviction \
  --iterations 3 \
  --coordinator-url http://127.0.0.1:19091
```

上述 coordinator 命令假设对应 vLLM CPU backup coordinator daemon 已经在 URL 上运行；benchmark 只负责把 URL 和 client/model id 传给 vLLM worker。

最新 curated runs：

- 本地 pinned pool baseline：`results/profiling/phase1_two_model_pool/20260702_165801/`
- metadata coordinator：`results/profiling/phase1_metadata_coordinator/20260707_084649/`
- daemon eviction：`results/profiling/phase2_daemon_eviction/20260707_093412/`

每个 run 的主要文件：

- `phase1_two_model_repeated_sleep_summary.json`：完整步骤和 sleep profile events。
- `phase1_two_model_repeated_sleep_steps.csv`：每个模型、每次 sleep 的扁平化指标。

本地 pool baseline 还保留了 `phase1_two_model_sleep_breakdown.png` / `.pdf`，用于可视化 sleep breakdown 与 inference latency。

绘图命令建议显式传入最新 CSV：

```bash
.venv/bin/python src/tool/plot_phase1_two_model_pool.py \
  --csv results/profiling/phase1_two_model_pool/20260702_165801/phase1_two_model_repeated_sleep_steps.csv \
  --out results/profiling/phase1_two_model_pool/20260702_165801/phase1_two_model_sleep_breakdown.pdf
```

## 实验设置

- 模型：`qwen2p5_0p5b` 与 `qwen2p5_1p5b`。
- 迭代：本地 pool baseline 为 5 轮；coordinator / eviction runs 为 3 轮。每轮按 0.5B、1.5B 顺序执行。
- 流程：首次为 load，后续为 `wake_up()`，然后推理并 `sleep(level=1)`。
- prompt：`short_short`。
- `gpu_memory_utilization=0.55`，`max_model_len=1024`，`dtype=float16`。
- 设置 `--coordinator-url` 时，脚本会启用 vLLM 的 CPU backup coordinator backend，并在 steps CSV 中记录 coordinator flush 与 eviction 指标。

## 本地 pool baseline

| model | first load | first sleep | first misses | reuse sleep mean | wake mean | infer mean | reuse count |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2p5_0p5b` | 8.553s | 0.422s | 41 | 0.050s | 0.104s | 0.088s | 41 |
| `qwen2p5_1p5b` | 9.866s | 0.991s | 76 | 0.102s | 0.318s | 0.119s | 76 |

首次 `sleep_l1` 会为 weights 分配 pinned CPU backup：0.5B 记录 41 次 miss，1.5B 记录 76 次 miss。后续同模型重复 sleep 时 `cpu_backup_pool_reuse_count` 分别稳定为 41 和 76，`copy_d2h_s` 与 `cpu_backup_alloc_s` 在 steps CSV 中降为 0，sleep 主要剩下 unmap/release 成本。

## Coordinator 与 eviction

metadata coordinator run 只上报本地 backup metadata，不主动驱逐本地 backup buffer。结果与本地 pool baseline 基本一致：后续 sleep 仍复用 41 / 76 个 backup，`copy_d2h_s=0`，coordinator flush error 为 0。

daemon eviction run 会在后续 sleep 前收到 coordinator eviction request，并释放已缓存 backup。最新 steps CSV 新增并记录了：

- `sleep_allocator_cpu_backup_coordinator_eviction_polls`
- `sleep_allocator_cpu_backup_coordinator_eviction_requests_received`
- `sleep_allocator_cpu_backup_eviction_released_count`
- `sleep_allocator_cpu_backup_eviction_released_bytes`

| run | model | repeated sleep mean | repeated D2H mean | repeated backup status | eviction evidence |
|---|---|---:|---:|---|---|
| metadata coordinator | `qwen2p5_0p5b` | 0.051s | 0.000s | reuse 41, miss 0 | events sent, no flush errors |
| metadata coordinator | `qwen2p5_1p5b` | 0.103s | 0.000s | reuse 76, miss 0 | events sent, no flush errors |
| daemon eviction | `qwen2p5_0p5b` | 0.087s | 0.053s | reuse 0, miss 41 | final released 82 / 2097152000 bytes |
| daemon eviction | `qwen2p5_1p5b` | 0.206s | 0.165s | reuse 0, miss 76 | final released 152 / 6501171200 bytes |

这说明 coordinator 元数据通路本身不会破坏本地 pinned backup 复用；真正触发 eviction 后，后续 sleep 会重新执行 weights D2H copy。由于 pinned allocation capacity 仍可快速复用，daemon eviction run 里的 `cpu_backup_alloc_s` 接近 0，但 `copy_d2h_s` 重新出现。

## 解释

这个结果支持 pin/no-pin 报告里的优化方向：不应简单把 backup 改成 pageable memory，因为 copy 会变慢；更有价值的是保留 pinned copy，同时复用已分配的 CPU backup buffer。两模型交替运行时，pool 仍能按模型权重 allocation 粒度复用，说明该方向可以覆盖多模型切换场景。

新增 coordinator / eviction runs 进一步把问题拆开：metadata-only coordinator 可以作为跨进程调度的观测面；daemon eviction 可以释放本地 backup，但会让后续 sleep 付出重新 D2H copy 的成本。因此后续策略应避免无条件 eviction，而是只在 CPU backup 内存压力或跨模型调度收益足够时驱逐。

当前脚本只把 sleep allocator event 展平成 steps CSV，wake 侧仍主要看 `wake_latency_s` 和 summary JSON 中的原始 profile events。若要继续做论文图，下一步可以把 wake allocator event 也扁平化到 steps CSV，直接展示 H2D copy 与 create/map 的复用后稳定成本。
