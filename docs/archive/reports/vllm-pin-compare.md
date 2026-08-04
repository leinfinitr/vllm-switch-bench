# vLLM Sleep Mode pin/no-pin 对比

本文比较 `sleep_l1` 和 `sleep_l2` 在 CPU backup 使用 pinned memory 与 non-pinned memory 时的 sleep/wake breakdown。实验由 `scripts/run_profiling.sh` 复现，原始结果位于 `results/profiling/sleep_l1_pin_compare/` 和 `results/profiling/sleep_l2_pin_compare/`。

## 复现命令

```bash
METHOD=sleep_l1 OUT_DIR=results/profiling/sleep_l1_pin_compare scripts/run_profiling.sh
METHOD=sleep_l2 OUT_DIR=results/profiling/sleep_l2_pin_compare scripts/run_profiling.sh
.venv/bin/python src/tool/plot_vllm_pin_compare.py
```

`bench_vllm_lifecycle.py` 会在 sleep profiling 可用时写出 `sleep_profile_summary.csv`。新版 `sleep_l2` restore 会拆成 `wake_weights`、`reload_weights`、`wake_kv_cache` 三段；绘图脚本会在结果目录提供这些字段时把 `reload_weights` 单独显示。

## 图

- `qwen2p5_0p5b`：`docs/reports/figures/vllm-pin-compare-qwen2p5_0p5b.png`
- `qwen2p5_1p5b`：`docs/reports/figures/vllm-pin-compare-qwen2p5_1p5b.png`
- `qwen2p5_3b`：`docs/reports/figures/vllm-pin-compare-qwen2p5_3b.png`

## 总切换时间

| method | model | pin switch | no-pin switch | no-pin 变化 | 结论 |
|---|---|---:|---:|---:|---|
| `sleep_l1` | `qwen2p5_0p5b` | 0.5666 | 0.4848 | -14.4% | no-pin 更快 |
| `sleep_l1` | `qwen2p5_1p5b` | 1.2421 | 1.3561 | +9.2% | no-pin 更慢 |
| `sleep_l1` | `qwen2p5_3b` | 2.6142 | 2.5484 | -2.5% | no-pin 更快 |
| `sleep_l2` | `qwen2p5_0p5b` | 0.2914 | 0.2897 | -0.6% | 基本持平 |
| `sleep_l2` | `qwen2p5_1p5b` | 0.6978 | 0.6978 | +0.0% | 基本持平 |
| `sleep_l2` | `qwen2p5_3b` | 1.5278 | 1.5836 | +3.7% | no-pin 更慢 |

## 关键 breakdown

### sleep_l1

| model | pin CPU alloc | no-pin CPU alloc | pin D2H | no-pin D2H | pin H2D | no-pin H2D |
|---|---:|---:|---:|---:|---:|---:|
| `qwen2p5_0p5b` | 0.3697 | 0.0003 | 0.0559 | 0.2896 | 0.0592 | 0.0830 |
| `qwen2p5_1p5b` | 0.7592 | 0.0006 | 0.1698 | 0.8888 | 0.1782 | 0.2525 |
| `qwen2p5_3b` | 1.7357 | 0.0009 | 0.3245 | 1.7155 | 0.3611 | 0.4946 |

`sleep_l1` 的 no-pin 几乎消除了 CPU backup allocation，但 D2H/H2D 拷贝明显变慢。0.5B 和 3B 总体略有收益，1.5B 反而变慢，因此不能把 no-pin 作为稳定优化。

### wake other 的含义

图中的 `wake other` 不是 vLLM 原生事件，而是绘图脚本根据现有 profiling 字段计算出的残差：

```text
wake other = allocator_wake_up_s - copy_h2d_s - create_map_s
```

关键代码在 `/home/ljl/research-systems/vllm/vllm/device_allocator/cumem.py:345` 附近：

`CuMemAllocator.wake_up()` 当前只细分了两段：

1. `create_and_map(handle)` 计入 `create_map_s`
2. `libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)` 计入 `copy_h2d_s`
3. 其他 wake 循环内开销都会落入 `wake other`

本轮结果中，`sleep_l1` no-pin 的 `wake other` 明显增加：

| model | pin wake other | no-pin wake other | restored weights |
|---|---:|---:|---:|
| `qwen2p5_0p5b` | 0.0001 | 0.0258 | 1.05 GB |
| `qwen2p5_1p5b` | 0.0002 | 0.0737 | 3.25 GB |
| `qwen2p5_3b` | 0.0002 | 0.1431 | 6.31 GB |

这部分增长与 restored weight bytes 基本同向，最可能来自 `data.cpu_backup_tensor = None` 触发的大块 pageable CPU backup tensor 释放/allocator 处理成本。

原因在于：wake_other 目前包含

- cpu_backup_tensor.numel() / element_size() / data_ptr() 等 Python 调用
- profiling 字典统计开销
- data.cpu_backup_tensor = None 引发的 CPU backup tensor 引用释放
- Python loop 的少量开销

但因为 pin 模式下 residual 只有 0.1~0.2ms，这些 Python/profiling 开销可以基本排除；no-pin 多出来的几十到一百多毫秒，大概率就是 pageable CPU tensor 释放/allocator 行为。

当前还没有把这段进一步拆成独立 profiling 字段。若后续需要确认，可以在 vLLM 插桩中补充 `cpu_backup_release_s`，专门包住 `data.cpu_backup_tensor = None`；同时记录 `wake_loop_unaccounted_s`，把 `wake other` 变成可直接归因的字段。

### sleep_l2

| model | pin evict | no-pin evict | pin restore | no-pin restore | 结论 |
|---|---:|---:|---:|---:|---|
| `qwen2p5_0p5b` | 0.0587 | 0.0590 | 0.2326 | 0.2307 | 基本持平 |
| `qwen2p5_1p5b` | 0.1150 | 0.1152 | 0.5828 | 0.5826 | 基本持平 |
| `qwen2p5_3b` | 0.1807 | 0.1811 | 1.3471 | 1.4025 | no-pin 更慢 |

`sleep_l2` 不做 CPU backup，因此 `--sleep-cpu-backup-pin-memory` 基本不影响 sleep 阶段。结果也显示 pin/no-pin 对 0.5B 和 1.5B 基本持平，3B no-pin restore 略慢。

#### reload weights 拆分

最近的 Qwen2.5-1.5B reload profiling 位于 `results/profiling/reload_weights_breakdown/qwen2p5_1p5b/20260611_205014/`。该 run 使用 `sleep_l2`、`short_short`、3 次重复，restore 平均约 0.616s，主要由 `reload_weights` 组成：

| restore step | mean latency | 说明 |
|---|---:|---|
| `wake_weights` | 0.0807 | remap weights memory，不从 CPU backup 拷贝 |
| `reload_weights` | 0.4956 | layerwise reload weights，读取 338 个 tensor，约 3.09GB |
| `wake_kv_cache` | 0.0297 | remap KV cache |

这说明 L2 restore 的核心不在 CPU backup pin/no-pin，而在权重 reload 路径。旧的 pin/no-pin `analysis_summary` 只直接统计 allocator wake/create-map；后续图表应优先使用带 `reload_weights` step 的结果目录。

## 结论

1. `sleep_l1` 的主要瓶颈是 pinned CPU backup tensor 分配。
2. 直接改成 no-pin 不是稳定优化，因为拷贝阶段会退化。
3. 更有价值的方向是复用或池化 pinned CPU backup buffer：保留 pinned copy 的速度，同时避免每次 sleep 都重新分配。
4. `sleep_l2` 的 pin/no-pin 对比说明该开关只影响 L1 权重备份路径，不应作为 L2 优化方向。
5. 后续 repeated sleep 和 microbench 结果已单独整理在 `docs/reports/phase1-two-model-pool.md` 与 `docs/reports/cumem-copy-microbench.md`，用于验证 pool 复用和 copy/allocation 成本来源。
