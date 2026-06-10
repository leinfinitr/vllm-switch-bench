# sleep_l2 pin/no-pin 对比

实验：`sleep_l2`、`short_short`、每个模型和 pin 模式 3 次重复。3B 使用 `gpu_memory_utilization=0.85`，0.5B/1.5B 使用 `0.55`。

## 平均总切换时间

| model | pin switch | no-pin switch | no-pin 变化 | 结论 |
|---|---:|---:|---:|---|
| qwen2p5_0p5b | 0.2914 | 0.2897 | -0.6% | 基本持平 |
| qwen2p5_1p5b | 0.6978 | 0.6978 | 0.0% | 基本持平 |
| qwen2p5_3b | 1.5278 | 1.5836 | +3.7% | no-pin 更慢 |

## 关键 breakdown

| model | pin unmap/release | no-pin unmap/release | pin create/map | no-pin create/map |
|---|---:|---:|---:|---:|
| qwen2p5_0p5b | 0.0471 | 0.0472 | 0.0231 | 0.0232 |
| qwen2p5_1p5b | 0.0989 | 0.0991 | 0.0705 | 0.0707 |
| qwen2p5_3b | 0.1643 | 0.1644 | 0.1299 | 0.1306 |

## 结论

`sleep_l2` 不走 CPU backup 路径，因此 pin/no-pin 开关基本不影响 sleep breakdown。L2 的主要成本是 unmap/release 与 wake 阶段 reload/create-map。
