# sleep_l1 pin/no-pin 对比

实验：`sleep_l1`、`short_short`、每个模型和 pin 模式 3 次重复。3B 使用 `gpu_memory_utilization=0.85`，0.5B/1.5B 使用 `0.55`。

## 平均总切换时间

| model | pin switch | no-pin switch | no-pin 变化 | 结论 |
|---|---:|---:|---:|---|
| qwen2p5_0p5b | 0.5666 | 0.4848 | -14.4% | no-pin 更快 |
| qwen2p5_1p5b | 1.2421 | 1.3561 | +9.2% | no-pin 更慢 |
| qwen2p5_3b | 2.6142 | 2.5484 | -2.5% | no-pin 略快 |

## 关键 breakdown

| model | pin CPU alloc | no-pin CPU alloc | pin D2H | no-pin D2H | pin H2D | no-pin H2D |
|---|---:|---:|---:|---:|---:|---:|
| qwen2p5_0p5b | 0.3697 | 0.0003 | 0.0559 | 0.2896 | 0.0592 | 0.0830 |
| qwen2p5_1p5b | 0.7592 | 0.0006 | 0.1698 | 0.8888 | 0.1782 | 0.2525 |
| qwen2p5_3b | 1.7357 | 0.0009 | 0.3245 | 1.7155 | 0.3611 | 0.4946 |

## 结论

no-pin 可以消除 CPU backup allocation，但会显著拖慢 D2H/H2D copy。它不是稳定优化；更合理的方向是复用 pinned CPU backup buffer。
