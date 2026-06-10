# Baseline3 对比报告：qwen2p5_0p5b

- 模型：`/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`
- 原始结果：`results/baselines/baseline3/qwen2p5_0p5b/20260605_145529`
- 指标单位：时间为秒，显存为 MiB。

## 聚合结果

| system | method | prompt | n | ok | startup | evict | restore | restore 估算 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `serverless_llm` | `delete_register` | `long_short` | 1 | 1 | - | 1.0046 | 13.1216 | 1 |
| `serverless_llm` | `delete_register` | `short_long` | 1 | 1 | - | 1.0074 | 13.1196 | 1 |
| `serverless_llm` | `delete_register` | `short_short` | 1 | 1 | - | 1.0089 | 12.3670 | 1 |
| `serverless_llm` | `scale_to_zero_restore` | `long_short` | 1 | 1 | - | 1.3992 | 12.0646 | 1 |
| `serverless_llm` | `scale_to_zero_restore` | `short_long` | 1 | 1 | - | 1.0964 | 12.0555 | 1 |
| `serverless_llm` | `scale_to_zero_restore` | `short_short` | 1 | 1 | - | 1.3529 | 12.0660 | 1 |
| `swapserve_llm` | `swapout_swapin` | `long_short` | 3 | 3 | - | 0.4515 | 0.4672 | 0 |
| `swapserve_llm` | `swapout_swapin` | `short_long` | 3 | 3 | - | 0.4452 | 0.4595 | 0 |
| `swapserve_llm` | `swapout_swapin` | `short_short` | 3 | 3 | - | 0.4444 | 0.4578 | 0 |
| `vllm` | `cold_reload` | `long_short` | 3 | 3 | 15.3538 | 0.3309 | 15.3539 | 0 |
| `vllm` | `cold_reload` | `short_long` | 3 | 3 | 15.1884 | 0.3642 | 15.5205 | 0 |
| `vllm` | `cold_reload` | `short_short` | 3 | 3 | 15.3544 | 0.3644 | 15.5204 | 0 |
| `vllm` | `sleep_l1` | `long_short` | 3 | 3 | 15.5227 | 0.4420 | 0.1097 | 0 |
| `vllm` | `sleep_l1` | `short_long` | 3 | 3 | 15.5233 | 0.4365 | 0.1127 | 0 |
| `vllm` | `sleep_l1` | `short_short` | 3 | 3 | 15.5224 | 0.4429 | 0.1100 | 0 |
| `vllm` | `sleep_l2` | `long_short` | 3 | 3 | 15.3549 | 0.0614 | 0.2496 | 0 |
| `vllm` | `sleep_l2` | `short_long` | 3 | 3 | 15.3550 | 0.0596 | 0.2431 | 0 |
| `vllm` | `sleep_l2` | `short_short` | 3 | 3 | 15.5217 | 0.0637 | 0.2525 | 0 |

## 主要观察

以 `short_short` 为代表，总切换时间从低到高：

- vLLM sleep_l2：0.3162s
- vLLM sleep_l1：0.5529s
- SwapServeLLM swapout/swapin：0.9022s
- ServerlessLLM delete/register：13.3759s
- ServerlessLLM scale-to-zero：13.4189s
- vLLM cold reload：15.8848s

## 失败或阻塞行

无。

## 解释

Baseline3 用于比较 vLLM 内部 sleep mode 与外部系统级切换方案。ServerlessLLM 的 restore 部分是由恢复请求和活跃请求差值估算，SwapServeLLM 的 TTFT/TPOT 语义与 vLLM 不完全一致，因此报告重点比较 evict/restore 级别的切换开销。
