# Baseline3 对比报告：qwen2p5_1p5b

- 模型：`/home/ljl/models/hf/Qwen2.5-1.5B-Instruct`
- 原始结果：`results/baselines/baseline3/qwen2p5_1p5b/20260605_144546`
- 指标单位：时间为秒，显存为 MiB。

## 聚合结果

| system | method | prompt | n | ok | startup | evict | restore | restore 估算 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `serverless_llm` | `delete_register` | `long_short` | 1 | 1 | - | 1.1844 | 13.3707 | 1 |
| `serverless_llm` | `delete_register` | `short_long` | 1 | 1 | - | 1.4496 | 14.1578 | 1 |
| `serverless_llm` | `delete_register` | `short_short` | 1 | 1 | - | 1.1603 | 14.1290 | 1 |
| `serverless_llm` | `scale_to_zero_restore` | `long_short` | 1 | 1 | - | 1.2407 | 12.1172 | 1 |
| `serverless_llm` | `scale_to_zero_restore` | `short_long` | 1 | 1 | - | 1.7509 | 12.0781 | 1 |
| `serverless_llm` | `scale_to_zero_restore` | `short_short` | 1 | 1 | - | 1.1770 | 12.0769 | 1 |
| `swapserve_llm` | `swapout_swapin` | `long_short` | 3 | 3 | - | 0.6711 | 0.7288 | 0 |
| `swapserve_llm` | `swapout_swapin` | `short_long` | 3 | 3 | - | 0.6711 | 0.7336 | 0 |
| `swapserve_llm` | `swapout_swapin` | `short_short` | 3 | 3 | - | 0.6692 | 0.7362 | 0 |
| `vllm` | `cold_reload` | `long_short` | 3 | 3 | 17.0233 | 0.3642 | 16.8567 | 0 |
| `vllm` | `cold_reload` | `short_long` | 3 | 3 | 16.6893 | 0.3642 | 16.5227 | 0 |
| `vllm` | `cold_reload` | `short_short` | 3 | 3 | 20.0265 | 0.3644 | 16.6893 | 0 |
| `vllm` | `sleep_l1` | `long_short` | 3 | 3 | 17.0249 | 0.9988 | 0.3459 | 0 |
| `vllm` | `sleep_l1` | `short_long` | 3 | 3 | 17.0239 | 0.9936 | 0.3453 | 0 |
| `vllm` | `sleep_l1` | `short_short` | 3 | 3 | 20.3604 | 0.9930 | 0.3469 | 0 |
| `vllm` | `sleep_l2` | `long_short` | 3 | 3 | 17.0241 | 0.1191 | 0.7246 | 0 |
| `vllm` | `sleep_l2` | `short_long` | 3 | 3 | 17.0234 | 0.1203 | 0.7023 | 0 |
| `vllm` | `sleep_l2` | `short_short` | 3 | 3 | 17.0235 | 0.1196 | 0.7091 | 0 |

## 主要观察

以 `short_short` 为代表，总切换时间从低到高：

- vLLM sleep_l2：0.8288s
- vLLM sleep_l1：1.3399s
- SwapServeLLM swapout/swapin：1.4055s
- ServerlessLLM scale-to-zero：13.2539s
- ServerlessLLM delete/register：15.2893s
- vLLM cold reload：17.0537s

## 失败或阻塞行

无。

## 解释

Baseline3 用于比较 vLLM 内部 sleep mode 与外部系统级切换方案。ServerlessLLM 的 restore 部分是由恢复请求和活跃请求差值估算，SwapServeLLM 的 TTFT/TPOT 语义与 vLLM 不完全一致，因此报告重点比较 evict/restore 级别的切换开销。
