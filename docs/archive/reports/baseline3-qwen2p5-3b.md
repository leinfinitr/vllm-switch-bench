# Baseline3 对比报告：qwen2p5_3b

- 模型：`/home/ljl/models/hf/Qwen2.5-3B-Instruct`
- 原始结果：`results/baselines/baseline3/qwen2p5_3b/20260605_144933`
- 指标单位：时间为秒，显存为 MiB。

## 聚合结果

| system | method | prompt | n | ok | startup | evict | restore | restore 估算 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `serverless_llm` | `delete_register` | `long_short` | 1 | 0 | - | - | - | 0 |
| `serverless_llm` | `delete_register` | `short_long` | 1 | 0 | - | - | - | 0 |
| `serverless_llm` | `delete_register` | `short_short` | 1 | 0 | - | - | - | 0 |
| `serverless_llm` | `scale_to_zero_restore` | `long_short` | 1 | 0 | - | - | - | 0 |
| `serverless_llm` | `scale_to_zero_restore` | `short_long` | 1 | 0 | - | - | - | 0 |
| `serverless_llm` | `scale_to_zero_restore` | `short_short` | 1 | 0 | - | - | - | 0 |
| `swapserve_llm` | `swapout_swapin` | `long_short` | 3 | 3 | - | 0.9405 | 1.1088 | 0 |
| `swapserve_llm` | `swapout_swapin` | `short_long` | 3 | 3 | - | 0.9412 | 1.1111 | 0 |
| `swapserve_llm` | `swapout_swapin` | `short_short` | 3 | 3 | - | 0.9396 | 1.1110 | 0 |
| `vllm` | `cold_reload` | `long_short` | 3 | 3 | 19.3611 | 0.3643 | 19.3594 | 0 |
| `vllm` | `cold_reload` | `short_long` | 3 | 3 | 19.3604 | 0.3308 | 19.3597 | 0 |
| `vllm` | `cold_reload` | `short_short` | 3 | 3 | 19.5277 | 0.3644 | 19.1942 | 0 |
| `vllm` | `sleep_l1` | `long_short` | 3 | 3 | 19.5276 | 2.1500 | 0.5118 | 0 |
| `vllm` | `sleep_l1` | `short_long` | 3 | 3 | 19.5273 | 2.1477 | 0.5071 | 0 |
| `vllm` | `sleep_l1` | `short_short` | 3 | 3 | 23.0320 | 2.1470 | 0.5137 | 0 |
| `vllm` | `sleep_l2` | `long_short` | 3 | 3 | 19.5277 | 0.1823 | 1.5232 | 0 |
| `vllm` | `sleep_l2` | `short_long` | 3 | 3 | 19.5272 | 0.1836 | 1.5817 | 0 |
| `vllm` | `sleep_l2` | `short_short` | 3 | 3 | 19.5276 | 0.1826 | 1.5034 | 0 |

## 主要观察

以 `short_short` 为代表，总切换时间从低到高：

- vLLM sleep_l2：1.6860s
- SwapServeLLM swapout/swapin：2.0506s
- vLLM sleep_l1：2.6607s
- vLLM cold reload：19.5586s

## 失败或阻塞行

- `serverless_llm` / `delete_register` / `short_short`：ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 in
- `serverless_llm` / `delete_register` / `long_short`：ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 in
- `serverless_llm` / `delete_register` / `short_long`：ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 in
- `serverless_llm` / `scale_to_zero_restore` / `short_short`：ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 in
- `serverless_llm` / `scale_to_zero_restore` / `long_short`：ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 in
- `serverless_llm` / `scale_to_zero_restore` / `short_long`：ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 in

## 解释

Baseline3 用于比较 vLLM 内部 sleep mode 与外部系统级切换方案。ServerlessLLM 的 restore 部分是由恢复请求和活跃请求差值估算，SwapServeLLM 的 TTFT/TPOT 语义与 vLLM 不完全一致，因此报告重点比较 evict/restore 级别的切换开销。
