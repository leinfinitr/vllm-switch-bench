# Baseline3 report

Model: /home/ljl/models/hf/Qwen2.5-3B-Instruct

## Aggregated rows

| system | method | prompt | n | ok_runs | mean_startup_s | mean_restore_s | mean_evict_s | estimated_restore_runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| serverless_llm | delete_register | long_short | 1 | 0 | - | - | - | 0 |
| serverless_llm | delete_register | short_long | 1 | 0 | - | - | - | 0 |
| serverless_llm | delete_register | short_short | 1 | 0 | - | - | - | 0 |
| serverless_llm | scale_to_zero_restore | long_short | 1 | 0 | - | - | - | 0 |
| serverless_llm | scale_to_zero_restore | short_long | 1 | 0 | - | - | - | 0 |
| serverless_llm | scale_to_zero_restore | short_short | 1 | 0 | - | - | - | 0 |
| swapserve_llm | swapout_swapin | long_short | 3 | 3 | - | 1.1088 | 0.9405 | 0 |
| swapserve_llm | swapout_swapin | short_long | 3 | 3 | - | 1.1111 | 0.9412 | 0 |
| swapserve_llm | swapout_swapin | short_short | 3 | 3 | - | 1.1110 | 0.9396 | 0 |
| vllm | cold_reload | long_short | 3 | 3 | 19.3611 | 19.3594 | 0.3643 | 0 |
| vllm | cold_reload | short_long | 3 | 3 | 19.3604 | 19.3597 | 0.3308 | 0 |
| vllm | cold_reload | short_short | 3 | 3 | 19.5277 | 19.1942 | 0.3644 | 0 |
| vllm | sleep_l1 | long_short | 3 | 3 | 19.5276 | 0.5118 | 2.1500 | 0 |
| vllm | sleep_l1 | short_long | 3 | 3 | 19.5273 | 0.5071 | 2.1477 | 0 |
| vllm | sleep_l1 | short_short | 3 | 3 | 23.0320 | 0.5137 | 2.1470 | 0 |
| vllm | sleep_l2 | long_short | 3 | 3 | 19.5277 | 1.5232 | 0.1823 | 0 |
| vllm | sleep_l2 | short_long | 3 | 3 | 19.5272 | 1.5817 | 0.1836 | 0 |
| vllm | sleep_l2 | short_short | 3 | 3 | 19.5276 | 1.5034 | 0.1826 | 0 |

Note: rows with restore_latency_estimated=True estimate restore latency from a warm restore request minus a second active request.

## Unsupported / blocked rows

- serverless_llm / delete_register / short_short: ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 instances with GPU memory resident but no completion.
- serverless_llm / delete_register / long_short: ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 instances with GPU memory resident but no completion.
- serverless_llm / delete_register / short_long: ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 instances with GPU memory resident but no completion.
- serverless_llm / scale_to_zero_restore / short_short: ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 instances with GPU memory resident but no completion.
- serverless_llm / scale_to_zero_restore / long_short: ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 instances with GPU memory resident but no completion.
- serverless_llm / scale_to_zero_restore / short_long: ServerlessLLM qwen2p5-3b inference did not complete: adapter warmup request timed out at 300s in the first run and a rerun with --request-timeout=900 was killed after prolonged hang; logs repeatedly showed qwen2p5-3b: 1 instances, need 1 instances with GPU memory resident but no completion.

## Stage breakdown excerpts

- None
