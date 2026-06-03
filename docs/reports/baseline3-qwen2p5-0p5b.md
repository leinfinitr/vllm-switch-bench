# Baseline3 report

Model: /home/ljl/models/hf/Qwen2.5-0.5B-Instruct

## Aggregated rows

| system | method | prompt | n | ok_runs | mean_startup_s | mean_restore_s | mean_evict_s | estimated_restore_runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| serverless_llm | delete_register | short_short | 1 | 0 | - | - | - | 0 |
| serverless_llm | scale_to_zero_restore | short_short | 1 | 0 | - | - | - | 0 |
| swapserve_llm | swapout_swapin | long_short | 3 | 3 | - | 0.4400 | 0.4456 | 0 |
| swapserve_llm | swapout_swapin | short_long | 3 | 3 | - | 0.4382 | 0.4419 | 0 |
| swapserve_llm | swapout_swapin | short_short | 3 | 3 | - | 0.4372 | 0.4498 | 0 |
| vllm | cold_reload | long_short | 3 | 3 | 15.3538 | 15.3539 | 0.3309 | 0 |
| vllm | cold_reload | short_long | 3 | 3 | 15.1884 | 15.5205 | 0.3642 | 0 |
| vllm | cold_reload | short_short | 3 | 3 | 15.3544 | 15.5204 | 0.3644 | 0 |
| vllm | sleep_l1 | long_short | 3 | 3 | 15.5227 | 0.1097 | 0.4420 | 0 |
| vllm | sleep_l1 | short_long | 3 | 3 | 15.5233 | 0.1127 | 0.4365 | 0 |
| vllm | sleep_l1 | short_short | 3 | 3 | 15.5224 | 0.1100 | 0.4429 | 0 |
| vllm | sleep_l2 | long_short | 3 | 3 | 15.3549 | 0.2496 | 0.0614 | 0 |
| vllm | sleep_l2 | short_long | 3 | 3 | 15.3550 | 0.2431 | 0.0596 | 0 |
| vllm | sleep_l2 | short_short | 3 | 3 | 15.5217 | 0.2525 | 0.0637 | 0 |

## Unsupported / blocked rows

- serverless_llm / delete_register / short_short: unsupported: current ServerlessLLM delete removes metadata but does not reliably stop runtime/router
- serverless_llm / scale_to_zero_restore / short_short: blocked in current rerun: ServerlessLLM Docker runtime health endpoint was up, but inference before measurement timed out after 300s; old 20260602 rows were not reused because they used first post-idle request as latency_after_s, inconsistent with the simplified schema requiring a second active request

## Stage breakdown excerpts

- None
