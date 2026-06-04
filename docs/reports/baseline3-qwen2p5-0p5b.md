# Baseline3 report

Model: /home/ljl/models/hf/Qwen2.5-0.5B-Instruct

## Aggregated rows

| system | method | prompt | n | ok_runs | mean_startup_s | mean_restore_s | mean_evict_s | estimated_restore_runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| serverless_llm | delete_register | long_short | 1 | 1 | 0.0000 | 0.0043 | 0.2321 | 0 |
| serverless_llm | delete_register | short_long | 1 | 1 | 0.0000 | 0.0042 | 0.2478 | 0 |
| serverless_llm | delete_register | short_short | 1 | 1 | 0.0000 | 0.0046 | 0.2323 | 0 |
| serverless_llm | scale_to_zero_restore | long_short | 1 | 1 | 1.0331 | 12.0402 | 1.0338 | 1 |
| serverless_llm | scale_to_zero_restore | short_long | 1 | 1 | 2.0533 | 12.0447 | 2.0536 | 1 |
| serverless_llm | scale_to_zero_restore | short_short | 1 | 1 | 1.0303 | 12.0271 | 2.0494 | 1 |
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

Note: rows with restore_latency_estimated=True estimate restore latency as first post-evict request latency minus second active request latency.

## Unsupported / blocked rows

- None

## Stage breakdown excerpts

- serverless_llm / scale_to_zero_restore / short_short: {"baseline_gpu_used_mib": 233, "first_post_evict_request_s": 13.202136278850958, "idle_gpu_threshold_mib": 538, "scale_to_zero_wait_s": 2.04938465799205, "second_active_request_s": 1.1750383209437132}
- serverless_llm / scale_to_zero_restore / long_short: {"baseline_gpu_used_mib": 233, "first_post_evict_request_s": 13.1857134019956, "idle_gpu_threshold_mib": 538, "scale_to_zero_wait_s": 1.0337874540127814, "second_active_request_s": 1.1455164120998234}
- serverless_llm / scale_to_zero_restore / short_long: {"baseline_gpu_used_mib": 233, "first_post_evict_request_s": 13.8444864009507, "idle_gpu_threshold_mib": 538, "scale_to_zero_wait_s": 2.053593240911141, "second_active_request_s": 1.7997578410431743}
