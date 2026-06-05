# Baseline3 report

Model: /home/ljl/models/hf/Qwen2.5-0.5B-Instruct

## Aggregated rows

| system | method | prompt | n | ok_runs | mean_startup_s | mean_restore_s | mean_evict_s | estimated_restore_runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| serverless_llm | delete_register | long_short | 1 | 1 | - | 13.1216 | 1.0046 | 1 |
| serverless_llm | delete_register | short_long | 1 | 1 | - | 13.1196 | 1.0074 | 1 |
| serverless_llm | delete_register | short_short | 1 | 1 | - | 12.3670 | 1.0089 | 1 |
| serverless_llm | scale_to_zero_restore | long_short | 1 | 1 | - | 12.0646 | 1.3992 | 1 |
| serverless_llm | scale_to_zero_restore | short_long | 1 | 1 | - | 12.0555 | 1.0964 | 1 |
| serverless_llm | scale_to_zero_restore | short_short | 1 | 1 | - | 12.0660 | 1.3529 | 1 |
| swapserve_llm | swapout_swapin | long_short | 3 | 3 | - | 0.4672 | 0.4515 | 0 |
| swapserve_llm | swapout_swapin | short_long | 3 | 3 | - | 0.4595 | 0.4452 | 0 |
| swapserve_llm | swapout_swapin | short_short | 3 | 3 | - | 0.4578 | 0.4444 | 0 |
| vllm | cold_reload | long_short | 3 | 3 | 15.3538 | 15.3539 | 0.3309 | 0 |
| vllm | cold_reload | short_long | 3 | 3 | 15.1884 | 15.5205 | 0.3642 | 0 |
| vllm | cold_reload | short_short | 3 | 3 | 15.3544 | 15.5204 | 0.3644 | 0 |
| vllm | sleep_l1 | long_short | 3 | 3 | 15.5227 | 0.1097 | 0.4420 | 0 |
| vllm | sleep_l1 | short_long | 3 | 3 | 15.5233 | 0.1127 | 0.4365 | 0 |
| vllm | sleep_l1 | short_short | 3 | 3 | 15.5224 | 0.1100 | 0.4429 | 0 |
| vllm | sleep_l2 | long_short | 3 | 3 | 15.3549 | 0.2496 | 0.0614 | 0 |
| vllm | sleep_l2 | short_long | 3 | 3 | 15.3550 | 0.2431 | 0.0596 | 0 |
| vllm | sleep_l2 | short_short | 3 | 3 | 15.5217 | 0.2525 | 0.0637 | 0 |

Note: rows with restore_latency_estimated=True estimate restore latency from a warm restore request minus a second active request.

## Unsupported / blocked rows

- None

## Stage breakdown excerpts

- serverless_llm / delete_register / short_short: {"delete_idle_wait_s": 0.26707895612344146, "initial_warm_request_s": 17.31408778997138, "restore_warm_request_s": 13.535224029095843, "second_active_request_s": 1.1682018339633942}
- serverless_llm / delete_register / long_short: {"delete_idle_wait_s": 0.2641056899446994, "initial_warm_request_s": 1.1489971459377557, "restore_warm_request_s": 14.25685322494246, "second_active_request_s": 1.1352869560942054}
- serverless_llm / delete_register / short_long: {"delete_idle_wait_s": 0.26893967506475747, "initial_warm_request_s": 1.7843536611180753, "restore_warm_request_s": 14.912156224017963, "second_active_request_s": 1.7925946819595993}
- serverless_llm / scale_to_zero_restore / short_short: {"baseline_gpu_used_mib": 233, "baseline_idle_wait_s": 0.9033492559101433, "idle_gpu_threshold_mib": 538, "initial_warm_request_s": 12.227715854067355, "restore_warm_request_s": 13.235005653928965, "scale_to_zero_wait_s": 1.3529279709327966, "second_active_request_s": 1.1690327040851116}
- serverless_llm / scale_to_zero_restore / long_short: {"baseline_gpu_used_mib": 233, "baseline_idle_wait_s": 1.3750223380047828, "idle_gpu_threshold_mib": 538, "initial_warm_request_s": 13.198495323071256, "restore_warm_request_s": 13.20318580698222, "scale_to_zero_wait_s": 1.399152937112376, "second_active_request_s": 1.1385548650287092}
- serverless_llm / scale_to_zero_restore / short_long: {"baseline_gpu_used_mib": 233, "baseline_idle_wait_s": 1.4470457299612463, "idle_gpu_threshold_mib": 538, "initial_warm_request_s": 13.848316129064187, "restore_warm_request_s": 13.838110622018576, "scale_to_zero_wait_s": 1.0963858501054347, "second_active_request_s": 1.782602114835754}
