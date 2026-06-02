# Baseline3 report

Model: /home/ljl/models/hf/Qwen2.5-0.5B-Instruct

## Aggregated rows

| system | method | prompt | n | ok_runs | mean_restore_s | mean_evict_s |
|---|---|---:|---:|---:|---:|---:|
| serverless_llm | delete_register | short_short | 1 | 0 | - | - |
| serverless_llm | scale_to_zero_restore | long_short | 3 | 3 | 12.8705 | 1.3711 |
| serverless_llm | scale_to_zero_restore | short_long | 3 | 3 | 13.5302 | 1.0336 |
| serverless_llm | scale_to_zero_restore | short_short | 3 | 3 | 12.8838 | 1.7117 |
| swapserve_llm | swapout_swapin | long_short | 3 | 3 | 0.4301 | 0.4410 |
| swapserve_llm | swapout_swapin | short_long | 3 | 3 | 0.4351 | 0.4479 |
| swapserve_llm | swapout_swapin | short_short | 3 | 3 | 0.4263 | 0.4440 |
| vllm | cold_reload | long_short | 3 | 3 | 15.0207 | 0.3143 |
| vllm | cold_reload | short_long | 3 | 3 | 15.0208 | 0.3141 |
| vllm | cold_reload | short_short | 3 | 3 | 15.1868 | 0.3144 |
| vllm | sleep_l1 | long_short | 3 | 3 | 0.1092 | 0.4330 |
| vllm | sleep_l1 | short_long | 3 | 3 | 0.1103 | 0.4250 |
| vllm | sleep_l1 | short_short | 3 | 3 | 0.1090 | 0.4277 |
| vllm | sleep_l2 | long_short | 3 | 3 | 0.2588 | 0.0605 |
| vllm | sleep_l2 | short_long | 3 | 3 | 0.2455 | 0.0590 |
| vllm | sleep_l2 | short_short | 3 | 3 | 0.2589 | 0.0602 |

## Unsupported / blocked rows

- serverless_llm / delete_register / short_short: controller delete only removes metadata; router.shutdown remains commented in ServerlessLLM/sllm/controller.py, so delete_register does not reliably free GPU for the next register

## Stage breakdown excerpts

- serverless_llm / scale_to_zero_restore / short_short: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 13.203646728070453, "scale_to_zero_wait_s": 2.050550946034491}
- serverless_llm / scale_to_zero_restore / short_short: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 12.231763103976846, "scale_to_zero_wait_s": 1.0319810688961297}
- serverless_llm / scale_to_zero_restore / short_short: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 13.215935688931495, "scale_to_zero_wait_s": 2.0525030461139977}
- serverless_llm / scale_to_zero_restore / long_short: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 12.208303287858143, "scale_to_zero_wait_s": 1.0348667290527374}
- serverless_llm / scale_to_zero_restore / long_short: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 13.180069061927497, "scale_to_zero_wait_s": 2.0476431611459702}
- serverless_llm / scale_to_zero_restore / long_short: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 13.223113998072222, "scale_to_zero_wait_s": 1.0309035759419203}
- serverless_llm / scale_to_zero_restore / short_long: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 12.849634455982596, "scale_to_zero_wait_s": 1.0322857710998505}
- serverless_llm / scale_to_zero_restore / short_long: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 13.870893829967827, "scale_to_zero_wait_s": 1.0332296560518444}
- serverless_llm / scale_to_zero_restore / short_long: {"baseline_gpu_used_mib": 238, "idle_gpu_threshold_mib": 538, "post_idle_request_s": 13.869970072060823, "scale_to_zero_wait_s": 1.0351697369478643}
