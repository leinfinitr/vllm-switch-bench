# Baseline3 report

Model: /home/ljl/models/hf/Qwen2.5-1.5B-Instruct

## Aggregated rows

| system | method | prompt | n | ok_runs | mean_startup_s | mean_restore_s | mean_evict_s | estimated_restore_runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| serverless_llm | delete_register | long_short | 1 | 1 | - | 13.3707 | 1.1844 | 1 |
| serverless_llm | delete_register | short_long | 1 | 1 | - | 14.1578 | 1.4496 | 1 |
| serverless_llm | delete_register | short_short | 1 | 1 | - | 14.1290 | 1.1603 | 1 |
| serverless_llm | scale_to_zero_restore | long_short | 1 | 1 | - | 12.1172 | 1.2407 | 1 |
| serverless_llm | scale_to_zero_restore | short_long | 1 | 1 | - | 12.0781 | 1.7509 | 1 |
| serverless_llm | scale_to_zero_restore | short_short | 1 | 1 | - | 12.0769 | 1.1770 | 1 |
| swapserve_llm | swapout_swapin | long_short | 3 | 3 | - | 0.7288 | 0.6711 | 0 |
| swapserve_llm | swapout_swapin | short_long | 3 | 3 | - | 0.7336 | 0.6711 | 0 |
| swapserve_llm | swapout_swapin | short_short | 3 | 3 | - | 0.7362 | 0.6692 | 0 |
| vllm | cold_reload | long_short | 3 | 3 | 17.0233 | 16.8567 | 0.3642 | 0 |
| vllm | cold_reload | short_long | 3 | 3 | 16.6893 | 16.5227 | 0.3642 | 0 |
| vllm | cold_reload | short_short | 3 | 3 | 20.0265 | 16.6893 | 0.3644 | 0 |
| vllm | sleep_l1 | long_short | 3 | 3 | 17.0249 | 0.3459 | 0.9988 | 0 |
| vllm | sleep_l1 | short_long | 3 | 3 | 17.0239 | 0.3453 | 0.9936 | 0 |
| vllm | sleep_l1 | short_short | 3 | 3 | 20.3604 | 0.3469 | 0.9930 | 0 |
| vllm | sleep_l2 | long_short | 3 | 3 | 17.0241 | 0.7246 | 0.1191 | 0 |
| vllm | sleep_l2 | short_long | 3 | 3 | 17.0234 | 0.7023 | 0.1203 | 0 |
| vllm | sleep_l2 | short_short | 3 | 3 | 17.0235 | 0.7091 | 0.1196 | 0 |

Note: rows with restore_latency_estimated=True estimate restore latency from a warm restore request minus a second active request.

## Unsupported / blocked rows

- None

## Stage breakdown excerpts

- serverless_llm / delete_register / short_short: {"delete_idle_wait_s": 0.3117161998525262, "initial_warm_request_s": 18.402479818090796, "restore_warm_request_s": 15.348544846987352, "second_active_request_s": 1.2195430838037282}
- serverless_llm / delete_register / long_short: {"delete_idle_wait_s": 0.32045545685105026, "initial_warm_request_s": 1.2288789600133896, "restore_warm_request_s": 14.551674225833267, "second_active_request_s": 1.1810160328168422}
- serverless_llm / delete_register / short_long: {"delete_idle_wait_s": 0.5868121080566198, "initial_warm_request_s": 2.080056873848662, "restore_warm_request_s": 16.222551607061177, "second_active_request_s": 2.064707597019151}
- serverless_llm / scale_to_zero_restore / short_short: {"baseline_gpu_used_mib": 233, "baseline_idle_wait_s": 1.4233225639909506, "idle_gpu_threshold_mib": 538, "initial_warm_request_s": 13.295302756130695, "restore_warm_request_s": 13.29530867910944, "scale_to_zero_wait_s": 1.1769596000667661, "second_active_request_s": 1.2183976019732654}
- serverless_llm / scale_to_zero_restore / long_short: {"baseline_gpu_used_mib": 233, "baseline_idle_wait_s": 1.227497679879889, "idle_gpu_threshold_mib": 538, "initial_warm_request_s": 13.298074188875034, "restore_warm_request_s": 13.296626038849354, "scale_to_zero_wait_s": 1.240708993980661, "second_active_request_s": 1.1793869389221072}
- serverless_llm / scale_to_zero_restore / short_long: {"baseline_gpu_used_mib": 233, "baseline_idle_wait_s": 1.2566036169882864, "idle_gpu_threshold_mib": 538, "initial_warm_request_s": 14.157731739105657, "restore_warm_request_s": 14.156733272830024, "scale_to_zero_wait_s": 1.7509371270425618, "second_active_request_s": 2.078583688940853}
