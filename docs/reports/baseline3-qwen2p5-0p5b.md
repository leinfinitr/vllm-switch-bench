# Baseline3 report

Model: /home/ljl/models/hf/Qwen2.5-0.5B-Instruct

ServerlessLLM rerun: `results/baseline3/serverless_llm_full_fixed/20260604_150528`.

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

## ServerlessLLM validation notes

- `delete_register` now completes for all three prompts after fixing ServerlessLLM router cleanup.
- `scale_to_zero_restore` now uses the second active request as `latency_after_s`, with `restore_latency_s` estimated from `first_post_evict_request_s - second_active_request_s`.
- ServerlessLLM still has no external streaming TTFT in the current API path, so `ttft_available=false` and `tpot_available=false` for these rows.
- The Docker image rebuild path was blocked by a `github.com` timeout while downloading Miniforge; the successful validation used the existing image with patched installed Python files copied into the created containers before start.

## Stage breakdown excerpts

| method | prompt | scale_to_zero_wait_s | first_post_evict_request_s | second_active_request_s | baseline_gpu_used_mib | idle_gpu_threshold_mib |
|---|---|---:|---:|---:|---:|---:|
| scale_to_zero_restore | short_short | 2.0494 | 13.2021 | 1.1750 | 233 | 538 |
| scale_to_zero_restore | long_short | 1.0338 | 13.1857 | 1.1455 | 233 | 538 |
| scale_to_zero_restore | short_long | 2.0536 | 13.8445 | 1.7998 | 233 | 538 |
