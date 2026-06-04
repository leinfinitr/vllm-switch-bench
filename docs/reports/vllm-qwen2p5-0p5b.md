# Qwen2.5-0.5B vLLM lifecycle benchmark results

Result directory: `results/baselines/vllm/qwen2p5_0p5b/20260603_150331`

## Summary by method / prompt / success

| method | prompt | ok | n | startup avg s | evict avg s | restore avg s | TTFT before avg s | TTFT after avg s | latency before avg s | latency after avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold_reload | long_short | True | 3 | 15.3538 | 0.3309 | 15.3539 | 0.0840 | 0.0837 | 0.1348 | 0.1356 |
| cold_reload | short_long | True | 3 | 15.1884 | 0.3642 | 15.5205 | 0.0746 | 0.0667 | 0.4343 | 0.4290 |
| cold_reload | short_short | True | 3 | 15.3544 | 0.3644 | 15.5204 | 0.0677 | 0.0673 | 0.1379 | 0.1367 |
| sleep_l1 | long_short | True | 3 | 15.5227 | 0.4420 | 0.1097 | 0.0825 | 0.0286 | 0.1341 | 0.0812 |
| sleep_l1 | short_long | True | 3 | 15.5233 | 0.4365 | 0.1127 | 0.0678 | 0.0092 | 0.4301 | 0.3698 |
| sleep_l1 | short_short | True | 3 | 15.5224 | 0.4429 | 0.1100 | 0.0673 | 0.0095 | 0.1390 | 0.0793 |
| sleep_l2 | long_short | True | 3 | 15.3549 | 0.0614 | 0.2496 | 0.0825 | 0.0284 | 0.1332 | 0.0820 |
| sleep_l2 | short_long | True | 3 | 15.3550 | 0.0596 | 0.2431 | 0.0675 | 0.0094 | 0.4300 | 0.3708 |
| sleep_l2 | short_short | True | 3 | 15.5217 | 0.0637 | 0.2525 | 0.0674 | 0.0097 | 0.1370 | 0.0817 |

## Ready / evicted memory

| method | prompt | ready gpu avg MiB | evicted gpu avg MiB | ready cpu avg MiB | evicted cpu avg MiB |
|---|---|---:|---:|---:|---:|
| cold_reload | long_short | 4899.0000 | 1.0000 | 3322.7083 | n/a |
| cold_reload | short_long | 4899.0000 | 1.0000 | 3319.9401 | n/a |
| cold_reload | short_short | 4899.0000 | 1.0000 | 3317.1628 | n/a |
| sleep_l1 | long_short | 4951.0000 | 909.0000 | 3318.9128 | 5207.2878 |
| sleep_l1 | short_long | 4951.0000 | 909.0000 | 3324.6888 | 5209.6172 |
| sleep_l1 | short_short | 4951.0000 | 909.0000 | 3318.7240 | 5204.9076 |
| sleep_l2 | long_short | 4951.0000 | 907.0000 | 3323.1263 | 3454.7917 |
| sleep_l2 | short_long | 4951.0000 | 907.0000 | 3318.8542 | 3449.7227 |
| sleep_l2 | short_short | 4951.0000 | 907.0000 | 3322.1029 | 3454.0768 |

## Failures

No failed rows.
