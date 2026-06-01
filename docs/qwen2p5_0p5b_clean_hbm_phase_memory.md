# Phase memory summary

Result directory: `results/qwen2p5_0p5b_clean_hbm_main/20260601_185457`

| method | phase | gpu used avg MiB | cpu used avg MiB | proc RSS avg MiB | proc USS avg MiB |
|---|---|---:|---:|---:|---:|
| cold_reload | run_start | 6.0000 | 7926.2222 |  |  |
| cold_reload | api_ready | 4904.0000 | 10427.2222 | 1281.8889 | 969.6667 |
| cold_reload | infer_before_end | 4904.0000 | 10533.3333 | 1282.3333 | 970.1111 |
| cold_reload | evict_end | 6.0000 | 8020.8889 |  |  |
| cold_reload | restore_end | 4904.0000 | 10380.2222 | 1282.3333 | 969.6667 |
| cold_reload | infer_after_end | 4904.0000 | 10498.3333 | 1282.8889 | 970.2222 |
| cold_reload | run_end | 6.0000 | 7944.2222 |  |  |
| sleep_l1 | run_start | 6.0000 | 7941.8889 |  |  |
| sleep_l1 | api_ready | 4956.0000 | 10478.3333 | 1282.0000 | 969.6667 |
| sleep_l1 | infer_before_end | 4956.0000 | 10580.0000 | 1282.6667 | 970.1111 |
| sleep_l1 | evict_end | 914.0000 | 12376.5556 | 1282.6667 | 970.1111 |
| sleep_l1 | restore_end | 4506.0000 | 12359.6667 | 1282.6667 | 970.1111 |
| sleep_l1 | infer_after_end | 4508.0000 | 12359.0000 | 1282.8889 | 970.1111 |
| sleep_l1 | run_end | 6.0000 | 7968.4444 |  |  |
| sleep_l2 | run_start | 6.0000 | 7970.2222 |  |  |
| sleep_l2 | api_ready | 4956.0000 | 10404.0000 | 1283.3333 | 970.7778 |
| sleep_l2 | infer_before_end | 4956.0000 | 10532.5556 | 1283.7778 | 971.3333 |
| sleep_l2 | evict_end | 912.0000 | 10533.4444 | 1283.7778 | 971.3333 |
| sleep_l2 | restore_end | 4764.0000 | 10691.7778 | 1283.7778 | 971.3333 |
| sleep_l2 | infer_after_end | 4766.0000 | 10702.2222 | 1283.7778 | 971.5556 |
| sleep_l2 | run_end | 6.0000 | 7978.0000 |  |  |
