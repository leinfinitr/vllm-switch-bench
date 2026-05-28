# Phase memory summary

Result directory: `results/qwen2p5_0p5b_server_final/20260528_merged`

| method | phase | gpu used avg MiB | cpu used avg MiB | proc RSS avg MiB | proc USS avg MiB |
|---|---|---:|---:|---:|---:|
| cold_reload | run_start | 4408.0000 | 14028.2222 |  |  |
| cold_reload | api_ready | 9303.0000 | 16581.0000 | 1281.2222 | 968.3333 |
| cold_reload | infer_before_end | 9303.0000 | 16699.5556 | 1281.7778 | 969.0000 |
| cold_reload | evict_end | 4408.0000 | 14137.4444 |  |  |
| cold_reload | restore_end | 9303.0000 | 16514.1111 | 1281.8889 | 968.8889 |
| cold_reload | infer_after_end | 9303.0000 | 16621.5556 | 1282.2222 | 969.8889 |
| cold_reload | run_end | 4408.0000 | 14117.7778 |  |  |
| sleep_l1 | run_start | 4408.0000 | 14219.3333 |  |  |
| sleep_l1 | api_ready | 9355.0000 | 16683.0000 | 1282.5556 | 969.8889 |
| sleep_l1 | infer_before_end | 9355.0000 | 16792.2222 | 1282.8889 | 970.4444 |
| sleep_l1 | evict_end | 5315.0000 | 18610.3333 | 1282.8889 | 970.4444 |
| sleep_l1 | restore_end | 8907.0000 | 18610.6667 | 1282.8889 | 970.4444 |
| sleep_l1 | infer_after_end | 8909.0000 | 18599.1111 | 1283.0000 | 970.5556 |
| sleep_l1 | run_end | 4408.0000 | 14204.4444 |  |  |
| sleep_l2 | run_start | 4408.0000 | 14242.1111 |  |  |
| sleep_l2 | api_ready | 9355.0000 | 16587.0000 | 1284.7778 | 972.1111 |
| sleep_l2 | infer_before_end | 9355.0000 | 16708.5556 | 1285.4444 | 972.6667 |
| sleep_l2 | evict_end | 5311.0000 | 16710.6667 | 1285.4444 | 972.6667 |
| sleep_l2 | restore_end | 9163.0000 | 16875.7778 | 1285.4444 | 972.6667 |
| sleep_l2 | infer_after_end | 9165.0000 | 16885.7778 | 1285.4444 | 972.7778 |
| sleep_l2 | run_end | 4408.0000 | 14267.5556 |  |  |
