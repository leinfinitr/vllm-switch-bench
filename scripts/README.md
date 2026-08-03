# `scripts/` 目录说明

`scripts/` 保存 shell 编排器以及为某个评测 artifact 提供证据的实验 driver。
可复用 benchmark 实现在 `src/`；纯后处理工具在 `src/tool/`。

## 可复用编排

- `run_baseline3.sh`：历史 Baseline3 配置入口。
- `run_profiling.sh`：vLLM pin/no-pin profiling 矩阵。
- `run_request_switch.sh`、`run_request_switch_matrix.py`：request-driven trace。
- `run_cross_system_matrix.py`：统一 OpenAI endpoint 的跨系统 trace 矩阵。
- `run_exact_disk_profile.py`：模型无关的 exact runtime disk-backup 命令
  wrapper；采集 engine JSONL、worker process-tree RSS、host `MemAvailable`、
  backup-root footprint 和 before/after output evidence，再生成严格断言。

### Exact disk profiling contract

`run_exact_disk_profile.py` 不导入尚在开发中的 vLLM API。它在 `--` 后执行任意
命令，并通过环境变量定义稳定的 benchmark/producer 边界：

- `VLLM_EXACT_DISK_BACKUP_ENABLED=1`：启用 exact disk tier；
- `VLLM_EXACT_DISK_BACKUP_DIR` / `VLLM_CPU_BACKUP_DISK_DIR`：exact backup
  根目录；默认 `/home/ljl/research-systems/vllm-model-switch-controller/tmp`；
- `VLLM_SLEEP_PROFILE_PATH`：producer 必须写入的 append-only JSONL；
- `LLM_SWITCH_BENCH_OUTPUT_OBSERVATION`：producer 必须写入的确定性推理
  before/after JSON；
- `LLM_SWITCH_BENCH_MODEL_NAME` / `LLM_SWITCH_BENCH_MODEL_PATH`：显式模型
  identity，脚本本身不硬编码任何模型。

最小调用形式：

```bash
.venv/bin/python scripts/run_exact_disk_profile.py \
  --model MODEL_NAME=/absolute/model/path \
  --out-dir results/tmp/exact-disk/MODEL_NAME/RUN_ID \
  -- \
  /absolute/python /absolute/path/to/model_agnostic_driver.py
```

backup root 默认必须为空，使 footprint growth 可归因到当前 run。若 producer 的
隔离子目录策略必须复用非空根目录，可显式传 `--allow-nonempty-backup-root`；此时
curated summary 仍只断言相对 pre-launch baseline 的 positive growth，操作者必须在
provenance 中说明隔离方式。

producer JSONL 中 exact-disk event 的稳定字段为：

- spill：`disk_spill_bytes` 与 `disk_spill_s` 必须同时出现；
- restore：`disk_read_bytes`、`disk_read_s`、`source_medium` 必须出现；
- restore 可加 `fallback`（boolean，缺省 false）和 `fallback_reason`。

successful command 还必须在 `LLM_SWITCH_BENCH_OUTPUT_OBSERVATION` 写入：

```json
{
  "schema_version": 1,
  "before": {"token_ids": [1, 2], "text": "..."},
  "after": {"token_ids": [1, 2], "text": "..."}
}
```

runner 默认要求 positive spill/read、唯一 `source_medium=disk`、zero fallback、
worker RSS、`MemAvailable`、positive disk-footprint growth 和 output equality。任何
门禁失败返回非零；command 本身失败时保留 blocker evidence，但不创建 curated
summary。

每个 run 严格分层：

```text
RUN_ID/
  raw/       # local_raw: producer records/logs/resources + checksum manifest
  curated/   # local_curated: derived summary.json + assertions.json only
```

`raw/` 不等同于仓库可引用 artifact，`curated/` 也不自动等同于论文 evidence。
两者默认都放在 ignored `results/tmp/`；只有经过人工选择、source/environment
identity 检查、fresh-checkout checksum 验证后才可复制到 tracked `results/`。
不要把失败 run 的数值写入 curated baseline。

`bench_exact_disk_allocator.py` 的 pipeline timing 语义：

- `disk_read_s` 是单 reader 的 `preadv` 时间累计；
- `disk_hash_worker_s` 是所有 SHA-256 worker 的 worker-seconds 总和，不是
  hash stage wall time，也不能与 `disk_pipeline_wall_s` 相加；
- `disk_hash_s` 是兼容旧消费者的同值别名；
- `disk_copy_h2d_s` 是 CUDA event 测得的 device copy duration；
- `disk_copy_wait_s` 是 host 等待 completion event 的墙钟累计；
- `disk_pipeline_depth` 是 pinned staging slot 数，不是同时在途的 H2D 数；
- `disk_pipeline_wall_s` 不含 manifest 预检和 VMM create/map。

聚合 timing 只能证明各阶段在同一 bounded pipeline 中调度；若要正式证明三阶段
在真实 GPU run 中的同时 overlap，需要保留 per-chunk monotonic interval trace。

已存在的完整 raw run 可以独立重建：

```bash
.venv/bin/python src/tool/collect_exact_disk_profile.py \
  --raw-dir results/tmp/exact-disk/MODEL_NAME/RUN_ID/raw \
  --curated-dir results/tmp/exact-disk/MODEL_NAME/RUN_ID/curated
```

## Curated artifact driver

- `measure_llama_swap_lifecycle.py`、`measure_llama_swap_switch.py`
- `measure_serverless_switch.py`
- `measure_swapserve_lifecycle.py`
- `measure_vllm_sleep_phases.py`
- `analyze_model_switch_eval.py`、`plot_model_switch_eval.py`
- `plot_osdi_switch_results.py`

这些文件与 `results/model_switch_eval/`、`results/cross_system/` 或
`results/osdi_20260723/` 的 provenance 绑定。即使当前有更通用的 `src/`
adapter，也不应在不重建 artifact 和 checksum 的情况下改名或移动。新实验应
优先扩展 `src/`/`src/tool/`，避免继续增加无测试的一次性脚本。
