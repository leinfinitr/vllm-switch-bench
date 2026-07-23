# `scripts/` 目录说明

`scripts/` 保存 shell 编排器以及为某个评测 artifact 提供证据的实验 driver。
可复用 benchmark 实现在 `src/`；纯后处理工具在 `src/tool/`。

## 可复用编排

- `run_baseline3.sh`：历史 Baseline3 配置入口。
- `run_profiling.sh`：vLLM pin/no-pin profiling 矩阵。
- `run_request_switch.sh`、`run_request_switch_matrix.py`：request-driven trace。
- `run_cross_system_matrix.py`：统一 OpenAI endpoint 的跨系统 trace 矩阵。

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
