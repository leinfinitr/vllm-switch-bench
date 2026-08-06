# Installed source package

`src/llm_switch_bench/` is the only Python implementation root and is installed by the project build.

## Modules

- `adapters/`: lifecycle/request adapters for vLLM, SwapServeLLM, llama-swap, and ServerlessLLM.
- `experiments/lifecycle_latency/`: vLLM lifecycle runner plus lifecycle analysis/plot entry points.
- `experiments/request_driven_switch/`: strict open-loop request runner, frozen-matrix orchestration, analysis, and plotting.
- `experiments/backup_reuse_reclaim/`: repeated-sleep/reclaim runner and mechanism microbench/plot modules.
- `experiments/exact_disk/`: exact-disk runner, lifecycle/allocator drivers, and evidence parser.
- `common/`: shared HTTP, schema, trace, sampling, resource, provenance, and analysis utilities.
- `plotting/style.py`: repository-wide paper figure style and deterministic PDF/PNG writer.
- `artifacts.py` / `build_all.py`: deterministic current-family summaries and figures.
- `validation/`: one semantic validator per family and the aggregate `validate_all` entry point.

Run modules from the repository root through uv, for example:

```bash
uv run python -m llm_switch_bench.build_all
uv run python -m llm_switch_bench.validation.validate_all
uv run python -m llm_switch_bench.experiments.lifecycle_latency.run --help
uv run python -m llm_switch_bench.experiments.request_driven_switch.run --help
uv run python -m llm_switch_bench.experiments.exact_disk.run --help
```

Do not recreate top-level Python scripts, legacy import aliases, `sys.path` mutation, or file-based dynamic imports. Tests import installed package modules directly.
