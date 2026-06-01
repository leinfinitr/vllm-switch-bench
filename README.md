# LLM Switch Bench

Standalone benchmark project for the first-stage model lifecycle measurements in
LLM serving / model switching research.

This project intentionally lives at:

`/home/ljl/research-systems/llm-switch-bench`

It is separate from `prism-research`, `vllm`, `SwapServeLLM`, and
`ServerlessLLM` so benchmark code, raw results, and environment notes do not
pollute other projects.

## Scope

Current harness measures:

- vLLM cold reload: infer -> stop process -> restart -> infer.
- vLLM Sleep level 1: infer -> sleep(level=1) -> wake_up -> infer.
- vLLM Sleep level 2: infer -> sleep(level=2) -> wake_up(tags=[weights]) -> collective_rpc(reload_weights) -> wake_up(tags=[kv_cache]) -> infer.

Baseline3 development adds shared `benchlib/` utilities and external-system config via:

- `configs/baseline3.local.example.yaml`
- `src/benchlib/config.py`

The baseline3 config records external ServerlessLLM / SwapServeLLM repo paths and commit metadata without vendoring them into this repository.

The maintained benchmark path is the server lifecycle harness in `src/bench_vllm_lifecycle.py`.
The earlier offline Python-API prototype and legacy compatibility shims were removed once
server-mode Sleep Mode was running reliably in the dedicated environment.

## Environment policy

Use a local uv environment under this directory. Do not modify system Python or
other project environments.

```bash
cd /home/ljl/research-systems/llm-switch-bench
uv venv --python 3.12 .venv
uv pip install pytest psutil requests pandas matplotlib
# Install a vLLM build that supports Sleep Mode. Prefer a recent release or the
# local /home/ljl/research-systems/vllm source tree if prebuilt wheels are not
# compatible with this host.
```

## Local model for first stage

The first-stage model is:

`/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`

It fits on the local RTX 3080 10 GiB and is sufficient to validate lifecycle
measurement correctness before scaling to larger models.

## Outputs

Each run directory contains:

- `metadata.json`: command/environment metadata.
- `summary.json`: nested per-run summary.
- `summary.csv`: flattened table for plotting.
- `*.events.jsonl`: timestamped state samples and lifecycle events.
- `*.server.log`: vLLM server logs when using the server backend.

## Known migrated result

Earlier, code was mistakenly prototyped under `prism-research`. Those results are
not the source of truth for this standalone project, but the useful finding was:
`vLLM 0.6.3.post1` does not support OpenAI-server Sleep Mode
`--enable-sleep-mode`, so a newer vLLM environment is required.
