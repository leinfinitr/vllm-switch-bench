# LLM Switch Bench

Standalone benchmark repository for model lifecycle / model switching baselines on the IPADS shared server aipc2.

The repository intentionally stays separate from the implementation repositories:

- `/home/ljl/research-systems/vllm`
- `/home/ljl/research-systems/ServerlessLLM`
- `/home/ljl/research-systems/SwapServeLLM`

This repo owns only benchmark harnesses, reproducibility notes, and curated baseline results.

## Current baseline definitions

The external Feishu wiki defines three conceptual baselines. The executable state in this repo is:

1. Baseline 1 — cold reload / on-demand vLLM serving.
   - Implemented by `src/bench_vllm_lifecycle.py --methods cold_reload`.
   - Repro doc: `docs/baselines/baseline1-vllm-cold-reload.md`.

2. Baseline 2 — separate vLLM processes with vLLM Sleep Mode.
   - Target definition: one vLLM process per model; inactive models sleep; switch by sleeping A and waking B.
   - Current harness: single-model lifecycle approximation using `sleep_l1` and `sleep_l2` in `src/bench_vllm_lifecycle.py`.
   - Repro doc: `docs/baselines/baseline2-vllm-sleep-mode.md`.

3. Baseline 3 — engine-agnostic process checkpoint / hotswap systems.
   - System comparison with ServerlessLLM and SwapServeLLM, plus imported vLLM baseline rows.
   - Implemented by `src/bench_baseline3.py`, `src/bench_serverless_llm.py`, and `src/bench_swapserve_llm.py`.
   - Repro doc: `docs/baselines/baseline3-engine-checkpoint-hotswap.md`.

## Repository layout

```text
configs/                 Machine-local configs and examples.
docs/baselines/          How to reproduce baseline1/2/3.
docs/systems/            ServerlessLLM and SwapServeLLM setup notes.
docs/reports/            Curated result reports and figures.
docs/plans/              Implementation plans kept for auditability.
results/baselines/       Curated baseline result data kept for future comparison.
src/                     Benchmark harnesses and analysis scripts.
tests/                   Unit tests for benchmark logic.
```

## Curated result data kept

The tracked result tree keeps the latest per-system runs plus the latest merged baseline3 comparison:

- `results/baselines/vllm/qwen2p5_0p5b/20260603_150331`
  - vLLM cold reload + sleep_l1 + sleep_l2 source run using the simplified schema.
- `results/baselines/serverless_llm/qwen2p5_0p5b/20260604_164857`
  - Latest ServerlessLLM full-fixed rerun with `delete_register` and `scale_to_zero_restore`.
- `results/baselines/swapserve_llm/qwen2p5_0p5b/20260603_155353`
  - Latest SwapServeLLM swapout/swapin simplified-schema run.
- `results/baselines/baseline3/qwen2p5_0p5b/20260604_164857`
  - Latest merged baseline3 comparison run used by `docs/reports/baseline3-qwen2p5-0p5b.md` and the comparison figure.

Older result directories were pruned after their useful findings were reflected in reports.

## Environment policy

Use a local uv environment under this directory. Do not modify system Python or other project environments.

```bash
cd /home/ljl/research-systems/llm-switch-bench
uv venv --python 3.12 .venv
uv pip install pytest psutil requests pandas matplotlib
```

The vLLM baselines require a vLLM build with OpenAI server Sleep Mode support. On this machine the successful runs used the local source checkout and CUDA 13 environment documented in `docs/baselines/baseline1-vllm-cold-reload.md`.

## Local model

The maintained small-model baseline uses:

`/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`

It fits on the local RTX 3080 10 GiB and is useful for validating lifecycle measurement correctness before scaling to larger models.

## Quick validation

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate
python -m pytest tests -q
```

## Reports

- Baseline3 comparison: `docs/reports/baseline3-qwen2p5-0p5b.md`
- Baseline3 comparison figure: `docs/reports/figures/baseline3-qwen2p5-0p5b-comparison.png`
- vLLM lifecycle and memory summary: `docs/reports/vllm-qwen2p5-0p5b.md`
