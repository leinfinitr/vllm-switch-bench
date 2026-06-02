# Baseline 1: Cold reload

Baseline 1 measures the most direct on-demand serving strategy:

1. Stop / unload model A.
2. Start a vLLM server for model B.
3. Load tokenizer / config / weights.
4. Initialize CUDA context and allocator.
5. Run CUDA graph capture / warmup.
6. Serve the first request.

This is memory efficient when no model is resident, but it has high readiness time and strongly affects first-request latency.

## Current implementation

Executable harness:

`src/bench_vllm_lifecycle.py --methods cold_reload`

This harness starts a local vLLM OpenAI API server, sends an inference request, terminates the process, starts a fresh vLLM server, waits for readiness, and sends a second inference request.

## Metrics

- `startup_to_health_s`: time for vLLM server startup to `/v1/models` readiness.
- `evict.latency_s`: process stop / unload time.
- `restore.latency_s`: full reload time until new server readiness.
- `infer_before.ttft_s` / `infer_after.ttft_s`: streaming TTFT before and after reload.
- `infer_before.client_latency_s` / `infer_after.client_latency_s`: full client latency.

## Environment used for the maintained baseline

- Project: `/home/ljl/research-systems/llm-switch-bench`
- venv: `/home/ljl/research-systems/llm-switch-bench/.venv`
- vLLM: local source checkout `/home/ljl/research-systems/vllm`, version recorded in run metadata
- PyTorch: `2.11.0+cu130`
- CUDA toolkit: `/home/ljl/cuda-13.0`
- Model: `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`
- Precheck for curated run: no compute apps; HBM used/free/util = `6 MiB / 9870 MiB / 0%`

## Reproduce baseline1

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate

PATH=$PWD/.venv/bin:/home/ljl/cuda-13.0/bin:$PATH \
CUDA_HOME=/home/ljl/cuda-13.0 \
python src/bench_vllm_lifecycle.py \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --workdir /home/ljl/research-systems/llm-switch-bench \
  --methods cold_reload \
  --prompts short_short long_short short_long \
  --repeats 3 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --port 0 \
  --out-dir results/baselines/vllm/qwen2p5_0p5b_clean_hbm
```

## Curated result

Current source result directory:

`results/baselines/vllm/qwen2p5_0p5b_clean_hbm/20260601_185457`

Report:

`docs/reports/vllm-qwen2p5-0p5b-clean-hbm.md`

All maintained rows succeeded: 3 prompt shapes × 3 repeats for `cold_reload`.

## Caveats

- This baseline is intentionally one model at a time; it does not implement a multi-model scheduler.
- `nvidia-smi` HBM is global GPU memory. The curated run prechecked no compute apps before launch, but driver/runtime accounting can still include non-process memory.
- Larger models may shift both readiness and TTFT behavior.
