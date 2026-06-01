# vLLM model lifecycle benchmark notes

This directory contains the standalone benchmark harness for local model
switching experiments on the IPADS shared server.

Scope:
- vLLM cold reload: infer -> stop process -> restart -> infer.
- vLLM Sleep level 1: infer -> /sleep?level=1 -> /wake_up -> infer.
- vLLM Sleep level 2: infer -> /sleep?level=2 -> /wake_up -> infer.

Current host selection:
- GPU: NVIDIA GeForce RTX 3080, 10 GiB HBM.
- Local full Hugging Face checkpoint available at `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`.
- Benchmarks run from `/home/ljl/research-systems/llm-switch-bench` with a dedicated uv venv.

Why no drop_caches:
- This is a shared public lab server. The harness intentionally does not run
  system-wide cache flushing or driver-level changes.

Outputs per run directory:
- metadata.json: command/environment metadata.
- summary.json: nested per-run summary.
- summary.csv: flattened table for plotting.
- *.events.jsonl: timestamped state samples and lifecycle events.
- *.server.log: vLLM server logs for later breakdown parsing.

Basic command:

```bash
cd /home/ljl/research-systems/llm-switch-bench
.venv/bin/python -m pytest tests -q
PATH=$PWD/.venv/bin:/home/ljl/cuda-13.0/bin:$PATH \
CUDA_HOME=/home/ljl/cuda-13.0 \
.venv/bin/python src/bench_vllm_lifecycle.py \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --workdir /home/ljl/research-systems/llm-switch-bench \
  --methods cold_reload sleep_l1 sleep_l2 \
  --prompts short_short long_short short_long \
  --repeats 3 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --port 0
```

The maintained code path is server-mode Sleep Mode. The older offline prototype
and compatibility shim were removed after the dedicated environment proved stable.
