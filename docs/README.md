# vLLM model lifecycle benchmark notes

This directory contains the first-stage measurement harness for local model
switching experiments on the IPADS shared server.

Scope:
- vLLM cold reload: infer -> stop process -> restart -> infer.
- vLLM Sleep level 1: infer -> /sleep?level=1 -> /wake_up -> infer.
- vLLM Sleep level 2: infer -> /sleep?level=2 -> /wake_up -> infer.

Current host selection:
- GPU: NVIDIA GeForce RTX 3080, 10 GiB HBM.
- Local full Hugging Face checkpoint available: Qwen/Qwen2.5-0.5B-Instruct
  config/tokenizer only in HF cache; weights are not fully cached.
- Local model files under /home/ljl/models are mostly GGUF. vLLM can support GGUF
  in some versions, but Sleep Mode experiments are more reliable with HF-format
  models. Therefore the default plan is to start with the small Qwen HF model,
  using local cache/download if needed, then only scale up if HBM and dependency
  constraints allow it.

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
cd /home/ljl/research-systems/prism-research
uv pip install pytest psutil requests matplotlib pandas
.venv/bin/python -m pytest benchmark/model-switching/tests -q
.venv/bin/python benchmark/model-switching/bench_vllm_lifecycle.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --methods cold_reload sleep_l1 sleep_l2 \
  --prompts short_short long_short short_long \
  --repeats 3
```

If vLLM import fails because the checked-out environment has mismatched torch and
vLLM versions, create a dedicated uv environment outside system Python and record
that in this document before running measurements.
