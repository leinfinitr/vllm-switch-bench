# vLLM sleep-mode baseline

**Status:** current runnable baseline; final v0.1 GPU rows are published.

The harness keeps lifecycle phases separate:

- **L1 sleep:** copy live runtime allocations to CPU backup, then unmap/release GPU allocations.
- **L1 wake:** restore the exact runtime bytes and complete the public wake call.
- **L2 sleep:** discard weight state and release GPU allocations.
- **L2 wake:** map weights, reload the checkpoint, then restore KV-cache mappings. The reported wake includes the complete supported transaction.

```bash
uv run python src/bench_vllm_lifecycle.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --workdir /path/to/vllm \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 5 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --port 0 \
  --out-dir results/tmp/vllm-sleep
```

Do not infer disk cold-start behavior from L2 unless page-cache and storage conditions are explicitly controlled. Verify post-wake output equality and retain allocator profiles for every claimed phase.
