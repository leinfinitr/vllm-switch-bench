# vLLM cold reload baseline

**Status:** current runnable baseline.

Cold reload terminates the serving process, starts a new process from the same frozen checkpoint, waits for readiness, and completes a correctness inference. It is a process-restart reference, not a sleep-mode phase.

From the repository root:

```bash
scripts/lifecycle-latency.sh \
  --model /path/to/model \
  --workdir /path/to/vllm \
  --python /path/to/vllm/.venv/bin/python \
  --methods cold_reload \
  --prompts short_short \
  --repeats 5 \
  --ready-timeout-s 360 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 1024 \
  --out-dir results/tmp/vllm-cold-reload
```

Record the exact vLLM source/import identity, model revision/checksum, dtype, context length, page-cache policy, GPU budget, and process post-condition. A "cold" process restart does not imply cold storage or a globally dropped page cache.
