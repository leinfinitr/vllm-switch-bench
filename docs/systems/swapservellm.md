# SwapServeLLM

**Status:** current external lifecycle baseline; final v0.1 GPU rows are published.

The canonical project name is **SwapServeLLM**. Historical machine-readable slugs and directories use `swapserve_llm`; those identifiers remain unchanged for provenance.

SwapServeLLM manages vLLM backends through a router, container runtime, and NVIDIA CUDA checkpoint/restore. It is not vLLM Sleep Mode.

## Requirements

- a frozen SwapServeLLM source commit and retained benchmark patch;
- a rootless container runtime and NVIDIA CDI/device access;
- a digest-pinned vLLM image;
- a frozen `cuda-checkpoint` commit/binary checksum;
- a local config derived from a sanitized example, with explicit model mounts and context length.

## Lifecycle boundary

- sleep ends after synchronous swap-out returns **and** the GPU process disappears/model GPU memory reaches the calibrated idle threshold and the container is paused;
- wake ends after resume, CUDA restore, vLLM load/readiness, and successful synchronous swap-in; post-wake inference must match the reference output.

Run the adapter only after starting the external router:

```bash
uv run python -m llm_switch_bench.adapters.swapservellm \
  --repo /path/to/SwapServeLLM \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model \
  --api-key dummy \
  --log-dir /path/to/swapserve/logs \
  --prompts short_short \
  --repeats 5 \
  --out-dir results/tmp/swapservellm
```

A 2xx control response alone is insufficient. Retain PID/GPU/container post-conditions and runtime-bound source/image metadata. If max model length or GPU utilization differs from another system, label the row as an operational comparison rather than a same-resource mechanism comparison.
