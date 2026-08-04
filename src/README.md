# Source entry points

This is a non-package uv project. Run entry points from the repository root with `uv run python ...`; `src/benchlib/` is added by script execution and is not installed as a public library.

## Benchmark runners

- `bench_vllm_lifecycle.py`: cold reload and vLLM L1/L2 lifecycle phases.
- `bench_vllm_repeated_sleep_l1.py`: same-process repeated sleep/wake, backup reuse, pressure release, RSS, and `MemAvailable`.
- `bench_vllm_pin_compare.py`: model-explicit pinned/pageable profiling matrix.
- `bench_request_driven_switch.py`: frozen open-loop OpenAI-compatible request traces with strict SSE success.
- `bench_baseline3.py`: historical cross-system aggregation compatibility harness.
- `bench_serverless_llm.py`: legacy ServerlessLLM adapter; not a publishable current scale-to-zero implementation.
- `bench_swapservellm.py`: SwapServeLLM adapter (canonical project spelling).

## Shared library

`benchlib/` provides configuration/provenance collection, HTTP helpers, resource sampling, result schemas, strict request-trace semantics, and exact-disk evidence parsing. It is internal to this repository.

## Analysis

`tool/` scripts read existing evidence and do not launch benchmark services. Artifact-specific v0.1 building is in `scripts/build_release_artifact.py`; it is kept under `scripts/` because its input contract is bound to the release bundle.

## Microbenchmarks

`microbench/` contains CUDA/vLLM allocator experiments for PCIe copy, CuMemAllocator synthetic allocations, and safetensors-like allocation sizes. They require a compatible CUDA/vLLM environment and are not part of CPU CI.

Examples:

```bash
uv run python src/bench_vllm_lifecycle.py \
  --model /path/to/model \
  --python .venv/bin/python \
  --workdir /path/to/vllm \
  --methods sleep_l1 sleep_l2 \
  --prompts short_short \
  --repeats 5 \
  --out-dir results/tmp/vllm-lifecycle

uv run python src/microbench/microbench_cumem_safetensor_sizes.py \
  --out-dir results/tmp/cumem-sizes/model \
  --repeats 1 \
  /path/to/model
```

Before a GPU run, record the actual interpreter, imported vLLM path/version, Git identity/dirty state, model revision/checksum, CUDA/driver/GPU, and all behavior-affecting parameters. Do not infer those identities from the development lockfile.
