# Qwen2.5-0.5B server-mode lifecycle experiment

## Environment

- Project: `/home/ljl/research-systems/llm-switch-bench`
- venv: `/home/ljl/research-systems/llm-switch-bench/.venv`
- vLLM: local source checkout `/home/ljl/research-systems/vllm`, version `0.1.dev16944+gb3269454b`
- PyTorch: `2.11.0+cu130`
- CUDA toolkit used for runtime/JIT: `/home/ljl/cuda-13.0`
- GPU: RTX 3080 10 GiB, sampled by `nvidia-smi`
- Model: `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`

## Result artifacts

- Final merged result directory: `/home/ljl/research-systems/llm-switch-bench/results/qwen2p5_0p5b_server_final/20260528_merged`
- Aggregate timing report: `/home/ljl/research-systems/llm-switch-bench/docs/qwen2p5_0p5b_server_results.md`
- Phase memory report: `/home/ljl/research-systems/llm-switch-bench/docs/qwen2p5_0p5b_phase_memory.md`

## Command shape

The main successful matrix was produced with:

```bash
cd /home/ljl/research-systems/llm-switch-bench
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

The initial long prompt exceeded the 1024-token context window by one token. I shortened the prompt and reran only `long_short`; `src/merge_results.py` merged the valid rerun rows with the other successful rows.

## Key results

All final rows succeeded: 3 methods × 3 prompt shapes × 3 repeats = 27 successful runs.

Lifecycle transition latency averages:

| method | evict/sleep avg | restore/wake avg |
|---|---:|---:|
| cold_reload | ~0.314 s process stop | ~15.02 s server reload |
| sleep_l1 | ~0.42 s sleep | ~0.14 s wake |
| sleep_l2 | ~0.06 s sleep | ~0.28-0.29 s staged wake/reload/kv wake |

Memory phase averages from `nvidia-smi`:

| method | ready GPU MiB | after evict/sleep GPU MiB | after restore GPU MiB |
|---|---:|---:|---:|
| cold_reload | ~9303 | ~4408 | ~9303 |
| sleep_l1 | ~9355 | ~5315 | ~8907 |
| sleep_l2 | ~9355 | ~5311 | ~9163 |

Interpretation:

- On this small 0.5B model, vLLM Sleep Mode wake is roughly two orders of magnitude faster than cold reload: ~0.14-0.29 s vs ~15 s.
- Both Sleep L1 and L2 release a large fraction of HBM after sleep, but not down to the process-stopped baseline.
- L1 shows larger CPU RAM growth after sleep because weights are backed up to CPU RAM; L2 keeps CPU RAM much closer to pre-sleep levels.
- First inference after wake is not slower here; TTFT after wake is often lower because the server/model has already completed warmup/JIT/profile work. Do not overgeneralize this to larger models without rerunning.

## Caveats

- The GPU baseline in these reports was already ~4408 MiB due to background usage/JIT/runtime state on the shared host.
- `nvidia-smi` samples total GPU memory on the visible GPU, not only this vLLM process.
- CPU memory columns include system-wide used memory plus API-server process RSS/USS; vLLM worker subprocess memory is partly reflected in system memory but not perfectly in `proc_rss_mib`.
- These results are only the vLLM Sleep/cold-reload baseline. SwapServeLLM and ServerlessLLM should be measured in separate, comparable harnesses next.
