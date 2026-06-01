# Qwen2.5-0.5B clean-HBM server-mode lifecycle experiment

## Why this rerun exists

The previous run was affected by unrelated GPU HBM consumers. Before this rerun,
`nvidia-smi --query-compute-apps` showed no compute processes and the GPU was at
~6 MiB used. This run should be used as the first-stage baseline.

## Environment

- Project: `/home/ljl/research-systems/llm-switch-bench`
- venv: `/home/ljl/research-systems/llm-switch-bench/.venv`
- vLLM: local source checkout `/home/ljl/research-systems/vllm`, version `0.1.dev16944+gb3269454b`
- PyTorch: `2.11.0+cu130`
- CUDA toolkit used for runtime/JIT: `/home/ljl/cuda-13.0`
- Model: `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`
- GPU precheck: no compute apps; HBM used/free/util = `6 MiB / 9870 MiB / 0%`

## Result artifacts

- Fresh result directory: `/home/ljl/research-systems/llm-switch-bench/results/qwen2p5_0p5b_clean_hbm_main/20260601_185457`
- Timing report: `/home/ljl/research-systems/llm-switch-bench/docs/qwen2p5_0p5b_clean_hbm_server_results.md`
- Phase memory report: `/home/ljl/research-systems/llm-switch-bench/docs/qwen2p5_0p5b_clean_hbm_phase_memory.md`

## Command

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
  --port 0 \
  --out-dir results/qwen2p5_0p5b_clean_hbm_main
```

## Results

All rows succeeded: 27 / 27.

Lifecycle transition latency averages:

| method | evict/sleep avg | restore/wake avg |
|---|---:|---:|
| cold_reload | ~0.314 s process stop | ~15.02-15.19 s server reload |
| sleep_l1 | ~0.425-0.433 s sleep | ~0.109-0.110 s wake |
| sleep_l2 | ~0.059-0.061 s sleep | ~0.246-0.259 s staged restore |

Phase HBM averages:

| method | ready HBM | after evict/sleep HBM | after restore HBM | post-cleanup HBM |
|---|---:|---:|---:|---:|
| cold_reload | ~4904 MiB | ~6 MiB | ~4904 MiB | ~6 MiB |
| sleep_l1 | ~4956 MiB | ~914 MiB | ~4506 MiB | ~6 MiB |
| sleep_l2 | ~4956 MiB | ~912 MiB | ~4764 MiB | ~6 MiB |

CPU RAM phase averages:

| method | ready CPU used | after evict/sleep CPU used | after restore CPU used |
|---|---:|---:|---:|
| cold_reload | ~10427 MiB | ~8021 MiB | ~10380 MiB |
| sleep_l1 | ~10478 MiB | ~12377 MiB | ~12360 MiB |
| sleep_l2 | ~10404 MiB | ~10533 MiB | ~10692 MiB |

Interpretation:

- Clean-HBM baseline confirms the prior high baseline was interference. Process-stopped baseline is now ~6 MiB rather than ~4408 MiB.
- vLLM Sleep wake is far faster than cold reload on this model: ~0.11 s for L1 and ~0.25 s for L2 versus ~15 s reload.
- Sleep L1 and L2 both reduce HBM to ~912-914 MiB, not to full process-stopped baseline.
- Sleep L1 shifts more state to CPU RAM: CPU used rises by about 1.8-1.9 GiB after sleep.
- Sleep L2 keeps CPU RAM much closer to ready-state CPU usage, but staged restore is about 2.3x slower than L1 wake.
- First post-wake request TTFT is lower than first pre-sleep request because startup/JIT/warmup work has already happened; this should be treated as a warmed-server effect, not a universal latency improvement.

## Caveats

- GPU HBM is sampled globally through `nvidia-smi`; the precheck verified no compute apps before launch, but driver/runtime accounting can still include non-process memory.
- CPU used is system-wide, while process RSS/USS is the API server process and does not perfectly include every worker's memory contribution.
- This is one small 0.5B HF checkpoint. Larger models may change the absolute and relative costs.
