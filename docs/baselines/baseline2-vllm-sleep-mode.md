# Baseline 2: Separate processes + vLLM Sleep Mode

Baseline 2 is the main practical vLLM baseline from the research plan.

Target definition:

1. Each model has its own vLLM server process.
2. Model A is awake.
3. Model B is resident as a sleeping vLLM process.
4. When a request for B arrives, A sleeps and B wakes.
5. The request is served by B.

## Sleep Mode levels

Level 1:

- Weights are offloaded to CPU RAM.
- KV cache is discarded.
- Wake moves weights back to GPU.
- Best when CPU RAM is sufficient and the same model sleeps/wakes frequently.

Level 2:

- Weights and KV are discarded from GPU.
- Only small buffers remain.
- Wake requires `reload_weights` and prefix-cache reset.
- Best when switching to different models or when CPU RAM is tight.

## Current implementation status

The current repository implements a single-model lifecycle approximation, not the full separate-process multi-model scheduler.

Executable harness:

`src/bench_vllm_lifecycle.py --methods sleep_l1 sleep_l2`

The harness starts one vLLM server, sends an inference request, calls vLLM sleep/wake endpoints, then sends another inference request. It measures the core vLLM Sleep Mode transition cost, but it does not yet run two simultaneous vLLM processes A/B and does not measure cross-model scheduler overhead.

This distinction is important for interpretation:

- Use current `sleep_l1` / `sleep_l2` results as the vLLM internal memory-management baseline.
- Do not claim they are full multi-model process switching results until a separate-process A/B harness is added.

## Metrics

- `evict.latency_s`: sleep API time.
- `restore.latency_s`: wake / staged restore time.
- `infer_before.ttft_s` / `infer_after.ttft_s`: first-token latency before and after sleep/wake.
- `infer_before.client_latency_s` / `infer_after.client_latency_s`: end-to-end request latency.
- `phase_memory.csv`: HBM/CPU memory phase samples.

## Reproduce current Sleep Mode approximation

```bash
cd /home/ljl/research-systems/llm-switch-bench
. .venv/bin/activate

PATH=$PWD/.venv/bin:/home/ljl/cuda-13.0/bin:$PATH \
CUDA_HOME=/home/ljl/cuda-13.0 \
python src/bench_vllm_lifecycle.py \
  --model /home/ljl/models/hf/Qwen2.5-0.5B-Instruct \
  --python .venv/bin/python \
  --workdir /home/ljl/research-systems/llm-switch-bench \
  --methods sleep_l1 sleep_l2 \
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

Reports:

- `docs/reports/vllm-qwen2p5-0p5b-clean-hbm.md`
- `docs/reports/vllm-qwen2p5-0p5b-clean-hbm-memory.md`

Observed on Qwen2.5-0.5B:

- Sleep L1 wake is about 0.109-0.110 s.
- Sleep L2 staged restore is about 0.246-0.259 s.
- Both are much faster than cold reload (~15 s), but this is for a small model and a warmed server process.

## Future work for exact Baseline 2

To match the target definition exactly, add a harness that:

1. Starts two vLLM servers on separate ports.
2. Loads model A and model B, or two aliases/models if GPU memory allows.
3. Puts B to sleep.
4. Measures A sleep + B wake + B inference under one switch operation.
5. Records cross-process scheduling and memory interference.

Until that exists, the maintained results are labeled as “single-model Sleep Mode approximation”.
