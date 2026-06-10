# sleep_l1 pinned vs non-pinned backup comparison

## Setup

- vLLM branch: `sleep-mode-optimize`
- Benchmark: `src/bench_vllm_lifecycle.py`
- Method: `sleep_l1`
- Prompt: `short_short`
- Repeats: 3 per model and pin mode
- Modes:
  - pinned: `--sleep-cpu-backup-pin-memory true`
  - non-pinned: `--sleep-cpu-backup-pin-memory false`
- Models:
  - `qwen2p5_0p5b`: `/home/ljl/models/hf/Qwen2.5-0.5B-Instruct`, `gpu_memory_utilization=0.55`
  - `qwen2p5_1p5b`: `/home/ljl/models/hf/Qwen2.5-1.5B-Instruct`, `gpu_memory_utilization=0.55`
  - `qwen2p5_3b`: `/home/ljl/models/hf/Qwen2.5-3B-Instruct`, `gpu_memory_utilization=0.85`

The first 3B attempt with `gpu_memory_utilization=0.55` failed for both modes because vLLM reported no available memory for KV cache blocks. The 3B retry at `0.85` completed successfully.

## Mean comparison

| model | gpu util | pinned switch | non-pinned switch | delta | delta % | pinned evict | non-pinned evict | pinned restore | non-pinned restore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2p5_0p5b | 0.55 | 0.5666 | 0.4848 | -0.0818 | -14.4% | 0.4605 | 0.3259 | 0.1061 | 0.1589 |
| qwen2p5_1p5b | 0.55 | 1.2421 | 1.3561 | +0.1141 | +9.2% | 0.9723 | 0.9343 | 0.2697 | 0.4218 |
| qwen2p5_3b | 0.85 | 2.6142 | 2.5484 | -0.0658 | -2.5% | 2.1147 | 1.7671 | 0.4995 | 0.7813 |

## Key breakdown means

| model | pin | n | cpu backup alloc | D2H copy | H2D copy | create map | unmap release |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2p5_0p5b | true | 3 | 0.3697 | 0.0559 | 0.0592 | 0.0441 | 0.0267 |
| qwen2p5_0p5b | false | 3 | 0.0003 | 0.2896 | 0.0830 | 0.0466 | 0.0269 |
| qwen2p5_1p5b | true | 3 | 0.7592 | 0.1698 | 0.1782 | 0.0885 | 0.0331 |
| qwen2p5_1p5b | false | 3 | 0.0006 | 0.8888 | 0.2525 | 0.0920 | 0.0332 |
| qwen2p5_3b | true | 3 | 1.7357 | 0.3245 | 0.3611 | 0.1352 | 0.0429 |
| qwen2p5_3b | false | 3 | 0.0009 | 1.7155 | 0.4946 | 0.1395 | 0.0382 |

## Interpretation

Non-pinned CPU backup is not a stable overall win for `evict + restore` across model sizes.

Observed pattern:

1. Non-pinned consistently reduces `evict_latency_s` by eliminating pinned CPU allocation cost.
2. Non-pinned consistently increases D2H copy and H2D copy time.
3. The total `evict + restore` outcome depends on whether saved pinned allocation time exceeds slower copy time.

Model-level verdict:

- `qwen2p5_0p5b`: non-pinned wins by 0.0818s / 14.4%.
- `qwen2p5_1p5b`: non-pinned loses by 0.1141s / 9.2%.
- `qwen2p5_3b`: non-pinned wins slightly by 0.0658s / 2.5%, but only after raising `gpu_memory_utilization` to 0.85.

The 1.5B result is the important counterexample: simply disabling pinned backup allocation does not robustly reduce total sleep/wake switch time.

## Next optimization direction

The promising direction is not a global non-pinned switch. A better optimization target is reducing pinned allocation overhead while preserving pinned-copy speed, e.g.:

1. Reuse/pool pinned CPU backup tensors across repeated sleep/wake cycles.
2. Allocate pinned backup memory lazily once per allocation and keep it until engine shutdown.
3. Experiment with size-bucketed pinned buffers to reduce allocation churn while limiting CPU memory retention.
4. Keep non-pinned as a fallback or threshold-based mode only if pinned allocation is too expensive for small models.

Raw summary files:

- `details.csv`
- `summary_by_mode.csv`
- `comparison.csv`
