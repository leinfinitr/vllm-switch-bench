# OSDI-style model switching evaluation (2026-07-23)

## Scope

This artifact contains a fresh single-GPU comparison requested for:

1. Qwen2.5-0.5B, Qwen2.5-1.5B, and Qwen2.5-3B model lifecycle phases. Sleep and wake are plotted separately; they are not summed into switch time.
2. One 20-request alternating A/B trace for Proposed and llama-swap. The plotted metric is each request's streamed completion latency, indexed by request sequence number.

## Systems and measurement boundaries

- **Proposed**: research vLLM commit `4a0e87b62`. Startup creates the CPU weight snapshot after warmup. L1 sleep and wake are timed at the public engine calls. Allocator evidence confirms zero D2H and positive pre-backup reuse on every measured sleep.
- **vLLM L1**: clean worktree commit `0decac0d9`. L1 sleep and wake use the same public engine-call boundaries. The existing checkout lacked local FlashAttention shared objects, so runtime-compatible symlinks to the identical research checkout build artifacts were used; no tracked source was changed.
- **llama-swap**: commit `c6adf57df`. Source inspection showed that the public unload handler synchronously awaits `StopProcesses`, but the installed Go 1.23.1 toolchain cannot rebuild the current Go 1.26.1 source. Therefore sleep uses the requested fallback: `POST /api/models/unload/{model}` through both stopped-process and idle-GPU post-conditions. Wake uses request dispatch through health readiness and successful streamed inference. The first compile/cache warmup wake is retained in raw data but excluded from the plotted post-warmup wake distribution.
- **ServerlessLLM** was re-gated but omitted from fresh bars: its local image was absent and its CUDA 12.1.1 base-image pull repeatedly failed with registry connection resets.
- **SwapServeLLM** was rebuilt from current source, and a current NVIDIA cuda-checkpoint binary was fetched. Its CUDA checkpoint path requires live-process compatibility and additional container setup; it was not added as an unverified number.

## Common conditions

- GPU: NVIDIA GeForce RTX 3080, 10 GiB, driver 580.95.05.
- Local Hugging Face model directories.
- Float16, max model length 1024.
- Lifecycle: five cycles per system/model. For llama-swap wake, four post-warmup cycles are summarized.
- Lifecycle plots: median point with IQR error bars on a logarithmic y-axis.
- E2E: frozen `configs/traces/request-switch-alternating.jsonl`, 20 requests, 1.5 s absolute arrival spacing, 32 output tokens, streaming enabled.
- Proposed E2E uses Qwen2.5-1.5B and Qwen2.5-3B long-lived engines managed by the model-switch controller.
- llama-swap E2E uses the same models and frozen trace but cold-start process switching.

## Key results

Lifecycle medians, seconds:

| Model | System | Sleep | Wake |
|---|---|---:|---:|
| Qwen2.5-0.5B | Proposed | 0.0728 | 0.1567 |
|  | vLLM L1 | 0.0897 | 0.1540 |
|  | llama-swap | 0.2970 | 12.3693 |
| Qwen2.5-1.5B | Proposed | 0.0983 | 0.2637 |
|  | vLLM L1 | 0.2001 | 0.2644 |
|  | llama-swap | 0.3010 | 12.3708 |
| Qwen2.5-3B | Proposed | 0.1559 | 0.4774 |
|  | vLLM L1 | 0.3529 | 0.4773 |
|  | llama-swap | 0.3019 | 13.3709 |

E2E alternating trace:

| System | Success | Median | Min | Max |
|---|---:|---:|---:|---:|
| Proposed | 20/20 | 0.796 s | 0.315 s | 1.038 s |
| llama-swap | 20/20 | 45.890 s | 25.609 s | 66.181 s |

The llama-swap E2E trace is intentionally open-loop. Requests arrive every 1.5 s while cold process starts take about 12--13 s, so requests queue behind serialized model switching; the plotted tens-of-seconds values are request-visible queueing plus cold-start and inference latency, not lifecycle wake alone.

## Figures

Vector PDF is the paper artifact; PNG is a review/export convenience.

- `figures/lifecycle-qwen-0.5b.pdf`
- `figures/lifecycle-qwen-1.5b.pdf`
- `figures/lifecycle-qwen-3b.pdf`
- `figures/e2e-alternating-request-latency.pdf`

The figures use single-column dimensions, embedded serif fonts, a color-blind-safe palette, redundant markers/line styles, and no legend overlap.

## Reproduction

```bash
# Rebuild plots and aggregate data from retained raw evidence.
.venv/bin/python scripts/plot_osdi_switch_results.py

# Validate scripts and formatting.
uvx ruff check scripts/measure_llama_swap_lifecycle.py \
  scripts/measure_vllm_l1_phases.py scripts/plot_osdi_switch_results.py
git diff --check
```

Raw rows, aggregate JSON/CSV, plot PDFs/PNGs, and SHA-256 checksums are under `results/osdi_20260723/`.
