# OSDI-style model switching evaluation (2026-07-23)

## Scope

This artifact contains a fresh single-GPU comparison requested for:

1. Qwen2.5-0.5B, Qwen2.5-1.5B, and Qwen2.5-3B model lifecycle phases. Sleep and wake are plotted separately; they are not summed into switch time.
2. One 20-request alternating A/B trace for Proposed and llama-swap. The plotted metric is each request's streamed completion latency, indexed by request sequence number.

## Systems and measurement boundaries

- **Proposed**: research vLLM commit `4a0e87b62`. Startup creates the CPU weight snapshot after warmup. L1 sleep and wake are timed at the public engine calls. Allocator evidence confirms zero D2H and positive pre-backup reuse on every measured sleep.
- **vLLM L1**: clean worktree commit `0decac0d9`. L1 sleep and wake use the same public engine-call boundaries. The existing checkout lacked local FlashAttention shared objects, so runtime-compatible symlinks to the identical research checkout build artifacts were used; no tracked source was changed.
- **vLLM L2**: clean worktree commit `0decac0d9`. Sleep uses `sleep(level=2)`. Wake is the complete supported restore transaction: map weights, `reload_weights`, then map KV cache; the plotted wake is the sum of those synchronous steps.
- **llama-swap**: current source plus the benchmark-only state-transition profiler patch recorded in `raw/llama-swap/lifecycle-profile.patch`. Sleep uses the source state interval `ready -> stopping -> stopped`; wake uses `stopped -> starting -> ready`, whose endpoint is a successful `/health` response. The public unload call and an external idle-GPU check are also retained for physical post-condition evidence.
- **ServerlessLLM** was re-gated but is not plotted: Podman repeatedly hit Docker Hub EOF/connection reset even with the requested proxy. Docker subsequently fetched both base manifests, but three current-source builds—including the exact host-networked buildx command with upper/lowercase proxy and `NO_PROXY` build args—either failed on an incomplete conda download or made no progress for about ten minutes during dependency installation; no current-source image was produced. The existing image loaded and inferred Qwen2.5-0.5B, but its delete-based gate removed the registry entry while `ray::VllmBackend` kept 5,810 MiB allocated. That gate is not used as scale-to-zero data: current-source ServerlessLLM has no public stop API, and valid sleep must instead wait for the registered model's automatic `keep_alive=0` path to remove the backend actor, return the scheduler reservation, and release GPU memory. No such current-source run was available, so no lifecycle number is reported.
- **SwapServeLLM** uses current commit `69f8aec0`, vLLM 0.22.0, and NVIDIA cuda-checkpoint commit `00d5cce8`. The retained benchmark-only patch mounts local models, fixes the swap API response boundary, and caps max model length; sleep/wake end when the synchronous swapout/swapin API returns, with zero model GPU memory after every measured swapout.

## Common conditions

- GPU: NVIDIA GeForce RTX 3080, 10 GiB, driver 580.95.05.
- Local Hugging Face model directories.
- Float16, max model length 1024.
- Lifecycle: five cycles per system/model. All plotted llama-swap phases are five source-instrumented state-machine intervals.
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
|  | vLLM L2 | **0.0622** | 0.2937 |
|  | SwapServeLLM | 0.4126 | 0.3969 |
|  | llama-swap | 0.2966 | 12.2596 |
| Qwen2.5-1.5B | Proposed | **0.0983** | **0.2637** |
|  | vLLM L1 | 0.2001 | 0.2644 |
|  | vLLM L2 | 0.1172 | 0.6536 |
|  | SwapServeLLM | 0.6204 | 0.6067 |
|  | llama-swap | 0.2981 | 12.2590 |
| Qwen2.5-3B | Proposed | **0.1559** | **0.4774** |
|  | vLLM L1 | 0.3529 | 0.4773 |
|  | vLLM L2 | 0.1701 | 1.5237 |
|  | SwapServeLLM | 0.9089 | 0.9852 |
|  | llama-swap | 0.2950 | 13.2610 |

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

# Validate both the publication subset and the complete artifact.
sha256sum -c results/osdi_20260723/checksums.sha256
sha256sum -c results/osdi_20260723/all-files.sha256

# Validate scripts and formatting.
uvx ruff check scripts/measure_llama_swap_lifecycle.py \
  scripts/measure_vllm_sleep_phases.py scripts/plot_osdi_switch_results.py
git diff --check
```

Raw rows, aggregate JSON/CSV, plot PDFs/PNGs, and SHA-256 checksums are under `results/osdi_20260723/`.
