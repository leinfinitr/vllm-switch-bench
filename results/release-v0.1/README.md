# v0.1 exploratory artifact (existing data)

**Status:** release-candidate structure complete; final GPU artifact reruns pending.

This is the single canonical v0.1 artifact root. Its current numerical content is carried forward from the 2026-07-23 exploratory single-GPU campaign to exercise the release builder, deterministic figures, and publication manifests. It must not be described as the final v0.1 confirmatory dataset.

## Systems and boundaries

Canonical names are **SwapServeLLM** and **llama-swap**.

- **Proposed:** research vLLM with eager pinned CPU weight backup. Public L1 sleep and wake calls are measured separately; profiles show zero D2H and positive backup reuse on measured repeated sleeps.
- **vLLM L1:** stock/reference L1 public sleep and wake boundaries.
- **vLLM L2:** level-2 sleep; wake includes map weights, checkpoint reload, and KV-cache mapping.
- **SwapServeLLM:** synchronous CUDA checkpoint/container swap-out and swap-in, with zero model GPU memory after measured swap-out and post-restore inference.
- **llama-swap:** automatic OpenAI-compatible request routing. Lifecycle rows use source-instrumented `ready -> stopped` and `stopped -> ready` process-state intervals. E2E rows include open-loop queueing, automatic process stop/start, and inference; they are not lifecycle wake values.
- **ServerlessLLM:** blocked and excluded from plots. Existing evidence does not prove a current-source automatic scale-to-zero cycle with actor/process removal, scheduler reservation return, idle GPU, and successful reload.
- **Exact disk:** in v0.1 scope but absent from this existing-data bundle; final GPU artifact rerun is required.

## Retained exploratory conditions

- GPU: NVIDIA GeForce RTX 3080, 10 GiB, driver 580.95.05.
- Models: local Qwen2.5-0.5B, 1.5B, and 3B checkpoints.
- Float16, maximum model length 1024 where supported.
- Lifecycle: five samples per system/model.
- E2E: one 20-request alternating trace per system at 1.5-second absolute arrival spacing.
- Figures: lifecycle median with IQR; request completion latency by sequence number; logarithmic axes.

## Existing exploratory results

Lifecycle medians in seconds:

| Model | System | Sleep | Wake |
|---|---|---:|---:|
| Qwen2.5-0.5B | Proposed | 0.0728 | 0.1567 |
|  | vLLM L1 | 0.0897 | 0.1540 |
|  | vLLM L2 | 0.0622 | 0.2937 |
|  | SwapServeLLM | 0.4126 | 0.3969 |
|  | llama-swap | 0.2966 | 12.2596 |
| Qwen2.5-1.5B | Proposed | 0.0983 | 0.2637 |
|  | vLLM L1 | 0.2001 | 0.2644 |
|  | vLLM L2 | 0.1172 | 0.6536 |
|  | SwapServeLLM | 0.6204 | 0.6067 |
|  | llama-swap | 0.2981 | 12.2590 |
| Qwen2.5-3B | Proposed | 0.1559 | 0.4774 |
|  | vLLM L1 | 0.3529 | 0.4773 |
|  | vLLM L2 | 0.1701 | 1.5237 |
|  | SwapServeLLM | 0.9089 | 0.9852 |
|  | llama-swap | 0.2950 | 13.2610 |

Existing open-loop alternating trace:

| System | Strict success | Median completion | Min | Max |
|---|---:|---:|---:|---:|
| Proposed | 20/20 | 0.796 s | 0.315 s | 1.038 s |
| llama-swap | 20/20 | 45.890 s | 25.609 s | 66.181 s |

Requests arrived much faster than llama-swap's 12–13-second cold process starts, so tens-of-seconds E2E values include serialized queueing. They do not imply a 45-second lifecycle wake.

## Layout and integrity

- `raw/`: immutable producer evidence; some files preserve producer-machine absolute paths.
- `configs/`: exact retained configurations and frozen trace. Files ending in `.local.yaml` are immutable historical run inputs here, not examples.
- `summary.json` and `lifecycle-summary.csv`: deterministic aggregates.
- `figures/`: vector paper artifacts and review PNGs.
- `checksums.sha256`: publication subset.
- `all-files.sha256`: every artifact file except the complete manifest itself.

## Rebuild and verify

```bash
uv run python scripts/build_release_artifact.py
uv run python scripts/build_release_checksums.py
uv run python scripts/verify_release_artifact.py
```

The complete final-rerun transaction and remaining blockers are defined in [`../../docs/release-artifact.md`](../../docs/release-artifact.md).
