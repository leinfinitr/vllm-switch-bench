# v0.1 Research Artifact

This directory is the canonical evidence bundle for the v0.1 release. It was
collected on 2026-08-04 on a single NVIDIA RTX 3080 (10 GiB), Linux, and is
intended as a **single-node, single-GPU research preview**, not a production or
cluster-scale evaluation.

## Included systems

- **Proposed**: the vLLM fork at `1b3919d8c210af05f6ea8b29fff33fb8d07e6c1d`.
- **vLLM L1 / L2**: upstream `v0.22.1` plus the instrumentation-only
  `research/sleep-mode-profiling` branch at
  `03e5ae257135073ddddbcd1264697f24c1c62e08`.
- **SwapServeLLM**: `69f8aec0b11e49124f70754dc5149c36fd8327a5`.
- **llama-swap**: `c6adf57df1ac2e3dff2402dbb479cd5a133b6afe`, exercised through its default
  exclusive swap group and request-driven automatic switching.
- **ServerlessLLM** remains excluded because retained evidence does not close a
  current-source automatic scale-to-zero/reload cycle.

Each lifecycle cell contains five successful cycles for Qwen2.5 0.5B, 1.5B,
and 3B. The Proposed and llama-swap request traces contain 20 strict-success
streaming requests with a fixed 1.5-second alternating schedule.

## v0.1 mechanism evidence

The Proposed allocator profiles show pinned clean-backup reuse in every cycle:
`copy_d2h_s == 0` and positive `cpu_backup_reused_bytes`. Controller evidence
binds the metadata-only protocol/capabilities; the pressure validator and
controller test suite cover dynamic reclaim. `raw/exact-disk/` retains the
exact-disk spill/demotion/restore profile, bundle manifest and commit marker,
full payload SHA-256 and size, and deterministic before/after inference. The
1,048,576,000-byte data payload is intentionally omitted from Git.

## Profiling semantics

- L1 restores exact runtime weight bytes from host backup.
- L2 maps weights, reloads the checkpoint, and recreates the KV cache.
- llama-swap lifecycle wake is `stopped -> starting -> ready`; lifecycle sleep
  is `ready -> stopping -> stopped`. Request-level E2E includes queueing and
  inference and is not substituted for lifecycle timing.
- SwapServeLLM sleep/wake is synchronous CUDA checkpoint/container swap-out and
  swap-in; every measured swapped-out boundary had zero model GPU memory.

## Integrity and rebuild

```bash
uv sync --frozen
uv run --frozen python scripts/build_release_artifact.py
uv run --frozen python scripts/build_release_checksums.py
uv run --frozen python scripts/verify_release_artifact.py
```

`checksums.sha256` covers publication outputs; `all-files.sha256` covers every
other file in this directory. The raw evidence is immutable after publication.
