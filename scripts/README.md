# Scripts

`scripts/` contains shell orchestration, artifact-bound measurement drivers, release builders, and repository gates. Reusable Python benchmark logic lives in `src/`; pure reusable analysis belongs in `src/tool/`.

## Current reusable entry points

- `run_profiling.sh`: vLLM pinned/pageable profiling matrix.
- `run_request_switch.sh` and `run_request_switch_matrix.py`: open-loop request traces.
- `run_exact_disk_profile.py`: model-agnostic exact-disk evidence wrapper.
- `check_bash.sh`: syntax-check all shell entry points.
- `check_docs.py`: current-doc link/language/path portability gate.

`run_baseline3.sh` is a historical compatibility entry point and does not define the v0.1 release protocol.

## Release artifact tooling

```bash
uv run python scripts/build_release_artifact.py
uv run python scripts/build_release_checksums.py
uv run python scripts/verify_release_artifact.py
```

- `build_release_artifact.py` reads retained raw evidence and deterministically rebuilds aggregate CSV/JSON and figures.
- `build_release_checksums.py` writes the publication subset and complete-bundle manifests.
- `verify_release_artifact.py` verifies checksums, Git tracked-path closure, and exact complete-manifest coverage.

Do not run the checksum builder before the derived artifact is final. Do not run the analysis builder during final GPU collection; publish only after the complete staged raw matrix validates.

## Exact disk wrapper

```bash
uv run python scripts/run_exact_disk_profile.py \
  --model model=/path/to/model \
  --backup-root runtime/exact-disk-backups \
  --out-dir results/tmp/exact-disk/model/RUN_ID \
  -- /path/to/python /path/to/model_agnostic_driver.py
```

The producer receives:

- `VLLM_EXACT_DISK_BACKUP_ENABLED=1`;
- `VLLM_EXACT_DISK_BACKUP_DIR` and `VLLM_CPU_BACKUP_DISK_DIR`;
- `VLLM_SLEEP_PROFILE_PATH` for append-only profile JSONL;
- `LLM_SWITCH_BENCH_OUTPUT_OBSERVATION` for deterministic before/after inference;
- model name/path environment fields.

The wrapper rejects existing output and, by default, a non-empty backup root. It captures command logs, process-tree RSS, host `MemAvailable`, disk footprint, source/fallback, and output equality. Failed commands remain raw-only.

`bench_exact_disk_allocator.py` and `drive_exact_disk_lifecycle.py` are GPU experiment drivers tied to the exact-disk research implementation. They are current v0.1 preparation tools but require a compatible external vLLM checkout and are not exercised by CPU CI.

## External baseline drivers

- `measure_llama_swap_lifecycle.py`: source-instrumented llama-swap process-state phases.
- `measure_llama_swap_switch.py`: legacy request-visible transition driver; prefer the shared trace runner for final E2E.
- `measure_swapserve_lifecycle.py`: SwapServeLLM synchronous swap-out/swap-in lifecycle.
- `measure_serverless_switch.py`: ServerlessLLM gate driver; numeric publication remains blocked until the automatic scale-to-zero contract passes.
- `measure_vllm_sleep_phases.py`: vLLM engine lifecycle phase collection.

These scripts assume externally started services. Final runs must retain the external commits, patches, binary/image digests, sanitized configs, and process/GPU post-conditions. The exact current protocol is [`../docs/release-artifact.md`](../docs/release-artifact.md).
