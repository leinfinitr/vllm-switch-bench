# v0.1 release artifact

## Status

`results/release-v0.1/` is the single canonical release artifact root. The
current bundle is the final 2026-08-04 v0.1 single-node, single-GPU campaign.
It is a research preview, not a production or cluster-scale result set.

## Layout

```text
results/release-v0.1/
  README.md                 Human-readable scope, boundaries, status, and blockers
  raw/                      Immutable retained producer evidence
  campaign.json             Scope and immutable source identities
  provenance/               Producer checksums and binary identities
  lifecycle-summary.csv     Deterministic lifecycle aggregate
  summary.json              Deterministic complete aggregate and provenance
  figures/                  Deterministic PDF/PNG outputs
  checksums.sha256          Publication subset
  all-files.sha256          Complete bundle except itself
```

## Existing-data rebuild

```bash
uv run python scripts/build_release_artifact.py
uv run python scripts/build_release_checksums.py
uv run python scripts/verify_release_artifact.py
```

The builder reads only tracked files below `results/release-v0.1/raw/`. It does not launch services or GPUs. `build_release_checksums.py` must run after all derived outputs. `verify_release_artifact.py` checks bytes, tracked-path closure, and exact complete-manifest coverage.

The retained Proposed E2E matrix predates the run-start executable-provenance
requirement below: it authenticates the benchmark checkout and frozen trace but
does not independently bind the controller/engine binaries, dirty states,
configuration, or model revision that served those requests. Release-level
source identities are post-run labels and must not be cited as if they were
captured by the per-run metadata.

## Final-rerun publication transaction

The final GPU campaign must use a fresh staging root such as `results/tmp/release-v0.1-<run-id>/`; it must not write into the canonical bundle incrementally.

1. Freeze model revisions/checksums, traces, prompts, generation fields, ports, deadlines, environment, and system order.
2. Capture run-start benchmark/controller/engine/baseline commits and dirty states, interpreter/import paths, container image digests, CUDA/driver/GPU/host identity, command line, and sanitized config hash.
3. Run each required cell in a fresh reset block. Retain strict request rows, phase events, correctness outputs, post-conditions, process/GPU samplers, and structured failures.
4. Validate the complete expected matrix before aggregation. Any missing, duplicate, unexpected, non-zero, or semantically failed cell invalidates publication.
5. Build summary and figures from the staged tracked-input set. Run the builder twice and require byte-identical outputs.
6. Generate the publication manifest first, then the complete manifest.
7. Verify in a detached fresh worktree and visually inspect final-size PDF/PNG figures.
8. Replace `results/release-v0.1/` atomically in one reviewable commit. Never merge old and new raw rows.

## v0.1 required cells

- Proposed, stock vLLM L1, and vLLM L2 lifecycle for all declared models;
- SwapServeLLM lifecycle with physical swap-out and post-restore inference;
- llama-swap source-state lifecycle profiling and automatic-routing traces;
- Proposed and llama-swap frozen request traces with identical request semantics;
- ServerlessLLM only after the current-source automatic scale-to-zero post-condition passes;
- exact disk tier lifecycle/profiling with disk-source, zero-fallback, physical reclaim, and output-equality assertions.

Any blocked system remains in a structured `blocked/` or status record and is excluded from numeric plots. Exact disk is a v0.1 requirement, not a future v0.2 feature.

## Measurement policy

Sleep/evict and wake/restore remain separate. Only an explicitly named switch-time metric may sum them. llama-swap's automatic routing E2E latency includes queueing and process switching, while its lifecycle rows come from process-state instrumentation; the two are not interchangeable.

The final campaign and deterministic rebuild are complete. Future corrections
must create a new staged bundle rather than modify checksummed raw evidence.
