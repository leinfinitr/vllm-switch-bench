# Backup reuse and reclaim experiment

## Question

For same-process repeated vLLM L1 sleep/wake, can immutable exact CPU weight backups be
reused without another device-to-host copy, and can coordinator-driven pressure release
those retained host bytes with an OS-visible effect?

## Metric

The reuse metrics are, per model and retained sleep event:

- `cpu_backup_reuse_count` and `cpu_backup_reused_bytes`;
- `copy_d2h_s`, which must be exactly zero for the retained reuse claim;
- five valid repeated sleep events per model.

The reclaim metrics are requested versus released bytes, pending release bytes/request
count, client RSS deltas, and host `MemAvailable` delta. A valid retained reclaim observation
requires requested bytes to equal released bytes, zero pending obligations, positive
`MemAvailable` change, and at least one client RSS decrease. Allocator counters alone do not
establish physical reclaim.

## Method

Retained reuse evidence covers `qwen-0.5b`, `qwen-1.5b`, and `qwen-3b` in repeated
same-process sleep/wake cycles. This lifecycle is essential: process restarts cannot test a
pool hit. The separate pressure observation records coordinator state before/after a
1,048,576,000-byte request, run-local client accounting, process RSS, and host memory.

The builder reports the minimum reused count/bytes across five events per model and the
maximum observed D2H time, then renders minimum reused GiB. The validator independently
requires positive reuse count/bytes and zero D2H for every event, settled byte accounting,
zero pending work, positive host-memory availability change, at least one RSS decrease, and
exact raw-to-summary recomputation.

## Retained result

Across each model's five retained sleep events, the minimum exact backup reuse is:

- 0.5B: 1,048,576,000 bytes across at least 41 allocations;
- 1.5B: 3,250,585,600 bytes across at least 76 allocations;
- 3B: 6,314,524,672 bytes across at least 112 allocations.

The maximum `copy_d2h_s` is zero in all three retained groups. In the pressure observation,
1,048,576,000 requested bytes were released with zero pending bytes/requests;
`MemAvailable` increased by 1,678,163,968 bytes and one recorded client RSS fell by
1,847,554,048 bytes.

- [Backup reuse figure (PNG)](../../../results/backup-reuse-reclaim/figures/backup-reuse.png)
- [Backup reuse figure (PDF)](../../../results/backup-reuse-reclaim/figures/backup-reuse.pdf)
- [Machine-readable summary](../../../results/backup-reuse-reclaim/summary.json)
- [Result-family notes](../../../results/backup-reuse-reclaim/README.md)

## Threats to validity

- Evidence comes from one local host/GPU and three model sizes.
- Host `MemAvailable` and RSS are noisy and can move because of unrelated processes,
  allocator caching, delayed accounting, or sampling time.
- The pressure evidence is one retained observation, not a distribution.
- Reuse assumes immutable runtime weight bytes; runtime mutation or expert rearrangement can
  invalidate a clean backup and require refresh.
- Logical pool release may overshoot targets at allocation granularity in other runs; the
  retained case happened to match requested and released bytes exactly.
- Zero measured D2H time depends on instrumentation boundaries and the retained runtime
  implementation.

## Limitations

No new data was generated, and the canonical GPU rerun is not complete. The refactor only
rebuilds the retained summary and figure. Broader v0.1 runtime provenance—including the E2E
producer's engine/controller commits, imported path, and configuration hash—was not bound at
execution, so related values remain a historical local observation rather than an exact
fresh-checkout runtime reproduction.

This family does not establish long-duration stability, multi-GPU/NUMA behavior, pressure
threshold selection, coordinator fairness, crash recovery, or reclaim latency under diverse
host workloads.

## Reproduce

### Deterministic CPU rebuild and validation

```bash
uv sync --frozen --group dev
scripts/build_all.sh
uv run python -m llm_switch_bench.validation.backup_reuse_reclaim.validate
scripts/validate_all.sh
git diff --exit-code -- results/backup-reuse-reclaim
```

Repeat the build/validation/diff sequence once more. It reads retained JSON only and performs
no GPU work.

### Live reuse control (GPU; not run in this refactor)

```bash
scripts/backup-reuse-reclaim.sh \
  --models small=/path/to/small-model large=/path/to/large-model \
  --iterations 5 \
  --no-expect-release \
  --expect-reuse \
  --out-dir results/tmp/backup-reuse-reclaim/control
```

A pressure run additionally needs a configured coordinator, `--expect-release`, a positive
post-wake observation interval, and an explicit `--min-worker-rss-reclaim-bytes` threshold.
Capture runtime source/import/config identities and inspect both application and OS-visible
post-conditions before promoting evidence.
