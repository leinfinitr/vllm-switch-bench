# Backup reuse and reclaim

## Question

Do repeated same-process sleeps reuse an existing clean CPU backup without another device-to-host copy, and does a pressure-triggered release complete both logically and physically?

## Metric boundary

Reuse is supported when every retained sleep event has positive reused bytes/count and `copy_d2h_s == 0`. Reclaim is supported when requested bytes equal the pool-size decrease, pending byte/request counters reach zero, the flush succeeds, host `MemAvailable` materially increases, and at least one client RSS value materially decreases.

## Method

The evidence retains five sleep events for each of three Qwen model sizes plus one controller pressure-release observation. The builder reports the minimum reused bytes/count and maximum repeated D2H time per model, then derives release deltas and OS-visible evidence. The validator independently checks every raw event and recomputes the summary.

## Result

All 15 retained sleep events report positive backup reuse and zero repeated D2H time. The pressure observation requests and releases 1,048,576,000 bytes, leaves no pending release accounting, increases `MemAvailable` by 1,678,163,968 bytes, and includes a 1,847,554,048-byte client RSS decrease.

![Minimum clean-backup reuse](../../../results/backup-reuse-reclaim/figures/backup-reuse.png)

## Threats and limitations

- The rows are a small, same-process mechanism observation, not a workload-level performance distribution.
- OS memory signals are host-noisy; the claim is limited to the retained material deltas and matching application accounting.
- The family does not prove behavior after arbitrary model mutation or across every allocator/runtime version.
- No new data was generated during the refactor, and the canonical GPU rerun is not complete.
