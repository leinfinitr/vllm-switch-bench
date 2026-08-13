# Backup reuse and reclaim experiment

## Question

For same-process repeated proposed vLLM L1 sleep/wake, can immutable exact CPU weight backups be
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

Retained reuse evidence covers `qwen2.5-0.5b`, `qwen2.5-1.5b`, and `qwen2.5-3b` in repeated
same-process sleep/wake cycles.

The builder reports the minimum reused count/bytes across five events per model and the
maximum observed D2H time, then renders minimum reused GiB. The validator independently
requires positive reuse count/bytes and zero D2H for every event, settled byte accounting,
zero pending work, positive host-memory availability change, at least one RSS decrease, and
exact raw-to-summary recomputation.

## Retained result

Across each model's five retained sleep events, the minimum exact backup reuse is:

- `0.5B`: `1,048,576,000 bytes` across at least 41 allocations;
- `1.5B`: `3,250,585,600 bytes` across at least 76 allocations;
- `3B`: `6,314,524,672 bytes` across at least 112 allocations.

The maximum `copy_d2h_s` is zero in all three retained groups. 

In the pressure observation using the same `0.5B` model,
`1,048,576,000` requested bytes were released with zero pending bytes/requests;
`MemAvailable` increased by `1,678,163,968` bytes and one recorded client RSS fell by
`1,847,554,048` bytes.

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

This family does not establish long-duration stability, multi-GPU/NUMA behavior, pressure
threshold selection, coordinator fairness, crash recovery, or reclaim latency under diverse
host workloads.

## Reproduce

### Deterministic CPU rebuild and validation

From the benchmark repository root:

```bash
uv sync --frozen --group dev
scripts/build_all.sh
uv run python -m llm_switch_bench.validation.backup_reuse_reclaim.validate
scripts/validate_all.sh
git diff --exit-code -- results/backup-reuse-reclaim
```

Repeat the build/validation/diff sequence once more. It reads retained JSON only and performs
no GPU work.

### Live same-process reuse and reclaim

The live harness imports `vllm` in-process. Therefore it must run with the compatible vLLM
Switch interpreter, while importing `llm_switch_bench` from this checkout. Stock vLLM lacks
the reuse/coordinator profile fields required by this experiment. On a fresh machine, create
or sync the compatible vLLM environment first, then install only the benchmark package into
that environment:

```bash
export BENCH_REPO=/path/to/llm-switch-bench
export VLLM_REPO=/path/to/vllm
export CONTROLLER_REPO=/path/to/vllm-switch-controller
export MODEL_ROOT=/path/to/huggingface-models
export RUN_ROOT="$BENCH_REPO/results/tmp/backup-reuse-reclaim"
export VLLM_PYTHON="$VLLM_REPO/.venv/bin/python"
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"

mkdir -p "$RUN_ROOT"

cd "$VLLM_REPO"
uv pip install --python "$VLLM_PYTHON" -e "$BENCH_REPO" --no-deps
"$VLLM_PYTHON" - <<'PY'
import llm_switch_bench
import os
import sys
import vllm

assert os.path.realpath(sys.executable) == os.path.realpath(os.environ["VLLM_PYTHON"])
print("benchmark:", llm_switch_bench.__file__)
print("vllm:", vllm.__file__)
PY
command -v ninja
test -x "$(dirname "$VLLM_PYTHON")/ninja"

nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader,nounits
```

The script creates a timestamped run below `--out-dir`. It retains every `LLM` object for the
entire model sequence: first visit is `load -> infer -> sleep`; later visits are
`wake -> infer -> sleep`. Do not replace this with process-restart runs because they cannot
observe per-process clean-backup reuse.

#### No-pressure reuse control

Use one model for a simple mechanism check, or multiple models only when host RAM can retain
all sleeping engine objects. The command below performs five same-process cycles and fails
unless a later sleep reports positive reuse with exactly zero D2H:

```bash
cd "$BENCH_REPO"
set -o pipefail
CONTROL_ROOT="$RUN_ROOT/reuse-control/$(date -u +%Y%m%dT%H%M%SZ)-$$"
test ! -e "$CONTROL_ROOT"
mkdir -p "$CONTROL_ROOT"
CONTROL_LOG="$CONTROL_ROOT/command.log"
"$VLLM_PYTHON" -m llm_switch_bench.experiments.backup_reuse_reclaim.run \
  --models qwen-0.5b="$MODEL_ROOT/Qwen2.5-0.5B-Instruct" \
  --iterations 5 \
  --no-expect-release \
  --expect-reuse \
  --gpu-memory-utilization 0.55 \
  --max-model-len 1024 \
  --dtype float16 \
  --out-dir "$CONTROL_ROOT" | tee "$CONTROL_LOG"
CONTROL_SUMMARY=$(find "$CONTROL_ROOT" -mindepth 2 -maxdepth 2 -type f \
  -name repeated_sleep_l1_summary.json -print -quit)
test -n "$CONTROL_SUMMARY"
test -f "$CONTROL_SUMMARY"
```

The runner creates one timestamped child directory under the unique per-command root. Inspect
the captured summary path rather than scanning a shared directory for a stale previous run:

```bash
export CONTROL_SUMMARY

"$VLLM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path(os.environ["CONTROL_SUMMARY"]).read_text())
assert data["ok"] is True, data.get("assertion_failures") or data.get("error")
assert not data["assertion_failures"]
reuse = [
    step for step in data["steps"]
    if int(step.get("sleep_allocator_cpu_backup_reused_bytes", 0)) > 0
    and float(step.get("sleep_allocator_copy_d2h_s", -1)) == 0.0
]
assert reuse, "no later sleep reused a clean CPU backup with zero D2H"
assert all(step.get("output_matches_reference") is not False for step in data["steps"])
print({"steps": len(data["steps"]), "reuse_steps": len(reuse), "ok": data["ok"]})
PY
```

`repeated_sleep_l1_steps.csv`, the complete JSON summary, and the raw sleep-profile events
are local evidence. The first load/sleep establishes the backup; only later sleep rows can
support the retained-clean-backup claim.

#### Pressure-driven reclaim treatment

Run this as a separate fresh controller process so stale process-incarnation records cannot
absorb a new run's release budget. Configure no models in the controller is invalid, so use
one placeholder backend entry; this offline harness uses only the coordinator endpoints and
does not ask the controller to route requests to that backend.

On a shared host, never allocate RAM toward OOM. Read current `MemTotal`/`MemAvailable` and
choose a temporary low/high pair **above current MemAvailable** by a safe margin so policy
pressure is synthetic and reversible. High must also exceed low and should exceed current
MemAvailable by at least the candidate backup size plus margin if full reclaim is required.
Record the chosen values. Example shell arithmetic for a 0.5B run:

```bash
read -r MEM_TOTAL MEM_AVAILABLE <<EOF
$(awk '
  /^MemTotal:/ {total=$2*1024}
  /^MemAvailable:/ {available=$2*1024}
  END {print total, available}
' /proc/meminfo)
EOF
BACKUP_BYTES=1048576000
MARGIN_BYTES=268435456
LOW_BYTES=$((MEM_AVAILABLE + MARGIN_BYTES))
HIGH_BYTES=$((MEM_AVAILABLE + BACKUP_BYTES + MARGIN_BYTES))
test "$HIGH_BYTES" -lt "$MEM_TOTAL"

mkdir -p "$RUN_ROOT/pressure"
cat > "$RUN_ROOT/pressure/controller.yaml" <<EOF
models:
  offline-benchmark:
    backend_url: http://127.0.0.1:19999
    served_model_name: offline-benchmark
    sleep_level: 1
    wake_tags: null
controller:
  host: 127.0.0.1
  port: 9000
  policy: always_sleep_previous
  startup_awake_model: null
  request_timeout_s: 600
  switch_timeout_s: 600
  metrics_path: "$RUN_ROOT/pressure/controller-events.jsonl"
  cpu_memory_reclaim_available_bytes: $LOW_BYTES
  cpu_memory_recovery_available_bytes: $HIGH_BYTES
  cpu_memory_poll_interval_s: 0.5
  cpu_memory_pressure_consecutive_samples: 3
  cpu_memory_reclaim_cooldown_s: 2.0
  cpu_backup_global_cap_bytes: null
EOF
```

Terminal 1 starts only this run's coordinator:

```bash
cd "$CONTROLLER_REPO"
uv sync --frozen --dev
uv run vllm-switch-controller --config "$RUN_ROOT/pressure/controller.yaml" \
  2>&1 | tee "$RUN_ROOT/pressure/controller.log"
```

Terminal 2 verifies that the monitor is active, then runs five cycles. The observation dwell
must cover pressure debounce (`3 × 0.5 s` here), the worker poll interval, and RSS
stabilization; `5 s` is the concrete minimum used below. The 256 MiB RSS threshold is an
explicit run acceptance threshold, not a universal constant.

```bash
curl --noproxy '*' -fsS http://127.0.0.1:9000/health
curl --noproxy '*' -fsS http://127.0.0.1:9000/admin/cpu-backup/stats \
  > "$RUN_ROOT/pressure/pre-run-stats.json"
export EXPECTED_LOW_BYTES="$LOW_BYTES" EXPECTED_HIGH_BYTES="$HIGH_BYTES"
"$VLLM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_ROOT"]) / "pressure/pre-run-stats.json"
data = json.loads(path.read_text())
pressure = data["memory_pressure"]
assert pressure["enabled"] is True
assert pressure["reclaim_available_bytes"] == int(os.environ["EXPECTED_LOW_BYTES"])
assert pressure["recovery_available_bytes"] == int(os.environ["EXPECTED_HIGH_BYTES"])
assert pressure["probe_errors"] == 0
assert data["stats"]["global_cap_bytes"] is None
PY

cd "$BENCH_REPO"
set -o pipefail
PRESSURE_ROOT="$RUN_ROOT/pressure-treatment/$(date -u +%Y%m%dT%H%M%SZ)-$$"
test ! -e "$PRESSURE_ROOT"
mkdir -p "$PRESSURE_ROOT"
PRESSURE_LOG="$PRESSURE_ROOT/command.log"
"$VLLM_PYTHON" -m llm_switch_bench.experiments.backup_reuse_reclaim.run \
  --models qwen-0.5b="$MODEL_ROOT/Qwen2.5-0.5B-Instruct" \
  --iterations 5 \
  --coordinator-url http://127.0.0.1:9000 \
  --coordinator-timeout-s 1.0 \
  --post-wake-observation-s 5 \
  --expect-release \
  --min-worker-rss-reclaim-bytes 268435456 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 1024 \
  --dtype float16 \
  --out-dir "$PRESSURE_ROOT" | tee "$PRESSURE_LOG"
PRESSURE_SUMMARY=$(find "$PRESSURE_ROOT" -mindepth 2 -maxdepth 2 -type f \
  -name repeated_sleep_l1_summary.json -print -quit)
test -n "$PRESSURE_SUMMARY"
test -f "$PRESSURE_SUMMARY"

curl --noproxy '*' -fsS http://127.0.0.1:9000/admin/cpu-backup/stats \
  > "$RUN_ROOT/pressure/post-run-stats.json"
```

The benchmark exits `2` if it sees no run-local release, any output mismatch, missing/nonzero
host-cache-flush error telemetry, unsettled requested/released bytes, or insufficient RSS
drop. Inspect the summary itself as well:

```bash
export PRESSURE_SUMMARY

"$VLLM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path(os.environ["PRESSURE_SUMMARY"]).read_text())
assert data["ok"] is True, data.get("assertion_failures") or data.get("error")
stats = data["coordinator_stats"]
assert stats["client_count"] >= 1
assert stats["requested_release_bytes_total"] > 0
assert stats["released_bytes_total"] >= stats["requested_release_bytes_total"]
assert stats["pending_release_bytes"] == 0
released = [s for s in data["steps"] if s.get("cpu_backup_release_delta_bytes", 0) > 0]
assert released
sleep_events = [
    event for event in data["sleep_profile_events"]
    if event.get("phase") == "allocator_sleep"
]
assert sleep_events and all(event.get("cpu_backup_pin_memory") is True for event in sleep_events)
assert any(
    int(value or 0) > 0
    for step in released
    for key, value in step.items()
    if key.endswith("cpu_backup_host_cache_flush_count")
)
before = data["environment"]["initial_meminfo_bytes"]["MemAvailable"]
after = data["environment"]["final_meminfo_bytes"]["MemAvailable"]
assert after > before, (before, after)
print({"release_steps": len(released), "coordinator": stats, "ok": data["ok"]})
PY
```

#### Cleanup

Stop terminal 1 with `Ctrl-C`. Verify port `9000`, benchmark-owned vLLM/EngineCore children,
and GPU compute processes are gone. The offline harness clears its engine references in a
`finally` block, but always inspect failed runs and clean only identified owned PIDs/PGIDs.

The pressure treatment proves application release only when cumulative coordinator
requested/released counters settle. A physical claim additionally needs zero host-cache
flush errors, a run-local worker RSS decrease, and corroborating `MemAvailable`; host-global
`MemAvailable` alone is noisy on a shared server. Capture source/import/config identities and
retain failed or partial evidence instead of aggregating it into successful numeric results.
