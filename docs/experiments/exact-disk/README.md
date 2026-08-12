# Exact-disk experiment

## Question

Can exact runtime weight bytes be spilled to a physically allocated disk payload, have their
CPU backup released, and then restore into the allocator with identical deterministic model
output?

## Metric

The retained claim-support metrics are:

- runtime payload, spill, demotion-release, and restore-read bytes;
- physical filesystem allocation (`allocated_bytes`) as well as logical size;
- the omitted payload's SHA-256 and the runtime manifest's per-chunk SHA-256 values,
  offsets, and sizes;
- zero pending release obligation after demotion;
- equality of deterministic output before and after restore.

These are runtime integrity checks, not repository-internal Git-file digest manifests. The
payload itself is intentionally omitted from Git, so its retained digest and per-chunk
manifest are essential evidence.

## Method

The retained run used Qwen2.5-0.5B-Instruct on a single RTX 3080 with an instrumented vLLM
exact-disk implementation. Before unmapping runtime weight memory, the allocator produced a
disk backup in 16 MiB chunks and recorded the runtime segment layout and chunk digests. It
then released the exact CPU backup, restored from disk, and repeated deterministic
inference. Filesystem observation records both logical and allocated bytes.

The current builder recomputes the byte totals, payload identity, and output equality from
seven retained evidence files and produces a compact figure. The semantic validator requires
the spill/demotion/sleep phases, exactly 1,048,576,000 payload bytes, segment-size closure,
nonempty 64-character chunk digests, physical allocation of at least the payload size,
settled release, equal outputs, and exact raw-to-summary recomputation.

## Retained result

The observation records 1,048,576,000 bytes spilled, released from CPU backup, and restored
from disk. The filesystem reported 1,048,580,096 allocated bytes for the
1,048,576,000-byte logical payload. The before/after completion text is identical. The
retained payload SHA-256 is
`4aec04d7b5d1a8a9ace300e239bc65381955b058f2dab0326b8a44dc3afbbdbb`.

- [Exact-disk figure (PNG)](../../../results/exact-disk/figures/exact-disk.png)
- [Exact-disk figure (PDF)](../../../results/exact-disk/figures/exact-disk.pdf)
- [Machine-readable summary](../../../results/exact-disk/summary.json)
- [Runtime segment/checksum manifest](../../../results/exact-disk/raw/exact-disk/bundle-manifest.json)
- [Result-family notes](../../../results/exact-disk/README.md)

## Threats to validity

- This is one local single-GPU observation for one model and approximately 0.98 GiB of
  runtime payload.
- Filesystem allocation does not prove storage media durability, cache coldness, or device
  read behavior; page cache and filesystem/kernel configuration can affect timings.
- Output equality covers one deterministic prompt/output observation, not arbitrary model
  behavior or every runtime tensor.
- Chunk hashes detect corruption against the retained manifest but do not establish that all
  runtime mutation paths correctly invalidate or refresh a backup.
- The exact-disk implementation is an external engine modification, not stock vLLM.

## Limitations

No new data was generated in this refactor, and a canonical GPU rerun is not complete. The
retained metadata identifies the collection/upstream engine commits and environment for this
observation, but it is not a broad performance study and the payload bytes are intentionally
not published. The wider v0.1 E2E producer did not runtime-bind engine/controller commits,
imported path, or configuration hash, so those E2E numbers remain a historical local
observation; the stronger exact-disk runtime checksum record does not retroactively repair
that separate provenance gap.

The current summary supports functional exact-byte spill/release/restore. It does not claim
steady-state latency, throughput, SSD endurance, crash consistency, multi-model capacity, or
superiority over another storage tier.

## Reproduce

### Deterministic CPU rebuild and validation

From the benchmark repository root:

```bash
uv sync --frozen --group dev
scripts/exact-disk-build.sh
scripts/exact-disk-validate.sh
scripts/validate_all.sh
git diff --exit-code -- results/exact-disk
```

Repeat the build/validation/diff sequence once more. It validates the retained manifest and
observations but does not recreate the omitted payload or run a GPU.

### Live exact-disk lifecycle capture

This is not a stock-vLLM command. Use a compatible vLLM Switch checkout that implements
`VLLM_EXACT_DISK_BACKUP_*`, `demote_weight_cpu_backup_to_disk`, exact-disk profile events,
and the developer lifecycle endpoints. Start the server as one captured process group, then
run the lifecycle driver inside that group. The collector samples the complete wrapper/server/
driver process tree before and after demotion.

Set explicit machine-local paths and install the benchmark package into the vLLM environment
so both server and driver use the same Python while the collector remains importable:

```bash
export BENCH_REPO=/path/to/llm-switch-bench
export VLLM_SWITCH_REPO=/path/to/vllm-switch
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
export VLLM_PYTHON="$VLLM_SWITCH_REPO/.venv/bin/python"
export RUN_ID=run-001
export OUT_DIR="$BENCH_REPO/results/tmp/exact-disk/$RUN_ID"
export BACKUP_ROOT="$BENCH_REPO/runtime/exact-disk-backups/$RUN_ID"

test -f "$MODEL_PATH/config.json"
test ! -e "$OUT_DIR"
test ! -e "$BACKUP_ROOT"
if ss -ltn 'sport = :8000' | grep -q LISTEN; then
  echo "port already in use: 8000" >&2
  exit 1
fi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits

cd "$VLLM_SWITCH_REPO"
uv pip install --python "$VLLM_PYTHON" -e "$BENCH_REPO" --no-deps
"$VLLM_PYTHON" - <<'PY'
import llm_switch_bench
import vllm
from vllm import envs
from vllm.v1.worker.gpu_worker import Worker

assert hasattr(envs, "VLLM_EXACT_DISK_BACKUP_ENABLED")
assert hasattr(Worker, "demote_weight_cpu_backup_to_disk")
print("benchmark:", llm_switch_bench.__file__)
print("vllm:", vllm.__file__)
PY

git -C "$BENCH_REPO" status --short --branch
git -C "$VLLM_SWITCH_REPO" status --short --branch
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader,nounits
df -T "$(dirname "$BACKUP_ROOT")"
```

The default requires an empty backup root so footprint growth belongs to this run. Do not use
`--allow-nonempty-backup-root` for claim-supporting evidence. Direct I/O is requested, but
the current vLLM profile does not expose the actual direct-I/O state: a pinned-staging failure
can fall back to buffered disk I/O while still reporting `fallback=false`. Until the engine
records actual `direct_io` and a fallback reason, retain and inspect the server warning and do
not claim that the current curator fails closed on direct-I/O fallback.

Run from the benchmark repository root. `LLM_SWITCH_BENCH_VLLM_REPO` binds engine provenance;
the collector exports the backup, profile, model, and output-observation paths. The
virtual-environment directory is prepended to `PATH` so runtime JIT helpers are available.
The server stays inside the collector-owned process group until the lifecycle driver exits:

```bash
cd "$BENCH_REPO"
export LLM_SWITCH_BENCH_VLLM_REPO="$VLLM_SWITCH_REPO"
export LLM_SWITCH_BENCH_VLLM_PYTHON="$VLLM_PYTHON"
export LLM_SWITCH_BENCH_VLLM_IMPORT_PATH=$(
  "$VLLM_PYTHON" -c 'import vllm; print(vllm.__file__)'
)
export LLM_SWITCH_BENCH_MODEL_REVISION=<exact-model-revision-or-content-id>
export LLM_SWITCH_BENCH_MODEL_CONFIG_SHA256=$(sha256sum "$MODEL_PATH/config.json" | cut -d' ' -f1)
export LLM_SWITCH_BENCH_BACKUP_FILESYSTEM=$(df -T "$(dirname "$BACKUP_ROOT")" | tail -n 1)
export LLM_SWITCH_BENCH_GPU_IDENTITY=$(
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader,nounits
)
export VLLM_SERVER_DEV_MODE=1
export VLLM_EXACT_DISK_BACKUP_ENABLED=1
export VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES=16777216
export VLLM_EXACT_DISK_BACKUP_DIRECT_IO=1

scripts/exact-disk-run.sh \
  --model qwen-0.5b="$MODEL_PATH" \
  --backup-root "$BACKUP_ROOT" \
  --out-dir "$OUT_DIR" \
  --sample-interval-s 0.25 \
  -- bash -c '
    set -euo pipefail
    export PATH="$VLLM_SWITCH_REPO/.venv/bin:$PATH"
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES=16777216
    export VLLM_EXACT_DISK_BACKUP_DIRECT_IO=1
    export no_proxy=127.0.0.1,localhost
    export NO_PROXY=127.0.0.1,localhost

    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$LLM_SWITCH_BENCH_MODEL_PATH" \
      --served-model-name "$LLM_SWITCH_BENCH_MODEL_NAME" \
      --host 127.0.0.1 \
      --port 8000 \
      --enable-sleep-mode \
      --gpu-memory-utilization 0.80 \
      --max-model-len 1024 \
      --dtype half \
      --enforce-eager &
    server_pid=$!
    trap '\''kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" || true'\'' EXIT

    "$VLLM_PYTHON" -m llm_switch_bench.experiments.exact_disk.lifecycle_driver \
      --base-url http://127.0.0.1:8000 \
      --served-model-name "$LLM_SWITCH_BENCH_MODEL_NAME" \
      --ready-timeout-s 300

    # Preserve the complete committed bundle before orderly vLLM teardown deletes it.
    mkdir -p "$OUT_DIR/runtime-bundle"
    cp -a "$VLLM_EXACT_DISK_BACKUP_DIR"/. "$OUT_DIR/runtime-bundle"/
  '
```

The lifecycle boundary is explicit:

```text
server ready -> deterministic inference
-> prepare exact disk bundle and release all CPU weight backup
-> verify positive released bytes and zero remaining/pending CPU bytes
-> sleep(level=1) -> wake_up -> disk restore
-> deterministic inference and exact output equality
```

The collector rejects the run unless the command exits zero and creates both
`raw/exact_disk_profile.jsonl` and `raw/output_observation.json`. Server and driver output are
captured together in `raw/command.stdout.log` and `raw/command.stderr.log`. The copied
`runtime-bundle/` is outside the collector's raw checksum manifest, so hash/copy it separately
before any promotion. The collector checksums its raw files, then creates
`curated/summary.json` and `curated/assertions.json`. Inspect the gates explicitly:

```bash
export OUT_DIR
"$VLLM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
run = json.loads((root / "raw/run.json").read_text())
summary = json.loads((root / "curated/summary.json").read_text())
checks = json.loads((root / "curated/assertions.json").read_text())

assert run["command_return_code"] == 0
assert checks["ok"] is True, checks["failures"]
assert summary["profile"]["disk_spill_bytes"] > 0
assert summary["profile"]["disk_read_bytes"] > 0
assert summary["profile"]["source_media"] == ["disk"]
assert summary["profile"]["fallback_count"] == 0
assert summary["resources"]["disk_footprint_peak_delta_bytes"] > 0
assert summary["resources"]["worker_rss_last_bytes"] \
    < summary["resources"]["worker_rss_peak_bytes"]
assert summary["output_equality"]["output_equal"] is True
print({"assertions": checks["ok"], "profile": summary["profile"]})
PY

find "$OUT_DIR/runtime-bundle" -type f -name manifest.json -print -exec sha256sum {} \;
find "$OUT_DIR/runtime-bundle" -type f -name COMMIT -print -exec sed -n '1p' {} \;
find "$OUT_DIR/runtime-bundle" -type f -name data.bin -exec stat \
  --printf='%n logical=%s blocks_512=%b\n' {} \;
"$VLLM_PYTHON" - "$OUT_DIR/runtime-bundle" <<'PY'
import sys
from pathlib import Path

payloads = list(Path(sys.argv[1]).rglob("data.bin"))
assert payloads, "runtime bundle contains no data.bin"
for path in payloads:
    stat = path.stat()
    print(path, {"logical_bytes": stat.st_size, "allocated_bytes": stat.st_blocks * 512})
PY
```

The runtime bundle's canonical `manifest.json`, `COMMIT` digest, chunk hashes, logical size,
and allocated blocks are primary exact-byte evidence. The local collector's
`evidence_manifest.json` separately authenticates its copied profile, resource samples,
output observation, logs, and run metadata.

#### Cleanup

The shell trap stops the captured server, and the collector then terminates any descendant
remaining in its process group. Afterward confirm port `8000`, vLLM/EngineCore descendants,
and GPU compute processes are gone. Preserve `$OUT_DIR/raw` for failed runs; failures are
intentionally raw-only and must not be promoted as a numeric result. Remove `$BACKUP_ROOT`
only after copying the claim-supporting runtime bundle/checksums you intend to retain.

Bind the actual engine commit/dirty state, imported path, model revision, filesystem/device,
CUDA/PyTorch/driver/GPU, complete argv/environment, page-cache policy, and direct-I/O result.
This command is a current functional capture; it does not recreate the retained payload
digest unless every runtime input and produced byte happens to match.
