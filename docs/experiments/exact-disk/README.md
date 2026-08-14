# Exact disk

## Research question and metrics

Can exact runtime weight bytes be spilled to a physically allocated disk payload, released
from CPU backup, restored into the allocator, and still produce identical deterministic
output?

The evidence retains payload, spill, demotion-release, and restore-read bytes; logical and
allocated filesystem size; omitted-payload SHA-256; per-chunk offset/size/SHA-256; manifest
commit; release settlement; and before/after output. Success requires all lifecycle phases,
positive spill/read bytes, no fallback, contiguous segment closure, an authenticated manifest,
allocated blocks covering the payload, complete CPU-backup release with zero pending bytes,
six successful cycles, and equal output. Runtime payload/chunk hashes are required because the
payload is intentionally omitted from Git.

The [claims contract](../../../results/exact-disk/config/claims.json) fixes a
`1,048,576,000` byte payload, 16 MiB chunks, six cycles, and five post-warm-up profiling
samples. The retained run used Qwen2.5-0.5B-Instruct and one RTX 3080.

## Retained result

The 2026-08-13 run spilled and released `1,048,576,000` bytes. Six restores read
`6,291,456,000` bytes in total, and deterministic output matched. The omitted payload SHA-256
is `4aec04d7b5d1a8a9ace300e239bc65381955b058f2dab0326b8a44dc3afbbdbb`.

- [PNG figure](../../../results/exact-disk/figures/exact-disk.png)
- [PDF figure](../../../results/exact-disk/figures/exact-disk.pdf)
- [JSON summary](../../../results/exact-disk/summary.json)
- [Runtime manifest](../../../results/exact-disk/raw/exact-disk/bundle-manifest.json)

## Reproduce the measurement

Use a compatible Proposed checkout that exposes exact-disk environment variables, the
demotion RPC, profile phases, and sleep/wake developer endpoints. Run from the benchmark root
on an idle GPU. The backup root must be empty so disk growth belongs to this run.

```bash
uv sync --frozen --group dev

BENCH_ROOT=$PWD
RUN_ROOT="$BENCH_ROOT/results/tmp/exact-disk/run-001"
BACKUP_ROOT="$BENCH_ROOT/runtime/exact-disk-backups/run-001"
PROPOSED_REPO=/path/to/vllm-switch
PROPOSED_PYTHON="$PROPOSED_REPO/.venv/bin/python"
MODEL=/path/to/Qwen2.5-0.5B-Instruct

"$PROPOSED_PYTHON" -m pip install -e . --no-deps

mkdir -p "$RUN_ROOT" "$(dirname "$BACKUP_ROOT")"

if test -e "$BACKUP_ROOT" && test -n "$(find "$BACKUP_ROOT" -mindepth 1 -print -quit)"
then
  echo "backup root is not empty: $BACKUP_ROOT" >&2
  exit 1
fi

export LLM_SWITCH_BENCH_VLLM_REPO="$PROPOSED_REPO"
export LLM_SWITCH_BENCH_VLLM_PYTHON="$PROPOSED_PYTHON"
export LLM_SWITCH_BENCH_VLLM_IMPORT_PATH=$(
  PYTHONPATH="$PROPOSED_REPO" "$PROPOSED_PYTHON" -c 'import vllm; print(vllm.__file__)'
)
export LLM_SWITCH_BENCH_MODEL_REVISION=local-files
export LLM_SWITCH_BENCH_MODEL_CONFIG_SHA256=$(sha256sum "$MODEL/config.json" | cut -d' ' -f1)
export LLM_SWITCH_BENCH_BACKUP_FILESYSTEM=$(df -T "$(dirname "$BACKUP_ROOT")" | tail -n 1)
export LLM_SWITCH_BENCH_GPU_IDENTITY=$(
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader,nounits
)
export VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES=16777216
export VLLM_EXACT_DISK_BACKUP_DIRECT_IO=1
```

Create an executable run-owned command at `$RUN_ROOT/lifecycle-command.sh` with this content.
The runtime bundle must be copied before server teardown removes its working state.

```bash
#!/usr/bin/env bash
set -euo pipefail

export PATH="/path/to/vllm-switch/.venv/bin:$PATH"
export PYTHONPATH=/path/to/vllm-switch
export VLLM_SERVER_DEV_MODE=1
export no_proxy=127.0.0.1,localhost
export NO_PROXY=127.0.0.1,localhost

/path/to/vllm-switch/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$LLM_SWITCH_BENCH_MODEL_PATH" \
  --served-model-name "$LLM_SWITCH_BENCH_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 19700 \
  --enable-sleep-mode \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --dtype half \
  --enforce-eager &
server_pid=$!
trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

/path/to/vllm-switch/.venv/bin/python \
  -m llm_switch_bench.experiments.exact_disk.lifecycle_driver \
  --base-url http://127.0.0.1:19700 \
  --served-model-name "$LLM_SWITCH_BENCH_MODEL_NAME" \
  --ready-timeout-s 360 \
  --cycles 6

mkdir -p "$LLM_SWITCH_BENCH_OUT_DIR/runtime-bundle"
cp -a "$VLLM_EXACT_DISK_BACKUP_DIR"/. "$LLM_SWITCH_BENCH_OUT_DIR/runtime-bundle"/
```

Run the evidence owner. It exports output/profile/backup paths to the child, samples resources,
verifies the command, builds curated assertions, and writes an evidence manifest.

```bash
chmod +x "$RUN_ROOT/lifecycle-command.sh"

scripts/exact-disk.sh \
  --model qwen-0.5b="$MODEL" \
  --backup-root "$BACKUP_ROOT" \
  --out-dir "$RUN_ROOT/run" \
  --sample-interval-s 0.25 \
  -- "$RUN_ROOT/lifecycle-command.sh"
```

Require a zero exit code and `curated/assertions.json` with `ok: true`. Review run metadata,
profile phases, resource samples, output observation, evidence manifest, runtime manifest and
COMMIT, chunk layout, allocated blocks, imported path/commit, and all six cycles. Stop and
preserve the run as diagnostic evidence if direct I/O falls back, a hash differs, output
changes, or release remains pending. Confirm port/process/GPU cleanup.

## Update `results/`

Dry-run promotion first. The promoter independently verifies the evidence manifest and
curated assertions, authenticates the runtime manifest, hashes the omitted payload, removes
process-local pointers, rebuilds artifacts, and runs the family validator.

```bash
scripts/promote.sh exact-disk \
  --candidate-root "$RUN_ROOT/candidate-dry" \
  --collected-at YYYY-MM-DD \
  --run "$RUN_ROOT/run"
```

Review the candidate and then apply with a new root:

```bash
scripts/promote.sh exact-disk \
  --apply \
  --candidate-root "$RUN_ROOT/candidate-apply" \
  --collected-at YYYY-MM-DD \
  --run "$RUN_ROOT/run"

scripts/build_all.sh exact-disk
uv run python -m llm_switch_bench.validation.exact_disk.validate
git diff -- results/exact-disk
```

Repeat build/validation and require no second-pass diff. The previous family is preserved in
`$RUN_ROOT/candidate-apply/previous/`; the 1 GiB runtime payload remains outside Git.

## Threats and limitations

This is one model, host, GPU, filesystem, and prompt. Allocated filesystem blocks do not prove
media durability or cold-cache device reads. Page cache and filesystem behavior affect the
path. Direct I/O was requested, but the retained profile does not record the kernel-selected
I/O mode; `fallback=false` alone must not be presented as proof of direct I/O. One deterministic
output cannot cover every tensor mutation/invalidation path. The result establishes functional
spill, release, and restore, not crash consistency, endurance, steady-state latency,
throughput, or superiority over another tier.
